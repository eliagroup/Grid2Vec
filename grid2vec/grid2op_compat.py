import json
import os
import warnings
from copy import deepcopy
from typing import List, Optional

import grid2op
import jax.numpy as jnp
import numpy as np
import pandas as pd
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.MakeEnv.PathUtils import DEFAULT_PATH_DATA
from jax_dataclasses import replace

from grid2vec.action_dump import ActionDump
from grid2vec.action_set import ActionSet, empty_action_set
from grid2vec.actions import (
    Action,
    assert_n_envs,
    concatenate_all_actions,
    do_nothing_action,
)
from grid2vec.grid import Grid, load_grid, validate_grid


def load_grid_grid2op(
    env_name: str, include_chronic_indices: Optional[List[int]] = None
) -> Grid:
    """Loads a grid2op scenario into a grid2elia Grid object

    Args:
        env_name (str): The grid2op environment name
        include_chronic_indices (Optional[List[int]], optional): Which chronics to include. This is
            an optimization to speed up loading of the grid in case you don't need all chronics.
            Defaults to all.

    Returns:
        Grid: The loaded grid.
    """
    # Make sure the scenario is downloaded by loading it
    g2o_env = grid2op.make(env_name)
    folder = os.path.join(DEFAULT_PATH_DATA, env_name)

    reorder_prods = None
    reorder_loads = None

    # No idea why grid2op reorders the generators, but we have to account for it
    if env_name == "l2rpn_case14_sandbox":
        reorder_prods = [1, 2, 3, 4, 5, 0]
    else:
        warnings.warn(f"Unknown scenario name {env_name}. No reordering applied.")

    grid = load_grid(
        folder,
        include_chronic_indices=include_chronic_indices,
        nminus1=False,
        reorder_prods=reorder_prods,
        reorder_loads=reorder_loads,
        dc=False,
    )

    # Duplicate busbars
    # Make substation affinity matrix
    substation_affinity = np.stack(
        [
            grid.substation_affinity[:, 0],
            grid.substation_affinity[:, 0] + len(grid.net.bus),
        ],
        axis=1,
    )

    # Copy the pandapower busbars
    net_copy = deepcopy(grid.net)
    bus_copy = net_copy.bus.copy()
    bus_copy.index = bus_copy.index + len(net_copy.bus)
    bus_copy.name = bus_copy.name.apply(lambda x: str(x) + "_copy")
    net_copy.bus = pd.concat([net_copy.bus, bus_copy], axis=0)

    # Set the thermal limits
    # From grid2op PandaPowerBackend:
    # self.thermal_limit_a = 1000. * np.concatenate(
    #     (
    #         self._grid.line["max_i_ka"].values,
    #         self._grid.trafo["sn_mva"].values
    #         / (np.sqrt(3) * self._grid.trafo["vn_hv_kv"].values),
    #     )
    # )
    # Unfortunately, setting sn_mva changes the whole load flow results as it fundamentally alters
    # the behaviour of the trafo, so we have to add a separate column grid2op_load_limit for the trafos
    # and use that instead in the powerflow computations.
    net_copy.line.max_i_ka = g2o_env.get_thermal_limit()[: grid.n_line] / 1000
    net_copy.trafo["grid2op_load_limit"] = (
        g2o_env.get_thermal_limit()[grid.n_line :] / 1000
    )

    grid = replace(
        grid,
        substation_affinity=jnp.array(substation_affinity),
        topo_vect_default=jnp.zeros_like(substation_affinity[:, 0]),
        net=net_copy,
    )
    validate_grid(grid)
    return grid


def import_grid2op_topo_vect(
    topo_vect: np.ndarray, g2o_env: BaseEnv, *, postprocess: bool = True
) -> np.ndarray:
    """Converts a grid2op topo vect to a grid2vec topo vect

    Args:
        topo_vect (np.ndarray): The topo vect in grid2op format
        g2o_env (BaseEnv): The grid2op environment
        postprocess (bool, optional): Whether to postprocess the topo vect. Defaults to True.

    Returns:
        np.ndarray: A grid2vec compatible topo_vect. Note that the grid2vec environment must
            use the same scenario, otherwise this doesn't make sense
    """
    n_line = len(g2o_env.backend._grid.line)
    n_trafo = len(g2o_env.backend._grid.trafo)
    assert n_line + n_trafo == g2o_env.n_line
    assert len(g2o_env.backend._grid.trafo3w) == 0
    assert len(g2o_env.backend._grid.sgen) == 0

    ret_topo_vect = np.concatenate(
        [
            topo_vect[g2o_env.line_or_pos_topo_vect[:n_line]],  # Line from_bus
            topo_vect[g2o_env.line_ex_pos_topo_vect[:n_line]],  # line to_bus
            topo_vect[g2o_env.line_or_pos_topo_vect[n_line:]],  # trafo hv_bus
            topo_vect[g2o_env.line_ex_pos_topo_vect[n_line:]],  # trafo lv_bus
            topo_vect[g2o_env.gen_pos_topo_vect],  # gen bus
            topo_vect[g2o_env.load_pos_topo_vect],
        ]
    )

    if postprocess:
        # Connect all disconnected elements to bus 1
        ret_topo_vect[ret_topo_vect == -1] = 1

        # Grid2op starts counting at 1, grid2elia at 0
        ret_topo_vect -= 1

    return ret_topo_vect


def import_grid2op_action(action: BaseAction, env: BaseEnv) -> Action:
    retval = do_nothing_action(n_envs=1)

    n_line_controllable = len(env.backend._grid.line)
    n_trafo_controllable = len(env.backend._grid.trafo)

    if np.any(action.set_line_status != 0):
        retval = replace(
            retval,
            new_line_state=empty_action_set((1, n_line_controllable), dtype=bool),
            new_trafo_state=empty_action_set((1, n_trafo_controllable), dtype=bool),
        )
        for idx, state in enumerate(action.set_line_status):
            if state == 0:
                continue
            if idx < n_line_controllable:
                assert retval.new_line_state is not None
                retval = replace(
                    retval,
                    new_line_state=ActionSet(
                        new_state=retval.new_line_state.new_state.at[0, idx].set(
                            state == 1
                        ),
                        mask=retval.new_line_state.mask.at[0, idx].set(True),
                    ),
                )
            else:
                idx -= n_line_controllable
                assert retval.new_trafo_state is not None
                retval = replace(
                    retval,
                    new_trafo_state=ActionSet(
                        new_state=retval.new_trafo_state.new_state.at[0, idx].set(
                            state == 1
                        ),
                        mask=retval.new_trafo_state.mask.at[0, idx].set(True),
                    ),
                )

    if np.any(action.set_bus != 0):
        if np.any(action.set_bus) == -1:
            raise NotImplementedError(
                "Disconnecting elements through topo vect not implemented"
            )

        topo_vect = np.expand_dims(
            import_grid2op_topo_vect(action.set_bus, env, postprocess=False),
            axis=0,
        )
        retval = replace(retval, new_topo_vect=ActionSet(topo_vect - 1, topo_vect != 0))

    # Check if the grid2op action had an impact that we can't model with grid2elia
    impact = action.impact_on_objects()
    if impact["injection"]["changed"] is True:
        raise NotImplementedError("Redispatch injections not implemented")
    if impact["redispatch"]["changed"] is True:
        raise NotImplementedError("Redispatch not implemented")
    if impact["storage"]["changed"] is True:
        raise NotImplementedError("Storage not implemented")
    if impact["curtailment"]["changed"] is True:
        raise NotImplementedError("Curtailment not implemented")
    if impact["switch_line"]["changed"] is True:
        raise NotImplementedError(
            "Switching lines not implemented, use set_line_status instead"
        )
    if len(impact["topology"]["bus_switch"]):
        raise NotImplementedError(
            "Change bus actions not implemented, use set_bus instead"
        )

    assert_n_envs(retval)

    return retval


def enumerate_all_actions(env: BaseEnv) -> List[dict]:
    """Enumerates all possible actions for a given grid2op environment.

    Args:
        env (BaseEnv): The grid2op environment to enumerate actions for.

    Returns:
        List[dict]: A list of all possible actions in serializable dict format
    """
    converter = IdToAct(env.action_space)
    converter.init_converter(
        set_line_status=True,
        change_line_status=False,
        set_topo_vect=True,
        change_topo_vect=False,
        change_bus_vect=False,
        redispatch=False,
        curtail=False,
        storage=False,
    )

    mapping = []
    for i in range(converter.n):
        action = converter.convert_act(i).as_serializable_dict()
        # Fixes a grid2op issue 527 action space sampling
        if len(action) == 1 and all([v == {} for v in action.values()]):
            action = {}
        mapping.append(action)
    return mapping


def load_grid2op_action_dump_from_file(dump_file: str, env: BaseEnv) -> ActionDump:
    """Loads a grid2op action dump from file and converts it to grid2vec format

    Args:
        dump_file (str): The file to load
        env (BaseEnv): The grid2op environment to convert the dump for

    Returns:
        ActionDump: A grid2vec action dump
    """
    with open(dump_file, "r") as f:
        dump = json.load(f)

    return import_grid2op_action_dump(dump, env)


def all_actions_action_dump(env: BaseEnv) -> ActionDump:
    """Enumerates all actions for a grid2op environment and makes an action dump from them

    Args:
        env (BaseEnv): The grid2op environment to enumerate actions for

    Returns:
        ActionDump: The resulting grid2vec action dump
    """
    return import_grid2op_action_dump(enumerate_all_actions(env), env)


def import_grid2op_action_dump(dump: List[dict], env: BaseEnv) -> ActionDump:
    """Converts a grid2op action dump to a grid2elia action dump

    Args:
        dump (List[dict]): The action dump as a list of dicts, obtained with
            Action::as_serializable_dict
        env (BaseEnv): The grid2op environment for which the action dump is intended

    Returns:
        ActionDump: The converted action dump
    """
    actions = []
    effect_on_substations = np.zeros((len(dump), env.n_sub), dtype=bool)
    effect_on_lines = np.zeros((len(dump), env.n_line), dtype=bool)
    for idx, action in enumerate(dump):
        g2o_act = env.action_space(action)
        line_impact, sub_impact = g2o_act.get_topological_impact()
        effect_on_substations[idx, :] = sub_impact
        effect_on_lines[idx, :] = line_impact
        actions.append(import_grid2op_action(g2o_act, env))

    # Fill the exclusion mask so that all actions with effect on the same substations are
    # masked out
    exclusion_mask = np.zeros((len(dump), len(dump)), dtype=bool)
    for idx in range(len(dump)):
        # Find out which other actions affect the same substation
        same_substation_affected = np.any(
            effect_on_substations[idx, :] & effect_on_substations, axis=1
        )
        exclusion_mask[idx, same_substation_affected] = True
        exclusion_mask[same_substation_affected, idx] = True

        # Find out which other actions affect the same lines
        same_lines_affected = np.any(effect_on_lines[idx, :] & effect_on_lines, axis=1)
        exclusion_mask[idx, same_lines_affected] = True
        exclusion_mask[same_lines_affected, idx] = True

    return ActionDump(
        actions=concatenate_all_actions(actions), exclusion_mask=exclusion_mask
    )
