import json

import jax.numpy as jnp
import numpy as np
from gymnasium.spaces.discrete import Discrete
from jax_dataclasses import pytree_dataclass

from grid2vec.action_set import ActionSet
from grid2vec.actions import (
    Action,
    do_nothing_action,
    make_action,
    pad_and_concatenate_actions,
)
from grid2vec.grid import Grid
from grid2vec.util import freeze_array


@pytree_dataclass
class ActionDump:
    actions: Action  # The action dumps, where the env dimension in the actions is the number of different actions available
    exclusion_mask: jnp.ndarray  # [bool] (n_actions, n_actions) A mask that indicates which actions are excluding which other actions, typically all from the same substation exclude each other

    def __len__(self):
        return len(self.actions)


def validate_action_dump(data: list, grid: Grid) -> None:
    """Validates the action dump file

    Raises a ValueError in case of inconsistencies in the action dump

    Args:
        data (list): The loaded action dump
        grid (Grid): The grid to validate against
    """
    for act_id, action in enumerate(data):
        if "set_switch" in action:
            if not isinstance(action["set_switch"], dict):
                raise ValueError(
                    f"Invalid action dump format, set_switch of action {act_id} not a dict"
                )

            for switch_id, switch_val in action["set_switch"].items():
                switch_id = int(switch_id)
                switch_val = bool(switch_val)
                if switch_id < 0 or switch_id > grid.n_switch_controllable:
                    raise ValueError(
                        f"Action {act_id}: Invalid switch index: {switch_id}"
                    )

        if "set_line" in action:
            if not isinstance(action["set_line"], dict):
                raise ValueError(
                    f"Invalid action dump format, set_line of action {act_id} not a dict"
                )

            for line_id, line_val in action["set_line"].items():
                line_id = int(line_id)
                line_val = bool(line_val)
                if line_id < 0 or line_id > grid.n_line_controllable:
                    raise ValueError(f"Action {act_id}: Invalid line index: {line_id}")

        if "set_trafo" in action:
            if not isinstance(action["set_trafo"], dict):
                raise ValueError(
                    f"Invalid action dump format, set_trafo of action {act_id} not a dict"
                )

            for trafo_id, trafo_val in action["set_trafo"].items():
                trafo_id = int(trafo_id)
                trafo_val = bool(trafo_val)
                if trafo_id < 0 or trafo_id > grid.n_trafo_controllable:
                    raise ValueError(
                        f"Action {act_id}: Invalid trafo index: {trafo_id}"
                    )

        if "set_trafo_tap" in action:
            if not isinstance(action["set_trafo_tap"], dict):
                raise ValueError(
                    f"Invalid action dump format, set_trafo_tap of action {act_id} not a dict"
                )

            for trafo_id, trafo_val in action["set_trafo_tap"].items():
                trafo_id = int(trafo_id)
                trafo_val = int(trafo_val)
                if trafo_id < 0 or trafo_id > grid.n_trafo_tap_controllable:
                    raise ValueError(
                        f"Action {act_id}: Invalid trafo index: {trafo_id}"
                    )
                if (
                    trafo_val < grid.trafo_tap_min[trafo_id]
                    or trafo_val > grid.trafo_tap_max[trafo_id]
                ):
                    raise ValueError(
                        f"Action {act_id}: Invalid trafo tap position: {trafo_val} for trafo {trafo_id}"
                    )

        if "set_trafo3w_tap" in action:
            if not isinstance(action["set_trafo3w_tap"], dict):
                raise ValueError(
                    f"Invalid action dump format, set_trafo3w_tap of action {act_id} not a dict"
                )

            for trafo3w_id, trafo3w_val in action["set_trafo3w_tap"].items():
                trafo3w_id = int(trafo3w_id)
                trafo3w_val = int(trafo3w_val)
                if trafo3w_id < 0 or trafo3w_id > grid.n_trafo3w_tap_controllable:
                    raise ValueError(
                        f"Action {act_id}: Invalid trafo3w index: {trafo3w_id}"
                    )
                if (
                    trafo3w_val < grid.trafo3w_tap_min[trafo3w_id]
                    or trafo_val > grid.trafo3w_tap_max[trafo3w_id]
                ):
                    raise ValueError(
                        f"Action {act_id}: Invalid trafo3w tap position: {trafo3w_val} for trafo {trafo3w_id}"
                    )

        if "set_topo_vect" in action:
            if not isinstance(action["set_topo_vect"], dict):
                raise ValueError(
                    f"Invalid action dump format, set_topo_vect of action {act_id} not a dict"
                )

            for topo_id, topo_val in action["set_topo_vect"].items():
                topo_id = int(topo_id)
                topo_val = int(topo_val)
                if topo_id < 0 or topo_id > grid.n_topo_vect_controllable:
                    raise ValueError(f"Action {act_id}: Invalid topo index: {topo_id}")
                if (
                    topo_val < grid.topo_vect_min[topo_id]
                    or topo_val > grid.topo_vect_max[topo_id]
                ):
                    raise ValueError(
                        f"Action {act_id}: Topo value out of bounds: {topo_val} for id {topo_id}"
                    )

        if "exclude" in action:
            if not isinstance(action["exclude"], list):
                raise ValueError(
                    f"Invalid action dump format, exclude of action {act_id} not a list"
                )
            for exclusion in action["exclude"]:
                if not isinstance(exclusion, int):
                    raise ValueError(
                        f"Invalid action dump format, exclude of action {act_id} not a list of ints"
                    )
                if exclusion < 0 or exclusion >= len(data):
                    raise ValueError(
                        f"Invalid action dump format, exclude of action {act_id} contains invalid exclusion id {exclusion}"
                    )


def load_action_dump(filename: str, grid: Grid) -> ActionDump:
    """Loads an action dump from a file.

    The file is expected to be a json file with the following structure:

    ```
    [
      {
        "set_switch": {<id>: <0/1>, ...},
        "set_line": {<id>: <0/1>, ...},
        "set_trafo": {<id>: <0/1>, ...},
        "set_trafo_tap": {<id>: <tap pos>, ...},
        "set_trafo3w_tap": {<id>: <tap pos>, ...},
        "set_topo_vect": {<id>: <bus>, ...}
        "exclude": [<exclusion1>, ...]
      },
      ...
    ]
    ```

    Each id in the file should reference the index in the grid.<element>_controllable array. IDs
    that are not set will remain unchanged.

    Args:
        filename (str): The filename to load
        grid (Grid): The grid to use for validation

    Raises:
        ValueError: Invalid action dump format

    Returns:
        ActionDump: An actiondump dataclass with all possible actions in the env dimension
    """
    with open(filename, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Invalid action dump format, not a list!")

    validate_action_dump(data, grid)

    has_switch = any([("set_switch" in x) for x in data])
    has_line = any([("set_line" in x) for x in data])
    has_trafo = any([("set_trafo" in x) for x in data])
    has_trafo_tap = any([("set_trafo_tap" in x) for x in data])
    has_trafo3w_tap = any([("set_trafo3w_tap" in x) for x in data])
    has_topo = any([("set_topo_vect" in x) for x in data])
    has_exclusion = any([("exclude" in x) for x in data])
    size = len(data)

    switch_set = None
    if has_switch:
        switch_actions = np.zeros((size, grid.n_switch_controllable), dtype=bool)
        switch_mask = np.zeros((size, grid.n_switch_controllable), dtype=bool)
        for act_id, action in enumerate(data):
            if "set_switch" in action:
                for switch_id, switch_val in action["set_switch"].items():
                    switch_mask[act_id, int(switch_id)] = True
                    switch_actions[act_id, int(switch_id)] = bool(switch_val)
        switch_set = ActionSet(freeze_array(switch_actions), freeze_array(switch_mask))

    line_set = None
    if has_line:
        line_actions = np.zeros((size, grid.n_line_controllable), dtype=bool)
        line_mask = np.zeros((size, grid.n_line_controllable), dtype=bool)
        for act_id, action in enumerate(data):
            if "set_line" in action:
                for line_id, line_val in action["set_line"].items():
                    line_mask[act_id, int(line_id)] = True
                    line_actions[act_id, int(line_id)] = bool(line_val)
        line_set = ActionSet(freeze_array(line_actions), freeze_array(line_mask))

    trafo_set = None
    if has_trafo:
        trafo_actions = np.zeros((size, grid.n_trafo_controllable), dtype=bool)
        trafo_mask = np.zeros((size, grid.n_trafo_controllable), dtype=bool)
        for act_id, action in enumerate(data):
            if "set_trafo" in action:
                for trafo_id, trafo_val in action["set_trafo"].items():
                    trafo_mask[act_id, int(trafo_id)] = True
                    trafo_actions[act_id, int(trafo_id)] = bool(trafo_val)
        trafo_set = ActionSet(freeze_array(trafo_actions), freeze_array(trafo_mask))

    trafo_tap_set = None
    if has_trafo_tap:
        trafo_actions = np.zeros((size, grid.n_trafo_tap_controllable), dtype=int)
        trafo_tap_mask = np.zeros((size, grid.n_trafo_tap_controllable), dtype=bool)
        for act_id, action in enumerate(data):
            if "set_trafo_tap" in action:
                for trafo_id, trafo_val in action["set_trafo_tap"].items():
                    trafo_tap_mask[act_id, int(trafo_id)] = True
                    trafo_actions[act_id, int(trafo_id)] = int(trafo_val)
        trafo_tap_set = ActionSet(
            freeze_array(trafo_actions), freeze_array(trafo_tap_mask)
        )

    trafo3w_tap_set = None
    if has_trafo3w_tap:
        trafo3w_actions = np.zeros((size, grid.n_trafo3w_tap_controllable), dtype=int)
        trafo3w_tap_mask = np.zeros((size, grid.n_trafo3w_tap_controllable), dtype=bool)
        for act_id, action in enumerate(data):
            if "set_trafo3w_tap" in action:
                for trafo3w_id, trafo3w_val in action["set_trafo3w_tap"].items():
                    trafo3w_tap_mask[act_id, int(trafo3w_id)] = True
                    trafo3w_actions[act_id, int(trafo3w_id)] = int(trafo3w_val)
        trafo3w_tap_set = ActionSet(
            freeze_array(trafo3w_actions), freeze_array(trafo3w_tap_mask)
        )

    topo_set = None
    if has_topo:
        topo_actions = np.zeros((size, grid.n_topo_vect_controllable), dtype=int)
        topo_mask = np.zeros((size, grid.n_topo_vect_controllable), dtype=bool)
        for act_id, action in enumerate(data):
            if "set_topo" in action:
                for topo_id, topo_val in action["set_topo"].items():
                    topo_mask[act_id, int(topo_id)] = True
                    topo_actions[act_id, int(topo_id)] = int(topo_val)
        topo_set = ActionSet(freeze_array(topo_actions), freeze_array(topo_mask))

    exclusion_mask = jnp.eye(size, dtype=bool)
    if has_exclusion:
        exclusion_mask = np.zeros((size, size), dtype=bool)
        for act_id, action in enumerate(data):
            if "exclude" in action:
                for exclusion in action["exclude"]:
                    exclusion_mask[act_id, exclusion] = True
                    # We only store the exclusions unidirectionally
                    # If symmetricallity is required, the action dump should contain both directions
                    # exclusion_mask[exclusion, act_id] = True
        exclusion_mask = jnp.array(exclusion_mask)

    action = make_action(
        new_switch_state=switch_set,
        new_line_state=line_set,
        new_trafo_state=trafo_set,
        new_trafo_taps=trafo_tap_set,
        new_trafo3w_taps=trafo3w_tap_set,
        new_topo_vect=topo_set,
    )
    return ActionDump(
        actions=action,
        exclusion_mask=exclusion_mask,
    )


def add_do_nothing_action(action_dump: ActionDump) -> ActionDump:
    """Adds the do nothing action to the beginning of the action dump

    Args:
        action_dump (ActionDump): The action dump to add the do-nothing-action to

    Returns:
        ActionDump: An action dump with the do-nothing-action added in the first place
    """
    new_action = pad_and_concatenate_actions(do_nothing_action(1), action_dump.actions)
    new_exclusion_mask = (
        jnp.zeros((len(new_action), len(new_action)), dtype=bool)
        .at[1:, 1:]
        .set(action_dump.exclusion_mask)
    )
    return ActionDump(
        actions=new_action,
        exclusion_mask=new_exclusion_mask,
    )


def get_action_space(action_dump: ActionDump) -> Discrete:
    """Returns a gymnasium action space description

    Args:
        action_dump (Action): The action to describe, where the env dimension is interpreted
            as the number of different actions available

    Returns:
        Discrete: A discrete action space with n = get_n_envs(action_dump)
    """
    return Discrete(n=len(action_dump))
