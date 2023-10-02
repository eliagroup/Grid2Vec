import json
import os

import grid2op
import numpy as np
from grid2op.Converter import IdToAct
from grid2op.Parameters import Parameters
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from grid2vec.actions import pad_out_like
from grid2vec.grid import Grid, split_substation_affinity
from grid2vec.grid2op_compat import (
    grid2op_action_dump_to_grid2elia,
    grid2op_action_to_grid2elia,
    grid2op_topo_vect_to_grid2elia,
    load_grid_grid2op,
)


def test_load_grid_grid2op(sandbox_grid: Grid) -> None:
    grid = sandbox_grid
    assert grid.net is not None
    assert len(grid.net.bus) == 28

    max_idx = (
        len(grid.net.line) * 2
        + len(grid.net.trafo) * 2
        + len(grid.net.load)
        + len(grid.net.gen)
        + len(grid.net.sgen)
    )
    assert grid.substation_affinity.shape == (max_idx, 2)
    assert np.all(grid.substation_affinity[:, 0] != grid.substation_affinity[:, 1])
    assert np.all(grid.substation_affinity >= 0)
    assert np.all(grid.substation_affinity < len(grid.net.bus))
    assert np.all(grid.topo_vect_controllable)
    assert grid.n_topo_vect_controllable == max_idx
    assert np.array_equal(grid.topo_vect_min, np.zeros(max_idx, dtype=int))
    assert np.array_equal(grid.topo_vect_max, np.ones(max_idx, dtype=int))


def test_grid2op_action_dump_to_grid2elia(sandbox_action_dump_file: str) -> None:
    env = grid2op.make("l2rpn_case14_sandbox")
    dump = grid2op_action_dump_to_grid2elia(sandbox_action_dump_file, env)

    assert len(dump.actions) == len(dump.exclusion_mask)
    # The diagonal is true, i.e. each element excludes itself
    # Only exception is the first action which is a do-nothing action
    assert np.all(np.diagonal(dump.exclusion_mask)[1:])
    # We have at least some actions that don't exclude each other
    assert not np.all(dump.exclusion_mask)
    # The exclusion mask is symmetric
    assert np.all(dump.exclusion_mask == dump.exclusion_mask.T)


@given(action=st.integers(min_value=0))
@settings(deadline=None, max_examples=25)
def test_grid2op_topo_vect_to_grid2elia(sandbox_grid: Grid, action: int) -> None:
    grid = sandbox_grid
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    g2o_env = grid2op.make("l2rpn_case14_sandbox", param=params)
    obs = g2o_env.reset()

    topo_vect = grid2op_topo_vect_to_grid2elia(obs.topo_vect, g2o_env)
    assert np.all(obs.topo_vect == 1)
    assert topo_vect.shape == (grid.n_topo_vect_controllable,)
    assert topo_vect.shape == (grid.len_topo_vect,)
    assert np.all(topo_vect == 0)

    # Use a converter to enumerate all actions
    converter = IdToAct(g2o_env.action_space)
    converter.init_converter(
        set_line_status=False,
        change_line_status=False,
        set_topo_vect=True,
        change_topo_vect=False,
        change_bus_vect=False,
        redispatch=False,
        curtail=False,
        storage=False,
    )
    action = action % converter.n
    g2o_act = converter.convert_act(action)

    obs, _, done, _ = g2o_env.step(g2o_act)
    assume(not done)
    assume(not np.any(obs.topo_vect == -1))

    topo_vect = grid2op_topo_vect_to_grid2elia(obs.topo_vect, g2o_env)

    bus_assignment = np.squeeze(
        np.take_along_axis(
            np.transpose(grid.substation_affinity), np.expand_dims(topo_vect, 0), axis=0
        )
    )
    topo_vect_split = split_substation_affinity(bus_assignment, grid.topo_vect_lookup)

    assert np.array_equal(
        g2o_env.backend._grid.line.from_bus.values,
        topo_vect_split[("line", "from_bus")],
    )
    assert np.array_equal(
        g2o_env.backend._grid.line.to_bus.values, topo_vect_split[("line", "to_bus")]
    )
    assert np.array_equal(
        g2o_env.backend._grid.trafo.hv_bus.values, topo_vect_split[("trafo", "hv_bus")]
    )
    assert np.array_equal(
        g2o_env.backend._grid.trafo.lv_bus.values, topo_vect_split[("trafo", "lv_bus")]
    )
    assert np.array_equal(
        g2o_env.backend._grid.gen.bus.values, topo_vect_split[("gen", "bus")]
    )
    assert np.array_equal(
        g2o_env.backend._grid.load.bus.values, topo_vect_split[("load", "bus")]
    )


@given(action=st.integers(min_value=0))
@settings(deadline=None, max_examples=25)
def test_grid2op_action_to_grid2elia_topology(action: int) -> None:
    grid = load_grid_grid2op(
        "l2rpn_case14_sandbox", include_chronic_indices=list(range(1))
    )
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    g2o_env = grid2op.make("l2rpn_case14_sandbox", param=params)

    converter = IdToAct(g2o_env.action_space)
    converter.init_converter(
        set_line_status=False,
        change_line_status=False,
        set_topo_vect=True,
        change_topo_vect=False,
        change_bus_vect=False,
        redispatch=False,
        curtail=False,
        storage=False,
    )
    action = action % converter.n
    g2o_act = converter.convert_act(action)

    obs = g2o_env.reset()
    obs, _, done, _ = g2o_env.step(g2o_act)
    assume(not done)
    assume(np.all(obs.line_status))

    elia_act = grid2op_action_to_grid2elia(g2o_act, g2o_env)

    if not np.any(g2o_act.set_bus != 0):
        assert elia_act.new_topo_vect.shape[1] == 0
    else:
        assert elia_act.new_topo_vect.shape[1] > 0
        expected_topo_vect = np.expand_dims(
            grid2op_topo_vect_to_grid2elia(obs.topo_vect, g2o_env), 0
        )

        assert np.array_equal(
            elia_act.new_topo_vect.new_state[elia_act.new_topo_vect.mask],
            expected_topo_vect[elia_act.new_topo_vect.mask],
        )
        assert np.any(elia_act.new_topo_vect.mask)
        assert elia_act.new_topo_vect.shape == (1, grid.n_topo_vect_controllable)


@given(action=st.integers(min_value=0))
@settings(deadline=None, max_examples=25)
def test_grid2op_action_to_grid2elia_set_line_status(action: int) -> None:
    grid = load_grid_grid2op(
        "l2rpn_case14_sandbox", include_chronic_indices=list(range(1))
    )
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    g2o_env = grid2op.make("l2rpn_case14_sandbox", param=params)

    converter = IdToAct(g2o_env.action_space)
    converter.init_converter(
        set_line_status=True,
        change_line_status=False,
        set_topo_vect=False,
        change_topo_vect=False,
        change_bus_vect=False,
        redispatch=False,
        curtail=False,
        storage=False,
    )
    action = action % converter.n
    g2o_act = converter.convert_act(action)

    obs = g2o_env.reset()
    expected_line_state = (obs + g2o_act).line_status

    obs, _, done, _ = g2o_env.step(g2o_act)
    assume(not done)
    assert np.array_equal(obs.line_status, expected_line_state)

    elia_act = grid2op_action_to_grid2elia(g2o_act, g2o_env)

    if not np.any(g2o_act.set_line_status != 0):
        assert elia_act.new_line_state.shape[1] == 0
    else:
        assert elia_act.new_line_state.shape == (1, grid.n_line_controllable)
        assert elia_act.new_trafo_state.shape == (1, grid.n_trafo_controllable)

        for idx, line_state in enumerate(g2o_act.set_line_status):
            if line_state == 0:
                if idx < grid.n_line_controllable:
                    assert elia_act.new_line_state.mask[0, idx].item() is False
                else:
                    idx -= grid.n_line_controllable
                    assert elia_act.new_trafo_state.mask[0, idx].item() is False
            else:
                if idx < grid.n_line_controllable:
                    assert elia_act.new_line_state.mask[0, idx].item() is True
                    assert elia_act.new_line_state.new_state[0, idx] == (
                        line_state == 1
                    )
                else:
                    idx -= grid.n_line_controllable
                    assert elia_act.new_trafo_state.mask[0, idx].item() is True
                    assert elia_act.new_trafo_state.new_state[0, idx] == (
                        line_state == 1
                    )

    dump_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "action_dumps", "sandbox", "all.json"
    )
    env = grid2op.make("l2rpn_case14_sandbox")
    dump = grid2op_action_dump_to_grid2elia(dump_file, env)

    with open(dump_file, "r") as f:
        dump_file_contents = json.load(f)

    assert dump.actions.n_envs == len(dump_file_contents)
    assert dump.actions.new_line_state.shape[1] > 0
    assert dump.actions.new_topo_vect.shape[1] > 0

    for idx, act_dict in enumerate(dump_file_contents):
        g2o_act = env.action_space(act_dict)
        elia_act = dump.actions[np.array([idx])]
        elia_act_ref = pad_out_like(grid2op_action_to_grid2elia(g2o_act, env), elia_act)
        assert elia_act == elia_act_ref
