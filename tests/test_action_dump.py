import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from grid2vec.action_dump import (
    add_do_nothing_action,
    get_action_space,
    load_action_dump,
)
from grid2vec.action_set import ActionSet
from grid2vec.grid import Grid


@pytest.fixture
def action_dump_file(tmpdir: str) -> str:
    data = [
        {
            "exclude": [2],
        },
        {
            "set_switch": {
                "0": True,
                "3": False,
            },
            "set_line": {
                "1": False,
                "2": False,
                "3": True,
            },
            "set_trafo": {
                "1": False,
            },
            "set_trafo_tap": {
                "0": 4,
                "1": 8,
            },
            "exclude": [1],
        },
        {
            "set_switch": {
                "1": True,
                "3": True,
            },
            "set_line": {
                "1": True,
                "2": False,
                "3": True,
            },
            "set_trafo_tap": {
                "0": 3,
                "1": -8,
            },
            "exclude": [2],
        },
    ]

    filename = os.path.join(tmpdir, "dump.json")
    with open(filename, "w") as f:
        json.dump(data, f)

    return filename


def test_load_action_dump(grid: Grid, action_dump_file: str) -> None:
    dump = load_action_dump(action_dump_file, grid)

    assert dump.actions.n_envs == 3

    assert dump.actions.new_switch_state is not None
    assert dump.actions.new_line_state is not None
    assert dump.actions.new_trafo_taps is not None
    assert dump.actions.new_trafo3w_taps is not None

    assert (
        dump.actions.new_switch_state.shape == dump.actions.new_switch_state.mask.shape
    )
    assert dump.actions.new_switch_state.shape == (3, grid.n_switch_controllable)

    assert dump.actions.new_line_state.shape == dump.actions.new_line_state.mask.shape
    assert dump.actions.new_line_state.shape == (3, grid.n_line_controllable)

    assert dump.actions.new_trafo_taps.shape == dump.actions.new_trafo_taps.mask.shape
    assert dump.actions.new_trafo_taps.shape == (3, grid.n_trafo_tap_controllable)

    assert (
        dump.actions.new_trafo3w_taps.shape == dump.actions.new_trafo3w_taps.mask.shape
    )
    assert dump.actions.new_trafo3w_taps.shape == (3, grid.n_trafo3w_tap_controllable)

    assert dump.exclusion_mask.shape == (len(dump.actions), len(dump.actions))
    assert jnp.array_equal(
        dump.exclusion_mask,
        jnp.array([[False, False, True], [False, True, False], [False, False, True]]),
    )
    assert len(dump) == len(dump.actions)


def test_get_action_space(grid: Grid, action_dump_file: str) -> None:
    dump = load_action_dump(action_dump_file, grid)

    action_space = get_action_space(dump)
    assert action_space.n == dump.actions.n_envs


def test_add_do_nothing_action(grid: Grid, action_dump_file: str) -> None:
    dump = load_action_dump(action_dump_file, grid)
    dump2 = add_do_nothing_action(dump)
    assert len(dump) == len(dump2) - 1
    assert dump2.exclusion_mask.shape == (len(dump2), len(dump2))

    assert jnp.all(dump2.exclusion_mask[:, 0] == 0)
    assert jnp.all(dump2.exclusion_mask[0, :] == 0)
    assert jnp.all(dump2.exclusion_mask[1:, 1:] == dump.exclusion_mask)


def test_convert_action(grid: Grid, action_dump_file: str) -> None:
    dump = load_action_dump(action_dump_file, grid)

    action = np.array([0, 1, 2])
    action_converted = dump.actions[action]

    assert action_converted.new_switch_state is not None
    assert action_converted.new_line_state is not None
    assert action_converted.new_trafo_taps is not None
    assert action_converted.new_trafo3w_taps is not None

    assert isinstance(action_converted.new_switch_state, ActionSet)
    assert isinstance(action_converted.new_line_state, ActionSet)
    assert isinstance(action_converted.new_trafo_taps, ActionSet)
    assert isinstance(action_converted.new_trafo3w_taps, ActionSet)

    assert action_converted.new_topo_vect.shape[1] == 0

    assert action_converted.new_switch_state.shape == (
        3,
        grid.n_switch_controllable,
    )
    assert action_converted.new_line_state.shape == (
        3,
        grid.n_line_controllable,
    )
    assert action_converted.new_trafo_taps.shape == (
        3,
        grid.n_trafo_tap_controllable,
    )
    assert action_converted.new_trafo3w_taps.shape == (
        3,
        grid.n_trafo3w_tap_controllable,
    )

    assert not np.any(action_converted.new_switch_state[0].mask)
    assert np.any(action_converted.new_switch_state[1].mask)
    assert np.any(action_converted.new_switch_state[2].mask)
