import numpy as np
import pytest

from grid2vec.action_set import ActionSet, action_set_unmasked
from grid2vec.actions import (
    assert_n_envs,
    concatenate_actions,
    concatenate_all_actions,
    do_nothing_action,
    is_do_nothing_action,
    make_action,
    merge_actions,
    pad_and_concatenate_actions,
    pad_out_like,
    random_action,
    replicate_action,
    split_action,
    vector_step_action,
)
from grid2vec.env import make_env, vector_reset
from grid2vec.grid import Grid


def test_vector_step_action(grid: Grid) -> None:
    env = make_env(grid, 2)
    env = vector_reset(env)

    action = random_action(grid, 2)
    assert action.n_envs == 2
    assert_n_envs(action)

    assert action.new_switch_state is not None
    assert action.new_line_state is not None
    assert action.new_trafo_taps is not None
    assert action.new_trafo3w_taps is not None
    assert action.new_topo_vect is not None

    assert action.n_envs == 2
    env = vector_step_action(env, action)

    assert np.array_equal(env.switch_state, action.new_switch_state.new_state)
    assert np.array_equal(env.line_state, action.new_line_state.new_state)
    assert np.array_equal(env.trafo_tap_pos, action.new_trafo_taps.new_state)
    assert np.array_equal(env.trafo3w_tap_pos, action.new_trafo3w_taps.new_state)
    assert np.array_equal(env.topo_vect, action.new_topo_vect.new_state)


def test_merge_actions() -> None:
    act_a = make_action(
        new_switch_state=ActionSet(
            np.array([[True, False], [False, True]]),
            np.array([[True, True], [False, False]]),
        )
    )

    act_b = make_action(
        new_switch_state=ActionSet(
            np.array([[False, True], [True, False]]),
            np.array([[False, False], [True, True]]),
        )
    )

    act = merge_actions(act_a, act_b)
    assert act.n_envs == 2
    assert_n_envs(act)
    assert act.new_switch_state is not None
    assert np.array_equal(
        act.new_switch_state.new_state, np.array([[True, False], [True, False]])
    )
    assert np.array_equal(
        act.new_switch_state.mask, np.array([[True, True], [True, True]])
    )

    act = merge_actions(act_b, act_a)
    assert act.new_switch_state is not None
    assert np.array_equal(
        act.new_switch_state.new_state, np.array([[True, False], [True, False]])
    )
    assert np.array_equal(
        act.new_switch_state.mask, np.array([[True, True], [True, True]])
    )

    act_a = make_action(
        new_line_state=action_set_unmasked(np.array([[True, False], [False, True]]))
    )
    act_b = make_action(
        new_line_state=ActionSet(
            np.array([[False, True], [True, False]]),
            np.array([[False, False], [True, True]]),
        ),
    )
    act = merge_actions(act_a, act_b)
    assert act.new_line_state is not None
    assert np.array_equal(
        act.new_line_state.new_state, np.array([[True, False], [True, False]])
    )
    assert np.all(act.new_line_state.mask)

    act = merge_actions(act_b, act_a)
    assert act.new_line_state is not None
    assert np.array_equal(
        act.new_line_state.new_state, np.array([[True, False], [False, True]])
    )
    assert np.all(act.new_line_state.mask)

    act_a = make_action(
        new_switch_state=action_set_unmasked(np.array([[True, False], [False, True]]))
    )
    act_b = make_action(
        new_line_state=action_set_unmasked(np.array([[False, True], [True, False]]))
    )
    with pytest.raises(ValueError):
        act = merge_actions(act_a, act_b)

    act = merge_actions(pad_out_like(act_a, act_b), pad_out_like(act_b, act_a))

    assert act.new_switch_state is not None
    assert act.new_line_state is not None
    assert np.array_equal(
        act.new_switch_state.new_state, np.array([[True, False], [False, True]])
    )
    assert np.array_equal(
        act.new_line_state.new_state, np.array([[False, True], [True, False]])
    )

    act_a = do_nothing_action(act_b.n_envs)
    with pytest.raises(ValueError):
        merge_actions(act_a, act_b)

    act = merge_actions(pad_out_like(act_a, act_b), act_b)
    assert len(act) == len(act_a)


def test_do_nothing_action() -> None:
    act = do_nothing_action(1)
    assert len(act) == 1
    assert act.new_switch_state.shape[1] == 0
    assert act.new_line_state.shape[1] == 0
    assert act.new_trafo_taps.shape[1] == 0
    assert act.new_trafo3w_taps.shape[1] == 0
    assert act.new_topo_vect.shape[1] == 0
    assert is_do_nothing_action(act)

    act = do_nothing_action(100)
    assert len(act) == 100


def test_pad_out_like() -> None:
    a = do_nothing_action(1)

    b = make_action(
        new_switch_state=ActionSet(
            np.array([[True, False], [False, True]]),
            np.array([[True, True], [False, False]]),
        )
    )
    assert not is_do_nothing_action(b)

    c = pad_out_like(a, b)
    assert len(c) == 1
    assert c.new_switch_state.shape == (1, 2)

    c = pad_out_like(b, b)
    assert c == b

    c = pad_out_like(a, a)
    assert c == a


def test_concatenate_actions() -> None:
    act_a = make_action(
        new_switch_state=ActionSet(
            np.array([[True, False], [False, True]]),
            np.array([[True, True], [False, False]]),
        )
    )

    act_b = make_action(
        new_switch_state=ActionSet(
            np.array([[False, True], [True, False]]),
            np.array([[False, False], [True, True]]),
        )
    )

    act_c = make_action(
        new_switch_state=ActionSet(
            np.array([[False, True], [True, False]]),
            np.array([[False, False], [True, False]]),
        )
    )

    act = concatenate_actions(act_a, act_b)
    assert act == pad_and_concatenate_actions(act_a, act_b)
    assert len(act) == 4
    assert_n_envs(act)

    assert act.new_switch_state is not None
    assert act_a.new_switch_state is not None
    assert act_b.new_switch_state is not None
    assert np.array_equal(
        act.new_switch_state.new_state,
        np.concatenate(
            [act_a.new_switch_state.new_state, act_b.new_switch_state.new_state], axis=0
        ),
    )
    assert np.array_equal(
        act.new_switch_state.mask,
        np.concatenate(
            [act_a.new_switch_state.mask, act_b.new_switch_state.mask], axis=0
        ),
    )

    act1 = concatenate_actions(act, act_c)
    act2 = concatenate_all_actions([act_a, act_b, act_c])
    act3 = pad_and_concatenate_actions(act, act_c)
    assert act1 == act2
    assert act1 == act3

    act = act1
    act_with_donothing = concatenate_actions(
        pad_out_like(do_nothing_action(1), act), act
    )
    assert len(act_with_donothing) == len(act) + 1

    assert act_with_donothing == pad_and_concatenate_actions(do_nothing_action(1), act)


def test_replicate_action() -> None:
    act = make_action(
        new_switch_state=ActionSet(
            np.array([[True, False], [False, True]]),
            np.array([[True, True], [False, False]]),
        )
    )

    res = replicate_action(act, 3)
    assert res.new_switch_state is not None
    assert res.new_line_state.shape == (6, 0)
    assert np.array_equal(
        res.new_switch_state.new_state,
        np.array(
            [
                [True, False],
                [False, True],
                [True, False],
                [False, True],
                [True, False],
                [False, True],
            ]
        ),
    )
    assert np.array_equal(
        res.new_switch_state.mask,
        np.array(
            [
                [True, True],
                [False, False],
                [True, True],
                [False, False],
                [True, True],
                [False, False],
            ]
        ),
    )


def test_split_actions(grid: Grid) -> None:
    action = random_action(grid, 10)
    assert action.new_line_state is not None
    split = split_action(action, 5)
    assert len(split) == 5
    for act in split:
        assert act.new_line_state is not None
        assert act.new_line_state.shape == (2, action.new_line_state.shape[1])
