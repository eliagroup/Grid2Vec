import jax.numpy as jnp
import numpy as np
import pytest

from grid2vec.action_set import (
    ActionSet,
    action_set_unmasked,
    concatenate_action_sets,
    empty_action_set,
    eye_mask_action_set,
    is_do_nothing_set,
    merge_action_sets,
    replicate_action_set,
    split_action_set,
)


def test_action_set() -> None:
    a = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
    )
    assert a.shape == (2, 10)
    assert a.new_state.shape == (2, 10)
    assert a.mask.shape == (2, 10)
    assert a.dtype == bool

    assert a[0].shape == (10,)
    assert a[0:1].shape == (1, 10)
    assert a[jnp.array([True, False])].shape == (1, 10)
    assert a[jnp.array([True, False]), :].shape == (1, 10)
    assert a[jnp.array([1, 0])].shape == (2, 10)

    assert not is_do_nothing_set(a)

    assert a == a

    b = ActionSet(
        new_state=jnp.array(np.random.randint(0, 5, size=(2, 10), dtype=int)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
    )
    assert a != b
    assert b != a

    a = empty_action_set((2, 0), bool)
    assert a == a
    assert is_do_nothing_set(a)


def test_concatenate_actions() -> None:
    a = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
    )
    b = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
    )
    c = concatenate_action_sets(a, b)
    assert c.shape == (4, 10)
    assert jnp.all(c.new_state[:2] == a.new_state)
    assert jnp.all(c.mask[:2] == a.mask)
    assert jnp.all(c.new_state[2:] == b.new_state)
    assert jnp.all(c.mask[2:] == b.mask)

    b = empty_action_set((2, 0))
    with pytest.raises(AssertionError):
        c = concatenate_action_sets(a, b)
    b = empty_action_set((2, 10))
    c = concatenate_action_sets(a, b)
    assert c.shape == (4, 10)
    assert jnp.all(c.new_state[2:] == 0)
    assert jnp.all(c.mask[2:] == 0)
    assert jnp.all(c.new_state[:2] == a.new_state)
    assert jnp.all(c.mask[:2] == a.mask)

    c = concatenate_action_sets(b, a)
    assert c.shape == (4, 10)
    assert jnp.all(c.new_state[:2] == 0)
    assert jnp.all(c.mask[:2] == 0)
    assert jnp.all(c.new_state[2:] == a.new_state)
    assert jnp.all(c.mask[2:] == a.mask)

    c = concatenate_action_sets(b, b)
    assert c == empty_action_set((4, 10))

    b = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(2, 11), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 11), dtype=bool)),
    )

    with pytest.raises(AssertionError):
        concatenate_action_sets(a, b)

    b = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(5, 10), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(5, 10), dtype=bool)),
    )
    c = concatenate_action_sets(a, b)
    assert c.shape == (7, 10)


def test_action_set_unmasked() -> None:
    state = jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool))
    act = action_set_unmasked(state)
    assert act.shape == (2, 10)
    assert np.array_equal(act.mask, np.ones((2, 10), dtype=bool))


def test_merge_action_sets() -> None:
    a = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
    )
    b = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
    )
    assert not a == b
    merged = merge_action_sets(a, b)
    assert merged is not None
    assert merged.shape == (2, 10)
    assert jnp.array_equal(merged.new_state[b.mask], b.new_state[b.mask])
    a_mask = jnp.logical_and(a.mask, ~b.mask)
    assert jnp.array_equal(merged.new_state[a_mask], a.new_state[a_mask])

    null_set = empty_action_set((2, 0), dtype=bool)

    merged = merge_action_sets(null_set, null_set)
    assert merged == null_set

    with pytest.raises(ValueError):
        merge_action_sets(a, null_set)

    with pytest.raises(ValueError):
        merge_action_sets(null_set, a)

    b = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(2, 11), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 11), dtype=bool)),
    )

    with pytest.raises(ValueError):
        merge_action_sets(a, b)


def test_replicate_action_set() -> None:
    a = ActionSet(
        new_state=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
        mask=jnp.array(np.random.randint(0, 2, size=(2, 10), dtype=bool)),
    )

    replicated = replicate_action_set(a, 3)
    assert not a == replicated
    assert replicated is not None
    assert replicated.shape == (6, 10)
    for i in range(3):
        assert jnp.array_equal(replicated.new_state[i * 2], a.new_state[0])
        assert jnp.array_equal(replicated.mask[i * 2], a.mask[0])
        assert jnp.array_equal(replicated.new_state[i * 2 + 1], a.new_state[1])
        assert jnp.array_equal(replicated.mask[i * 2 + 1], a.mask[1])

    a = ActionSet(
        new_state=jnp.array([[0, 1, 2]]),
        mask=jnp.array([[True, False, True]]),
    )
    replicated = replicate_action_set(a, 2)
    assert not a == replicated
    assert replicated.shape == (2, 3)


def test_eye_mask_action_set() -> None:
    act = eye_mask_action_set(np.random.randint(0, 2, size=(1, 10), dtype=bool))
    assert act.shape == (10, 10)
    assert np.array_equal(act.mask, np.eye(10, dtype=bool))


def test_split_action_set() -> None:
    act = eye_mask_action_set(np.random.randint(0, 2, size=(1, 10), dtype=bool))
    split = split_action_set(act, 5)
    assert len(split) == 5
    for s in split:
        assert s is not None
        assert s.shape == (2, 10)
