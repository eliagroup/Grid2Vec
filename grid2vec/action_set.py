from __future__ import annotations

from functools import partial
from typing import List, Optional, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax_dataclasses import pytree_dataclass


@pytree_dataclass(eq=False)
class ActionSet:
    new_state: jnp.ndarray
    mask: jnp.ndarray

    @jax.jit
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionSet):
            return NotImplemented
        val_eq = jnp.array_equal(
            jnp.where(self.mask, self.new_state, 0),
            jnp.where(other.mask, other.new_state, 0),
        )
        mask_eq = jnp.array_equal(self.mask, other.mask)
        shape_eq = self.shape == other.shape
        return val_eq & mask_eq & shape_eq

    def __post_init__(self) -> None:
        assert self.new_state.shape == self.mask.shape

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> ActionSet:
        retval = ActionSet(self.new_state[key], self.mask[key])

        # If you somehow managed to pass in a single index, we raise
        chex.assert_rank(retval.new_state, 2)
        chex.assert_rank(retval.mask, 2)
        return retval

    def __len__(self) -> int:
        return self.new_state.shape[0]

    @property
    def shape(self) -> tuple:
        return self.new_state.shape

    @property
    def dtype(self) -> np.dtype:
        return self.new_state.dtype

    @property
    def has_zero_element_dim(self) -> bool:
        return self.new_state.shape[1] == 0


# @partial(jax.jit, static_argnames=("shape", "dtype"))
def empty_action_set(shape: tuple, dtype: jnp.dtype = jnp.bool_) -> ActionSet:
    """Returns an ActionSet with all zeros

    Args:
        shape (tuple): The shape for both the mask and the new_state
        dtype (np.dtype, optional): A datatype of the new_state array. Defaults to np.bool.

    Returns:
        ActionSet: An ActionSet with all zeros which does nothing when applied
    """
    return ActionSet(
        new_state=jnp.zeros(shape, dtype=dtype), mask=jnp.zeros(shape, dtype=np.bool_)
    )


def is_do_nothing_set(set: ActionSet) -> bool:
    """Checks if an action set has no effect if applied

    An action set has no effect if either the element dim is 0 or the mask is all empty

    Args:
        set (ActionSet): The action set to check

    Returns:
        bool: If this is the do-nothing action set
    """
    return set.shape[1] == 0 or not jnp.any(set.mask)


def action_set_unmasked(new_state: jnp.ndarray) -> ActionSet:
    """Returns an ActionSet with a mask that is all ones, meaning nothing is masked out

    Args:
        new_state (np.ndarray): The new_state array that will be used entirely

    Returns:
        ActionSet: An ActionSet with a mask that is all ones
    """
    return ActionSet(
        new_state=new_state, mask=jnp.ones(new_state.shape, dtype=jnp.bool_)
    )


def get_mask_or_none(set: Optional[ActionSet]) -> Optional[jnp.ndarray]:
    """Returns the mask of an ActionSet or None if the ActionSet is None"""
    if set is None or set.mask.shape[1] == 0:
        return None
    return set.mask


def get_new_state_or_none(set: Optional[ActionSet]) -> Optional[jnp.ndarray]:
    """Returns the new_state of an ActionSet or None if the ActionSet is None"""
    if set is None or set.new_state.shape[1] == 0:
        return None
    return set.new_state


def merge_action_sets(a: ActionSet, b: ActionSet) -> ActionSet:
    """Merges two actions, where if both actions set the same element, the action from b wins

    Args:
        a (ActionSet): Action A
        b (ActionSet): Action B

    Raises:
        ValueError: Action sets have different shapes

    Returns:
        ActionSet: An action that is the merge of A and B, or None if both are None
    """
    if a.shape != b.shape:
        raise ValueError(f"Action sets have different shapes: {a.shape} and {b.shape}")

    setting = jnp.where(b.mask, b.new_state, a.new_state)
    return ActionSet(new_state=setting, mask=jnp.logical_or(a.mask, b.mask))


def concatenate_action_sets(a: ActionSet, b: ActionSet) -> ActionSet:
    """Concatenates two action sets into one

    Requires action sets to be of same shape, if one of them has an element dimension of 0,
    please use empty_action_set to create an effectless action set of the correct shape

    Args:
        a (ActionSet): The first action set
        b (ActionSet): The second action set

    Raises:
        ValueError: Both action sets have an element dimension != 0 and those dimensions don't match

    Returns:
        ActionSet: A concatenated action set with env dimension = a.env_dimension + b.env_dimension
    """
    chex.assert_equal_shape([a.new_state, b.new_state, a.mask, b.mask], dims=1)
    return ActionSet(
        new_state=jnp.concatenate([a.new_state, b.new_state], axis=0),
        mask=jnp.concatenate([a.mask, b.mask], axis=0),
    )


@partial(jax.jit, static_argnames=("repetitions",))
def replicate_action_set(action_set: ActionSet, repetitions: int) -> ActionSet:
    """Replicates an action set across multiple repetitions

    Args:
        action_set (ActionSet): The action set to replicate
        repetitions (int): The number of repetitions to replicate the action set across

    Returns:
        ActionSet: The replicated action set, or None if the input was None
    """
    return ActionSet(
        new_state=jnp.tile(action_set.new_state, (repetitions, 1)),
        mask=jnp.tile(action_set.mask, (repetitions, 1)),
    )


def eye_mask_action_set(new_state: jnp.ndarray) -> ActionSet:
    """Returns an action set of shape n*n with an mask that has all ones on the diagonal and zeros elsewhere

    This is useful for enumerating actions

    Args:
        new_state (np.ndarray): The new_state array to use, must be (n) or (1, n)

    Returns:
        ActionSet: An action set of shape (n, n) with the eye mask
    """
    if len(new_state.shape) == 1:
        new_state = jnp.expand_dims(new_state, axis=0)
    if new_state.shape[0] != 1:
        raise ValueError(f"Expected shape (1, n), got {new_state.shape}")
    return ActionSet(
        new_state=jnp.repeat(new_state, new_state.shape[1], axis=0),
        mask=jnp.eye(new_state.shape[1], dtype=jnp.bool_),
    )


@partial(jax.jit, static_argnames=("n_splits",))
def split_action_set(act_set: ActionSet, n_splits: int) -> List[ActionSet]:
    """Divides an action set of shape (n, m) into n_splits action sets of shape (n/n_splits, m)

    Args:
        set (Optional[ActionSet]): The action set to split up. If it's None, it will return
            a list of Nones with length n_splits
        n_splits (int): How many splits to make, must divide n evenly

    Returns:
        List[ActionSet]: A list of length n_splits, each with an action set
    """
    masks = jnp.split(act_set.mask, n_splits, axis=0)
    new_states = jnp.split(act_set.new_state, n_splits, axis=0)
    return [ActionSet(new_state=ns, mask=ms) for ns, ms in zip(new_states, masks)]
