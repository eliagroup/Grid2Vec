from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax_dataclasses import pytree_dataclass

from grid2vec.action_set import (
    ActionSet,
    action_set_unmasked,
    concatenate_action_sets,
    empty_action_set,
    get_mask_or_none,
    get_new_state_or_none,
    is_do_nothing_set,
    merge_action_sets,
    replicate_action_set,
    split_action_set,
)
from grid2vec.env import VecEnvState, vector_step
from grid2vec.grid import Grid


@pytree_dataclass(eq=False)
class Action:
    """
    A Dataclass encapsulating an action on the vector-environment. This only exists because
    the number of parameters to vector_step grew quite large over time
    """

    # I should refactor this in some way, as the shape of the action can't be determined statically
    # One way would be to statically store a n_envs variable in the action, which is computed upon
    # creation.
    # The other way would be to store arrays of shape [n_envs, 0] for unused categories
    # Both ways require knowing the shape for do-nothing actions
    # The latter holds the advantage of not needing recompiles for different None/not-None
    # combinations. The former holds the advantage of being more intuitive, as arithmetics with
    # dimension zero are quite counter-intuitive in jax
    new_switch_state: ActionSet
    new_line_state: ActionSet
    new_trafo_state: ActionSet
    new_trafo_taps: ActionSet
    new_trafo3w_taps: ActionSet
    new_topo_vect: ActionSet

    def __eq__(self, other: object) -> bool:
        """Two actions are equal if all their action sets are equal"""
        if not isinstance(other, Action):
            raise NotImplementedError(f"Cannot compare Action to type {type(other)}")
        return (
            self.new_switch_state == other.new_switch_state
            and self.new_line_state == other.new_line_state
            and self.new_trafo_state == other.new_trafo_state
            and self.new_trafo_taps == other.new_trafo_taps
            and self.new_trafo3w_taps == other.new_trafo3w_taps
            and self.new_topo_vect == other.new_topo_vect
        )

    def __getitem__(self, key: Union[int, slice, jnp.ndarray]) -> Action:
        """Accesses a subset of the action in the environment dimension"""
        return Action(
            new_switch_state=self.new_switch_state[key],
            new_line_state=self.new_line_state[key],
            new_trafo_state=self.new_trafo_state[key],
            new_trafo_taps=self.new_trafo_taps[key],
            new_trafo3w_taps=self.new_trafo3w_taps[key],
            new_topo_vect=self.new_topo_vect[key],
        )

    @property
    def n_envs(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return len(self.new_switch_state)

    def asdict(self) -> Dict[str, jnp.ndarray]:
        """Returns the action as a dictionary compatible to vector_step

        Note that reconstructing from this dict is not possible in the case of do-nothing actions
        as the environment dimension gets lost.

        Returns:
            Dict[str, jnp.ndarray]: The action as a dictionary
        """
        return {
            "new_switch_state": get_new_state_or_none(self.new_switch_state),
            "switch_mask": get_mask_or_none(self.new_switch_state),
            "new_line_state": get_new_state_or_none(self.new_line_state),
            "line_mask": get_mask_or_none(self.new_line_state),
            "new_trafo_state": get_new_state_or_none(self.new_trafo_state),
            "trafo_mask": get_mask_or_none(self.new_trafo_state),
            "new_trafo_taps": get_new_state_or_none(self.new_trafo_taps),
            "trafo_tap_mask": get_mask_or_none(self.new_trafo_taps),
            "new_trafo3w_taps": get_new_state_or_none(self.new_trafo3w_taps),
            "trafo3w_tap_mask": get_mask_or_none(self.new_trafo3w_taps),
            "new_topo_vect": get_new_state_or_none(self.new_topo_vect),
            "topo_vect_mask": get_mask_or_none(self.new_topo_vect),
        }


def make_action(
    new_switch_state: Optional[ActionSet] = None,
    new_line_state: Optional[ActionSet] = None,
    new_trafo_state: Optional[ActionSet] = None,
    new_trafo_taps: Optional[ActionSet] = None,
    new_trafo3w_taps: Optional[ActionSet] = None,
    new_topo_vect: Optional[ActionSet] = None,
) -> Action:
    """Creates an action, but tries to determine the environment dimension from the shapes of the
    action components

    Args:
        new_switch_state (Optional[ActionSet], optional): The new switch state. Defaults to None.
        new_line_state (Optional[ActionSet], optional): The new line state. Defaults to None.
        new_trafo_state (Optional[ActionSet], optional): The new trafo state. Defaults to None.
        new_trafo_taps (Optional[ActionSet], optional): The new trafo tap position. Defaults to None.
        new_trafo3w_taps (Optional[ActionSet], optional): The new trafo3w tap position. Defaults to None.
        new_topo_vect (Optional[ActionSet], optional): The new topo vector. Defaults to None.

    Raises:
        ValueError: In case all components are None

    Returns:
        Action: The action with inferred environment dimension
    """
    n_envs = (
        (len(new_switch_state) if new_switch_state is not None else None)
        or (len(new_line_state) if new_line_state is not None else None)
        or (len(new_trafo_state) if new_trafo_state is not None else None)
        or (len(new_trafo_taps) if new_trafo_taps is not None else None)
        or (len(new_trafo3w_taps) if new_trafo3w_taps is not None else None)
        or (len(new_topo_vect) if new_topo_vect is not None else None)
    )

    if n_envs is None:
        raise ValueError("Cannot infer environment dimension from action components")

    new_switch_state = (
        empty_action_set((n_envs, 0), dtype=bool)
        if new_switch_state is None
        else new_switch_state
    )
    new_line_state = (
        empty_action_set((n_envs, 0), dtype=bool)
        if new_line_state is None
        else new_line_state
    )
    new_trafo_state = (
        empty_action_set((n_envs, 0), dtype=bool)
        if new_trafo_state is None
        else new_trafo_state
    )
    new_trafo_taps = (
        empty_action_set((n_envs, 0), dtype=jnp.int32)
        if new_trafo_taps is None
        else new_trafo_taps
    )
    new_trafo3w_taps = (
        empty_action_set((n_envs, 0), dtype=jnp.int32)
        if new_trafo3w_taps is None
        else new_trafo3w_taps
    )
    new_topo_vect = (
        empty_action_set((n_envs, 0), dtype=jnp.int32)
        if new_topo_vect is None
        else new_topo_vect
    )

    return Action(
        new_switch_state=new_switch_state,
        new_line_state=new_line_state,
        new_trafo_state=new_trafo_state,
        new_trafo_taps=new_trafo_taps,
        new_trafo3w_taps=new_trafo3w_taps,
        new_topo_vect=new_topo_vect,
    )


def assert_n_envs(action: Action) -> None:
    """Asserts that all components of the action have similar dimension
    Args:
        action (Action): The action to check

    Raises:
        ValueError: If any component of the action has a different number of environments than n_envs
    """
    n_envs = action.new_switch_state.shape[0]
    if action.new_line_state.shape[0] != n_envs:
        raise ValueError(
            f"Action has different n_envs for line_state: {action.new_line_state.shape[0]} != {n_envs}"
        )
    if action.new_trafo_state.shape[0] != n_envs:
        raise ValueError(
            f"Action has different n_envs for trafo_state: {action.new_trafo_state.shape[0]} != {n_envs}"
        )
    if action.new_trafo_taps.shape[0] != n_envs:
        raise ValueError(
            f"Action has different n_envs for trafo_taps: {action.new_trafo_taps.shape[0]} != {n_envs}"
        )
    if action.new_trafo3w_taps.shape[0] != n_envs:
        raise ValueError(
            f"Action has different n_envs for trafo3w_taps: {action.new_trafo3w_taps.shape[0]} != {n_envs}"
        )
    if action.new_topo_vect.shape[0] != n_envs:
        raise ValueError(
            f"Action has different n_envs for topo_vect: {action.new_topo_vect.shape[0]} != {n_envs}"
        )


def random_action(grid: Grid, n_envs: int) -> Action:
    """Creates a random action for the vectorized environment

    Args:
        grid (Grid): The grid to create the action for
        n_envs (int): Number of environments

    Returns:
        Action: The random action
    """
    return Action(
        new_switch_state=action_set_unmasked(
            np.random.randint(
                0, 2, (n_envs, grid.n_switch_controllable), dtype=np.bool_
            )
        ),
        new_line_state=action_set_unmasked(
            np.random.randint(0, 2, (n_envs, grid.n_line_controllable), dtype=np.bool_)
        ),
        new_trafo_state=action_set_unmasked(
            np.random.randint(0, 2, (n_envs, grid.n_trafo_controllable), dtype=np.bool_)
        ),
        new_trafo_taps=action_set_unmasked(
            np.random.randint(
                grid.trafo_tap_min,
                grid.trafo_tap_max,
                (n_envs, grid.n_trafo_tap_controllable),
                dtype=np.int32,
            )
        ),
        new_trafo3w_taps=action_set_unmasked(
            np.random.randint(
                grid.trafo3w_tap_min,
                grid.trafo3w_tap_max,
                (n_envs, grid.n_trafo3w_tap_controllable),
                dtype=np.int32,
            )
        ),
        new_topo_vect=action_set_unmasked(
            np.random.randint(
                grid.topo_vect_min,
                grid.topo_vect_max,
                (n_envs, grid.n_topo_vect_controllable),
                dtype=np.int32,
            )
        ),
    )


def do_nothing_action(
    n_envs: int,
) -> Action:
    """Creates a generic do-nothing action with everything empty actions (i.e. element dim 0)

    This still requires to know the environment dimension, as every action needs an environment
    dimension.
    """
    return Action(
        new_switch_state=empty_action_set((n_envs, 0), dtype=bool),
        new_line_state=empty_action_set((n_envs, 0), dtype=bool),
        new_trafo_state=empty_action_set((n_envs, 0), dtype=bool),
        new_trafo_taps=empty_action_set((n_envs, 0), dtype=jnp.int32),
        new_trafo3w_taps=empty_action_set((n_envs, 0), dtype=jnp.int32),
        new_topo_vect=empty_action_set((n_envs, 0), dtype=jnp.int32),
    )


def is_do_nothing_action(action: Action) -> bool:
    """Checks if an action is a do-nothing action

    Args:
        action (Action): The action to check

    Returns:
        bool: True if the action is a do-nothing action
    """
    return (
        is_do_nothing_set(action.new_switch_state)
        and is_do_nothing_set(action.new_line_state)
        and is_do_nothing_set(action.new_trafo_state)
        and is_do_nothing_set(action.new_trafo_taps)
        and is_do_nothing_set(action.new_trafo3w_taps)
        and is_do_nothing_set(action.new_topo_vect)
    )


def pad_out_like(action: Action, like: Action) -> Action:
    """Pads out an action that has action sets of element shape 0 to have element shapes
    like the other action with all masked out elements

    Args:
        action (Action): The action to pad out
        like (Action): The action to get the element dimensions from

    Returns:
        Action: An action that is padded out to have the same element dimensions as like
    """
    return Action(
        new_switch_state=empty_action_set(
            (action.new_switch_state.shape[0], *like.new_switch_state.shape[1:]),
            like.new_switch_state.dtype,
        )
        if action.new_switch_state.shape[1] == 0
        else action.new_switch_state,
        new_line_state=empty_action_set(
            (action.new_line_state.shape[0], *like.new_line_state.shape[1:]),
            like.new_line_state.dtype,
        )
        if action.new_line_state.shape[1] == 0
        else action.new_line_state,
        new_trafo_state=empty_action_set(
            (action.new_trafo_state.shape[0], *like.new_trafo_state.shape[1:]),
            like.new_trafo_state.dtype,
        )
        if action.new_trafo_state.shape[1] == 0
        else action.new_trafo_state,
        new_trafo_taps=empty_action_set(
            (action.new_trafo_taps.shape[0], *like.new_trafo_taps.shape[1:]),
            like.new_trafo_taps.dtype,
        )
        if action.new_trafo_taps.shape[1] == 0
        else action.new_trafo_taps,
        new_trafo3w_taps=empty_action_set(
            (action.new_trafo3w_taps.shape[0], *like.new_trafo3w_taps.shape[1:]),
            like.new_trafo3w_taps.dtype,
        )
        if action.new_trafo3w_taps.shape[1] == 0
        else action.new_trafo3w_taps,
        new_topo_vect=empty_action_set(
            (action.new_topo_vect.shape[0], *like.new_topo_vect.shape[1:]),
            like.new_topo_vect.dtype,
        )
        if action.new_topo_vect.shape[1] == 0
        else action.new_topo_vect,
    )


def merge_actions(a: Action, b: Action) -> Action:
    """Merge two actions into one. If both actions act on the same element, the second action
    is used for that element

    Actions need to have the same shape in all action sets, if they don't, use pad_out_like to
    pad them out to the same shape: `merge_actions(pad_out_like(a, b), pad_out_like(b, a))`

    Args:
        a (Action): The first action
        b (Action): The second action

    Returns:
        Action: A resulting action
    """
    if len(a) != len(b):
        raise ValueError("Cannot merge actions with different n_envs")

    return Action(
        new_switch_state=merge_action_sets(a.new_switch_state, b.new_switch_state),
        new_line_state=merge_action_sets(a.new_line_state, b.new_line_state),
        new_trafo_state=merge_action_sets(a.new_trafo_state, b.new_trafo_state),
        new_trafo_taps=merge_action_sets(a.new_trafo_taps, b.new_trafo_taps),
        new_trafo3w_taps=merge_action_sets(a.new_trafo3w_taps, b.new_trafo3w_taps),
        new_topo_vect=merge_action_sets(a.new_topo_vect, b.new_topo_vect),
    )


def concatenate_actions(a: Action, b: Action) -> Action:
    """Concatenate two actions to form a joint action of len(res) = len(a) + len(b)

    Raises if the actions have incompatible shapes - if one of them has dimension 0 use
    pad_out_like or pad_and_concatenate_actions

    Args:
        a (Action): The first action
        b (Action): The second action

    Returns:
        Action: A concatenated action
    """
    return Action(
        concatenate_action_sets(a.new_switch_state, b.new_switch_state),
        concatenate_action_sets(a.new_line_state, b.new_line_state),
        concatenate_action_sets(a.new_trafo_state, b.new_trafo_state),
        concatenate_action_sets(a.new_trafo_taps, b.new_trafo_taps),
        concatenate_action_sets(a.new_trafo3w_taps, b.new_trafo3w_taps),
        concatenate_action_sets(a.new_topo_vect, b.new_topo_vect),
    )


def pad_and_concatenate_actions(a: Action, b: Action) -> Action:
    """First pads and then concatenates actions

    With this, you can concatenate actions that have 0-dimension action sets

    Args:
        a (Action): The first action
        b (Action): The second action

    Returns:
        Action: A concatenated action
    """
    return concatenate_actions(pad_out_like(a, b), pad_out_like(b, a))


def concatenate_all_actions(actions: List[Action]) -> Action:
    """Concatenates a list of actions into one

    Pads them out to the same shape if necessary

    Args:
        actions (List[Action]): The list of actions

    Returns:
        Action: The concatenated action
    """
    return reduce(pad_and_concatenate_actions, actions)


def split_action(action: Action, n_splits: int) -> List[Action]:
    """Splits an action into n_splits equal sized actions along the environment dimension

    Args:
        action (Action): An action to split
        n_splits (int): Into how many sub-actions to split it, must divide action.n_envs

    Returns:
        List[Action]: A list of actions of length n_splits
    """
    if action.n_envs % n_splits != 0:
        raise ValueError(
            f"Cannot split action with n_envs {action.n_envs} into {n_splits} splits"
        )

    line_states = split_action_set(action.new_line_state, n_splits)
    switch_states = split_action_set(action.new_switch_state, n_splits)
    trafo_states = split_action_set(action.new_trafo_state, n_splits)
    trafo_taps = split_action_set(action.new_trafo_taps, n_splits)
    trafo3w_taps = split_action_set(action.new_trafo3w_taps, n_splits)
    topo_vects = split_action_set(action.new_topo_vect, n_splits)

    return [
        Action(
            new_switch_state=switch_states[i],
            new_line_state=line_states[i],
            new_trafo_state=trafo_states[i],
            new_trafo_taps=trafo_taps[i],
            new_trafo3w_taps=trafo3w_taps[i],
            new_topo_vect=topo_vects[i],
        )
        for i in range(n_splits)
    ]


def replicate_action(action: Action, repetitions: int) -> Action:
    """Returns an action with a replicated environment dimension

    Args:
        action (Action): The input action
        repetitions (int): How often to replicate the action

    Returns:
        Action: An action with n_envs equal to action.n_envs * repetitions
    """
    return Action(
        new_switch_state=replicate_action_set(action.new_switch_state, repetitions),
        new_line_state=replicate_action_set(action.new_line_state, repetitions),
        new_trafo_state=replicate_action_set(action.new_trafo_state, repetitions),
        new_trafo_taps=replicate_action_set(action.new_trafo_taps, repetitions),
        new_trafo3w_taps=replicate_action_set(action.new_trafo3w_taps, repetitions),
        new_topo_vect=replicate_action_set(action.new_topo_vect, repetitions),
    )


def vector_step_action(
    env: VecEnvState, action: Action, which: Optional[np.ndarray] = None
) -> VecEnvState:
    """This is a wrapper around vector_step that takes an Action dataclass instead of the
    individual arguments

    Args:
        env (VecEnvState): The vectorized environment state
        action (Action): The action to take
        which (Optional[np.ndarray], optional): Which environments to act on. Defaults to all.

    Returns:
        VecEnvState: The updated environment state
    """
    return vector_step(env, which=which, **action.asdict())
