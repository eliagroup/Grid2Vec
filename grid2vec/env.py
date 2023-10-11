from typing import Callable, Dict, Optional

import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax_dataclasses import Static, pytree_dataclass, replace

from grid2vec.batched_pfc import PFCResults
from grid2vec.grid import Grid
from grid2vec.result_spec import ResultSpec, ResultsValue


@pytree_dataclass(frozen=True, eq=False)
class VecEnvState:
    """A class that holds the state of a vectorized set of environments, where fundamentally each
    environment state is classified by the timestep and accumulated actions
    """

    grid: Static[Grid]  # A shared grid among all environments
    timestep: jnp.ndarray  # [int] (n_envs) - Which timestep the environment is on
    chronic: jnp.ndarray  # [int] (n_envs) - Which chronic the environment is on
    switch_state: jnp.ndarray  # [bool] (n_envs, n_switch_controllable) - State of the (controllable) switches
    line_state: jnp.ndarray  # [bool] (n_envs, n_line_controllable) - State of the (controllable) lines
    trafo_state: jnp.ndarray  # [bool] (n_envs, n_trafo_controllable) - State of the (controllable) trafos
    trafo_tap_pos: jnp.ndarray  # [int] (n_envs, n_trafo_tap_controllable) - Tap position of the (controllable) trafos
    trafo3w_tap_pos: jnp.ndarray  # [int] (n_envs, n_trafo3w_tap_controllable) - Tap position of the (controllable) trafos3w
    topo_vect: jnp.ndarray  # [int] (n_envs, len_topo_vect) - Topology vector of the environment, indicating bus assignments of each element

    @property
    def n_envs(self) -> int:
        return len(self.timestep)

    def __len__(self) -> int:
        return self.n_envs

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VecEnvState):
            return NotImplemented
        if not self.grid == other.grid:
            raise ValueError("Can not compare VecEnvStates with different grids")

        return (
            jnp.array_equal(self.timestep, other.timestep)
            & jnp.array_equal(self.chronic, other.chronic)
            & jnp.array_equal(self.switch_state, other.switch_state)
            & jnp.array_equal(self.line_state, other.line_state)
            & jnp.array_equal(self.trafo_state, other.trafo_state)
            & jnp.array_equal(self.trafo_tap_pos, other.trafo_tap_pos)
            & jnp.array_equal(self.trafo3w_tap_pos, other.trafo3w_tap_pos)
            & jnp.array_equal(self.topo_vect, other.topo_vect)
        )


def expand_and_repeat(arr: jnp.ndarray, n_envs: int) -> jnp.ndarray:
    """Expands an array to a given number of environments and repeats it along the first axis

    Args:
        arr (jnp.ndarray): The array to expand, shape (n_elements)
        n_envs (int): The number of environments to expand to

    Returns:
        jnp.ndarray: The expanded and repeated array, shape (n_envs, n_elements)
    """
    return jnp.expand_dims(arr, 0).repeat(n_envs, axis=0)


# @partial(
#     jax.jit,
#     static_argnames=(
#         "grid",
#         "n_envs",
#     ),
# )
def make_env(grid: Grid, n_envs: int) -> VecEnvState:
    """Creates a vectorized environment state

    Args:
        grid: A loaded grid, use load_grid() for obtaining one
        n_envs (int): Number of environments

    Returns:
        VecEnvState: The vectorized environment state
    """
    return VecEnvState(  # type: ignore
        grid=grid,
        timestep=jnp.zeros(n_envs, dtype=jnp.int32),
        chronic=jnp.arange(n_envs, dtype=jnp.int32) % grid.n_chronics,
        switch_state=expand_and_repeat(grid.switch_default, n_envs),
        line_state=expand_and_repeat(grid.line_default, n_envs),
        trafo_state=expand_and_repeat(grid.trafo_default, n_envs),
        trafo_tap_pos=expand_and_repeat(grid.trafo_tap_default, n_envs),
        trafo3w_tap_pos=expand_and_repeat(grid.trafo3w_tap_default, n_envs),
        topo_vect=expand_and_repeat(grid.topo_vect_default, n_envs),
    )


def vector_reset(
    env: VecEnvState,
    which: Optional[jnp.ndarray] = None,
    target_chronics: Optional[jnp.ndarray] = None,
    target_timesteps: Optional[jnp.ndarray] = None,
) -> VecEnvState:
    """Resets the vectorized environment state

    Args:
        env (VecEnvState): The vectorized environment state
        which (Optional[np.ndarray[bool]], optional): Shape (n_env) Which environments to reset.
            Defaults to All.
        target_chronics (Optional[np.ndarray[int]], optional): Shape (n_envs Which chronics to
            reset to. Defaults to env.chronic + 1.
            If which is passed, values for which=False will be ignored.
        target_timesteps (Optional[np.ndarray[int]], optional): Shape (n_envs) Which timesteps to
            reset to. Defaults to 0.


    Returns:
        VecEnvState: The vectorized environment state
    """
    if which is None:
        which = jnp.ones(env.n_envs, dtype=bool)

    if target_chronics is None:
        target_chronics = env.chronic + 1

    if target_timesteps is None:
        target_timesteps = jnp.zeros(env.n_envs, dtype=jnp.int32)

    assert target_chronics.shape == (env.n_envs,)

    target_chronics = jnp.array(target_chronics) % env.grid.n_chronics

    new_timestep = jnp.where(which, target_timesteps, env.timestep)
    new_chronic = jnp.where(which, target_chronics, env.chronic)
    new_switch_state = jnp.where(
        which[:, None],
        expand_and_repeat(env.grid.switch_default, env.n_envs),
        env.switch_state,
    )
    new_line_state = jnp.where(
        which[:, None],
        expand_and_repeat(env.grid.line_default, env.n_envs),
        env.line_state,
    )
    new_trafo_state = jnp.where(
        which[:, None],
        expand_and_repeat(env.grid.trafo_default, env.n_envs),
        env.trafo_state,
    )
    new_trafo_tap_pos = jnp.where(
        which[:, None],
        expand_and_repeat(env.grid.trafo_tap_default, env.n_envs),
        env.trafo_tap_pos,
    )
    new_trafo3w_tap_pos = jnp.where(
        which[:, None],
        expand_and_repeat(env.grid.trafo3w_tap_default, env.n_envs),
        env.trafo3w_tap_pos,
    )
    new_topo_vect = jnp.where(
        which[:, None],
        expand_and_repeat(env.grid.topo_vect_default, env.n_envs),
        env.topo_vect,
    )

    return VecEnvState(  # type: ignore
        grid=env.grid,
        timestep=new_timestep,
        chronic=new_chronic,
        switch_state=new_switch_state,
        line_state=new_line_state,
        trafo_state=new_trafo_state,
        trafo_tap_pos=new_trafo_tap_pos,
        trafo3w_tap_pos=new_trafo3w_tap_pos,
        topo_vect=new_topo_vect,
    )


def masked_set(
    mask: Optional[jnp.ndarray],
    which: jnp.ndarray,
    new_state: jnp.ndarray,
    current_state: jnp.ndarray,
) -> jnp.ndarray:
    """Performs a masked set, returning an array where the mask is True, the new_state is used,
    otherwise the current_state is used.

    Furthermore, the which array can be used to only update a subset of the environments

    Args:
        mask (jnp.ndarray): Shape (n_env, n_elements) The mask into the state array
        which (jnp.ndarray): Shape (n_env) Which environments to update
        new_state (jnp.ndarray): Shape (n_env, n_elements) The new state
        current_state (jnp.ndarray): Shape (n_env, n_elements) The current state

    Returns:
        jnp.ndarray: A new state array
    """
    if mask is None:
        mask = jnp.ones_like(new_state, dtype=bool)

    chex.assert_equal_shape([new_state, mask, current_state])
    chex.assert_shape(which, (current_state.shape[0],))

    # Environments that are not in which will have a mask of only False
    mask = jnp.where(which[:, None], mask, False)

    # Where mask is False, use the current state
    # Where mask is True, use the new state
    return jnp.where(mask, new_state, current_state)


def vector_step(
    env: VecEnvState,
    *,
    new_switch_state: Optional[jnp.ndarray] = None,
    switch_mask: Optional[jnp.ndarray] = None,
    new_line_state: Optional[jnp.ndarray] = None,
    line_mask: Optional[jnp.ndarray] = None,
    new_trafo_state: Optional[jnp.ndarray] = None,
    trafo_mask: Optional[jnp.ndarray] = None,
    new_trafo_taps: Optional[jnp.ndarray] = None,
    trafo_tap_mask: Optional[jnp.ndarray] = None,
    new_trafo3w_taps: Optional[jnp.ndarray] = None,
    trafo3w_tap_mask: Optional[jnp.ndarray] = None,
    new_topo_vect: Optional[jnp.ndarray] = None,
    topo_vect_mask: Optional[jnp.ndarray] = None,
    which: Optional[jnp.ndarray] = None,
    step_time: int = 1,
) -> VecEnvState:
    """Steps the vectorized environment with a new set of switch states

    Args:
        env (VecEnvState): The environment state
        new_switch_state (np.ndarray[bool], optional): Shape (n_env, n_contollable_switch) The new
            switch states for controllable switches. If None, switch states will not be changed.
        switch_mask (np.ndarray[bool], optional] Shape equal to new_switch_state, if
            provided, it will leave all switches where switch_mask is False unchanged
        new_line_state (np.ndarray[bool], optional): Shape (n_env, n_line) The new line statuses
            representing whether the line is active. If None, line statuses will not be changed.
        line_mask (np.ndarray[bool], optional] Shape equal to new_line_state, if
            provided, it will leave all lines where line_mask is False unchanged
        new_trafo_state (np.ndarray[bool], optional): Shape (n_env, n_trafo) The new trafo
            statuses representing whether the trafo is active. If None, trafo statuses will not be
            changed.
        trafo_mask (np.ndarray[bool], optional] Shape equal to new_trafo_state, if
            provided, it will leave all trafos where trafo_mask is False unchanged
        new_trafo_taps (np.ndarray[int], optional): Shape (n_env, n_trafo) The new tap positions
            for 2w transformers. If None, tap positions will not be changed.
        trafo_tap_mask (np.ndarray[bool], optional] Shape equal to new_trafo_taps, if
            provided, it will leave all trafos taps where trafo_tap_mask is False unchanged
        new_trafo3w_taps (np.ndarray[int], optional): Shape (n_env, n_trafo3w) The new tap
            positions for 3w transformers. If None, tap positions will not be changed.
        trafo3w_tap_mask (np.ndarray[bool], optional] Shape equal to new_trafo3w_taps, if
            provided, it will leave all trafo3ws where trafo3w_tap_mask is False unchanged
        new_topo_vect (np.ndarray[int], optional): Shape (n_env, len_topo_vect) The new topology
            vector. If None, topology vector will not be changed.
        topo_vect_mask (np.ndarray[bool], optional] Shape equal to new_topo_vect, if
            provided, it will leave all topo_vect entries where topo_vect_mask is False unchanged
        which (np.ndarray[bool], optional): Shape (n_env) which environments to step, if None all
            will be stepped. If provided, all previous new_* arguments will change their required
            shape from (n_env, ...) to (sum(which), ...).
        step_time (int, optional): The number of timesteps to step the environment. Defaults to 1.
            Really, the only sensible values are likely 0 and 1.

    Returns:
        VecEnvState: The updated environment state
    """
    if step_time < 0:
        raise ValueError("Negative step time provided")

    if which is None:
        which = jnp.ones(env.n_envs, dtype=bool)
    assert which.shape == (env.n_envs,)

    set_timestep = env.timestep + which * int(step_time)
    # If the environment is already past the end of chronics, raise an error
    set_timestep = eqx.error_if(
        set_timestep,
        set_timestep >= env.grid.chronics.n_timesteps[env.chronic],
        "You forgot to reset the environment after the end of the episode!",
    )

    set_switch_state = env.switch_state
    if new_switch_state is not None:
        set_switch_state = masked_set(
            switch_mask, which, new_switch_state, env.switch_state
        )

    set_line_state = env.line_state
    if new_line_state is not None:
        set_line_state = masked_set(line_mask, which, new_line_state, env.line_state)

    set_trafo_state = env.trafo_state
    if new_trafo_state is not None:
        set_trafo_state = masked_set(
            trafo_mask, which, new_trafo_state, env.trafo_state
        )

    set_trafo_taps = env.trafo_tap_pos
    if new_trafo_taps is not None:
        set_trafo_taps = masked_set(
            trafo_tap_mask, which, new_trafo_taps, env.trafo_tap_pos
        )
        set_trafo_taps = jnp.clip(
            set_trafo_taps,
            a_min=env.grid.trafo_tap_min,
            a_max=env.grid.trafo_tap_max,
        )
        set_trafo_taps = eqx.error_if(
            set_trafo_taps,
            set_trafo_taps > env.grid.trafo_tap_max,
            "Transformer tap assignment too high",
        )
        set_trafo_taps = eqx.error_if(
            set_trafo_taps,
            set_trafo_taps < env.grid.trafo_tap_min,
            "Transformer tap assignment too low",
        )

    set_trafo3w_taps = env.trafo3w_tap_pos
    if new_trafo3w_taps is not None:
        set_trafo3w_taps = masked_set(
            trafo3w_tap_mask, which, new_trafo3w_taps, env.trafo3w_tap_pos
        )
        set_trafo3w_taps = jnp.clip(
            set_trafo3w_taps,
            a_min=env.grid.trafo3w_tap_min,
            a_max=env.grid.trafo3w_tap_max,
        )
        set_trafo3w_taps = eqx.error_if(
            set_trafo3w_taps,
            set_trafo3w_taps > env.grid.trafo3w_tap_max,
            "3w Transformer tap assignment too high",
        )
        set_trafo3w_taps = eqx.error_if(
            set_trafo3w_taps,
            set_trafo3w_taps < env.grid.trafo3w_tap_min,
            "3w Transformer tap assignment too low",
        )

    set_topo_vect = env.topo_vect
    if new_topo_vect is not None:
        set_topo_vect = masked_set(topo_vect_mask, which, new_topo_vect, env.topo_vect)
        set_topo_vect = jnp.clip(
            set_topo_vect,
            a_min=env.grid.topo_vect_min,
            a_max=env.grid.topo_vect_max,
        )
        set_topo_vect = eqx.error_if(
            set_topo_vect,
            set_topo_vect > env.grid.topo_vect_max,
            "Topology vector assignment too high",
        )
        set_topo_vect = eqx.error_if(
            set_topo_vect,
            set_topo_vect < env.grid.topo_vect_min,
            "Topology vector assignment too low",
        )

    # Create a new env to remain immutable
    return VecEnvState(  # type: ignore
        grid=env.grid,
        chronic=env.chronic,
        timestep=set_timestep,
        switch_state=set_switch_state,
        line_state=set_line_state,
        trafo_state=set_trafo_state,
        trafo_tap_pos=set_trafo_taps,
        trafo3w_tap_pos=set_trafo3w_taps,
        topo_vect=set_topo_vect,
    )


def timesteps_in_current_chronic(env: VecEnvState) -> jnp.ndarray:
    """Returns how many timesteps are left in the chronic of each env, meaning how many times
    you can call vector_step before get_truncated will return True

    Args:
        env (VecEnvState): The vectorized environment state

    Returns:
        np.ndarray: An array of shape (n_env) [int] with the number of timesteps left in the chronic
    """
    return env.grid.chronics.n_timesteps[env.chronic] - env.timestep - 1


def get_truncated(env: VecEnvState, check_increment: int = 1) -> jnp.ndarray:
    """Returns whether an environment is at the chronic end

    Args:
        env (VecEnvState): The vectorized environment state
        check_increment (int, optional): How far away can the end of the chronics be, i.e. what is
            the timestep increment you're using for stepping. Defaults to 1, i.e. it will return
            true on the last timestep and all (invalid) timesteps after that.

    Returns:
        np.ndarray[bool]: Shape (n_env) Whether the environment is at the chronic end
    """
    return env.timestep >= env.grid.chronics.n_timesteps[env.chronic] - check_increment


def postprocess_obs(obs: PFCResults, res_spec: ResultSpec) -> Dict[str, np.ndarray]:
    """Postprocesses the observation, remove nan values and convert to correct type

    Args:
        obs (PFCResults): The observation to postprocess
        res_spec (ResultSpec): The result spec to use for postprocessing, should match that for
            get_observation_space

    Returns:
        PFCResults: The postprocessed observation
    """

    def postprocess_value(value: np.ndarray, spec: ResultsValue) -> np.ndarray:
        value = value.astype(spec.dtype)
        value = np.nan_to_num(value)
        if spec.low is not None:
            value[value < spec.low] = spec.low
        if spec.high is not None:
            value[value > spec.high] = spec.high
        return value

    return {spec.key: postprocess_value(obs[spec.key], spec) for spec in res_spec}


def get_done(results: PFCResults) -> np.ndarray:
    """Check the death condition of an overloaded element

    Args:
        results (PFCResults): A PFC result from a steady state pfc

    Returns:
        np.ndarray[bool]: Shape (n_env) Whether the environment has died
    """
    failed_converge = ~results["converged"]
    overloaded_line = np.any(results["loading_line"] > 100, axis=1)
    overloaded_trafo = np.any(results["loading_trafo"] > 100, axis=1)
    overloaded_trafo3w = np.any(results["loading_trafo3w"] > 100, axis=1)

    return failed_converge | overloaded_line | overloaded_trafo | overloaded_trafo3w


def get_reward(
    grid: Grid,
    results: PFCResults,
    constant_offset: float = 1,
    crit_threshold: Optional[float] = None,
) -> jnp.ndarray:
    """Get either the n-1 or n-0 reward based on the grid type

    Args:
        grid (Grid): The grid the results are computed on
        results (PFCResults): A pfc result from compute_obs
        constant_offset (float, optional): A constant offset to add to the reward. Defaults to 1.
        crit_threshold (Optional[float], optional): A threshold for the reward. If None, the
            default_crit_threshold of the grid is used. Defaults to None.

    Returns:
        np.ndarray: A reward for each environment
    """
    if crit_threshold is None:
        crit_threshold = grid.default_crit_threshold

    if grid.uses_nminus1:
        return get_reward_nminus1(
            results=results,
            crit_threshold=crit_threshold,
            line_capacity=grid.line_capacity_masked_for_reward,
            trafo_capacity=grid.trafo_capacity_masked_for_reward,
            trafo3w_capacity=grid.trafo3w_capacity_masked_for_reward,
            constant_offset=constant_offset,
        )
    else:
        return get_reward_n(
            results=results,
            crit_threshold=crit_threshold,
            mask_line=grid.line_for_reward,
            mask_trafo=grid.trafo_for_reward,
            mask_trafo3w=grid.trafo3w_for_reward,
            constant_offset=constant_offset,
        )


def compute_n_penalty(loading: jnp.ndarray, crit_threshold: float) -> jnp.ndarray:
    """Process an element loading vector as returned from pandapower to a penalty vector

    These vectors might contain non-finite values, which are replaced
    (nan=0, posinf=3, neginf=0) and then clipped to the crit_threshold
    so that everything below the crit_threshold is zero and everything
    above rises up linearly

    Args:
        loading (jnp.ndarray): The loading vector from the pfc results
        crit_threshold (float): The crit threshold to where to clip it

    Returns:
        jnp.ndarray: The loading vector processed for the reward
    """
    return (
        jnp.clip(
            jnp.nan_to_num(loading / 100, nan=0, posinf=3, neginf=0),
            a_min=crit_threshold,
            a_max=None,
        )
        - crit_threshold
    )


def get_reward_n(
    results: PFCResults,
    crit_threshold: float,
    mask_line: Optional[jnp.ndarray] = None,
    mask_trafo: Optional[jnp.ndarray] = None,
    mask_trafo3w: Optional[jnp.ndarray] = None,
    constant_offset: float = 1,
) -> jnp.ndarray:
    """Get a reward for a n based environment

    The reward is 1 minus the sum of loading above the critical threshold

    Args:
        results (PFCResults): The results of a loadflow
        crit_threshold (float): The critical threshold
        mask_line (Optional[np.ndarray], optional): Shape broadcastable to (n_line) Which lines to
            include in the reward. Defaults to all.
        mask_trafo (Optional[np.ndarray], optional): Shape broadcastable to (n_trafo) Which trafos
            to include in the reward. Defaults to all.
        mask_trafo3w (Optional[np.ndarray], optional): Shape broadcastable to (n_trafo3w) Which 3w
            trafos to include in the reward. Defaults to all.
        constant_offset (float, optional): A constant offset to add to the reward. Defaults to 1.

    Returns:
        np.ndarray: [float] Shape (n_env) The reward for each environment
    """
    if mask_line is None:
        mask_line = jnp.ones_like(results["loading_line"], dtype=bool)
    if mask_trafo is None:
        mask_trafo = jnp.ones_like(results["loading_trafo"], dtype=bool)
    if mask_trafo3w is None:
        mask_trafo3w = jnp.ones_like(results["loading_trafo3w"], dtype=bool)

    line_penalty = jnp.sum(
        compute_n_penalty(
            jnp.where(mask_line, results["loading_line"], 0), crit_threshold
        ),
        axis=1,
    )
    trafo_penalty = jnp.sum(
        compute_n_penalty(
            jnp.where(mask_trafo, results["loading_trafo"], 0), crit_threshold
        ),
        axis=1,
    )
    trafo3w_penalty = jnp.sum(
        compute_n_penalty(
            jnp.where(mask_trafo3w, results["loading_trafo3w"], 0), crit_threshold
        ),
        axis=1,
    )
    return constant_offset - line_penalty - trafo_penalty - trafo3w_penalty


def compute_nminus1_penalty(
    loading: jnp.ndarray,
    capacity: jnp.ndarray,
    crit_threshold: float,
    clip_loading_above: float,
) -> jnp.ndarray:
    """Process a n-1 matrix into a penalty vector

    Args:
        loading (jnp.ndarray): The n-1 matrix, shape (n_env, n_outage, n_element)
        capacity (jnp.ndarray): The capacity of the elements, shape (n_element)
        crit_threshold (float): A criticality threshold
        clip_loading_above (float): Where to clip loading if it gets unreasonably high

    Returns:
        jnp.ndarray: A penalty vector, shape (n_env)
    """
    loading_clipped_and_scaled = jnp.clip(
        jnp.nan_to_num((loading - crit_threshold) / 100, nan=0, posinf=99999, neginf=0),
        a_min=0,
        a_max=(clip_loading_above - crit_threshold) / 100,
    )
    l_max = jnp.max(
        loading_clipped_and_scaled, axis=1
    )  # (n_env, n_outage, n_element) -> (n_env, n_element)
    return jnp.sum(l_max * capacity, axis=1)  # (n_env, n_element) -> (n_env)


def get_reward_nminus1(
    results: PFCResults,
    crit_threshold: float,
    line_capacity: np.ndarray,
    trafo_capacity: np.ndarray,
    trafo3w_capacity: np.ndarray,
    clip_loading_above: float = 200,
    constant_offset: float = 1,
) -> np.ndarray:
    """Get a reward for a n-1 based environment

    The reward is defined as -clip(L_max-100, 0) * S_n where
    L_max is the maximum line load for each line along the outage case dimension
    S_n is the capacity of the line
    And the clipping is in place to set the penalty to zero for lines that do not exhibit any
    overload

    This is done for both lines, trafos and trafo3ws.

    Args:
        results (PFCResults): The results of a loadflow
        crit_threshold (float): The critical threshold in percent
        line_capacity (np.ndarray): Shape (n_line), [float] The capacity of each line, scales the
            penalty for that line. Pass zero if you want to ignore the line
        trafo_capacity (np.ndarray): Shape (n_trafo), [float] The capacity of each trafo, scales the
            penalty for that trafo. Pass zero if you want to ignore the trafo
        trafo3w_capacity (np.ndarray): Shape (n_trafo3w), [float] The capacity of each trafo3w,
            scales the penalty for that trafo3w. Pass zero if you want to ignore the trafo3w
        clip_loading_above (float, optional): Clip element loads that are above this value to this
            value. Defaults to 200%
        constant_offset (float, optional): A constant offset to add to the reward. Defaults to 1.

    Returns:
        np.ndarray: [float] Shape (n_env) The reward for each environment
    """
    line_penalty = compute_nminus1_penalty(
        results["line_loading_per_failure"],
        line_capacity,
        crit_threshold=crit_threshold,
        clip_loading_above=clip_loading_above,
    )
    trafo_penalty = compute_nminus1_penalty(
        results["trafo_loading_per_failure"],
        trafo_capacity,
        crit_threshold=crit_threshold,
        clip_loading_above=clip_loading_above,
    )
    trafo3w_penalty = compute_nminus1_penalty(
        results["trafo3w_loading_per_failure"],
        trafo3w_capacity,
        crit_threshold=crit_threshold,
        clip_loading_above=clip_loading_above,
    )

    return constant_offset - (line_penalty + trafo_penalty + trafo3w_penalty)


def replicate_env(env: VecEnvState, repetitions: int) -> VecEnvState:
    """Replicates an environment repetitions times using jnp.tile

    The resulting environment layout will be something like:
    1, 2, 3, 1, 2, 3, 1, 2, 3, ...

    Args:
        env (VecEnvState): The input environment
        repetitions (int): The number of repetitions

    Returns:
        VecEnvState: An environment of n_envs = env.n_envs * repetitions
    """
    return VecEnvState(  # type: ignore
        grid=env.grid,
        timestep=jnp.tile(env.timestep, (repetitions,)),
        chronic=jnp.tile(env.chronic, (repetitions,)),
        switch_state=jnp.tile(env.switch_state, (repetitions, 1)),
        line_state=jnp.tile(env.line_state, (repetitions, 1)),
        trafo_state=jnp.tile(env.trafo_state, (repetitions, 1)),
        trafo_tap_pos=jnp.tile(env.trafo_tap_pos, (repetitions, 1)),
        trafo3w_tap_pos=jnp.tile(env.trafo3w_tap_pos, (repetitions, 1)),
        topo_vect=jnp.tile(env.topo_vect, (repetitions, 1)),
    )


def timebatch_env(
    env: VecEnvState, n_timesteps: int, end_of_chronic_behaviour: str = "ignore"
) -> VecEnvState:
    """Timebatches an environment so you can compute multiple timesteps at once

    It uses jnp.tile to do this, so the resulting environment layout will be something like:
    ```
    env1, ts1
    env2, ts1,
    env3, ts1,
    env1, ts2,
    env2, ts2,
    env3, ts2,
    ...
    ```

    Args:
        env (VecEnvState): The input environment
        n_timesteps (int): The number of timesteps to batch
        end_of_chronic_behaviour (str, optional): What to do when the end of the chronic is reached
            by one or more timesteps. Can be one of "ignore", "wrap" or "clip", defaults to "ignore"
            - "wrap" means timesteps restart at zero when they are past the chronic end
            - "clip" means timesteps are clipped to the chronic end
            - "ignore" means no special handling for the end of the chronic is done and the function
            - "raise" means a chex assertion is raised - wrap with chex.chexify to catch this
            might return a truncated environment

    Returns:
        VecEnvState: An environment of n_envs = env.n_envs * n_timesteps
    """
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be at least 1")
    if n_timesteps == 1:
        return env

    orig_n_envs = env.n_envs
    env = replicate_env(env, n_timesteps)
    # Timestep increments look like 0, 0, 0, 1, 1, 1, 2, 2, 2, ...
    timestep_increments = jnp.repeat(jnp.arange(n_timesteps), orig_n_envs)
    assert env.n_envs == len(timestep_increments)
    new_timestep = env.timestep + timestep_increments

    if end_of_chronic_behaviour == "wrap":
        new_timestep = new_timestep % env.grid.chronics.n_timesteps[env.chronic]
    elif end_of_chronic_behaviour == "clip":
        new_timestep = jnp.clip(
            new_timestep,
            a_min=0,
            a_max=env.grid.chronics.n_timesteps[env.chronic] - 1,
        )
    elif end_of_chronic_behaviour == "ignore":
        pass
    elif end_of_chronic_behaviour == "raise":
        new_timestep = eqx.error_if(
            new_timestep,
            new_timestep >= env.grid.chronics.n_timesteps[env.chronic],
            "Timestep is past the end of the chronic, use a different end_of_chronic behaviour or a lower n_timesteps",
        )
    else:
        raise ValueError(f"Unknown end_of_chronic_behaviour {end_of_chronic_behaviour}")

    env = replace(env, timestep=new_timestep)
    return env


def aggregate_timebatched(
    data: jnp.ndarray, n_timesteps: int, agg_fn: Callable
) -> jnp.ndarray:
    """Aggregate a timebatched quantity across the time dimension

    Args:
        data (jnp.ndarray): Some data with shape (n_envs * n_timesteps) which is arranged in a
            tiled layout as could be the results of using timebatched
        n_timesteps (int): The number of timesteps used for timebatching
        agg_fn (Callable): The aggregation function to use, should take an axis parameter

    Returns:
        jnp.ndarray: The aggregated array
    """
    chex.assert_rank(data, 1)
    if n_timesteps == 1:
        return data
    if data.shape[0] % n_timesteps != 0:
        raise ValueError(
            f"Data shape {data.shape} is not divisible by n_timesteps {n_timesteps}"
        )
    tmp_data = jnp.reshape(data, (n_timesteps, data.shape[0] // n_timesteps))
    return agg_fn(tmp_data, axis=0)
