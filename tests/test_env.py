import os
from collections import Counter
from itertools import combinations

import chex
import grid2op
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from grid2op.Converter import IdToAct
from grid2op.Observation import BaseObservation
from grid2op.Parameters import Parameters
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from jax_dataclasses import replace

from grid2vec.actions import random_action
from grid2vec.env import (
    PFCResults,
    aggregate_timebatched,
    get_done,
    get_reward,
    get_reward_n,
    get_reward_nminus1,
    get_truncated,
    make_env,
    masked_set,
    postprocess_obs,
    replicate_env,
    timebatch_env,
    timesteps_in_current_chronic,
    vector_reset,
    vector_step,
)
from grid2vec.grid import Grid, chronics_current_timestep, load_grid
from grid2vec.grid2op_compat import import_grid2op_topo_vect, load_grid_grid2op
from grid2vec.result_spec import set_env_dim
from grid2vec.solver_interface import action_to_pandapower, compute_obs
from grid2vec.spaces import get_observation_space


def test_make_env(grid: Grid) -> None:
    env = make_env(grid, 2)
    env2 = jax.jit(make_env, static_argnums=(0, 1))(grid, 2)
    assert env == env2

    assert env is not None
    assert jnp.all(env.timestep == jnp.zeros(2))
    assert jnp.all(env.chronic == jnp.arange(2))  # only true for n_chronics >= n_envs
    assert env.switch_state.shape == (2, grid.n_switch_controllable)


def test_vector_reset(grid: Grid) -> None:
    env = make_env(grid, 2)

    env2 = jax.jit(vector_reset)(env)
    env = vector_reset(env)
    assert env == env2

    assert env is not None
    assert jnp.all(env.timestep == jnp.zeros(2))
    assert jnp.all(
        env.chronic == (jnp.arange(2) + 1) % grid.n_chronics
    )  # only true for n_chronics >= n_envs
    assert env.switch_state.shape == (2, grid.n_switch_controllable)

    env = vector_step(env)
    assert jnp.all(env.timestep == 1)
    env_old = env
    env = vector_reset(env)
    assert jnp.all(env.timestep == 0)
    assert jnp.all(env_old.timestep == 1)

    env = vector_reset(env, target_chronics=jnp.array([0, 0]))
    assert jnp.all(env.chronic == 0)

    env_new = vector_step(env)
    assert env != env_new

    env_new = vector_reset(env_new, target_chronics=jnp.array([0, 0]))
    assert env == env_new


def test_masked_set() -> None:
    mask = jnp.array([[True, False], [False, True]])
    which = jnp.array([True, True])
    new_state = jnp.array([[True, False], [True, False]])
    current_state = jnp.array([[False, True], [False, True]])

    assert jnp.array_equal(
        masked_set(mask, which, new_state, current_state),
        jnp.array([[True, True], [False, False]]),
    )

    which = jnp.array([True, False])
    assert jnp.array_equal(
        masked_set(mask, which, new_state, current_state),
        jnp.array([[True, True], [False, True]]),
    )

    mask = jnp.array([[True, False, True], [False, True, True]])
    with pytest.raises(Exception):
        masked_set(mask, which, new_state, current_state)


def test_vector_step(grid: Grid) -> None:
    env = make_env(grid, 3)
    res_spec = set_env_dim(grid.res_spec, 3)
    obs_space = get_observation_space(res_spec, False)

    env = vector_reset(env)

    # Enable all switches on envs 1 and 2, disable all on env 3
    action = jnp.ones((3, grid.n_switch_controllable), dtype=bool)
    action = action.at[2, :].set(False)
    env_old = env
    env2 = jax.jit(chex.chexify(vector_step))(env, new_switch_state=action)
    env = vector_step(env, new_switch_state=action)
    assert env == env2

    assert jnp.all(env_old.switch_state == 1)
    assert jnp.array_equal(env.timestep, jnp.array([1, 1, 1]))
    assert jnp.all(env.switch_state[0:1] == 1)
    assert jnp.all(env.switch_state[2] == 0)

    # Reset only the first env
    env = vector_reset(env, jnp.array([True, False, False]))
    assert jnp.array_equal(env.timestep, jnp.array([0, 1, 1]))

    # Compute observations
    obs = compute_obs(env)
    obs_proc = postprocess_obs(obs, res_spec)
    assert obs_space.contains(obs_proc)

    # Check if observation space and output match
    obs_proc = postprocess_obs(obs, res_spec)
    for key, value in obs_proc.items():
        assert key in list(obs_space.keys())
        assert obs_space[key].shape == value.shape, f"Shape mismatch for {key}"
        assert obs_space[key].dtype == value.dtype, f"Dtype mismatch for {key}"

    done = get_done(obs)

    assert done.shape == (3,)

    # Recompute only the first env
    obs_part = compute_obs(env, which=jnp.array([True, False, False]))
    assert jnp.allclose(obs_part["p_or_line"][0], obs["p_or_line"][0], equal_nan=True)
    assert jnp.allclose(
        obs_part["loading_line"][0], obs["loading_line"][0], equal_nan=True
    )
    assert jnp.allclose(
        obs_part["loading_trafo"][0], obs["loading_trafo"][0], equal_nan=True
    )
    assert jnp.allclose(
        obs_part["loading_trafo3w"][0], obs["loading_trafo3w"][0], equal_nan=True
    )


def test_vector_step_which(grid: Grid) -> None:
    env = make_env(grid, 3)
    env = vector_reset(env)

    # Disable all switches
    action = jnp.zeros((3, grid.n_switch_controllable), dtype=bool)
    env = vector_step(
        env, new_switch_state=action, which=jnp.array([True, False, False])
    )
    assert not jnp.any(env.switch_state[0])
    assert jnp.all(env.switch_state[1:2])
    assert jnp.array_equal(env.timestep, jnp.array([1, 0, 0]))

    # Behaves well together with masks
    env = vector_reset(env)
    action = jnp.zeros((3, grid.n_switch_controllable), dtype=bool)
    mask = jnp.array(np.random.randn(3, grid.n_switch_controllable) > 0)
    env = vector_step(
        env,
        new_switch_state=action,
        switch_mask=mask,
        which=jnp.array([True, False, False]),
    )
    # Where mask is true, switch state should be false
    assert jnp.array_equal(~mask[0], env.switch_state[0])


def test_vector_step_mask(grid: Grid) -> None:
    env = make_env(grid, 3)

    # Disable all switches
    env = vector_reset(env)
    action = jnp.zeros((3, grid.n_switch_controllable), dtype=bool)
    mask = jnp.array(np.random.randn(3, grid.n_switch_controllable) > 0)
    env = vector_step(env, new_switch_state=action, switch_mask=mask)
    # Where mask is true, switch state should be false
    assert jnp.array_equal(~mask, env.switch_state)

    # Disable all lines
    env = vector_reset(env)
    action = jnp.zeros((3, grid.n_line_controllable), dtype=bool)
    mask = jnp.array(np.random.randn(3, grid.n_line_controllable) > 0)
    env = vector_step(env, new_line_state=action, line_mask=mask)
    # Where mask is true, line state should be false
    assert jnp.array_equal(~mask, env.line_state)

    # Set trafo taps - first set them all to their lowest value
    # Then set them masked to their highest
    env = vector_reset(env)
    action = jnp.repeat(jnp.expand_dims(grid.trafo_tap_min, 0), 3, axis=0)
    env = vector_step(env, new_trafo_taps=action)
    assert jnp.array_equal(env.trafo_tap_pos, action)

    action = jnp.repeat(jnp.expand_dims(grid.trafo_tap_max, 0), 3, axis=0)
    mask = jnp.array(np.random.randn(3, grid.n_trafo_tap_controllable) > 0)
    env = vector_step(env, new_trafo_taps=action, trafo_tap_mask=mask)
    # Where mask is true, trafo taps should be at max
    assert jnp.array_equal(mask, env.trafo_tap_pos == grid.trafo_tap_max)
    # Where mask is false, trafo taps should be at min
    assert jnp.array_equal(~mask, env.trafo_tap_pos == grid.trafo_tap_min)

    # Set trafo3w taps - first set them all to their lowest value
    # Then set them masked to their highest
    env = vector_reset(env)
    action = jnp.repeat(jnp.expand_dims(grid.trafo3w_tap_min, 0), 3, axis=0)
    env = vector_step(env, new_trafo3w_taps=action)
    assert jnp.array_equal(env.trafo3w_tap_pos, action)

    action = jnp.repeat(jnp.expand_dims(grid.trafo3w_tap_max, 0), 3, axis=0)
    mask = jnp.array(np.random.randn(3, grid.n_trafo3w_tap_controllable) > 0)
    env = vector_step(env, new_trafo3w_taps=action, trafo3w_tap_mask=mask)
    # Where mask is true, trafo3w taps should be at max
    assert jnp.array_equal(mask, env.trafo3w_tap_pos == grid.trafo3w_tap_max)
    # Where mask is false, trafo3w taps should be at min
    assert jnp.array_equal(~mask, env.trafo3w_tap_pos == grid.trafo3w_tap_min)


def test_nminus1_step(grid_folder: str) -> None:
    grid = load_grid(grid_folder, nminus1=True)
    assert grid.nminus1_definition is not None
    env = make_env(grid, 1)
    res_spec = set_env_dim(grid.res_spec, 1)
    obs_space = get_observation_space(res_spec, False)

    assert "line_loading_per_failure" in obs_space.spaces.keys()
    assert "trafo_loading_per_failure" in obs_space.spaces.keys()
    assert "trafo3w_loading_per_failure" in obs_space.spaces.keys()
    assert "nminus1_converged" in obs_space.spaces.keys()

    env = vector_reset(env)

    obs = compute_obs(env)
    obs_proc = postprocess_obs(obs, res_spec)
    assert obs_space.contains(obs_proc)
    assert "line_loading_per_failure" in obs
    assert obs["line_loading_per_failure"].shape == (
        1,
        len(grid.nminus1_definition),
        grid.n_line,
    )

    if grid.n_line:
        counter: Counter = Counter()
        idx_range = list(range(0, obs["line_loading_per_failure"].shape[1]))
        for idx1, idx2 in combinations(idx_range, 2):
            if not jnp.array_equal(
                obs["line_loading_per_failure"][0][idx1],
                obs["line_loading_per_failure"][0][idx2],
                equal_nan=True,
            ):
                counter["not_all_equal"] += 1
                break
            else:
                counter["all_equal"] += 1

        # Only tolerate a few equal results in the N-1 computation results
        assert counter["all_equal"] <= obs["line_loading_per_failure"].shape[1] // 10

    assert "trafo_loading_per_failure" in obs
    assert obs["trafo_loading_per_failure"].shape == (
        1,
        len(grid.nminus1_definition),
        grid.n_trafo,
    )

    if grid.n_trafo:
        counter = Counter()
        idx_range = list(range(0, obs["trafo_loading_per_failure"].shape[1]))
        for idx1, idx2 in combinations(idx_range, 2):
            if not jnp.array_equal(
                obs["trafo_loading_per_failure"][0][idx1],
                obs["trafo_loading_per_failure"][0][idx2],
                equal_nan=True,
            ):
                counter["not_all_equal"] += 1
                break
            else:
                counter["all_equal"] += 1

        # Only tolerate a few equal results in the N-1 computation results
        assert counter["all_equal"] <= obs["trafo_loading_per_failure"].shape[1] // 10

    assert "trafo3w_loading_per_failure" in obs
    assert obs["trafo3w_loading_per_failure"].shape == (
        1,
        len(grid.nminus1_definition),
        grid.n_trafo3w,
    )

    if grid.n_trafo3w:
        counter = Counter()
        idx_range = list(range(0, obs["trafo3w_loading_per_failure"].shape[1]))
        for idx1, idx2 in combinations(idx_range, 2):
            if not np.array_equal(
                obs["trafo3w_loading_per_failure"][0][idx1],
                obs["trafo3w_loading_per_failure"][0][idx2],
                equal_nan=True,
            ):
                counter["not_all_equal"] += 1
                break
            else:
                counter["all_equal"] += 1

        # Only tolerate a few equal results in the N-1 computation results
        assert counter["all_equal"] <= obs["trafo3w_loading_per_failure"].shape[1] // 10

    assert "nminus1_converged" in obs
    assert obs["nminus1_converged"].shape == (1, len(grid.nminus1_definition))
    assert np.sum(obs["nminus1_converged"]) > 0


def test_truncated(grid: Grid) -> None:
    grid = grid
    env = make_env(grid, 1)
    env = vector_reset(env, target_chronics=jnp.array([0]))
    assert not jnp.any(get_truncated(env))
    for _ in range(grid.chronics.n_timesteps[0] - 1):
        assert get_truncated(env).item() is False
        assert timesteps_in_current_chronic(env).item() > 0
        env = vector_step(env)

    assert get_truncated(env).item() is True
    assert timesteps_in_current_chronic(env).item() == 0
    with pytest.raises(Exception):
        vector_step_chexified = chex.chexify(vector_step)
        env = vector_step_chexified(env)
        vector_step_chexified.wait_checks()


def grid2op_matches_grid2elia(
    obs: PFCResults, g2o_obs: BaseObservation, n_line: int
) -> None:
    assert np.allclose(obs["gen_p"].flatten(), g2o_obs.gen_p, rtol=1e-3, atol=1e-6)
    assert np.allclose(obs["gen_q"].flatten(), g2o_obs.gen_q, rtol=1e-3, atol=1e-6)
    assert np.allclose(obs["load_p"].flatten(), g2o_obs.load_p, rtol=1e-3, atol=1e-6)
    assert np.allclose(obs["load_q"].flatten(), g2o_obs.load_q, rtol=1e-3, atol=1e-6)
    # Have a little loose tolerances, as the powerflow might not converge to exactly the same values
    assert np.allclose(
        obs["p_or_line"].flatten(), g2o_obs.p_or[:n_line], rtol=1e-3, atol=1e-6
    )
    assert np.allclose(
        obs["p_ex_line"].flatten(), g2o_obs.p_ex[:n_line], rtol=1e-3, atol=1e-6
    )
    assert np.allclose(
        obs["p_hv_trafo"].flatten(), g2o_obs.p_or[n_line:], rtol=1e-3, atol=1e-6
    )
    assert np.allclose(
        obs["p_lv_trafo"].flatten(), g2o_obs.p_ex[n_line:], rtol=1e-3, atol=1e-6
    )
    assert np.allclose(
        obs["q_or_line"].flatten(), g2o_obs.q_or[:n_line], rtol=1e-3, atol=1e-6
    )
    assert np.allclose(
        obs["q_ex_line"].flatten(), g2o_obs.q_ex[:n_line], rtol=1e-3, atol=1e-6
    )
    assert np.allclose(
        obs["q_hv_trafo"].flatten(), g2o_obs.q_or[n_line:], rtol=1e-3, atol=1e-6
    )
    assert np.allclose(
        obs["q_lv_trafo"].flatten(), g2o_obs.q_ex[n_line:], rtol=1e-3, atol=1e-6
    )

    # For some reason, grid2op implemented a different trafo loading computation
    # Actually, I trust the pandapower one more...
    # assert np.allclose(obs.trafo_loading.flatten() / 100, g2o_obs.rho[n_line:])
    assert np.allclose(
        obs["loading_line_grid2op"].flatten(),
        g2o_obs.rho[:n_line],
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.allclose(
        obs["loading_trafo_grid2op"].flatten(),
        g2o_obs.rho[n_line:],
        rtol=1e-3,
        atol=1e-6,
    )


@given(chronic_id=st.integers(min_value=0, max_value=9), timestep=st.integers(0, 10))
@settings(deadline=None, max_examples=25)
@example(chronic_id=3, timestep=0)
def test_grid2op_compatible_do_nothing(chronic_id: int, timestep: int) -> None:
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    g2o_env = grid2op.make("l2rpn_case14_sandbox", param=params)
    grid = load_grid_grid2op("l2rpn_case14_sandbox", list(range(10)))

    # Check if power flow inputs and outputs are the same between grid2op and grid2elia
    g2o_env.set_id(chronic_id)
    g2o_obs = g2o_env.reset()
    if timestep > 0:
        for _ in range(timestep):
            g2o_obs, _, done, _ = g2o_env.step(g2o_env.action_space({}))
            assume(not done)
            assume(np.all(g2o_obs.line_status))

    env = make_env(grid, 1)
    env = vector_reset(env, target_chronics=jnp.array([chronic_id]))

    if timestep > 0:
        for _ in range(timestep):
            env = vector_step(env)

    obs = compute_obs(env)

    load_p, load_q, prod_p, prod_v = chronics_current_timestep(
        env.timestep, env.chronic, env.grid.chronics
    )

    # Inputs match
    assert np.allclose(g2o_env.backend._grid.load.p_mw.values, load_p)
    assert np.allclose(g2o_env.backend._grid.load.q_mvar.values, load_q)
    assert np.allclose(g2o_env.backend._grid.gen.p_mw.values, prod_p)
    vn_kv = env.grid.net.bus.loc[env.grid.net.gen.bus]["vn_kv"].values
    assert np.allclose(g2o_env.backend._grid.gen.vm_pu.values, prod_v / vn_kv)

    # Outputs match
    grid2op_matches_grid2elia(obs, g2o_obs, n_line=grid.n_line)


@given(
    chronic_id=st.integers(min_value=0, max_value=9),
    timestep=st.integers(0, 10),
    line_to_disable=st.integers(min_value=0, max_value=13),
)
@settings(deadline=None, max_examples=25)
def test_grid2op_compatible_line_actions(
    chronic_id: int, timestep: int, line_to_disable: int
) -> None:
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    g2o_env = grid2op.make("l2rpn_case14_sandbox", param=params)
    grid = load_grid_grid2op("l2rpn_case14_sandbox", list(range(10)))

    g2o_env.set_id(chronic_id)
    g2o_env.reset()

    if timestep > 0:
        for _ in range(timestep):
            g2o_obs, _, done, _ = g2o_env.step(g2o_env.action_space({}))
            assume(not done)
            assume(np.all(g2o_obs.line_status))

    env = make_env(grid, 1)
    env = vector_reset(env, target_chronics=jnp.array([chronic_id]))

    if timestep > 0:
        for _ in range(timestep):
            env = vector_step(env)

    # Apply the action on grid2elia
    new_line_state = env.line_state.at[0, line_to_disable].set(False)
    env = vector_step(env, new_line_state=new_line_state)
    pp_action = action_to_pandapower(env)
    all_line_state = pp_action[("line", "in_service")]
    obs = compute_obs(env)

    # Apply the action on grid2op
    action = g2o_env.action_space({})
    action.line_set_status = (line_to_disable, -1)
    g2o_obs, _, g2o_done, _ = g2o_env.step(action)
    assert not g2o_obs.line_status[line_to_disable]

    assume(not g2o_done)
    expected_line_status = np.ones(g2o_env.n_line, dtype=bool)
    expected_line_status[line_to_disable] = False
    assume(np.array_equal(g2o_obs.line_status, expected_line_status))

    assert np.array_equal(
        np.squeeze(all_line_state), expected_line_status[: grid.n_line]
    )
    # Outputs match
    grid2op_matches_grid2elia(obs, g2o_obs, n_line=grid.n_line)


@given(
    chronic_id=st.integers(min_value=0, max_value=9),
    timestep=st.integers(0, 10),
    action=st.integers(min_value=0),
)
@settings(deadline=None, max_examples=25)
def test_grid2op_compatible_topology_actions(
    chronic_id: int, timestep: int, action: int
) -> None:
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    g2o_env = grid2op.make("l2rpn_case14_sandbox", param=params)
    grid = load_grid_grid2op("l2rpn_case14_sandbox", list(range(10)))

    # Forward grid2op to the desired timestep and chronic
    g2o_env.set_id(chronic_id)
    g2o_env.reset()
    if timestep > 0:
        for _ in range(timestep):
            g2o_obs, _, done, _ = g2o_env.step(g2o_env.action_space({}))
            assume(not done)
            assume(np.all(g2o_obs.line_status))

    # Forward grid2elia to the desired timestep and chronic
    env = make_env(grid, 1)
    env = vector_reset(env, target_chronics=jnp.array([chronic_id]))
    if timestep > 0:
        for _ in range(timestep):
            env = vector_step(env)

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

    # Apply the action on grid2op
    g2o_obs, _, g2o_done, _ = g2o_env.step(g2o_act)
    assume(not g2o_done)
    assume(np.all(g2o_obs.line_status))

    # Apply the action on grid2elia
    topo_vect = import_grid2op_topo_vect(g2o_obs.topo_vect, g2o_env)
    env = vector_step(env, new_topo_vect=np.expand_dims(topo_vect, 0))
    obs = compute_obs(env)

    grid2op_matches_grid2elia(obs, g2o_obs, n_line=grid.n_line)


def test_reward_n(grid: Grid) -> None:
    env = make_env(grid, 2)
    env = vector_reset(env)
    obs = compute_obs(env)
    reward = get_reward_n(obs, crit_threshold=0.01)

    assert reward.shape == (2,)
    assert not jnp.any(jnp.isnan(reward))

    reward = get_reward_n(
        obs,
        crit_threshold=0.01,
        mask_line=jnp.zeros_like(grid.line_for_reward),
        mask_trafo=jnp.zeros_like(grid.trafo_for_reward),
        mask_trafo3w=jnp.zeros_like(grid.trafo3w_for_reward),
    )
    assert jnp.array_equal(reward, jnp.array([1, 1]))

    reward_2 = get_reward_n(
        obs,
        crit_threshold=0.01,
        mask_line=grid.line_for_reward,
        mask_trafo=grid.trafo_for_reward,
        mask_trafo3w=grid.trafo3w_for_reward,
    )

    assert jnp.all(reward != reward_2)


def test_reward_nminus1(grid_folder: str) -> None:
    grid = load_grid(grid_folder, nminus1=True)
    env = make_env(grid, 2)
    env = vector_reset(env)
    obs = compute_obs(env)
    reward = get_reward_nminus1(
        obs,
        grid.default_crit_threshold,
        line_capacity=grid.line_capacity_masked_for_reward,
        trafo_capacity=grid.trafo_capacity_masked_for_reward,
        trafo3w_capacity=grid.trafo3w_capacity_masked_for_reward,
    )

    assert reward.shape == (2,)
    assert not jnp.any(jnp.isnan(reward))
    assert jnp.all(reward != 1)

    reward = get_reward_nminus1(
        obs,
        grid.default_crit_threshold,
        line_capacity=jnp.zeros_like(grid.line_capacity_masked_for_reward),
        trafo_capacity=jnp.zeros_like(grid.trafo_capacity_masked_for_reward),
        trafo3w_capacity=jnp.zeros_like(grid.trafo3w_capacity_masked_for_reward),
    )
    assert jnp.array_equal(reward, jnp.array([1, 1]))


def test_replicate_env(grid: Grid) -> None:
    env = make_env(grid, 3)
    env = vector_reset(env, target_chronics=jnp.array([0, 1, 0]))
    action = random_action(grid, 3)
    assert action.new_line_state is not None
    action = replace(
        action,
        new_line_state=replace(
            action.new_line_state, mask=jnp.ones_like(action.new_line_state.mask)
        ),
    )
    env = vector_step(env, **action.asdict(), step_time=0)
    assert action.new_line_state is not None
    assert np.array_equal(env.line_state, action.new_line_state.new_state)

    env = replicate_env(env, 4)
    assert env.n_envs == 3 * 4
    assert np.array_equal(env.chronic, np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]))
    assert np.array_equal(
        env.line_state, np.tile(action.new_line_state.new_state, (4, 1))
    )


def test_timebatch_env(grid: Grid) -> None:
    env = make_env(grid, 3)
    env = vector_reset(env, target_chronics=jnp.array([0, 1, 0]))
    env = vector_step(env, which=jnp.array([True, False, False]))

    assert jnp.array_equal(env.timestep, jnp.array([1, 0, 0]))
    assert jnp.array_equal(env.chronic, jnp.array([0, 1, 0]))

    env_batched = timebatch_env(env, 2, end_of_chronic_behaviour="ignore")
    assert env_batched.n_envs == 3 * 2
    assert jnp.array_equal(env_batched.timestep, jnp.array([1, 0, 0, 2, 1, 1]))

    env_batched = timebatch_env(env, 1)
    assert env == env_batched

    with pytest.raises(ValueError):
        timebatch_env(env, 0)

    n_batches = int(max(env.grid.chronics.n_timesteps))
    env_batched = timebatch_env(env, n_batches, end_of_chronic_behaviour="ignore")
    assert env_batched.n_envs == 3 * n_batches
    assert jnp.max(env_batched.timestep) == n_batches
    out_of_bounds = (
        env_batched.timestep
        >= env_batched.grid.chronics.n_timesteps[env_batched.chronic]
    )

    assert jnp.any(out_of_bounds)

    env_batched = timebatch_env(env, n_batches, end_of_chronic_behaviour="clip")
    assert env_batched.n_envs == 3 * n_batches
    assert jnp.all(
        env_batched.timestep[out_of_bounds]
        == env_batched.grid.chronics.n_timesteps[env_batched.chronic][out_of_bounds] - 1
    )
    assert not jnp.any(env_batched.timestep[out_of_bounds] == 0)

    env_batched = timebatch_env(env, n_batches, end_of_chronic_behaviour="wrap")
    assert env_batched.n_envs == 3 * n_batches
    assert not jnp.any(
        env_batched.timestep[out_of_bounds]
        == env_batched.grid.chronics.n_timesteps[env_batched.chronic][out_of_bounds] - 1
    )
    # Not jnp.all but jnp.any here, as some might have wrapped around further
    assert jnp.any(env_batched.timestep[out_of_bounds] == 0)

    timebatch_env_chexified = chex.chexify(timebatch_env)
    with pytest.raises(Exception):
        env_batched = timebatch_env_chexified(
            env, n_batches, end_of_chronic_behaviour="raise"
        )
        timebatch_env_chexified.wait_checks()

    env_batched = timebatch_env_chexified(env, 2, end_of_chronic_behaviour="raise")
    timebatch_env_chexified.wait_checks()
    assert env_batched.n_envs == 3 * 2
    assert jnp.array_equal(env_batched.timestep, jnp.array([1, 0, 0, 2, 1, 1]))


def test_timebatch_env_grid2op() -> None:
    grid = load_grid_grid2op(
        "l2rpn_case14_sandbox", include_chronic_indices=[0, 1, 2, 3]
    )
    env = make_env(grid, 2)
    env = vector_reset(env, target_chronics=jnp.array([0, 1]))
    env = timebatch_env(env, 3)
    assert jnp.array_equal(env.chronic, jnp.array([0, 1, 0, 1, 0, 1]))
    assert jnp.array_equal(env.timestep, jnp.array([0, 0, 1, 1, 2, 2]))
    obs = compute_obs(env)
    reward = get_reward(env.grid, obs, crit_threshold=0)
    assert reward.shape == (6,)
    assert len(np.unique(reward)) == 6
    agg_reward = aggregate_timebatched(reward, 3, jnp.sum)
    assert agg_reward.shape == (2,)
    assert jnp.array_equal(
        agg_reward,
        jnp.array(
            [reward[0] + reward[2] + reward[4], reward[1] + reward[3] + reward[5]]
        ),
    )


@pytest.mark.skip(
    reason="Can only be tested with the grid planning scenario, which is currently not in the CI"
)
def test_grid_planning_data() -> None:
    folder = os.path.join(
        os.path.dirname(__file__), "../../data_grid2op/grid_planning/"
    )
    grid = load_grid(folder, nminus1=False, dc=True)
    env = vector_reset(make_env(grid, 1))
    obs = compute_obs(env)

    assert obs["loading_line"].shape == (1, grid.n_line)
    assert np.sum(np.isnan(obs["loading_line"])) < grid.n_line / 2
    assert obs["loading_trafo"].shape == (1, grid.n_trafo)
    assert np.sum(np.isnan(obs["loading_trafo"])) < grid.n_trafo / 2
    assert obs["loading_trafo3w"].shape == (1, grid.n_trafo3w)
    assert np.sum(np.isnan(obs["loading_trafo3w"])) < grid.n_trafo3w / 2

    reward = get_reward(grid, obs)
    assert reward.shape == (1,)
    assert not np.any(np.isnan(reward))


@pytest.mark.skip(
    reason="Can only be tested with the grid planning scenario, which is currently not in the CI"
)
def test_grid_planning_data_nminus1() -> None:
    folder = os.path.join(
        os.path.dirname(__file__), "../../data_grid2op/grid_planning/"
    )
    grid = load_grid(folder, nminus1=True, dc=True)
    assert grid.nminus1_definition is not None

    # Replace the nminus1 definition with a subset to make the computation f
    new_line_mask = np.zeros_like(grid.nminus1_definition.line_mask)
    new_line_mask[np.argwhere(grid.nminus1_definition.line_mask).flatten()[0:3]] = 1
    new_definition = replace(
        grid.nminus1_definition,
        trafo_mask=np.zeros_like(grid.nminus1_definition.trafo_mask),
        trafo3w_mask=np.zeros_like(grid.nminus1_definition.trafo3w_mask),
        line_mask=new_line_mask,
    )
    grid = replace(grid, nminus1_definition=new_definition)
    assert grid.nminus1_definition is not None
    assert grid.nminus1_definition.n_failures == 3

    env = vector_reset(make_env(grid, 1))
    obs = compute_obs(env)

    assert obs["loading_line"].shape == (1, grid.n_line)
    assert np.sum(np.isnan(obs["loading_line"])) < grid.n_line / 2
    assert not np.all(obs["loading_line"] == 0)
    assert obs["loading_trafo"].shape == (1, grid.n_trafo)
    assert np.sum(np.isnan(obs["loading_trafo"])) < grid.n_trafo / 2
    assert not np.all(obs["loading_trafo"] == 0)
    assert obs["loading_trafo3w"].shape == (1, grid.n_trafo3w)
    assert np.sum(np.isnan(obs["loading_trafo3w"])) < grid.n_trafo3w / 2
    assert not np.all(obs["loading_trafo3w"] == 0)

    assert obs["line_loading_per_failure"].shape == (1, 3, grid.n_line)
    assert np.sum(np.isnan(obs["line_loading_per_failure"])) < 3 * grid.n_line / 2
    assert not np.all(obs["line_loading_per_failure"] == 0)

    assert obs["trafo_loading_per_failure"].shape == (1, 3, grid.n_trafo)
    assert np.sum(np.isnan(obs["trafo_loading_per_failure"])) < 3 * grid.n_trafo / 2
    assert not np.all(obs["trafo_loading_per_failure"] == 0)

    assert obs["trafo3w_loading_per_failure"].shape == (1, 3, grid.n_trafo3w)
    assert np.sum(np.isnan(obs["trafo3w_loading_per_failure"])) < 3 * grid.n_trafo3w / 2
    assert not np.all(obs["trafo3w_loading_per_failure"] == 0)

    reward = get_reward(grid, obs)
    assert reward.shape == (1,)
    assert not np.any(np.isnan(reward))
