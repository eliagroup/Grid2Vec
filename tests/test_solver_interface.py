from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest

from grid2vec.env import make_env, vector_reset
from grid2vec.grid import Grid, load_grid
from grid2vec.grid2op_compat import load_grid_grid2op
from grid2vec.result_spec import describe_chronics, set_env_dim, spec_to_jax
from grid2vec.solver_interface import (
    action_to_pandapower,
    collect_pfc_inputs,
    compute_obs,
    invoke_solver,
    make_profile,
)


def test_action_to_pandapower(grid: Grid) -> None:
    env = make_env(grid, 2)
    env = vector_reset(env)
    pp_action = action_to_pandapower(env)

    # With a default action, everything should be as in the default grid
    for (table, key), value in pp_action.items():
        assert jnp.array_equal(value[0], grid.net[table][key].values, equal_nan=True)
        assert jnp.array_equal(value[1], grid.net[table][key].values, equal_nan=True)


def test_make_profile(grid: Grid) -> None:
    env = make_env(grid, 1)
    env = vector_reset(env, target_chronics=jnp.array([0]))
    profile = make_profile(env)
    profile2 = jax.jit(make_profile)(env)

    chex.assert_trees_all_close(profile2, profile)

    for key, value in profile.items():
        assert value.shape[0] == env.n_envs
        (table, column) = key
        assert value.shape[1] == len(grid.net[table][column])
        assert jnp.allclose(value[0], grid.net[table][column].values, equal_nan=True)


def test_collect_pfc_inputs(grid: Grid) -> None:
    env = make_env(grid, 2)
    env = vector_reset(env)
    profile = make_profile(env)
    specs = describe_chronics(
        n_envs=env.n_envs,
        n_load=env.grid.n_load,
        n_gen=env.grid.n_gen,
        n_sgen=env.grid.n_sgen,
        n_switch=env.grid.n_switch,
        n_line=env.grid.n_line,
        n_trafo=env.grid.n_trafo,
        n_trafo3w=env.grid.n_trafo3w,
    )

    pfc_inputs = collect_pfc_inputs(env, profile)

    assert pfc_inputs is not None
    assert set(pfc_inputs.keys()) == set([r.key for r in specs])
    for r in specs:
        assert r.key in pfc_inputs
        assert r.shape == pfc_inputs[r.key].shape
        assert r.dtype == pfc_inputs[r.key].dtype

    assert jnp.all(pfc_inputs["topo_vect"] >= 0)
    assert jnp.all(pfc_inputs["topo_vect"] < grid.max_bus_per_sub)


# TODO figure this out
@pytest.mark.xfail(reason="I couldn't figure this out yet...")
def test_compute_obs_jit(grid: Grid) -> None:
    env = make_env(grid, 1)
    env = vector_reset(env)

    obs2 = jax.jit(compute_obs)(env)
    obs = compute_obs(env)
    chex.assert_trees_all_close(obs2, obs, rtol=1e-3, atol=1e-6)


def test_compute_obs_jit_sandbox() -> None:
    grid = load_grid_grid2op(
        "l2rpn_case14_sandbox", include_chronic_indices=list(range(2))
    )
    env = make_env(grid, 1)
    env = vector_reset(env)

    obs2 = jax.jit(compute_obs)(env)
    obs = compute_obs(env)
    chex.assert_trees_all_close(obs2, obs, rtol=1e-3, atol=1e-6)


def test_compute_obs(grid: Grid) -> None:
    env = make_env(grid, 2)
    env = vector_reset(env)
    obs = compute_obs(env)

    res_spec = set_env_dim(env.grid.res_spec, env.n_envs)
    jax_spec = spec_to_jax(res_spec)

    extra_keys = [
        "load_p_input",
        "load_q_input",
        "prod_p_input",
        "prod_v_input",
        "switch_state",
        "line_state",
        "trafo_state",
        "trafo_tap_pos",
        "trafo3w_tap_pos",
        "topo_vect",
    ]

    assert obs is not None
    assert set(obs.keys()) == set(jax_spec.keys()).union(set(extra_keys))
    for key, value in jax_spec.items():
        assert key in obs
        assert value.shape == obs[key].shape
        assert value.dtype == obs[key].dtype

    obs_old = obs

    # Using which we expect the same result shapes but filled with zeros
    which = jnp.array([True, False])
    obs = compute_obs(env, which=which)
    assert obs is not None
    assert set(obs.keys()) == set(jax_spec.keys()).union(set(extra_keys))
    for key, value in jax_spec.items():
        assert key in obs
        assert value.shape == obs[key].shape
        assert value.dtype == obs[key].dtype
        assert jnp.allclose(obs_old[key][0], obs[key][0], equal_nan=True)
        if key not in extra_keys:
            assert jnp.array_equal(jnp.zeros_like(obs[key][0]), obs[key][1])


def test_invoke_solver(grid: Grid) -> None:
    env = make_env(grid, 2)
    env = vector_reset(env)

    profile = make_profile(env)

    res_spec = set_env_dim(env.grid.res_spec, env.n_envs)

    obs = invoke_solver(
        profile=profile,
        net=env.grid.net,
        nminus1_definition=env.grid.nminus1_definition,
        res_spec=res_spec,
        dc=env.grid.dc,
        which=None,
    )

    jax_spec = spec_to_jax(res_spec)

    obs2 = jax.pure_callback(
        partial(
            invoke_solver,
            net=env.grid.net,
            res_spec=res_spec,
            dc=env.grid.dc,
            nminus1_definition=env.grid.nminus1_definition,
        ),
        result_shape_dtypes=jax_spec,
        profile=profile,
        which=None,
    )
    chex.assert_trees_all_close(obs2, obs, rtol=1e-3, atol=1e-6)

    assert obs is not None
    assert set(obs.keys()) == set([r.key for r in res_spec])
    for r in res_spec:
        assert r.key in obs
        assert r.shape == obs[r.key].shape
        assert r.dtype == obs[r.key].dtype

    obs_old = obs

    obs = invoke_solver(
        profile=profile,
        net=env.grid.net,
        nminus1_definition=env.grid.nminus1_definition,
        res_spec=res_spec,
        dc=env.grid.dc,
        which=jnp.array([True, False]),
    )

    assert obs is not None
    assert set(obs.keys()) == set([r.key for r in res_spec])
    for r in res_spec:
        assert r.key in obs
        assert r.shape == obs[r.key].shape
        assert r.dtype == obs[r.key].dtype
        assert jnp.allclose(obs_old[r.key][0], obs[r.key][0], equal_nan=True)
        assert jnp.array_equal(jnp.zeros_like(obs[r.key][0]), obs[r.key][1])


def test_invoke_solver_nminus1(grid_folder: str) -> None:
    grid = load_grid(grid_folder, nminus1=True)
    env = make_env(grid, 1)
    env = vector_reset(env)

    profile = make_profile(env)

    res_spec = set_env_dim(env.grid.res_spec, env.n_envs)

    obs = invoke_solver(
        profile=profile,
        net=env.grid.net,
        nminus1_definition=env.grid.nminus1_definition,
        res_spec=res_spec,
        dc=env.grid.dc,
        which=None,
    )

    assert obs is not None
    assert set(obs.keys()) == set([r.key for r in res_spec])
    for r in res_spec:
        assert r.key in obs
        assert r.shape == obs[r.key].shape
        assert r.dtype == obs[r.key].dtype
