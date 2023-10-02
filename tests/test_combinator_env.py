import grid2op
import jax
import jax.numpy as jnp

from grid2vec.actions import is_do_nothing_action
from grid2vec.combinator_env import (
    compute_obs,
    get_reward,
    is_action_excluded,
    make_combinator_env,
    random_action,
    vector_step,
)
from grid2vec.env import make_env, vector_reset
from grid2vec.grid import Grid
from grid2vec.grid2op_compat import grid2op_action_dump_to_grid2elia


def test_combinator_env_grid2op(
    sandbox_action_dump_file: str, sandbox_grid: Grid
) -> None:
    grid = sandbox_grid
    env = make_env(grid, 2)
    env = vector_reset(env)

    g2o_env = grid2op.make("l2rpn_case14_sandbox")
    dump = grid2op_action_dump_to_grid2elia(sandbox_action_dump_file, g2o_env)

    comb_env = make_combinator_env(env, dump, 2)
    assert is_do_nothing_action(comb_env.accumulated_action)

    for i in range(10):
        action = random_action(jax.random.PRNGKey(i), comb_env)
        assert not jnp.any(is_action_excluded(comb_env, action))
        comb_env = vector_step(comb_env, action)

    obs = compute_obs(comb_env)
    reward = get_reward(comb_env, obs)
    assert jnp.all(jnp.isfinite(reward))
