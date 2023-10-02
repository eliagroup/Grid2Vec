import chex
import jax
import jax.numpy as jnp

from grid2vec.action_search import action_search
from grid2vec.action_set import eye_mask_action_set
from grid2vec.actions import make_action
from grid2vec.env import make_env, vector_reset
from grid2vec.grid import Grid


def test_action_search(grid: Grid) -> None:
    env = make_env(grid, 1)
    env = vector_reset(env)
    # this enumerates all line change actions
    actions = make_action(new_line_state=eye_mask_action_set(~env.line_state))[0:10]
    res = action_search(env, actions)
    assert res is not None
    assert len(res) == 10

    res2 = jax.jit(chex.chexify(action_search), static_argnames=("reward_fn",))(
        env, actions
    )
    assert jnp.allclose(res, res2)
