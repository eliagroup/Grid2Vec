# The idea is to evaluate a set of actions at a time and find the best one. This is done by
# replicating the environment in to a batch of environments, stepping each and computing the rewards

from typing import Callable

import jax.numpy as jnp

from grid2vec.actions import Action
from grid2vec.env import PFCResults, VecEnvState, get_reward, replicate_env, vector_step
from grid2vec.grid import Grid
from grid2vec.solver_interface import compute_obs


def action_search(
    env: VecEnvState,
    actions: Action,
    reward_fn: Callable[[Grid, PFCResults], jnp.ndarray] = get_reward,
) -> jnp.ndarray:
    """Performs a bruteforce action search for a vector of environments and actions

    Replicates the environment in case the number of actions is a multiple of the environment dim

    Args:
        env (VecEnvState): A vectorized environment state
        actions (Action): An action vector with n_envs being a multiple of env.n_envs

    Returns:
        np.ndarray: A numpy array of shape (actions.n_envs,) with the rewards for each action
    """

    if actions.n_envs == env.n_envs:
        joint_env = env
    else:
        if actions.n_envs % env.n_envs != 0:
            raise ValueError(
                f"Number of environments in action vector ({actions.n_envs}) is not a multiple of "
                f"number of environments in environment vector ({env.n_envs})"
            )
        joint_env = replicate_env(env, int(actions.n_envs // env.n_envs))

    joint_env = vector_step(joint_env, **actions.asdict(), step_time=0)
    obs = compute_obs(joint_env)
    return reward_fn(joint_env.grid, obs)
