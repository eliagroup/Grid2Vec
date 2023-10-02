# The idea of this module is to enable a search over an action dump to find a good joint action,
# combined of a set of unitary actions. The search is done over a fixed window into the chronics
# and not as a time-based MDP across the chronics.

import chex
import jax
import jax.numpy as jnp
from jax_dataclasses import Static, pytree_dataclass, replace

from grid2vec.action_dump import ActionDump
from grid2vec.actions import Action, do_nothing_action, merge_actions, pad_out_like
from grid2vec.env import PFCResults, VecEnvState, aggregate_timebatched
from grid2vec.env import get_reward as get_reward_vanilla
from grid2vec.env import timebatch_env
from grid2vec.solver_interface import compute_obs as compute_obs_vanilla


@pytree_dataclass
class CombinatorEnv:
    env_state: VecEnvState  # A wrapped vecenv state holding the current combination, env dim = timebatching * n_envs
    action_dump: Static[ActionDump]  # The action dump this combinator is working on
    accumulated_action: Action
    exclusion_mask: jnp.ndarray  # The accumulated exclusion mask, this is effectively an action mask

    timebatching: Static[int]  # How many timesteps to batch together

    def __len__(self):
        return len(self.env_state)


def make_combinator_env(
    env_state: VecEnvState, action_dump: ActionDump, timebatching: int
) -> CombinatorEnv:
    """Creates a combinator environment

    Args:
        env_state (VecEnvState): The initial environment state
        action_dump (ActionDump): The action dump to work on
        timebatching (int): How many timesteps to batch together

    Returns:
        CombinatorEnv: The combinator environment
    """
    return CombinatorEnv(
        env_state=env_state,
        action_dump=action_dump,
        accumulated_action=pad_out_like(
            do_nothing_action(len(env_state)), action_dump.actions
        ),
        exclusion_mask=jnp.zeros((len(env_state), len(action_dump)), dtype=jnp.bool_),
        timebatching=timebatching,
    )


def vector_step(env: CombinatorEnv, action: jnp.ndarray) -> CombinatorEnv:
    """Adds another action to the environment state

    Args:
        env (CombinatorEnv): The current environment state
        action (jnp.ndarray): An integer in the range of 0, len(action_dump) that selects an action
            to merge.

    Returns:
        CombinatorEnv: An updated environment
    """
    action_from_dump = env.action_dump.actions[action]

    return replace(
        env,
        accumulated_action=merge_actions(env.accumulated_action, action_from_dump),
        exclusion_mask=env.exclusion_mask | env.action_dump.exclusion_mask[action],
    )


def compute_obs(env: CombinatorEnv) -> PFCResults:
    """Computes the observation from the current environment state

    The PFC Results will have dimension n_envs * timebatching

    Args:
        env (CombinatorEnv): The current environment state

    Returns:
        jnp.ndarray: The observation
    """
    env_expanded = timebatch_env(
        env.env_state, env.timebatching, end_of_chronic_behaviour="clip"
    )
    return compute_obs_vanilla(env_expanded)


def get_reward(env: CombinatorEnv, results: PFCResults) -> jnp.ndarray:
    """Computes the current reward for each enviromnent

    This sums over timebatches, hence the resulting shape is (n_envs,)

    Args:
        env (CombinatorEnv): The current environment state
        results (PFCResults): The loadflow results as obtained by compute_obs

    Returns:
        jnp.ndarray: A reward for each environment, shape (n_envs,)
    """
    rewards = get_reward_vanilla(env.env_state.grid, results)
    return aggregate_timebatched(rewards, env.timebatching, jnp.sum)


def random_action(key: jax.random.PRNGKey, env: CombinatorEnv) -> jnp.ndarray:
    """Samples a random action for the environment, respecting the action mask

    Args:
        key (jax.random.PRNGKey): The random key to use for sampling
        env (CombinatorEnv): The combinatior environment to sample for

    Returns:
        jnp.ndarray: [int] (n_envs,) An integer action for each environment
    """

    def sample_single_action(key, mask) -> jnp.ndarray:
        return jax.random.choice(
            key,
            jnp.arange(len(env.action_dump)),
            shape=(),
            replace=False,
            p=1 - mask,
        )

    key = jax.random.split(key, len(env))
    return jax.vmap(sample_single_action)(key, env.exclusion_mask)


def is_action_excluded(env: CombinatorEnv, action: jnp.ndarray) -> jnp.ndarray:
    """Checks if the chose action is excluded in the mask

    Args:
        env (CombinatorEnv): The environment
        action (jnp.ndarray): An action vector of shape (n_envs,), dtype [int] holding the actions to check

    Returns:
        jnp.ndarray: [bool] (n_envs,) True if the action is excluded
    """
    chex.assert_shape(action, (len(env),))
    chex.assert_type(action, int)
    return jnp.take_along_axis(
        env.exclusion_mask, jnp.expand_dims(action, axis=-1), axis=1
    )
