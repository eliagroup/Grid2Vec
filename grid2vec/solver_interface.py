# The task of the solver interface is to expand the environment to the solver's dimension and
# to find batchings of the environment that can be solved in parallel.
from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandapower as pp

from grid2vec.batched_nminus1 import batched_n_minus_1
from grid2vec.batched_pfc import PFCResults, batched_pfc
from grid2vec.env import VecEnvState
from grid2vec.grid import Grid, chronics_current_timestep, split_substation_affinity
from grid2vec.nminus1_definition import NMinus1Definition
from grid2vec.result_spec import ResultSpec, set_env_dim, spec_to_jax


def invoke_solver(
    profile: dict,
    net: pp.pandapowerNet,
    nminus1_definition: Optional[NMinus1Definition],
    res_spec: ResultSpec,
    dc: bool,
    which: Optional[jnp.ndarray] = None,
) -> PFCResults:
    """Invokes the solver on the host

    This part is not to be jitted and should be replaced with a native jax version as soon as
    we have ported the solver to jax.

    Args:
        profile (dict): A profile as returned by make_profile(env), a dict containing tuples of
            (table, key) as keys and arrays of shape (n_envs, n_elements) as values. The n_env
            dimension is the batch dimension of the solver
        net (pp.pandapowerNet): The pandapower net underlying the computations
        nminus1_definition (Optional[NMinus1Definition]): Either None (meaning simple N solving) or
            a NMinus1Definition describing the N-1 cases to be solved
        res_spec (ResultSpec): Which results are required to be extracted from the computation
        dc (bool): Whether to use DC approximation
        which (Optional[jnp.ndarray], optional: If given, should have shape (n_envs) and dtype bool
            Describes which environments to compute the observation for. This can be used to skip
            computations for individual parts of the batch. Defaults to None, which means all.
            If some environment batches are skipped, undefined contents will be in the result values
            for those environment dimensions.

    Returns:
        PFCResults: A dict with powerflow results, holding the results as described by res_spec
    """

    if which is not None:
        profile = {k: v[which] for (k, v) in profile.items()}
        res_spec = set_env_dim(res_spec, jnp.sum(which))

    if nminus1_definition is None:
        pfc_results = batched_pfc(
            net=net,
            profile=profile,
            res_spec=res_spec,
            dc=dc,
        )
    else:
        pfc_results = batched_n_minus_1(
            net=net,
            profile=profile,
            nminus1_definition=nminus1_definition,
            res_spec=res_spec,
            dc=dc,
        )

    if which is not None:
        for key, value in pfc_results.items():
            tmp_value = jnp.zeros(
                (which.shape[0],) + value.shape[1:], dtype=value.dtype
            )
            tmp_value = tmp_value.at[which].set(value)
            pfc_results[key] = tmp_value

    return pfc_results


def compute_obs(env: VecEnvState, which: Optional[jnp.ndarray] = None) -> PFCResults:
    """Computes the observation for the environment by performing a loadflow (and n-1 if necessary)

    Infers from the grid.nminus1_definition whether to use the n-1 or n-0 loadflow

    Args:
        env (VecEnvState): The vectorized environment state
        which (Optional[np.ndarray[bool]], optional): Shape (n_env) Which environments to compute
            the observation for. Defaults to all. If passed, the outputs of the non-computed
            elements will hold arbitrary values, however the shape of the output remains the same.

    Returns:
        PFCResults: The raw PFC result, use obs_postprocess to make a valid observation
    """
    profile = make_profile(env)

    res_spec = set_env_dim(env.grid.res_spec, env.n_envs)
    jax_spec = spec_to_jax(res_spec)

    pfc_results = jax.pure_callback(
        partial(
            invoke_solver,
            net=env.grid.net,
            res_spec=res_spec,
            dc=env.grid.dc,
            nminus1_definition=env.grid.nminus1_definition,
        ),
        result_shape_dtypes=jax_spec,
        profile=profile,
        which=which,
    )

    # Update the results with relevant inputs to the solver
    pfc_results.update(collect_pfc_inputs(env, profile))

    return pfc_results


def collect_pfc_inputs(env: VecEnvState, profile: dict) -> Dict[str, jnp.ndarray]:
    """Returns the inputs to the PFC solver that are interesting as observations

    This matches the specs from result_spec.describe_chronics()

    Args:
        env (VecEnvState): The vectorized environment state
        profile (dict): The profile passed to the solver

    Returns:
        Dict[str, jnp.ndarray]: A dict that can be concatenated to the pfc_results
    """
    # To extract the topo vect from the profile, we need go through the lookup and check which values
    # have been assigned to each element
    topo_vect_assignment = jnp.zeros(
        (env.n_envs, env.grid.len_topo_vect), dtype=jnp.int32
    )
    for section_length, last_idx, table, key in env.grid.topo_vect_lookup:
        element_bus_assignment = get_from_profile_or_net(
            table, key, profile, env.grid, env.n_envs
        )
        # Now the topo vect holds the bus assignment in pandapower bus idx
        topo_vect_assignment = topo_vect_assignment.at[
            :, last_idx - section_length : last_idx
        ].set(element_bus_assignment)
    # Map this to indices in the substation affinity
    topo_vect = jnp.zeros((env.n_envs, env.grid.len_topo_vect), dtype=jnp.int32)
    for bus_idx in range(env.grid.substation_affinity.shape[1]):
        topo_vect = jnp.where(
            topo_vect_assignment == env.grid.substation_affinity[:, bus_idx],
            bus_idx,
            topo_vect,
        )

    vn_kv = jnp.array(env.grid.net.bus.loc[env.grid.net.gen.bus]["vn_kv"].values)

    return {
        "load_p_input": get_from_profile_or_net(
            "load", "p_mw", profile, env.grid, env.n_envs
        ),
        "load_q_input": get_from_profile_or_net(
            "load", "q_mvar", profile, env.grid, env.n_envs
        ),
        "prod_p_input": get_from_profile_or_net(
            "gen", "p_mw", profile, env.grid, env.n_envs
        ),
        "prod_v_input": get_from_profile_or_net(
            "gen", "vm_pu", profile, env.grid, env.n_envs
        )
        * vn_kv,
        "switch_state": get_from_profile_or_net(
            "switch", "closed", profile, env.grid, env.n_envs
        ),
        "line_state": get_from_profile_or_net(
            "line", "in_service", profile, env.grid, env.n_envs
        ),
        "trafo_state": get_from_profile_or_net(
            "trafo", "in_service", profile, env.grid, env.n_envs
        ),
        "trafo_tap_pos": jnp.nan_to_num(
            get_from_profile_or_net("trafo", "tap_pos", profile, env.grid, env.n_envs),
            nan=-1,
        ).astype(jnp.int32),
        "trafo3w_tap_pos": jnp.nan_to_num(
            get_from_profile_or_net(
                "trafo3w", "tap_pos", profile, env.grid, env.n_envs
            ),
            nan=-1,
        ).astype(jnp.int32),
        "topo_vect": topo_vect,
    }


def make_profile(env: VecEnvState) -> Dict[Tuple[str, str], jnp.ndarray]:
    """Makes a profile for the batched solver

    The profile contains tuples which reference the table and key in the pandapower net
    and arrays, each of dimension (n_envs, n_elements) where n_elements is the number of
    elements in the table.

    Args:
        env (VecEnvState): The VecEnvState to convert

    Returns:
        Dict[Tuple[str, str], jnp.ndarray]: The profile
    """
    profile = action_to_pandapower(env)
    load_p, load_q, prod_p, prod_v = chronics_current_timestep(
        env.timestep, env.chronic, env.grid.chronics
    )
    profile[("load", "p_mw")] = load_p
    profile[("load", "q_mvar")] = load_q
    profile[("gen", "p_mw")] = prod_p
    vn_kv = jnp.array(env.grid.net.bus.loc[env.grid.net.gen.bus]["vn_kv"].values)
    profile[("gen", "vm_pu")] = prod_v / vn_kv

    return profile


# @partial(jax.jit, static_argnames=("table", "key", "grid", "n_envs"))
def get_from_profile_or_net(
    table: str,
    key: str,
    profile: Dict[Tuple[str, str], jnp.ndarray],
    grid: Grid,
    n_envs: int,
) -> jnp.ndarray:
    """Returns a value from profile if it's in profile, else from the pandapower net

    Args:
        table (str): The pandapower table to access
        key (str): The column in that table to access
        profile (Dict[Tuple[str, str], jnp.ndarray]): The profile as from make_profile(env)
        grid (Grid): The grid holding the pandapower net. Note that due to pp.pandapowerNet being
            unhashable, we have to pass in the entire grid
        n_envs (int): The environment dimension

    Returns:
        jnp.ndarray: The value
    """
    if (table, key) in profile:
        return profile[(table, key)]
    else:
        return jnp.expand_dims(grid.net[table][key].values, axis=0).repeat(
            n_envs, axis=0
        )


def action_to_pandapower(env: VecEnvState) -> Dict[Tuple[str, str], jnp.ndarray]:
    """Takes the actions from the VecEnv, expands them and returns them as a dict for
    pandapower assigment

    It expands controllable elements to all elements and splits the topo vect

    The dict can the be used like this:
    action_dict = action_to_pandapower(env)
    for (table, key) in action_dict.keys():
        pp.net[table][key] = action_dict[(table, key)]

    Args:
        env (VecEnvState): The vectorized environment state

    Returns:
        Dict[Tuple[str, str], np.ndarray]: The actions as a dict for pandapower assignment
    """
    retval: Dict[Tuple[str, str], jnp.ndarray] = {}

    if env.grid.has_switch_actions:
        # expand to (n_env, n_switch)
        # then overwrite the controllable switches with the env switch state
        all_switch_state = (
            jnp.expand_dims(env.grid.net.switch.closed.values, axis=0)
            .repeat(env.n_envs, axis=0)
            .at[:, env.grid.switch_controllable]
            .set(env.switch_state)
        )

        retval[("switch", "closed")] = all_switch_state

    if env.grid.has_line_actions:
        # Do the same for the line status
        all_line_state = (
            jnp.expand_dims(env.grid.net.line.in_service.values, axis=0)
            .repeat(env.n_envs, axis=0)
            .at[:, env.grid.line_controllable]
            .set(env.line_state)
        )

        retval[("line", "in_service")] = all_line_state

    if env.grid.has_trafo_actions:
        # Do the same for the trafo status
        all_trafo_state = (
            jnp.expand_dims(env.grid.net.trafo.in_service.values, axis=0)
            .repeat(env.n_envs, axis=0)
            .at[:, env.grid.trafo_controllable]
            .set(env.trafo_state)
        )

        retval[("trafo", "in_service")] = all_trafo_state

    if env.grid.has_trafo_tap_actions:
        # Do the same for the trafo tap position
        all_trafo_tap_pos = (
            jnp.expand_dims(env.grid.net.trafo.tap_pos.values, axis=0)
            .repeat(env.n_envs, axis=0)
            .at[:, env.grid.trafo_tap_controllable]
            .set(env.trafo_tap_pos)
        )

        retval[("trafo", "tap_pos")] = all_trafo_tap_pos

    if env.grid.has_trafo3w_tap_actions:
        # Do the same for the trafo3w tap position
        all_trafo3w_tap_pos = (
            jnp.expand_dims(env.grid.net.trafo3w.tap_pos.values, axis=0)
            .repeat(env.n_envs, axis=0)
            .at[:, env.grid.trafo3w_tap_controllable]
            .set(env.trafo3w_tap_pos)
        )

        retval[("trafo3w", "tap_pos")] = all_trafo3w_tap_pos

    if env.grid.has_topo_vect_actions:
        # Get the expanded topo vect too
        all_topo_vect = (
            jnp.zeros((env.n_envs, env.grid.len_topo_vect), dtype=np.int32)
            .at[:, env.grid.topo_vect_controllable]
            .set(env.topo_vect)
        )

        bus_assignment = jnp.take_along_axis(
            jnp.transpose(env.grid.substation_affinity), all_topo_vect, axis=0
        )
        # It would be nicer to throw an exception here, however we can not add a @chex.chexify
        # as this is not a toplevel function
        # chex.assert_trees_all_equal(jnp.any(bus_assignment == -1), jnp.array(False))
        bus_assignment = jnp.where(
            bus_assignment == -1, env.grid.substation_affinity[:, 0], bus_assignment
        )
        retval.update(
            split_substation_affinity(bus_assignment, env.grid.topo_vect_lookup)
        )

    return retval
