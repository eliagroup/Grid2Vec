from copy import deepcopy
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandapower as pp

from grid2vec.result_spec import ResultSpec, describe_results, find_spec
from grid2vec.util import freeze_array

PFCResults = Dict[str, jnp.ndarray]


def prepare_results(res_spec: ResultSpec) -> dict:
    """Prepares the result dict for the powerflow calculation.

    Args:
        res_spec (ResultSpec): The result specification

    Returns:
        dict: A dict with zero-initialized arrays for the results
    """
    return {spec.key: np.zeros(spec.shape, dtype=spec.dtype) for spec in res_spec}


def extract_env_results(
    net: pp.pandapowerNet, env: int, results: dict, res_spec: ResultSpec
) -> dict:
    """Extracts the results for a single powerflow computation for one environment

    Args:
        net (pp.pandapowerNet): The pandapower network with filled res_ tables
        env (int): The environment index
        results (dict): The results dict
        res_spec (ResultSpec): The result specification

    Returns:
        dict: The results dict, with the results for the environment added
    """
    for spec in res_spec:
        if spec.pp_res_table is None or spec.pp_res_key is None:
            continue
        table = getattr(net, spec.pp_res_table)
        res = table[spec.pp_res_key].values
        results[spec.key][env] = spec.transformer(res)
    return results


def batched_pfc(
    net: pp.pandapowerNet,
    profile: Dict[Tuple[str, str], np.ndarray],
    res_spec: Optional[ResultSpec] = None,
    dc: bool = False,
) -> PFCResults:
    """Performs a batched power flow calculation.

    This performs a single steady-state AC powerflow computation for each environment in the batch
    and returns the most important results.
    TODO optimize with results from HPC track

    Args:
        net (pp.pandapowerNet): The base pandapower network
        profile (Dict[Tuple[str, str], np.ndarray]): - The actions for each dim, can be one of
            ("load", "p_mw"), ("load", "q_mvar"), ("gen", "p_mw"), ("gen", "vm_pu"),
            ("line", "in_service"), ("switch", "closed"), ("trafo", "tap_pos"),
            ("trafo3w", "tap_pos"), ("line", "from_bus"), ("line", "to_bus"), ("trafo", "bus_hv"),
            ("trafo", "bus_lv"), ("trafo3w", "bus_hv"), ("trafo3w", "bus_mv"), ("trafo3w", "bus_lv"),
            ("gen", "bus"), ("load", "bus"), ("sgen", "bus")
        res_spec (Optional[ResultSpec], optional): The result specification for deciding
            which observations to return in the results dictionary. Defaults to all specs from
            describe_results().
        dc (bool, optional): Whether to use the DC power flow. Defaults to False.

    Returns:
        A dict of powerflow results, holding the results as described by res_spec
    """
    n_envs = next(iter(profile.values())).shape[0]

    for val in profile.values():
        assert val.shape[0] == n_envs

    if res_spec is None:
        res_spec = describe_results(
            n_envs=n_envs,
            n_line=len(net.line),
            n_trafo=len(net.trafo),
            n_trafo3w=len(net.trafo3w),
            n_gen=len(net.gen),
            n_load=len(net.load),
        )

    res_dict = prepare_results(res_spec)
    converged = np.ones((n_envs,), dtype=bool)

    for i in range(n_envs):
        net_copy = deepcopy(net)

        # Apply actions
        for (table, key), setting in profile.items():
            net_copy[table][key] = setting[i]

        # Run power flow
        try:
            if dc:
                pp.rundcpp(net_copy, numba=True)
            else:
                pp.runpp(net_copy, init="dc", numba=True)

            # Extract results
            res_dict = extract_env_results(
                net=net_copy, env=i, results=res_dict, res_spec=res_spec
            )
            # Extract grid2op compatible loadings
            spec = find_spec(res_spec, "loading_line_grid2op")
            if spec is not None:
                res_dict[spec.key][i] = spec.transformer(
                    net_copy.res_line.i_from_ka / net_copy.line.max_i_ka
                )
            spec = find_spec(res_spec, "loading_trafo_grid2op")
            if spec is not None and "grid2op_load_limit" in net_copy.trafo:
                res_dict[spec.key][i] = spec.transformer(
                    net_copy.res_trafo.i_hv_ka / net_copy.trafo.grid2op_load_limit
                )

        except Exception:
            converged[i] = False

    # Add converged flag
    spec = find_spec(res_spec, "converged")
    if spec is not None:
        res_dict[spec.key] = spec.transformer(converged)

    # Freeze res dict
    res_dict = {k: freeze_array(v) for k, v in res_dict.items()}
    return res_dict
