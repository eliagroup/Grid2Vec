from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import pandapower as pp

from grid2vec.batched_pfc import batched_pfc
from grid2vec.nminus1_definition import (
    FailureType,
    NMinus1Definition,
    full_nminus1_definition,
)
from grid2vec.result_spec import (
    ResultSpec,
    describe_nminus1,
    describe_results,
    find_spec,
)
from grid2vec.util import freeze_array

PFCResults = Dict[str, np.ndarray]


def batched_n_minus_1(
    net: pp.pandapowerNet,
    profile: Dict[Tuple[str, str], np.ndarray],
    res_spec: Optional[ResultSpec] = None,
    nminus1_definition: Optional[NMinus1Definition] = None,
    dc: bool = False,
) -> PFCResults:
    """Performs a batched n-1 power flow calculation.

    Args:
        net (pp.pandapowerNet): The base pandapower network
        load_p (np.ndarray): Shape (n_envs, n_load) - The active power consumption of each load
        load_q (np.ndarray): Shape (n_envs, n_load) - The reactive power consumption of each load
        prod_p (np.ndarray): Shape (n_envs, n_gen) - The active power production of each generator
        prod_v (np.ndarray): Shape (n_envs, n_gen) - The voltage setpoint of each generator
        profile (Dict[Tuple[str, str], np.ndarray]): - The actions for each dim, can be one of
            ("load", "p_mw"), ("load", "q_mvar"), ("gen", "p_mw"), ("gen", "vm_pu"),
            ("line", "in_service"), ("switch", "closed"), ("trafo", "tap_pos"),
            ("trafo3w", "tap_pos"), ("line", "from_bus"), ("line", "to_bus"), ("trafo", "bus_hv"),
            ("trafo", "bus_lv"), ("trafo3w", "bus_hv"), ("trafo3w", "bus_mv"), ("trafo3w", "bus_lv"),
            ("gen", "bus"), ("load", "bus"), ("sgen", "bus")
        res_spec (Optional[ResultSpec], optional): The result specification for deciding
            which observations to return in the results dictionary. Defaults to all specs from
            describe_results().
        nminus1_definition (Optional[NMinus1Definition], optional): The definition of the n-1
            failures to be simulated. Defaults to None, which means that all failures are simulated.
        dc (bool, optional): Whether to use the DC power flow solver. Defaults to False.

    Returns:
        A dict of powerflow results, holding the results as described by res_spec
    """
    n_envs = next(iter(profile.values())).shape[0]

    if nminus1_definition is None:
        nminus1_definition = full_nminus1_definition(net)

    if res_spec is None:
        res_spec = describe_results(
            n_envs=n_envs,
            n_line=len(net.line),
            n_trafo=len(net.trafo),
            n_trafo3w=len(net.trafo3w),
            n_gen=len(net.gen),
            n_load=len(net.load),
        )
        res_spec += describe_nminus1(
            n_envs=n_envs,
            n_failures=len(nminus1_definition),
            n_line=len(net.line),
            n_trafo=len(net.trafo),
            n_trafo3w=len(net.trafo3w),
        )

    # Do a steady-state power flow first for obtaining base observations
    base_res = batched_pfc(
        net=net,
        profile=profile,
        res_spec=res_spec,
        dc=dc,
    )

    n_failures = len(nminus1_definition)
    n_line = len(net.line)
    n_trafo = len(net.trafo)
    n_trafo3w = len(net.trafo3w)

    line_loading_per_failure = np.zeros((n_envs, n_failures, n_line))
    line_loading_grid2op_per_failure = np.zeros((n_envs, n_failures, n_line))
    trafo_loading_per_failure = np.zeros((n_envs, n_failures, n_trafo))
    trafo_loading_grid2op_per_failure = np.zeros((n_envs, n_failures, n_trafo))
    trafo3w_loading_per_failure = np.zeros((n_envs, n_failures, n_trafo3w))
    converged = np.ones((n_envs, n_failures), dtype=np.bool_)

    for env in range(n_envs):
        net_copy = deepcopy(net)

        # Apply actions
        for (table, key), setting in profile.items():
            net_copy[table][key] = setting[env]

        for index, (fail_type, fail_id) in enumerate(
            nminus1_definition.iter_failures()
        ):
            # Set element out of service
            was_in_service = True
            if fail_type == FailureType.LINE:
                was_in_service = net_copy.line.loc[
                    net.line.index[fail_id], "in_service"
                ]
                net_copy.line.loc[net.line.index[fail_id], "in_service"] = False
            elif fail_type == FailureType.TRAFO:
                was_in_service = net_copy.trafo.loc[
                    net.trafo.index[fail_id], "in_service"
                ]
                net_copy.trafo.loc[net.trafo.index[fail_id], "in_service"] = False
            elif fail_type == FailureType.TRAFO3W:
                was_in_service = net_copy.trafo3w.loc[
                    net.trafo3w.index[fail_id], "in_service"
                ]
                net_copy.trafo3w.loc[net.trafo3w.index[fail_id], "in_service"] = False
            elif fail_type == FailureType.BUS:
                was_in_service = net_copy.bus.loc[net.bus.index[fail_id], "in_service"]
                net_copy.bus.loc[net.bus.index[fail_id], "in_service"] = False
            else:
                raise ValueError(f"Unknown failure type {fail_type}")

            # Run power flow
            try:
                if dc:
                    pp.rundcpp(net_copy, numba=True)
                else:
                    pp.runpp(net_copy, init="results", numba=True)

                # Extract results
                line_loading_per_failure[
                    env, index, :
                ] = net_copy.res_line.loading_percent
                line_loading_grid2op_per_failure[env, index, :] = (
                    net_copy.res_line.i_from_ka / net_copy.line.max_i_ka
                )
                trafo_loading_per_failure[
                    env, index, :
                ] = net_copy.res_trafo.loading_percent
                if "grid2op_load_limit" in net_copy.trafo:
                    trafo_loading_grid2op_per_failure[env, index, :] = (
                        net_copy.res_trafo.i_hv_ka / net_copy.trafo.grid2op_load_limit
                    )
                trafo3w_loading_per_failure[
                    env, index, :
                ] = net_copy.res_trafo3w.loading_percent

            except Exception:
                converged[env, index] = False
            finally:
                # Set bus back to original state
                if fail_type == FailureType.LINE:
                    net_copy.line.loc[
                        net.line.index[fail_id], "in_service"
                    ] = was_in_service
                elif fail_type == FailureType.TRAFO:
                    net_copy.trafo.loc[
                        net.trafo.index[fail_id], "in_service"
                    ] = was_in_service
                elif fail_type == FailureType.TRAFO3W:
                    net_copy.trafo3w.loc[
                        net.trafo3w.index[fail_id], "in_service"
                    ] = was_in_service
                elif fail_type == FailureType.BUS:
                    net_copy.bus.loc[
                        net.bus.index[fail_id], "in_service"
                    ] = was_in_service

    # Add results to base results
    spec = find_spec(res_spec, "line_loading_per_failure")
    if spec is not None:
        base_res["line_loading_per_failure"] = spec.transformer(
            line_loading_per_failure.astype(spec.dtype)
        )
    spec = find_spec(res_spec, "line_loading_grid2op_per_failure")
    if spec is not None:
        base_res["line_loading_grid2op_per_failure"] = spec.transformer(
            line_loading_grid2op_per_failure.astype(spec.dtype)
        )
    spec = find_spec(res_spec, "trafo_loading_per_failure")
    if spec is not None:
        base_res["trafo_loading_per_failure"] = spec.transformer(
            trafo_loading_per_failure.astype(spec.dtype)
        )
    spec = find_spec(res_spec, "trafo_loading_grid2op_per_failure")
    if spec is not None:
        base_res["trafo_loading_grid2op_per_failure"] = spec.transformer(
            trafo_loading_grid2op_per_failure.astype(spec.dtype)
        )
    spec = find_spec(res_spec, "trafo3w_loading_per_failure")
    if spec is not None:
        base_res["trafo3w_loading_per_failure"] = spec.transformer(
            trafo3w_loading_per_failure.astype(spec.dtype)
        )
    spec = find_spec(res_spec, "nminus1_converged")
    if spec is not None:
        base_res["nminus1_converged"] = spec.transformer(converged)

    base_res = {k: freeze_array(v) for k, v in base_res.items()}

    return base_res
