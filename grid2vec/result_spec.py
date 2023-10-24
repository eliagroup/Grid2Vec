from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ResultsValue:
    key: str
    shape: tuple
    unit: str
    dtype: np.dtype
    description: str
    pp_res_table: Optional[str] = None
    pp_res_key: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    mask: Optional[np.ndarray] = None


ResultSpec = Tuple[ResultsValue, ...]


def apply_mask(data: jnp.ndarray, mask: Optional[np.ndarray]) -> jnp.ndarray:
    """Applies a mask to data

    Args:
        data (jnp.ndarray): The data to mask
        mask (Optional[np.ndarray]): The mask. If None, no masking is applied

    Returns:
        jnp.ndarray: The masked data
    """
    if mask is None:
        return data
    return data[mask]


def find_spec(specs: ResultSpec, key: str) -> Optional[ResultsValue]:
    """Finds a spec by key in a list of specs

    Args:
        specs (ResultSpec): The list of specs to search in
        key (str): The key to search for

    Returns:
        Optional[ResultsValue]: The spec if found, None otherwise
    """
    for spec in specs:
        if spec.key == key:
            return spec
    return None


def set_env_dim(specs: ResultSpec, n_envs: int) -> ResultSpec:
    """Sets the first dimension of all specs to n_envs

    Args:
        specs (ResultSpec): The specs to modify
        n_envs (int): The number of environments

    Returns:
        ResultSpec: The modified specs
    """
    return tuple(
        [
            ResultsValue(
                **{k: v for k, v in asdict(spec).items() if k != "shape"},
                shape=(n_envs,) + spec.shape[1:],
            )
            for spec in specs
        ]
    )


def spec_to_jax(specs: ResultSpec) -> Dict[str, jax.ShapeDtypeStruct]:
    """Converts a result spec to a jax shape-dtype struct

    Args:
        specs (ResultSpec): The result spec

    Returns:
        Dict[jax.ShapeDtypeStruct]: The jax shape-dtype struct
    """
    return {spec.key: jax.ShapeDtypeStruct(spec.shape, spec.dtype) for spec in specs}


def describe_pfc_results(
    *,
    n_line: int,
    n_trafo: int,
    n_trafo3w: int,
    n_gen: int,
    n_load: int,
    n_switch: int,
    n_sgen: int,
    n_nminus1_cases: Optional[int] = None,
    n_envs: int = 1,
) -> ResultSpec:
    """Describes the expected results for a net

    Args:
        n_line (int): Number of lines in the net
        n_trafo (int): Number of transformers in the net
        n_trafo3w (int): Number of 3W transformers in the net
        n_gen (int): Number of generators in the net
        n_sgen (int): Number of static generators in the net
        n_load (int): Number of loads in the net
        n_switch (int): Number of switches in the net
        n_nminus1_cases (Optional[int], optional): Number of N-1 cases. If None, no n-1 results will
            be assumed
        n_envs (int, optional): The environment dimension. You can change this later with
            set_env_dim. Defaults to 1.

    Returns:
        ResultSpec: The resulting result spec
    """
    res_spec = list(
        describe_results(
            n_envs=n_envs,
            n_line=n_line,
            n_trafo=n_trafo,
            n_trafo3w=n_trafo3w,
            n_gen=n_gen,
            n_load=n_load,
        )
    ) + list(
        describe_chronics(
            n_envs=n_envs,
            n_load=n_load,
            n_gen=n_gen,
            n_switch=n_switch,
            n_line=n_line,
            n_trafo=n_trafo,
            n_trafo3w=n_trafo3w,
            n_sgen=n_sgen,
        )
    )
    if n_nminus1_cases is not None:
        res_spec += list(
            describe_nminus1(
                n_envs=n_envs,
                n_failures=n_nminus1_cases,
                n_line=n_line,
                n_trafo=n_trafo,
                n_trafo3w=n_trafo3w,
            )
        )
    return tuple(res_spec)


def describe_line_results(n_envs: int, n_line: int) -> ResultSpec:
    return (
        ResultsValue(
            key="p_or_line",
            shape=(n_envs, n_line),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each origin side of the line",
            pp_res_table="res_line",
            pp_res_key="p_from_mw",
        ),
        ResultsValue(
            key="p_ex_line",
            shape=(n_envs, n_line),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each exit side of the line",
            pp_res_table="res_line",
            pp_res_key="p_to_mw",
        ),
        ResultsValue(
            key="q_or_line",
            shape=(n_envs, n_line),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each origin side of the line",
            pp_res_table="res_line",
            pp_res_key="q_from_mvar",
        ),
        ResultsValue(
            key="q_ex_line",
            shape=(n_envs, n_line),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each exit side of the line",
            pp_res_table="res_line",
            pp_res_key="q_to_mvar",
        ),
    )


def describe_trafo_results(n_envs: int, n_trafo: int) -> ResultSpec:
    return (
        ResultsValue(
            key="p_hv_trafo",
            shape=(n_envs, n_trafo),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each high-voltage side of the transformer",
            pp_res_table="res_trafo",
            pp_res_key="p_hv_mw",
        ),
        ResultsValue(
            key="p_lv_trafo",
            shape=(n_envs, n_trafo),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each low-voltage side of the transformer",
            pp_res_table="res_trafo",
            pp_res_key="p_lv_mw",
        ),
        ResultsValue(
            key="q_hv_trafo",
            shape=(n_envs, n_trafo),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each high-voltage side of the transformer",
            pp_res_table="res_trafo",
            pp_res_key="q_hv_mvar",
        ),
        ResultsValue(
            key="q_lv_trafo",
            shape=(n_envs, n_trafo),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each low-voltage side of the transformer",
            pp_res_table="res_trafo",
            pp_res_key="q_lv_mvar",
        ),
    )


def describe_trafo3w_results(n_envs: int, n_trafo3w: int) -> ResultSpec:
    return (
        ResultsValue(
            key="p_hv_trafo3w",
            shape=(n_envs, n_trafo3w),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each high-voltage side of the 3W transformer",
            pp_res_table="res_trafo3w",
            pp_res_key="p_hv_mw",
        ),
        ResultsValue(
            key="p_mv_trafo3w",
            shape=(n_envs, n_trafo3w),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each mid-voltage side of the 3W transformer",
            pp_res_table="res_trafo3w",
            pp_res_key="p_mv_mw",
        ),
        ResultsValue(
            key="p_lv_trafo3w",
            shape=(n_envs, n_trafo3w),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each low-voltage side of the 3W transformer",
            pp_res_table="res_trafo3w",
            pp_res_key="p_lv_mw",
        ),
        ResultsValue(
            key="q_hv_trafo3w",
            shape=(n_envs, n_trafo3w),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each high-voltage side of the 3W transformer",
            pp_res_table="res_trafo3w",
            pp_res_key="q_hv_mvar",
        ),
        ResultsValue(
            key="q_mv_trafo3w",
            shape=(n_envs, n_trafo3w),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each mid-voltage side of the 3W transformer",
            pp_res_table="res_trafo3w",
            pp_res_key="q_mv_mvar",
        ),
        ResultsValue(
            key="q_lv_trafo3w",
            shape=(n_envs, n_trafo3w),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each low-voltage side of the 3W transformer",
            pp_res_table="res_trafo3w",
            pp_res_key="q_lv_mvar",
        ),
    )


def describe_gen_results(n_envs: int, n_gen: int) -> ResultSpec:
    return (
        ResultsValue(
            key="gen_p",
            shape=(n_envs, n_gen),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each generator, after pfc",
            pp_res_table="res_gen",
            pp_res_key="p_mw",
        ),
        ResultsValue(
            key="gen_q",
            shape=(n_envs, n_gen),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each generator",
            pp_res_table="res_gen",
            pp_res_key="q_mvar",
        ),
    )


def describe_load_results(n_envs: int, n_load: int) -> ResultSpec:
    return (
        ResultsValue(
            key="load_p",
            shape=(n_envs, n_load),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each load",
            pp_res_table="res_load",
            pp_res_key="p_mw",
        ),
        ResultsValue(
            key="load_q",
            shape=(n_envs, n_load),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each load",
            pp_res_table="res_load",
            pp_res_key="q_mvar",
        ),
    )


def describe_loading_results(
    n_envs: int, n_line: int, n_trafo: int, n_trafo3w: int
) -> ResultSpec:
    return (
        ResultsValue(
            key="loading_line",
            shape=(n_envs, n_line),
            unit="%",
            dtype=np.float32,
            description="The loading of each line, computed by pandapower",
            low=0.0,
            pp_res_table="res_line",
            pp_res_key="loading_percent",
        ),
        ResultsValue(
            key="loading_trafo",
            shape=(n_envs, n_trafo),
            unit="%",
            dtype=np.float32,
            description="The loading of each transformer, computed by pandapower",
            low=0.0,
            pp_res_table="res_trafo",
            pp_res_key="loading_percent",
        ),
        ResultsValue(
            key="loading_trafo3w",
            shape=(n_envs, n_trafo3w),
            unit="%",
            dtype=np.float32,
            description="The loading of each 3W transformer, computed by pandapower",
            low=0.0,
            pp_res_table="res_trafo3w",
            pp_res_key="loading_percent",
        ),
        ResultsValue(
            key="loading_line_grid2op",
            shape=(n_envs, n_line),
            unit="%",
            dtype=np.float32,
            description="The loading of each line, computed according to grid2op",
            low=0.0,
        ),
        ResultsValue(
            key="loading_trafo_grid2op",
            shape=(n_envs, n_trafo),
            unit="%",
            dtype=np.float32,
            description="The loading of each transformer, computed according to grid2op",
            low=0.0,
        ),
    )


def describe_converged_results(n_envs: int) -> ResultSpec:
    return (
        ResultsValue(
            key="converged",
            shape=(n_envs,),
            unit="",
            dtype=np.bool_,
            description="Whether the power flow converged",
        ),
    )


def describe_results(
    n_envs: int, n_line: int, n_trafo: int, n_trafo3w: int, n_gen: int, n_load: int
) -> ResultSpec:
    return tuple(
        list(describe_line_results(n_envs=n_envs, n_line=n_line))
        + list(describe_trafo_results(n_envs=n_envs, n_trafo=n_trafo))
        + list(describe_trafo3w_results(n_envs=n_envs, n_trafo3w=n_trafo3w))
        + list(describe_gen_results(n_envs=n_envs, n_gen=n_gen))
        + list(describe_load_results(n_envs=n_envs, n_load=n_load))
        + list(
            describe_loading_results(
                n_envs=n_envs, n_line=n_line, n_trafo=n_trafo, n_trafo3w=n_trafo3w
            )
        )
        + list(describe_converged_results(n_envs=n_envs))
    )


def describe_nminus1(
    *, n_envs: int, n_failures: int, n_line: int, n_trafo: int, n_trafo3w: int
) -> ResultSpec:
    return (
        ResultsValue(
            key="line_loading_per_failure",
            shape=(n_envs, n_failures, n_line),
            unit="%",
            dtype=np.float32,
            description="The line loading per failure",
            low=0.0,
        ),
        ResultsValue(
            key="line_loading_grid2op_per_failure",
            shape=(n_envs, n_failures, n_line),
            unit="%",
            dtype=np.float32,
            description="The line loading per failure, computed according to grid2op",
            low=0.0,
        ),
        ResultsValue(
            key="trafo_loading_per_failure",
            shape=(n_envs, n_failures, n_trafo),
            unit="%",
            dtype=np.float32,
            description="The transformer loading per failure",
            low=0.0,
        ),
        ResultsValue(
            key="trafo_loading_grid2op_per_failure",
            shape=(n_envs, n_failures, n_trafo),
            unit="%",
            dtype=np.float32,
            description="The transformer loading per failure, computed according to grid2op",
            low=0.0,
        ),
        ResultsValue(
            key="trafo3w_loading_per_failure",
            shape=(n_envs, n_failures, n_trafo3w),
            unit="%",
            dtype=np.float32,
            description="The 3W transformer loading per failure",
            low=0.0,
        ),
        ResultsValue(
            key="nminus1_converged",
            shape=(n_envs, n_failures),
            unit="",
            dtype=np.bool_,
            description="Whether the power flow converged in this failure scenario",
        ),
    )


def describe_chronics(
    *,
    n_envs: int,
    n_load: int,
    n_gen: int,
    n_switch: int,
    n_line: int,
    n_trafo: int,
    n_trafo3w: int,
    n_sgen: int,
) -> ResultSpec:
    topo_vect_len = n_line * 2 + n_trafo * 2 + n_trafo3w * 3 + n_load + n_gen + n_sgen

    return (
        ResultsValue(
            key="load_p_input",
            shape=(n_envs, n_load),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each load, as input into the pfc",
        ),
        ResultsValue(
            key="load_q_input",
            shape=(n_envs, n_load),
            unit="MVar",
            dtype=np.float32,
            description="The MVar reactive power at each load, as input into the pfc",
        ),
        ResultsValue(
            key="prod_p_input",
            shape=(n_envs, n_gen),
            unit="MW",
            dtype=np.float32,
            description="The MW active power at each generator, as input into the pfc",
        ),
        ResultsValue(
            key="prod_v_input",
            shape=(n_envs, n_gen),
            unit="vm_pu",
            dtype=np.float32,
            description="The voltage setpoint at each generator, as input into the pfc",
        ),
        ResultsValue(
            key="switch_state",
            shape=(n_envs, n_switch),
            unit="",
            dtype=np.bool_,
            description="The state of each switch, as input into the pfc (including non-controllable switches)",
        ),
        ResultsValue(
            key="line_state",
            shape=(n_envs, n_line),
            unit="",
            dtype=np.bool_,
            description="The state of each line, as input into the pfc (including non-controllable lines)",
        ),
        ResultsValue(
            key="trafo_state",
            shape=(n_envs, n_trafo),
            unit="",
            dtype=np.bool_,
            description="The in-service state of each transformer, as input into the pfc (including non-controllable transformers)",
        ),
        ResultsValue(
            key="trafo_tap_pos",
            shape=(n_envs, n_trafo),
            unit="",
            dtype=np.int32,
            description="The tap position of each transformer, as input into the pfc (including non-controllable transformers)",
        ),
        ResultsValue(
            key="trafo3w_tap_pos",
            shape=(n_envs, n_trafo3w),
            unit="",
            dtype=np.int32,
            description="The tap position of each 3W transformer, as input into the pfc (including non-controllable 3W transformers)",
        ),
        ResultsValue(
            key="topo_vect",
            shape=(n_envs, topo_vect_len),
            unit="busbar",
            dtype=np.int32,
            description="The index of the busbar, ranging from 0 to n_busbar-1, for each element in the topo_vect",
        ),
    )
