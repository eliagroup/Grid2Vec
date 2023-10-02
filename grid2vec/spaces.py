from typing import Any

import numpy as np
from gymnasium.spaces.box import Box as BoxSpace
from gymnasium.spaces.dict import Dict as DictSpace

from grid2vec.grid import Grid
from grid2vec.result_spec import (
    ResultSpec,
    ResultsValue,
    describe_chronics,
    describe_nminus1,
    describe_results,
)


def get_action_space(grid: Grid) -> DictSpace:
    """Returns the action space in gym format

    Args:
        grid (Grid): The grid to get the action space for

    Returns:
        DictSpace: A gym action space with entries for switch, line, trafo and trafo3w actions
    """
    switch_actions = BoxSpace(
        low=0, high=1, shape=(grid.n_switch_controllable,), dtype=np.bool_
    )
    line_actions = BoxSpace(
        low=0, high=1, shape=(grid.n_line_controllable,), dtype=np.bool_
    )
    trafo_actions = BoxSpace(
        low=np.array(grid.trafo_tap_min),
        high=np.array(grid.trafo_tap_max),
        shape=(grid.n_trafo_tap_controllable,),
        dtype=np.int32,
    )
    trafo3w_actions = BoxSpace(
        low=np.array(grid.trafo3w_tap_min),
        high=np.array(grid.trafo3w_tap_max),
        shape=(grid.n_trafo3w_tap_controllable,),
        dtype=np.int32,
    )
    topo_actions = BoxSpace(
        low=np.array(grid.topo_vect_min),
        high=np.array(grid.topo_vect_max),
        shape=(int(grid.n_topo_vect_controllable),),
        dtype=np.int32,
    )
    return DictSpace(
        {
            "switch": switch_actions,
            "line": line_actions,
            "trafo": trafo_actions,
            "trafo3w": trafo3w_actions,
            "topo_vect": topo_actions,
        }
    )


def dtype_min(dtype: type) -> Any:
    """Returns the minimum value of a given dtype

    Args:
        dtype (type): The dtype to get the minimum value for

    Returns:
        Any: The minimum value of the dtype
    """
    if dtype == np.bool_:
        return 0  # gymnasium doesn't support bools
    elif dtype == np.int32:
        return np.iinfo(np.int32).min
    elif dtype == np.float32:
        return np.finfo(np.float32).min
    elif dtype == np.float64:
        return np.finfo(np.float64).min
    else:
        raise ValueError(f"Unknown dtype {dtype}")


def dtype_max(dtype: type) -> Any:
    """Returns the maximum value of a given dtype

    Args:
        dtype (type): The dtype to get the maximum value for

    Returns:
        Any: The maximum value of the dtype
    """
    if dtype == np.bool_:
        return 1  # gymnasium doesn't support bools
    elif dtype == np.int32:
        return np.iinfo(np.int32).max
    elif dtype == np.float32:
        return np.finfo(np.float32).max
    elif dtype == np.float64:
        return np.finfo(np.float64).max
    else:
        raise ValueError(f"Unknown dtype {dtype}")


def spec_to_boxspace(spec: ResultsValue, omit_env_dim: bool) -> BoxSpace:
    """Converts a ResultsValue spec to a gym BoxSpace

    Args:
        spec (ResultsValue): The spec to convert

    Returns:
        BoxSpace: The resulting gym BoxSpace
    """
    return BoxSpace(
        low=spec.low if spec.low is not None else dtype_min(spec.dtype),
        high=spec.high if spec.high is not None else dtype_max(spec.dtype),
        shape=spec.shape[1:] if omit_env_dim else spec.shape,
        dtype=spec.dtype,
    )


def default_results_spec(grid: Grid, n_envs: int = 1) -> ResultSpec:
    """Returns the default results spec for a given grid

    The default is to use all available observations

    Args:
        grid (Grid): The grid to get the results spec for
        n_envs [int]: The number of environments to create the specs space for. Results specs
            requires an environment dimension, but if you want to obtain the observation space for
            a single environment, you can set this to any value and use omit_env_dim on
            get_observation_space

    Returns:
        ResultSpec: The resulting results spec
    """
    res_spec = describe_results(
        n_envs=n_envs,
        n_line=grid.n_line,
        n_trafo=grid.n_trafo,
        n_trafo3w=grid.n_trafo3w,
        n_gen=grid.n_gen,
        n_load=grid.n_load,
    ) + describe_chronics(
        n_envs=n_envs,
        n_load=grid.n_load,
        n_gen=grid.n_gen,
        n_switch=grid.n_switch,
        n_line=grid.n_line,
        n_trafo=grid.n_trafo,
        n_trafo3w=grid.n_trafo3w,
        n_sgen=grid.n_sgen,
    )
    if grid.nminus1_definition is not None:
        res_spec += describe_nminus1(
            n_envs=n_envs,
            n_failures=len(grid.nminus1_definition),
            n_line=grid.n_line,
            n_trafo=grid.n_trafo,
            n_trafo3w=grid.n_trafo3w,
        )
    return res_spec


def get_observation_space(res_spec: ResultSpec, omit_env_dim=True) -> DictSpace:
    """Returns the observation space as a dict space

    Args:
        res_spec (ResultSpec): A result spec, defining which entries to include in the observation
            space
        omit_env_dim (bool, optional): Whether to omit the environment dimension from the results
            spec. Defaults to True.

    Returns:
        DictSpace: The resulting observation space, holding the entries of res_spec
    """
    return DictSpace(
        {spec.key: spec_to_boxspace(spec, omit_env_dim) for spec in res_spec}
    )
