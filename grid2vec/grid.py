import bz2
import datetime
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandapower as pp
import pandas as pd
from jax_dataclasses import Static, pytree_dataclass

from grid2vec.nminus1_definition import NMinus1Definition, load_nminus1_definition
from grid2vec.result_spec import ResultSpec, describe_pfc_results
from grid2vec.util import load_mask

# A list of (section_length, last_idx, table, key) sorted by last_idx
# which can be interpreted as topo_vect[idx] belongs to (table, key) iff idx < last_idx
# and idx > prev_last_idx
TopoVectLookup = Tuple[Tuple[int, int, str, str], ...]


@pytree_dataclass
class Chronics:
    """A class that holds the chronic load and generation data in flattened numpy arrays

    It can be read from grid2op format through load_chronics().
    """

    load_p: jnp.ndarray  # [float] (n_timesteps_total, n_load) - Load active power
    load_q: jnp.ndarray  # [float] (n_timesteps_total, n_load) - Load reactive power
    prod_p: jnp.ndarray  # [float] (n_timesteps_total, n_load) - Generation active power
    prod_v: jnp.ndarray  # [float] (n_timesteps_total, n_load) - Generation voltage setpoints
    n_timesteps: jnp.ndarray  # [int] (n_chronics) - Number of timesteps in each chronic

    def __eq__(self, other: Any):
        """The eq function compares all arrays"""
        if not isinstance(other, Chronics):
            return False

        return (
            jnp.array_equal(self.load_p, other.load_p)
            and jnp.array_equal(self.load_q, other.load_q)
            and jnp.array_equal(self.prod_p, other.prod_p)
            and jnp.array_equal(self.prod_v, other.prod_v)
            and jnp.array_equal(self.n_timesteps, other.n_timesteps)
        )


@pytree_dataclass
class Grid:
    """A static grid information class that is shared among all environments and should never be changed."""

    net: Static[pp.pandapowerNet]
    topo_vect_lookup: Static[TopoVectLookup]

    chronics: Chronics

    # Controllable element masks
    switch_controllable: jnp.ndarray  # [bool] (n_switches) - Whether the switch is controllable, i.E. operating > 150kV
    line_controllable: jnp.ndarray  # [bool] (n_line) - Whether the line can be switched on/off, i.E. is not in maintenance
    trafo_controllable: jnp.ndarray  # [bool] (n_trafo) - Whether the trafo can be switched on/off, i.E. is not in maintenance
    trafo_tap_controllable: jnp.ndarray  # [bool] (n_trafo) - Whether the trafo is controllable, i.E. has tap settings
    trafo3w_tap_controllable: jnp.ndarray  # [bool] (n_trafo3w) - Whether the trafo3w is controllable, i.E. has tap settings

    # Default values of the elements
    switch_default: jnp.ndarray  # [bool] (n_switches_controllable) - The default position of the switch
    line_default: jnp.ndarray  # [bool] (n_line_controllable) - Whether the line is in service by default
    trafo_default: jnp.ndarray  # [bool] (n_trafo_controllable) - Whether the trafo is in service by default
    trafo_tap_default: jnp.ndarray  # [bool] (n_trafo_tap_controllable) - The trafo's default tap position
    trafo_tap_min: jnp.ndarray  # [int] (n_trafo_tap_controllable) - The trafo's minimum tap position
    trafo_tap_max: jnp.ndarray  # [int] (n_trafo_tap_controllable) - The trafo's maximum tap position
    trafo3w_tap_default: jnp.ndarray  # [bool] (n_trafo3w_controllable) - The trafo3w's default tap position
    trafo3w_tap_min: jnp.ndarray  # [int] (n_trafo3w_tap_controllable) - The trafo3w's minimum tap position
    trafo3w_tap_max: jnp.ndarray  # [int] (n_trafo3w_tap_controllable) - The trafo3w's maximum tap position
    topo_vect_default: jnp.ndarray  # [int] (n_topo_vect_controllable) - The default busbar assignment for each element

    # Handling busbar assignments
    substation_affinity: jnp.ndarray  # [int] (len_topo_vect, max_bus_per_sub) - The affinity of each element in the topo vect to busbars. Each column holds the possible busbars an element can be set to, with -1 indicating this entry is not used.

    # Reward masking
    line_capacity_masked_for_reward: jnp.ndarray  # [float] (n_line) - 0 if the line is not used for reward computation, max_i_ka otherwise
    trafo_capacity_masked_for_reward: jnp.ndarray  # [float] (n_trafo) - 0 if the trafo is not used for reward computation, sn_mva otherwise
    trafo3w_capacity_masked_for_reward: jnp.ndarray  # [float] (n_trafo3w) - 0 if the trafo3w is not used for reward computation, sn_hv_mva otherwise

    # General grid information
    nminus1_definition: Optional[
        NMinus1Definition
    ]  # The n-1 definition to use for the grid, None means no n-1, i.e. plain n
    default_crit_threshold: float  # The default critical threshold to use for the rewards
    dc: bool  # Whether to do all loadflows in DC or AC
    timestep_minutes: int  # The length of a timestep in minutes

    res_spec: Static[ResultSpec]  # The result spec for the grid

    @property
    def n_switch_controllable(self) -> int:
        return self.switch_default.shape[0]

    @property
    def has_switch_actions(self) -> bool:
        return self.n_switch_controllable > 0

    @property
    def n_line_controllable(self) -> int:
        return self.line_default.shape[0]

    @property
    def has_line_actions(self) -> bool:
        return self.n_line_controllable > 0

    @property
    def n_trafo_controllable(self) -> int:
        return self.trafo_default.shape[0]

    @property
    def has_trafo_actions(self) -> bool:
        return self.n_trafo_controllable > 0

    @property
    def n_trafo_tap_controllable(self) -> int:
        return self.trafo_tap_default.shape[0]

    @property
    def has_trafo_tap_actions(self) -> bool:
        return self.n_trafo_tap_controllable > 0

    @property
    def n_trafo3w_tap_controllable(self) -> int:
        return self.trafo3w_tap_default.shape[0]

    @property
    def has_trafo3w_tap_actions(self) -> bool:
        return self.n_trafo3w_tap_controllable > 0

    @property
    def n_chronics(self) -> int:
        return len(self.chronics.n_timesteps)

    @property
    def uses_nminus1(self) -> bool:
        return self.nminus1_definition is not None

    @cached_property
    def topo_vect_controllable(self) -> jnp.ndarray:
        return jnp.sum(self.substation_affinity != -1, axis=1) > 1

    @property
    def n_topo_vect_controllable(self) -> int:
        return self.topo_vect_default.shape[0]

    @property
    def max_bus_per_sub(self) -> int:
        return self.substation_affinity.shape[1]

    @property
    def has_topo_vect_actions(self) -> bool:
        return self.max_bus_per_sub > 1

    @property
    def topo_vect_min(self) -> jnp.ndarray:
        return jnp.zeros(self.n_topo_vect_controllable, dtype=jnp.int32)

    @property
    def topo_vect_max(self) -> jnp.ndarray:
        return (
            jnp.sum(self.substation_affinity[self.topo_vect_controllable] != -1, axis=1)
            - 1
        )

    @cached_property
    def line_for_reward(self) -> jnp.ndarray:
        """Whether a line is used for reward computation"""
        return self.line_capacity_masked_for_reward != 0

    @cached_property
    def trafo_for_reward(self) -> jnp.ndarray:
        """Whether a trafo is used for reward computation"""
        return self.trafo_capacity_masked_for_reward != 0

    @cached_property
    def trafo3w_for_reward(self) -> jnp.ndarray:
        """Whether a trafo3w is used for reward computation"""
        return self.trafo3w_capacity_masked_for_reward != 0

    def __hash__(self):
        """Returns a hash of the grid, so it can be a static arg for jax jit"""
        # We never expect to change the grid, so we can safely use its id
        # If it does change, as it's immutable, the id will change.
        return id(self)

    def __eq__(self, other: Any):
        """The eq function acts according to the hash"""
        return self is other

    @property
    def n_switch(self) -> int:
        return len(self.net.switch)

    @property
    def n_load(self) -> int:
        return len(self.net.load)

    @property
    def n_gen(self) -> int:
        return len(self.net.gen)

    @property
    def n_sgen(self) -> int:
        return len(self.net.sgen)

    @property
    def n_line(self) -> int:
        return len(self.net.line)

    @property
    def n_trafo(self) -> int:
        return len(self.net.trafo)

    @property
    def n_trafo3w(self) -> int:
        return len(self.net.trafo3w)

    @property
    def len_topo_vect(self) -> int:
        return self.topo_vect_lookup[-1][1]


def chronics_current_timestep(
    timestep: jnp.ndarray,
    chronic_index: jnp.ndarray,
    chronics: Chronics,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns the current timestep in the chronics

    Args:
        timestep (jnp.ndarray): The timestep to look up, shape (n_env,)
        chronic_index (jnp.ndarray): The chronic index to look up, shape (n_env,)
        chronics (Chronics): The chronics dataclass

    Raises:
        ValueError: If a chronic is past its end

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: load_p, load_q, prod_p, prod_v each
            of shape (n_env, n_load/n_gen)
    """
    # Cannot compute observation for an environment that has ended
    timestep = eqx.error_if(
        timestep,
        timestep >= chronics.n_timesteps[chronic_index],
        "you passed the end of a chronic",
    )

    # Compute the beginning indices for each chronic
    beginning_index = jnp.cumsum(
        jnp.concatenate([jnp.array([0]), chronics.n_timesteps[:-1]])
    )

    flat_timestamp_idx = beginning_index[chronic_index] + timestep

    load_p = chronics.load_p[flat_timestamp_idx]
    load_q = chronics.load_q[flat_timestamp_idx]
    prod_p = chronics.prod_p[flat_timestamp_idx]
    prod_v = chronics.prod_v[flat_timestamp_idx]

    return load_p, load_q, prod_p, prod_v


def topo_vect_to_pandapower(net: pp.pandapowerNet) -> TopoVectLookup:
    """Returns a lookup table for the pandapower table and key for each index in the topo vect

    In grid2op, the topo vect is laid out by substation, i.e. env._topo_vect_to_sub is sorted.
    In grid2elia, we decide against this representation, and instead sort by pandapower element
    - line from
    - line to
    - trafo hv
    - trafo lv
    - trafo3w hv
    - trafo3w mv
    - trafo3w lv
    - gen
    - load
    - sgen


    Args:
        net (pp.pandapowerNet): The net to get size information from

    Returns:
        TopoVectLookup: A list of (last_idx, table, key) sorted by last_idx
            which can be interpreted as topo_vect[idx] belongs to (table, key) iff idx < last_idx
            and idx > prev_last_idx
    """
    retval: List[Tuple[int, int, str, str]] = []
    idx = len(net.line)
    retval.append((len(net.line), idx, "line", "from_bus"))
    idx += len(net.line)
    retval.append((len(net.line), idx, "line", "to_bus"))
    idx += len(net.trafo)
    retval.append((len(net.trafo), idx, "trafo", "hv_bus"))
    idx += len(net.trafo)
    retval.append((len(net.trafo), idx, "trafo", "lv_bus"))
    idx += len(net.trafo3w)
    retval.append((len(net.trafo3w), idx, "trafo3w", "hv_bus"))
    idx += len(net.trafo3w)
    retval.append((len(net.trafo3w), idx, "trafo3w", "mv_bus"))
    idx += len(net.trafo3w)
    retval.append((len(net.trafo3w), idx, "trafo3w", "lv_bus"))
    idx += len(net.gen)
    retval.append((len(net.gen), idx, "gen", "bus"))
    idx += len(net.load)
    retval.append((len(net.load), idx, "load", "bus"))
    idx += len(net.sgen)
    retval.append((len(net.sgen), idx, "sgen", "bus"))
    return tuple(retval)


def topo_vect_lookup(index: int, lookup: TopoVectLookup) -> Tuple[str, str, int]:
    """Finds the table and key for an index into the topo vect

    Args:
        index (int): The index to look up

    Returns:
        Tuple[str, str]: pandapower table, pandapower, table row (.iloc)
    """
    if index < 0:
        raise ValueError("Negative index provided to topo_vect_lookup")
    if not len(lookup):
        raise ValueError("Empty topo vect lookup table!")

    idx_subtracted = index
    for sec_len, idx, table, key in lookup:
        if index < idx:
            return table, key, idx_subtracted
        idx_subtracted -= sec_len

    raise ValueError(f"No topo vect entry for index {index}")


def empty_substation_affinity(net: pp.pandapowerNet) -> jnp.ndarray:
    """Returns a substation affinity where no element can be set to another bus

    Args:
        net (pp.pandapowerNet): The pandapower network

    Returns:
        np.ndarray: A substation_affinity array of shape (len_topo_vect, 1) where there is no
            alternative substation for any element on the network
    """
    return jnp.array(
        np.expand_dims(
            np.concatenate(
                [
                    net.line.from_bus,
                    net.line.to_bus,
                    net.trafo.hv_bus,
                    net.trafo.lv_bus,
                    net.trafo3w.hv_bus,
                    net.trafo3w.mv_bus,
                    net.trafo3w.lv_bus,
                    net.gen.bus,
                    net.load.bus,
                    net.sgen.bus,
                ]
            ),
            axis=-1,
        )
    )


# @partial(jax.jit, static_argnames=("lookup",))
def split_substation_affinity(
    substation_affinity: jnp.ndarray, lookup: TopoVectLookup
) -> Dict[Tuple[str, str], jnp.ndarray]:
    """Split an indexed substation affinity into bus assignments for each element type

    use like assignments = split_substation_affinity(np.take(substation_affinity, topo_vect))

    Args:
        substation_affinity (np.ndarray): An indexed substation affinity, shape (..., len_topo_vect)
            Note that this is not the full substation affinity matrix

    Returns:
        Dict[Tuple[str, str], np.ndarray]: A mapping of pandapower table and key to a bus assignment
            , i.e. ("line", "from_bus") -> [0, 1, 20, 3, ...]
    """
    starting_idx = 0
    retval: Dict[Tuple[str, str], np.ndarray] = {}
    for sec_len, last_idx, table, key in lookup:
        assert last_idx - starting_idx == sec_len
        retval[(table, key)] = substation_affinity[..., starting_idx:last_idx]
        starting_idx = last_idx
    return retval


def load_chronics(
    folder: str, convert_to_numpy_on_disk: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Loads a single chronic folder

    Args:
        folder (str): The chronic folder to load
        convert_to_numpy_on_disk (bool, optional): If true, convert the csv files to .npy file so
            the next time you open them, loading is faster. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Load_p, load_q, prod_p, prod_v
    """

    def load_csv(file_prefix: str) -> np.ndarray:
        filename = file_prefix + ".npy"
        if os.path.exists(os.path.join(folder, filename)):
            return jnp.load(os.path.join(folder, filename))

        filename = file_prefix + ".csv.bz2"
        with bz2.BZ2File(os.path.join(folder, filename), "r") as f:
            data = pd.read_csv(f, sep=";").values
            if convert_to_numpy_on_disk:
                np.save(os.path.join(folder, file_prefix + ".npy"), data)
            return jnp.array(data)

    load_p = load_csv("load_p")
    load_q = load_csv("load_q")
    prod_p = load_csv("prod_p")
    prod_v = load_csv("prod_v")

    assert load_p.shape == load_q.shape
    assert prod_p.shape == prod_v.shape
    assert len(load_p) == len(prod_p)

    return load_p, load_q, prod_p, prod_v


def load_substation_affinity(filename: str, net: pp.pandapowerNet) -> jnp.ndarray:
    """Loads and validates a substation affinity array

    If no file was found, uses an empty substation affinity mask, i.e. no topo actions possible

    Args:
        filename (str): The numpy file to load
        net (pp.pandapowerNet): The pandapower network to validate against

    Returns:
        np.ndarray: A substation affinity mask
    """
    try:
        substation_affinity = jnp.load(filename)
    except FileNotFoundError:
        substation_affinity = empty_substation_affinity(net)
    max_idx = (
        len(net.line) * 2
        + len(net.trafo) * 2
        + len(net.trafo3w) * 3
        + len(net.load)
        + len(net.gen)
        + len(net.sgen)
    )
    if len(substation_affinity) != max_idx:
        raise ValueError(
            f"Invalid substation affinity array, it has length: {len(substation_affinity)}"
        )

    # Validate that in the substation affinity, once an element has a -1 entry all consecutive
    # slices also have a -1
    last_slice = substation_affinity[:, 0] == -1
    for col in range(substation_affinity.shape[1]):
        new_slice = substation_affinity[:, col] == -1
        if jnp.any(last_slice & ~new_slice):
            raise ValueError(
                "The substation affinity array is invalid, it has a -1 entry and then a non -1 entry"
            )
        last_slice = new_slice | last_slice

    # Validate that all default entries are equal to the setup on the net at the moment
    default_slice = substation_affinity[:, 0]
    split = split_substation_affinity(default_slice, topo_vect_to_pandapower(net))
    for (table, key), default in split.items():
        if np.any(default != net[table][key].values):
            raise ValueError(
                f"The default substation affinity for {table}.{key} does not match the current setup"
            )

    return substation_affinity


def load_grid_info(folder: str | Path) -> Tuple[float, int]:
    """Loads the default critical threshold and timestep length from a folder

    Args:
        folder (str): The folder to load from

    Returns:
        Tuple[float, int]: The default critical threshold and timestep length
    """
    timestep_minutes = 5  # The grid2op default
    if os.path.exists(os.path.join(folder, "time_interval.info")):
        with open(os.path.join(folder, "time_interval.info"), "r") as f:
            timestep_minutes = datetime.datetime.strptime(f.read(), "%H:%M").minute

    default_crit_threshold = 1.0
    if os.path.exists(os.path.join(folder, "default_crit_threshold.info")):
        with open(os.path.join(folder, "default_crit_threshold.info"), "r") as f:
            default_crit_threshold = float(f.read())

    return default_crit_threshold, timestep_minutes


def load_grid(
    folder: str | Path,
    *,
    include_chronic_indices: Optional[List[int]] = None,
    nminus1: bool = False,
    dc: bool = False,
    reorder_loads: Optional[np.ndarray] = None,
    reorder_prods: Optional[np.ndarray] = None,
) -> Grid:
    """Loads a grid from a folder.

    Args:
        folder (str): Folder to load the grid from.
        include_chronic_indices (Optional[List[int]], optional): If given, only loads the chronics with the given indices. Defaults to None.
        nminus1 (bool, optional): Whether to load a nminus1 definition. Defaults to False.
        dc (bool, optional): Whether to run DC powerflows instead of AC. Defaults to False.
        reorder_loads (Optional[np.ndarray], optional): If given, reorders the loads according to the given indices. Defaults to None.
        reorder_prods (Optional[np.ndarray], optional): If given, reorders the prods according to the given indices. Defaults to None.

    Returns:
        Grid: The loaded grid.
    """
    net = pp.from_json(os.path.join(folder, "grid.json"))

    if reorder_loads is None:
        reorder_loads = np.arange(len(net.load))
    if reorder_prods is None:
        reorder_prods = np.arange(len(net.gen))

    # Load chronics
    load_p = []
    load_q = []
    prod_p = []
    prod_v = []
    n_timesteps = []
    chronics_to_load = os.listdir(os.path.join(folder, "chronics"))
    if not len(chronics_to_load):
        raise ValueError(f"No chronics found in folder {folder}/chronics")

    chronics_to_load.sort()
    if include_chronic_indices is not None:
        chronics_to_load = [chronics_to_load[i] for i in include_chronic_indices]

    for chronic_folder in chronics_to_load:
        load_p_, load_q_, prod_p_, prod_v_ = load_chronics(
            os.path.join(folder, "chronics", chronic_folder)
        )

        # Reorder loads and prods
        load_p.append(load_p_[:, reorder_loads])
        load_q.append(load_q_[:, reorder_loads])
        prod_p.append(prod_p_[:, reorder_prods])
        prod_v.append(prod_v_[:, reorder_prods])
        n_timesteps.append(len(load_p_))

    # Concatenate chronics
    chronics = Chronics(  # type: ignore
        load_p=jnp.array((np.concatenate(load_p, axis=0))),
        load_q=jnp.array((np.concatenate(load_q, axis=0))),
        prod_p=jnp.array((np.concatenate(prod_p, axis=0))),
        prod_v=jnp.array((np.concatenate(prod_v, axis=0))),
        n_timesteps=jnp.array(n_timesteps),
    )
    assert chronics.load_p.shape[1] == len(net.load)
    assert chronics.load_q.shape[1] == len(net.load)
    assert chronics.prod_p.shape[1] == len(net.gen)
    assert chronics.prod_v.shape[1] == len(net.gen)

    # Find out which switch is controllable
    switch_controllable = load_mask(
        os.path.join(folder, "switch_controllable.npy"),
        fallback=net.switch.closed.values,
    )

    line_controllable = load_mask(
        os.path.join(folder, "line_controllable.npy"),
        fallback=net.line.in_service.values,
    )

    trafo_controllable = load_mask(
        os.path.join(folder, "trafo_controllable.npy"),
        fallback=net.trafo.in_service.values,
    )

    trafo_tap_controllable = load_mask(
        os.path.join(folder, "trafo_tap_controllable.npy"),
        fallback=~np.isnan(net.trafo.tap_pos.values)
        & (net.trafo.tap_min.values < net.trafo.tap_max.values),
    )

    trafo3w_tap_controllable = load_mask(
        os.path.join(folder, "trafo3w_tap_controllable.npy"),
        fallback=~np.isnan(net.trafo3w.tap_pos.values)
        & (net.trafo3w.tap_min.values < net.trafo3w.tap_max.values),
    )

    line_for_reward = load_mask(
        os.path.join(folder, "line_for_reward.npy"), fallback=net.line.in_service.values
    )
    line_capacity_masked_for_reward = (
        jnp.zeros_like(net.line.max_i_ka.values)
        .at[line_for_reward]
        .set(net.line.max_i_ka.values[line_for_reward])
    )

    trafo_for_reward = load_mask(
        os.path.join(folder, "trafo_for_reward.npy"),
        fallback=net.trafo.in_service.values,
    )
    trafo_capacity_masked_for_reward = (
        jnp.zeros_like(net.trafo.sn_mva.values)
        .at[trafo_for_reward]
        .set(net.trafo.sn_mva.values[trafo_for_reward])
    )

    trafo3w_for_reward = load_mask(
        os.path.join(folder, "trafo3w_for_reward.npy"),
        fallback=net.trafo3w.in_service.values,
    )
    trafo3w_capacity_masked_for_reward = (
        jnp.zeros_like(net.trafo3w.sn_hv_mva.values)
        .at[trafo3w_for_reward]
        .set(net.trafo3w.sn_hv_mva.values[trafo3w_for_reward])
    )

    substation_affinity = load_substation_affinity(
        os.path.join(folder, "substation_affinity.npy"), net
    )
    topo_vect_default = jnp.zeros(
        jnp.sum(jnp.sum(substation_affinity != -1, axis=1) > 1)
    )

    nminus1_definition = load_nminus1_definition(net, folder) if nminus1 else None

    res_spec = describe_pfc_results(
        n_line=len(net.line),
        n_trafo=len(net.trafo),
        n_trafo3w=len(net.trafo3w),
        n_load=len(net.load),
        n_gen=len(net.gen),
        n_sgen=len(net.sgen),
        n_switch=len(net.switch),
        n_nminus1_cases=len(nminus1_definition)
        if nminus1_definition is not None
        else None,
    )

    default_crit_threshold, timestep_minutes = load_grid_info(folder)

    grid = Grid(  # type: ignore
        net=net,
        topo_vect_lookup=topo_vect_to_pandapower(net),
        chronics=chronics,
        # Element masks
        switch_controllable=switch_controllable,
        line_controllable=line_controllable,
        trafo_controllable=trafo_controllable,
        trafo_tap_controllable=trafo_tap_controllable,
        trafo3w_tap_controllable=trafo3w_tap_controllable,
        # Default values
        switch_default=jnp.array(net.switch.closed.values[switch_controllable]),
        line_default=jnp.array(net.line.in_service.values[line_controllable]),
        trafo_default=jnp.array(net.trafo.in_service.values[trafo_controllable]),
        trafo_tap_default=jnp.array(net.trafo.tap_pos.values[trafo_tap_controllable]),
        trafo3w_tap_default=jnp.array(
            net.trafo3w.tap_pos.values[trafo3w_tap_controllable]
        ),
        trafo_tap_min=jnp.array(net.trafo.tap_min.values[trafo_tap_controllable]),
        trafo_tap_max=jnp.array(net.trafo.tap_max.values[trafo_tap_controllable]),
        trafo3w_tap_min=jnp.array(net.trafo3w.tap_min.values[trafo3w_tap_controllable]),
        trafo3w_tap_max=jnp.array(net.trafo3w.tap_max.values[trafo3w_tap_controllable]),
        topo_vect_default=topo_vect_default,
        # Substation affinity
        substation_affinity=substation_affinity,
        # Reward masking
        line_capacity_masked_for_reward=line_capacity_masked_for_reward,
        trafo_capacity_masked_for_reward=trafo_capacity_masked_for_reward,
        trafo3w_capacity_masked_for_reward=trafo3w_capacity_masked_for_reward,
        # N-1
        nminus1_definition=nminus1_definition,
        default_crit_threshold=default_crit_threshold,
        dc=dc,
        timestep_minutes=timestep_minutes,
        res_spec=res_spec,
    )

    validate_grid(grid)
    return grid


def validate_grid(grid: Grid) -> None:
    """Runs assertions on the grid that are required for grid2elia to function

    Raises an AssertionError in case an assertion triggers

    Args:
        grid (Grid): The grid to check
    """
    # Check trafo taps
    assert jnp.all(
        jnp.isfinite(grid.trafo_tap_min)
    ), "Trafo tap min contains NaNs or infs"
    assert jnp.all(
        jnp.isfinite(grid.trafo_tap_max)
    ), "Trafo tap max contains NaNs or infs"
    assert jnp.all(grid.trafo_tap_min < grid.trafo_tap_max), "Trafo tap min >= max"
    assert jnp.all(
        grid.trafo_tap_default <= grid.trafo_tap_max
    ), "Trafo tap default > max"
    assert jnp.all(
        grid.trafo_tap_default >= grid.trafo_tap_min
    ), "Trafo tap default < min"

    # Check trafo3w taps
    assert jnp.all(
        jnp.isfinite(grid.trafo3w_tap_min)
    ), "Trafo3w tap min contains NaNs or infs"
    assert jnp.all(
        jnp.isfinite(grid.trafo3w_tap_max)
    ), "Trafo3w tap max contains NaNs or infs"
    assert jnp.all(
        grid.trafo3w_tap_min < grid.trafo3w_tap_max
    ), "Trafo3w tap min >= max"
    assert jnp.all(
        grid.trafo3w_tap_default <= grid.trafo3w_tap_max
    ), "Trafo3w tap default > max"
    assert jnp.all(
        grid.trafo3w_tap_default >= grid.trafo3w_tap_min
    ), "Trafo3w tap default < min"

    assert jnp.all(
        jnp.isfinite(grid.line_capacity_masked_for_reward)
    ), "Line capacity contains NaNs or infs"
    assert jnp.all(grid.line_capacity_masked_for_reward >= 0), "Line capacity < 0"
    assert jnp.all(
        jnp.isfinite(grid.trafo_capacity_masked_for_reward)
    ), "Trafo capacity contains NaNs or infs"
    assert jnp.all(grid.trafo_capacity_masked_for_reward >= 0), "Trafo capacity < 0"
    assert jnp.all(
        jnp.isfinite(grid.trafo3w_capacity_masked_for_reward)
    ), "Trafo3w capacity contains NaNs or infs"
    assert jnp.all(grid.trafo3w_capacity_masked_for_reward >= 0), "Trafo3w capacity < 0"

    topo_vect_controllable = jnp.sum(grid.substation_affinity != -1, axis=1) > 1
    assert jnp.array_equal(topo_vect_controllable, grid.topo_vect_controllable)
    assert grid.topo_vect_default.shape == (jnp.sum(topo_vect_controllable),)
