import os

import chex
import grid2op
import jax.numpy as jnp
import numpy as np
import pandapower.networks as pn
import pytest
from grid2op.MakeEnv.PathUtils import DEFAULT_PATH_DATA

from grid2vec.grid import (
    chronics_current_timestep,
    empty_substation_affinity,
    load_chronics,
    load_grid,
    load_grid_info,
    load_substation_affinity,
    split_substation_affinity,
    topo_vect_lookup,
    topo_vect_to_pandapower,
)


def test_chronics_current_timestep(grid_folder: str) -> None:
    grid = load_grid(grid_folder, nminus1=True)

    timestep = jnp.array([0, 1])
    chronic = jnp.array([0, 1])

    chexified_chronics_current_timestep = chex.chexify(chronics_current_timestep)

    load_p, load_q, prod_p, prod_v = chexified_chronics_current_timestep(
        timestep, chronic, grid.chronics
    )

    assert load_p.shape == (2, grid.n_load)
    assert load_q.shape == (2, grid.n_load)
    assert prod_p.shape == (2, grid.n_gen)
    assert prod_v.shape == (2, grid.n_gen)

    with pytest.raises(Exception):
        chexified_chronics_current_timestep(
            timestep + 100000, chronic, grid.chronics, grid.chronics
        )
        chexified_chronics_current_timestep.wait_checks()


def test_topo_vect_to_pandapower() -> None:
    # Has some sgens
    net = pn.case145()
    lookup_table = topo_vect_to_pandapower(net)
    indices = [x[1] for x in lookup_table]
    # Assert sorted
    for i in range(len(indices) - 1):
        assert indices[i] <= indices[i + 1]

    sec_len_sum = sum([x[0] for x in lookup_table])
    assert indices[-1] == sec_len_sum

    max_idx = (
        len(net.line) * 2
        + len(net.trafo) * 2
        + len(net.trafo3w) * 3
        + len(net.load)
        + len(net.gen)
        + len(net.sgen)
    )
    assert indices[-1] == max_idx


def test_topo_vect_lookup() -> None:
    net = pn.case145()
    lookup_table = topo_vect_to_pandapower(net)

    max_idx = (
        len(net.line) * 2
        + len(net.trafo) * 2
        + len(net.trafo3w) * 3
        + len(net.load)
        + len(net.gen)
        + len(net.sgen)
    )
    for _ in range(100):
        idx = np.random.randint(0, max_idx)
        res = topo_vect_lookup(idx, lookup_table)
        assert res[2] >= 0
        assert res[2] < len(net[res[0]])
        assert res[1] in net[res[0]].columns


def test_empty_substation_affinity() -> None:
    net = pn.case145()
    substation_affinity = empty_substation_affinity(net)

    max_idx = (
        len(net.line) * 2
        + len(net.trafo) * 2
        + len(net.trafo3w) * 3
        + len(net.load)
        + len(net.gen)
        + len(net.sgen)
    )
    assert substation_affinity.shape == (max_idx, 1)
    assert np.all(substation_affinity >= 0)
    assert np.all(substation_affinity < len(net.bus))

    # Should only hold default values
    split = split_substation_affinity(
        substation_affinity[:, 0], topo_vect_to_pandapower(net)
    )
    for table, key in split.keys():
        assert np.array_equal(split[(table, key)], net[table][key].values)


def test_split_substation_affinity() -> None:
    net = pn.case145()
    substation_affinity = empty_substation_affinity(net)
    lookup_table = topo_vect_to_pandapower(net)

    split = split_substation_affinity(substation_affinity[:, 0], lookup_table)
    assert jnp.all(split[("line", "from_bus")] == net.line.from_bus.values)
    assert jnp.all(split[("line", "to_bus")] == net.line.to_bus.values)
    assert jnp.all(split[("trafo", "hv_bus")] == net.trafo.hv_bus.values)
    assert jnp.all(split[("trafo", "lv_bus")] == net.trafo.lv_bus.values)
    assert jnp.all(split[("trafo3w", "hv_bus")] == net.trafo3w.hv_bus.values)
    assert jnp.all(split[("trafo3w", "mv_bus")] == net.trafo3w.mv_bus.values)
    assert jnp.all(split[("trafo3w", "lv_bus")] == net.trafo3w.lv_bus.values)
    assert jnp.all(split[("load", "bus")] == net.load.bus.values)
    assert jnp.all(split[("gen", "bus")] == net.gen.bus.values)
    assert jnp.all(split[("sgen", "bus")] == net.sgen.bus.values)

    indices = jnp.zeros((5, substation_affinity.shape[0]), dtype=jnp.int32)
    indexed = jnp.take(substation_affinity, indices)
    assert indexed.shape == (5, substation_affinity.shape[0])
    split = split_substation_affinity(indexed, lookup_table)
    for table, key in split.keys():
        assert split[(table, key)].shape == (5, len(net[table]))


def test_load_substation_affinity(tmp_path: str) -> None:
    net = pn.case145()

    # Load and save the empty substation affinity
    substation_affinity = empty_substation_affinity(net)
    jnp.save(os.path.join(tmp_path, "substation_affinity.npy"), substation_affinity)
    loaded_substation_affinity = load_substation_affinity(
        os.path.join(tmp_path, "substation_affinity.npy"), net
    )
    assert jnp.array_equal(substation_affinity, loaded_substation_affinity)

    substation_affinity = empty_substation_affinity(net)[:, 0]
    # Pretend each element can be assigned to the next bus as well
    substation_affinity = jnp.stack(
        [
            substation_affinity,
            (substation_affinity + 1) % len(net.bus),
        ]
    ).transpose()
    # Set some random elements in the second row to -1
    minus1_indices = np.random.choice(substation_affinity.shape[0], 10)
    substation_affinity = substation_affinity.at[minus1_indices, 1].set(-1)
    jnp.save(os.path.join(tmp_path, "substation_affinity.npy"), substation_affinity)

    loaded_substation_affinity = load_substation_affinity(
        os.path.join(tmp_path, "substation_affinity.npy"), net
    )

    assert jnp.array_equal(substation_affinity, loaded_substation_affinity)

    # Make substation affinity invalid by adding another row without -1s
    substation_affinity = jnp.stack(
        [
            substation_affinity[:, 0],
            substation_affinity[:, 1],
            substation_affinity[:, 0],
        ]
    ).transpose()
    jnp.save(os.path.join(tmp_path, "substation_affinity.npy"), substation_affinity)

    with pytest.raises(ValueError):
        load_substation_affinity(os.path.join(tmp_path, "substation_affinity.npy"), net)

    # Set one element in the default row to a non-default bus, assert raise
    substation_affinity = empty_substation_affinity(net)
    substation_affinity = substation_affinity.at[0, 0].set(
        substation_affinity[0, 0] + 1
    )
    jnp.save(os.path.join(tmp_path, "substation_affinity.npy"), substation_affinity)

    with pytest.raises(ValueError):
        load_substation_affinity(os.path.join(tmp_path, "substation_affinity.npy"), net)


def test_load_grid_info(grid_folder: str) -> None:
    crit_threshold, timestep = load_grid_info(grid_folder)
    assert timestep > 0
    assert crit_threshold > 0
    assert crit_threshold <= 1


def test_load_grid(grid_folder: str) -> None:
    grid = load_grid(grid_folder, nminus1=True)

    assert grid.chronics is not None
    assert jnp.sum(grid.chronics.n_timesteps) == grid.chronics.load_p.shape[0]
    assert grid.chronics.load_p.shape == grid.chronics.load_q.shape
    assert grid.chronics.prod_p.shape == grid.chronics.prod_v.shape
    assert len(grid.net.load) == grid.chronics.load_p.shape[1]
    assert len(grid.net.gen) == grid.chronics.prod_p.shape[1]

    # We expect the first timestep in the first chronic to be equal to the pandapower grid
    assert jnp.array_equal(
        grid.chronics.load_p[0], grid.net.load.p_mw.values.astype(jnp.float32)
    )
    assert jnp.array_equal(
        grid.chronics.load_q[0], grid.net.load.q_mvar.values.astype(jnp.float32)
    )
    assert jnp.array_equal(
        grid.chronics.prod_p[0], grid.net.gen.p_mw.values.astype(jnp.float32)
    )
    vn_kv = jnp.array(grid.net.bus.loc[grid.net.gen.bus]["vn_kv"].values)
    assert jnp.array_equal(
        grid.chronics.prod_v[0], grid.net.gen.vm_pu.values.astype(jnp.float32) * vn_kv
    )

    assert grid.n_chronics == len(grid.chronics.n_timesteps)

    assert grid.n_switch_controllable <= grid.n_switch
    assert grid.n_line_controllable <= grid.n_line
    assert grid.n_trafo_controllable <= grid.n_trafo
    assert grid.n_trafo_tap_controllable <= grid.n_trafo
    assert grid.n_trafo3w_tap_controllable <= grid.n_trafo3w

    assert grid.nminus1_definition is not None
    assert grid.nminus1_definition.n_failures == grid.n_line

    max_idx = (
        len(grid.net.line) * 2
        + len(grid.net.trafo) * 2
        + len(grid.net.trafo3w) * 3
        + len(grid.net.load)
        + len(grid.net.gen)
        + len(grid.net.sgen)
    )
    assert grid.substation_affinity.shape == (max_idx, 1)
    assert grid.max_bus_per_sub == 1

    assert grid.topo_vect_lookup == topo_vect_to_pandapower(grid.net)
    assert jnp.sum(grid.topo_vect_controllable) == 0
    assert grid.n_topo_vect_controllable == 0
    assert grid.topo_vect_controllable.shape == (max_idx,)
    assert grid.len_topo_vect == max_idx

    assert hash(grid) is not None
    assert grid == grid

    grid2 = load_grid(grid_folder, nminus1=True)
    assert grid != grid2


def test_load_chronics(grid_folder: str) -> None:
    folder = os.path.join(grid_folder, "chronics/0000")
    load_p, load_q, prod_p, prod_v = load_chronics(folder)

    assert load_p.shape == load_q.shape
    assert prod_p.shape == prod_v.shape
    assert len(load_p) == len(prod_p)
    assert jnp.sum(load_p == 0) < (load_p.shape[1] / 2) * load_p.shape[0]
    assert jnp.sum(load_q == 0) < (load_p.shape[1] / 2) * load_p.shape[0]
    assert jnp.sum(prod_p == 0) < (prod_p.shape[1] / 2) * prod_p.shape[0]
    assert jnp.sum(prod_v == 0) < (prod_p.shape[1] / 2) * prod_p.shape[0]


def test_load_grid2op() -> None:
    # Make sure the folder exists
    g2o_env = grid2op.make("l2rpn_case14_sandbox")
    folder = os.path.join(DEFAULT_PATH_DATA, "l2rpn_case14_sandbox")
    grid = load_grid(folder, include_chronic_indices=list(range(10)))

    assert grid.n_chronics == 10
    assert grid.n_gen == g2o_env.n_gen
    assert grid.n_line + grid.n_trafo == g2o_env.n_line
    assert grid.n_load == g2o_env.n_load

    assert grid.chronics.n_timesteps.shape == (grid.n_chronics,)
    n_timesteps = sum(grid.chronics.n_timesteps)
    assert grid.chronics.load_p.shape == (n_timesteps, grid.n_load)
    assert grid.chronics.load_q.shape == (n_timesteps, grid.n_load)
    assert grid.chronics.prod_p.shape == (n_timesteps, grid.n_gen)
    assert grid.chronics.prod_v.shape == (n_timesteps, grid.n_gen)

    assert grid.nminus1_definition is None
