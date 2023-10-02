import numpy as np

from grid2vec.batched_pfc import batched_pfc
from grid2vec.grid import Grid
from grid2vec.result_spec import describe_results


def test_batched_pfc(grid: Grid) -> None:
    n_batch = 2
    load_p = grid.chronics.load_p[0:n_batch]
    load_q = grid.chronics.load_q[0:n_batch]
    prod_p = grid.chronics.prod_p[0:n_batch]
    vn_kv = grid.net.bus.loc[grid.net.gen.bus]["vn_kv"].values
    prod_v = grid.chronics.prod_v[0:n_batch]
    profile = {
        ("switch", "closed"): np.expand_dims(grid.net.switch.closed.values, 0).repeat(
            n_batch, axis=0
        ),
        ("line", "in_service"): np.expand_dims(
            grid.net.line.in_service.values, 0
        ).repeat(n_batch, axis=0),
        ("trafo", "tap_pos"): np.expand_dims(grid.net.trafo.tap_pos.values, 0).repeat(
            n_batch, axis=0
        ),
        ("trafo3w", "tap_pos"): np.expand_dims(
            grid.net.trafo3w.tap_pos.values, 0
        ).repeat(n_batch, axis=0),
        ("load", "p_mw"): load_p,
        ("load", "q_mvar"): load_q,
        ("gen", "p_mw"): prod_p,
        ("gen", "vm_pu"): prod_v / vn_kv,
    }

    res = batched_pfc(grid.net, profile=profile)

    res_spec = describe_results(
        n_batch,
        grid.n_line,
        grid.n_trafo,
        grid.n_trafo3w,
        grid.n_gen,
        grid.n_load,
    )

    for spec in res_spec:
        assert res[spec.key].shape == spec.shape

    for i in range(n_batch):
        # At least half of the elements in the net should have some load
        # In theory, it should be all of them, but the json file has some islanded elements
        # Hence, not all are expected to be loaded
        assert np.sum(res["loading_line"][i] > 0) >= len(grid.net.line) / 2
        assert np.sum(res["loading_trafo3w"][i] > 0) >= len(grid.net.trafo3w) / 2
        assert np.sum(res["loading_trafo"][i] > 0) >= len(grid.net.trafo) / 2
