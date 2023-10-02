import numpy as np

from grid2vec.batched_nminus1 import FailureType, batched_n_minus_1
from grid2vec.grid import Grid
from grid2vec.nminus1_definition import from_list_of_failures


def test_batched_n_minus_1(grid: Grid) -> None:
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

    failure_definition = from_list_of_failures(
        grid.net,
        [
            (FailureType.LINE, 0),
            (FailureType.LINE, 1),
            (FailureType.LINE, 2),
            (FailureType.LINE, 3),
            (FailureType.LINE, 4),
            (FailureType.LINE, 5),
            (FailureType.TRAFO, 0),
            (FailureType.TRAFO, 1),
            # (FailureType.TRAFO, 2),
            # (FailureType.TRAFO, 3),
            # (FailureType.TRAFO, 4),
            # (FailureType.TRAFO, 5),
            # (FailureType.TRAFO3W, 0),
            # (FailureType.TRAFO3W, 1),
            # (FailureType.TRAFO3W, 2),
            # (FailureType.TRAFO3W, 3),
            # (FailureType.TRAFO3W, 4),
            # (FailureType.TRAFO3W, 5),
            (FailureType.BUS, 0),
            (FailureType.BUS, 1),
            (FailureType.BUS, 2),
            (FailureType.BUS, 3),
            (FailureType.BUS, 4),
            (FailureType.BUS, 5),
        ],
    )

    res = batched_n_minus_1(
        grid.net,
        profile=profile,
        nminus1_definition=failure_definition,
    )

    assert res["nminus1_converged"].shape == (2, 14)

    # At least some of the computations should have converged
    assert np.any(res["nminus1_converged"])

    assert np.any(res["line_loading_per_failure"] > 0)
    assert np.any(res["trafo_loading_per_failure"] > 0)
    if grid.n_trafo3w > 0:
        assert np.any(res["trafo3w_loading_per_failure"] > 0)
    assert np.any(res["line_loading_grid2op_per_failure"] > 0)
    # All zero because net.trafo.grid2op_load_limit is not supplied
    # assert np.any(res["trafo_loading_grid2op_per_failure"] > 0)

    assert res["line_loading_per_failure"].shape == (2, 14, grid.n_line)
    assert res["trafo_loading_per_failure"].shape == (2, 14, grid.n_trafo)
    assert res["trafo3w_loading_per_failure"].shape == (2, 14, grid.n_trafo3w)
    assert res["line_loading_grid2op_per_failure"].shape == (2, 14, grid.n_line)
    assert res["trafo_loading_grid2op_per_failure"].shape == (
        2,
        14,
        grid.n_trafo,
    )
