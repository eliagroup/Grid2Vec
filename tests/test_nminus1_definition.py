import numpy as np
import pandapower as pp
import pandapower.networks as pn

from grid2vec.nminus1_definition import (
    FailureType,
    from_list_of_failures,
    full_nminus1_definition,
    load_nminus1_definition,
    subset_nminus1_definition,
)


def test_full_nminus1_definition(net: pp.pandapowerNet):
    net = pn.case118()
    definition = full_nminus1_definition(net)
    assert (
        definition.n_failures
        == net.line.shape[0] + net.trafo.shape[0] + net.trafo3w.shape[0]
    )

    line_counter = 0
    trafo_counter = 0
    trafo3w_counter = 0

    for global_failure_idx, (failure_type, failure_idx) in enumerate(
        definition.iter_failures()
    ):
        if failure_type == FailureType.LINE:
            assert failure_idx == line_counter
            line_counter += 1
        elif failure_type == FailureType.TRAFO:
            assert failure_idx == trafo_counter
            trafo_counter += 1
        elif failure_type == FailureType.TRAFO3W:
            assert failure_idx == trafo3w_counter
            trafo3w_counter += 1
        else:
            raise ValueError(f"Unknown failure type {failure_type}")

        assert definition[global_failure_idx] == (failure_type, failure_idx)

    assert line_counter == net.line.shape[0]
    assert trafo_counter == net.trafo.shape[0]
    assert trafo3w_counter == net.trafo3w.shape[0]


def test_load_nminus1_definition(net: pp.pandapowerNet, grid_folder: str) -> None:
    definition = load_nminus1_definition(net, folder=grid_folder)

    assert definition.n_failures == len(net.line)
    assert definition.line_mask.shape == (len(net.line),)
    assert definition.trafo_mask.shape == (len(net.trafo),)
    assert definition.trafo3w_mask.shape == (len(net.trafo3w),)
    assert definition.bus_mask.shape == (len(net.bus),)


def test_from_list_of_failures(net: pp.pandapowerNet, grid_folder: str) -> None:
    definition = []
    fail_lines = np.random.choice(len(net.line), 10)
    fail_trafos = np.random.choice(len(net.trafo), 10)
    fail_bus = np.random.choice(len(net.bus), 10)
    for line in fail_lines:
        definition.append((FailureType.LINE, line))
    for trafo in fail_trafos:
        definition.append((FailureType.TRAFO, trafo))
    for bus in fail_bus:
        definition.append((FailureType.BUS, bus))
    converted_definition = from_list_of_failures(net, definition)

    assert set(definition) == set(converted_definition.iter_failures())


def test_subset_nminus1_definition(net: pp.pandapowerNet, grid_folder: str) -> None:
    definition = load_nminus1_definition(net, folder=grid_folder)
    subset = subset_nminus1_definition(definition, [1, 3, 5])

    assert definition.line_mask.shape == subset.line_mask.shape
    assert definition.trafo_mask.shape == subset.trafo_mask.shape
    assert definition.trafo3w_mask.shape == subset.trafo3w_mask.shape
    assert definition.bus_mask.shape == subset.bus_mask.shape

    assert len(subset) == 3
    list_def = list(definition.iter_failures())
    list_subset = list(subset.iter_failures())
    assert list_def[1] == list_subset[0]
    assert list_def[3] == list_subset[1]
    assert list_def[5] == list_subset[2]
