from __future__ import annotations

import os
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Iterable, Tuple

import jax.numpy as jnp
import numpy as np
import pandapower as pp
from jax_dataclasses import pytree_dataclass

from grid2vec.util import load_mask


class FailureType(Enum):
    """The type of a failure"""

    LINE = 0
    TRAFO = 1
    TRAFO3W = 2
    BUS = 3


Failure = Tuple[FailureType, int]


@pytree_dataclass
class NMinus1Definition:
    """A definition of N-1 cases to be simulated. All masks are of shape (n_envs, n_elements) and
    are True for elements that should be failed in the respective case.
    """

    line_mask: jnp.ndarray
    trafo_mask: jnp.ndarray
    trafo3w_mask: jnp.ndarray
    bus_mask: jnp.ndarray

    @cached_property
    def n_failures(self) -> int:
        """The number of failures to be simulated"""
        return (
            jnp.sum(self.line_mask)
            + jnp.sum(self.trafo_mask)
            + jnp.sum(self.trafo3w_mask)
            + jnp.sum(self.bus_mask)
        )

    def __len__(self) -> int:
        return self.n_failures

    def __iter__(self) -> Iterable[Failure]:
        """Iterates over all failures in this definition"""
        return self.iter_failures()

    def __getitem__(self, idx: int) -> Failure:
        """Returns the idx-th failure in this definition"""
        n_line_failures = jnp.sum(self.line_mask)
        n_trafo_failures = jnp.sum(self.trafo_mask)
        n_trafo3w_failures = jnp.sum(self.trafo3w_mask)
        n_bus_failures = jnp.sum(self.bus_mask)

        if idx < n_line_failures:
            return (FailureType.LINE, jnp.where(self.line_mask.flatten())[0][idx])
        elif idx < n_line_failures + n_trafo_failures:
            return (
                FailureType.TRAFO,
                jnp.where(self.trafo_mask.flatten())[0][idx - n_line_failures],
            )
        elif idx < n_line_failures + n_trafo_failures + n_trafo3w_failures:
            return (
                FailureType.TRAFO3W,
                jnp.where(self.trafo3w_mask.flatten())[0][
                    idx - n_line_failures - n_trafo_failures
                ],
            )
        elif (
            idx
            < n_line_failures + n_trafo_failures + n_trafo3w_failures + n_bus_failures
        ):
            return (
                FailureType.BUS,
                jnp.where(self.bus_mask.flatten())[0][
                    idx - n_line_failures - n_trafo_failures - n_trafo3w_failures
                ],
            )
        else:
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

    def iter_failures(self) -> Iterable[Failure]:
        for i, mask in enumerate(
            [self.line_mask, self.trafo_mask, self.trafo3w_mask, self.bus_mask]
        ):
            for j, failed in enumerate(mask):
                if failed:
                    yield (FailureType(i), j)


def load_nminus1_definition(
    net: pp.pandapowerNet, folder: str | Path
) -> NMinus1Definition:
    """Loads an n-1 definition from a folder

    Uses empty masks in case of missing files

    Args:
        net (pp.pandapowerNet): The pandapower network to get the mask shapes from
        folder (str): The folder to load from

    Returns:
        NMinus1Definition: A definition of all failures to be simulated
    """
    return NMinus1Definition(
        line_mask=load_mask(
            os.path.join(folder, "line_for_nminus1.npy"),
            jnp.zeros(len(net.line), dtype=bool),
        ),
        trafo_mask=load_mask(
            os.path.join(folder, "trafo_for_nminus1.npy"),
            jnp.zeros(len(net.trafo), dtype=bool),
        ),
        trafo3w_mask=load_mask(
            os.path.join(folder, "trafo3w_for_nminus1.npy"),
            jnp.zeros(len(net.trafo3w), dtype=bool),
        ),
        bus_mask=load_mask(
            os.path.join(folder, "bus_for_nminus1.npy"),
            jnp.zeros(len(net.bus), dtype=bool),
        ),
    )


def full_nminus1_definition(
    net: pp.pandapowerNet,
    fail_lines: bool = True,
    fail_trafos: bool = True,
    fail_trafos3w: bool = True,
    fail_busses: bool = False,
) -> NMinus1Definition:
    """Creates a definition for a full n-1 analysis

    By default, this includes all line, trafo and trafo3w failures, but no bus failures

    Args:
        net (pp.pandapowerNet): The pandapower network
        fail_lines (bool, optional): Whether to include line failures. Defaults to True.
        fail_trafos (bool, optional): Whether to include trafo failures. Defaults to True.
        fail_trafos3w (bool, optional): Whether to include trafo3w failures. Defaults to True.
        fail_busses (bool, optional): Whether to include bus failures. Defaults to False.

    Returns:
        NMinus1Definition: A definition of all failures to be simulated
    """
    return NMinus1Definition(
        line_mask=(
            jnp.ones(len(net.line), dtype=bool)
            if fail_lines
            else jnp.zeros(len(net.line), dtype=bool)
        ),
        trafo_mask=(
            jnp.ones(len(net.trafo), dtype=bool)
            if fail_trafos
            else jnp.zeros(len(net.trafo), dtype=bool)
        ),
        trafo3w_mask=(
            jnp.ones(len(net.trafo3w), dtype=bool)
            if fail_trafos3w
            else jnp.zeros(len(net.trafo3w), dtype=bool)
        ),
        bus_mask=(
            jnp.ones(len(net.bus), dtype=bool)
            if fail_busses
            else jnp.zeros(len(net.bus), dtype=bool)
        ),
    )


def from_list_of_failures(
    net: pp.pandapowerNet, failures: Iterable[Failure]
) -> NMinus1Definition:
    """Converts a list of failures into a N-1 definition

    Warning, this will reorder the failures to be in ascending index. If you already passed the
    failures in ascending indices, then failures == result.iter_failures(). Otherwise, only
    set(failures) == set(result.iter_failures()).

    Args:
        net (pp.pandapowerNet): The pandapower network
        failures (Iterable[Failure]): A list of failures

    Returns:
        NMinus1Definition: A nminus1 definition with those elements in the mask set to true where
            there is a failure in the list
    """
    line_mask = np.zeros(len(net.line), dtype=bool)
    trafo_mask = np.zeros(len(net.trafo), dtype=bool)
    trafo3w_mask = np.zeros(len(net.trafo3w), dtype=bool)
    bus_mask = np.zeros(len(net.bus), dtype=bool)
    for failure_type, failure_idx in failures:
        if failure_type == FailureType.LINE:
            line_mask[failure_idx] = True
        elif failure_type == FailureType.TRAFO:
            trafo_mask[failure_idx] = True
        elif failure_type == FailureType.TRAFO3W:
            trafo3w_mask[failure_idx] = True
        elif failure_type == FailureType.BUS:
            bus_mask[failure_idx] = True
        else:
            raise ValueError(f"Unknown failure type {failure_type}")
    return NMinus1Definition(
        line_mask=jnp.array(line_mask),
        trafo_mask=jnp.array(trafo_mask),
        trafo3w_mask=jnp.array(trafo3w_mask),
        bus_mask=jnp.array(bus_mask),
    )


def subset_nminus1_definition(
    definition: NMinus1Definition, indices: Iterable[int]
) -> NMinus1Definition:
    """Returns a subset of the n-1 definition

    It will return only the failures that are in the indices array, in

    Args:
        definition (NMinus1Definition): The n-1 definition to subset
        indices (Iterable[int]): The indices to subset to, must be in range [0, len(definition))

    Returns:
        NMinus1Definition: A subset of the n-1 definition
    """
    list_definition = list(definition.iter_failures())

    line_mask = np.zeros_like(definition.line_mask)
    trafo_mask = np.zeros_like(definition.trafo_mask)
    trafo3w_mask = np.zeros_like(definition.trafo3w_mask)
    bus_mask = np.zeros_like(definition.bus_mask)

    for idx in indices:
        if idx < 0 or idx >= len(list_definition):
            raise ValueError(f"Index {idx} out of range [0, {len(list_definition)})")

        failure_type, failure_idx = list_definition[idx]
        if failure_type == FailureType.LINE:
            line_mask[failure_idx] = True
        elif failure_type == FailureType.TRAFO:
            trafo_mask[failure_idx] = True
        elif failure_type == FailureType.TRAFO3W:
            trafo3w_mask[failure_idx] = True
        elif failure_type == FailureType.BUS:
            bus_mask[failure_idx] = True
        else:
            raise ValueError(f"Unknown failure type {failure_type}")

    return NMinus1Definition(
        line_mask=jnp.array(line_mask),
        trafo_mask=jnp.array(trafo_mask),
        trafo3w_mask=jnp.array(trafo3w_mask),
        bus_mask=jnp.array(bus_mask),
    )
