from multiprocessing import Pool, cpu_count
from multiprocessing.pool import AsyncResult
from typing import Iterable, List
import argparse
from pathlib import Path
import warnings

import numpy as np

from grid2vec.env import make_env, vector_step, vector_reset
from grid2vec.solver_interface import compute_obs
from grid2vec.grid import Grid, load_grid
import os
import contextlib

warnings.simplefilter("ignore", category=FutureWarning)


def filter_loading_per_failure_to_reward_elements(
    loading_per_failure: np.ndarray, reward_elements: np.ndarray
) -> np.ndarray:
    """Filters the loading per failure array to only contain the elements that are
    relevant for the reward calculation.

    Args:
        loading_per_failure (np.ndarray): loading_per_failure array for given asset type from nminus1 calculation
        reward_elements (np.ndarray): _for_reward array for given asset type

    Returns:
        np.ndarray: filtered array with shape (n_envs, n_reward_elements, n_reward_elements)
    """    
    return loading_per_failure[:, :, reward_elements]


def run_nminus1(grid: Grid, steps: Iterable[int], destination_path: Path) -> None:
    """Runs a powerflow analysis for a given grid and a given set of timesteps. Each
    step is saved as a numpy array in the destination path.

    Args:
        grid (Grid): loaded Grid2Vec grid
        steps (Iterable[int]): List-like object of integer timesteps to run the powerflow analysis for
        destination_path (Path): path to save the results to
    """
    print(f"Running nminus1 from {steps[0]} to {steps[-1]}")
    env = make_env(grid, 1)
    env = vector_reset(env)
    env = vector_step(env, step_time=int(steps[0]))

    keys_to_keep = [
        "converged",
        "nminus1_converged",
        "line_loading_per_failure",
        "trafo3w_loading_per_failure",
        "trafo_loading_per_failure",
    ]

    for step in steps:
        obs = compute_obs(env)
        env = vector_step(env)
        obs_smaller = {key: obs[key] for key in keys_to_keep}
        obs_smaller["line_loading_per_failure"] = filter_loading_per_failure_to_reward_elements(
            obs_smaller["line_loading_per_failure"], grid.line_for_reward)
        obs_smaller["trafo_loading_per_failure"] = filter_loading_per_failure_to_reward_elements(
            obs_smaller["trafo_loading_per_failure"], grid.trafo_for_reward)
        obs_smaller["trafo3w_loading_per_failure"] = filter_loading_per_failure_to_reward_elements(
            obs_smaller["trafo3w_loading_per_failure"], grid.trafo3w_for_reward)

        np.save(destination_path / f"{int(step)}.npy", obs_smaller)


def main(
    data_path: Path,
    nminus1: bool = True,
    dc: bool = True,
    n_procs: int = None,
    surpress_print: bool = False,
):
    """This function loads a grid from a given data directory and runs a powerflow analysis
    for all timesteps. The results are saved in the data directory under
    data_path/powerflow_analysis/nminus1_dc or data_path/powerflow_analysis/nminus1_ac or
    data_path/powerflow_analysis/n_dc or data_path/powerflow_analysis/n_ac depending on the
    parameters.

    Args:
        data_path (Path): path containing grid and chronics data
        nminus1 (bool, optional): whether or not to run nminus1 analysis. Defaults to True.
        dc (bool, optional): when True, run dc analysis. When False, run ac analysis. Defaults to True.
        n_procs (int, optional): number of forked processes to create. If not given, this is set at multiprocessing.cpu_count(). Defaults to None.
        surpress_print (bool, optional): surpress the print output in the case of dependencies giving large amounts of print output. Defaults to False.
    """
    if n_procs is None:
        n_procs = cpu_count()
    grid = load_grid(data_path, nminus1=nminus1, dc=dc)

    destination_path = data_path / "powerflow_analysis"
    path_suffix = (
        "nminus1_dc"
        if nminus1 and dc
        else "nminus1_ac"
        if nminus1
        else "n_dc"
        if dc
        else "n_ac"
    )
    destination_path = destination_path / path_suffix

    if not destination_path.exists():
        destination_path.mkdir()

    # check for existing completed files and exclude them from processing
    list_existing_files = list(destination_path.glob("*.npy"))
    already_processed = [int(file.stem) for file in list_existing_files]
    list_steps = np.arange(grid.chronics.n_timesteps)[:10]
    list_steps = np.setdiff1d(list_steps, already_processed)

    # give each process a list of timesteps
    step_partitions = np.array_split(list_steps, n_procs)

    async_results_list: List[AsyncResult] = []
    with Pool(n_procs) as p:
        for steps in step_partitions:
            async_results_list.append(
                p.apply_async(
                    run_nminus1,
                    kwds=dict(
                        grid=grid, steps=steps, destination_path=destination_path
                    ),
                )
            )
        if surpress_print:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                results = [async_result.get() for async_result in async_results_list]
        else:
            results = [async_result.get() for async_result in async_results_list]
        print("Finished all processes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path)
    parser.add_argument("--n_procs", type=int, default=None)
    parser.add_argument("--nminus1", action="store_true")
    parser.add_argument("--dc", action="store_true")
    parser.add_argument("--surpress_print", action="store_true")
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        nminus1=args.nminus1,
        dc=args.dc,
        n_procs=args.n_procs,
        surpress_print=args.surpress_print,
    )
