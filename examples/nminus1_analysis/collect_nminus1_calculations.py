import argparse
import contextlib
import os
import warnings
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from grid2vec.env import make_env, vector_reset
from grid2vec.grid import Grid, load_grid
from grid2vec.nminus1_definition import NMinus1Definition
from grid2vec.solver_interface import compute_obs

warnings.simplefilter("ignore", category=FutureWarning)


def filter_loading_per_failure_to_reward_elements(
    loading_per_failure: np.ndarray, nminus1_elements: np.ndarray
) -> np.ndarray:
    """Filters the loading per failure array to only contain the elements that are
    relevant for the reward calculation.

    Args:
        loading_per_failure (np.ndarray): loading_per_failure array for given asset type from nminus1 calculation
        nminus1_elements (np.ndarray): array of elements that are relevant for the nminus1 calculation

    Returns:
        np.ndarray: filtered array with shape (n_envs, n_reward_elements, n_reward_elements)
    """
    return loading_per_failure[:, :, nminus1_elements]


def run_nminus1(
    grid: Grid, chronic_steps: List[Tuple[int, int]], destination_path: Path
) -> None:
    """Runs a powerflow analysis for a given grid and a given set of timesteps. Each
    step is saved as a numpy array in the destination path.

    Args:
        grid (Grid): loaded Grid2Vec grid
        chronic_steps (List[Tuple[int, int]]): list of tuples of the form (chronic, step) to run the powerflow analysis for
        destination_path (Path): path to save the results to
    """
    env = make_env(grid, n_envs=len(chronic_steps))

    chronics = np.array([chronic for chronic, _ in chronic_steps])
    steps = np.array([step for _, step in chronic_steps])

    env = vector_reset(env, target_chronics=chronics, target_timesteps=steps)

    keys_to_keep = [
        "converged",
        "nminus1_converged",
        "line_loading_per_failure",
        "trafo3w_loading_per_failure",
        "trafo_loading_per_failure",
    ]
    obs = compute_obs(env)
    obs_smaller = {key: obs[key] for key in keys_to_keep}

    if grid.nminus1_definition is None:
        raise ValueError(
            "No nminus1 definition found. Please re-run the analysis with nminus1=True"
        )
    else:
        nminus1_definition: NMinus1Definition = grid.nminus1_definition

    obs_smaller[
        "line_loading_per_failure"
    ] = filter_loading_per_failure_to_reward_elements(
        obs_smaller["line_loading_per_failure"], nminus1_definition.line_mask
    )
    obs_smaller[
        "trafo_loading_per_failure"
    ] = filter_loading_per_failure_to_reward_elements(
        obs_smaller["trafo_loading_per_failure"], nminus1_definition.trafo_mask
    )
    obs_smaller[
        "trafo3w_loading_per_failure"
    ] = filter_loading_per_failure_to_reward_elements(
        obs_smaller["trafo3w_loading_per_failure"], nminus1_definition.trafo3w_mask
    )

    for i in range(len(chronic_steps)):
        chronic, step = chronic_steps[i]
        obs_chronic_step = {key: obs_smaller[key][i] for key in keys_to_keep}
        np.save(
            destination_path / f"{chronic:04d}/{step}.npy",
            obs_chronic_step,
        )
    return None


def main(
    data_path: Path,
    dc: bool = True,
    n_procs: Optional[int] = None,
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
    grid = load_grid(data_path, nminus1=True, dc=dc)

    destination_path = data_path / "powerflow_analysis"
    path_suffix = "nminus1_dc" if dc else "nminus1_ac"
    destination_path = destination_path / path_suffix

    n_chronics = len(grid.chronics.n_timesteps)
    async_results_list: List[AsyncResult] = []
    chronic_steps: List[Tuple[int, int]] = []

    for chronic in range(n_chronics):
        for step in range(grid.chronics.n_timesteps[chronic]):
            chronic_steps.append((chronic, step))

    already_processed_chronic_steps = []

    # check for existing completed files and exclude them from processing
    for chronic in range(n_chronics):
        chronic_destination_path = destination_path / f"{chronic:04d}"

        if not chronic_destination_path.exists():
            chronic_destination_path.mkdir(parents=True)
            continue

        list_existing_files = list(chronic_destination_path.glob("*.npy"))
        already_processed_chronic_steps.extend(
            [(chronic, int(file.stem)) for file in list_existing_files]
        )

    chronic_steps = [
        chronic_step
        for chronic_step in chronic_steps
        if chronic_step not in already_processed_chronic_steps
    ]
    if not len(chronic_steps):
        raise ValueError(
            f"No new chronic steps to process. Try deleting {destination_path} and rerunning."
        )
    if n_procs == 1:
        run_nminus1(
            grid=grid, chronic_steps=chronic_steps, destination_path=destination_path
        )
        return
    with Pool(n_procs) as p:
        # give each process a list of timesteps
        chronic_step_chunks = np.array_split(chronic_steps, n_procs)

        for chronic_step_chunk in chronic_step_chunks:
            async_results_list.append(
                p.apply_async(
                    run_nminus1,
                    kwds=dict(
                        grid=grid,
                        chronic_steps=chronic_step_chunk,
                        destination_path=destination_path,
                    ),
                )
            )
        if surpress_print:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                [async_result.get() for async_result in async_results_list]
        else:
            [async_result.get() for async_result in async_results_list]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path)
    parser.add_argument("--n_procs", type=int, default=None)
    parser.add_argument("--dc", action="store_true")
    parser.add_argument("--surpress_print", action="store_true")
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        dc=args.dc,
        n_procs=args.n_procs,
        surpress_print=args.surpress_print,
    )
