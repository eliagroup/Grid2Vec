import argparse
import warnings
from functools import partial
from itertools import chain
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from grid2vec.grid import Grid, NMinus1Definition, load_grid

warnings.simplefilter("ignore", category=FutureWarning)


def process_nminus1_observation_file(
    file: Path, grid: Grid, crit_threshold: float = 90.0
) -> List[Dict[str, float]]:
    """Takes a single observation file from the powerflow analysis and returns a list
    of dicts containing the relevant information for the overloads.

    Args:
        file (Path): path to the observation file
        grid (Grid): loaded Grid2Vec grid
        crit_threshold (float, optional): threshold above which is considered a critical loading value. Defaults to 90.0.

    Returns:
        _type_: _description_
    """
    obs: dict = np.load(file, allow_pickle=True).item()

    line_loading_per_failure = obs["line_loading_per_failure"]
    trafo_loading_per_failure = obs["trafo_loading_per_failure"]
    trafo3w_loading_per_failure = obs["trafo3w_loading_per_failure"]

    # get grid indices instead of numpy element indices
    grid_line_index = grid.net.line.loc[pd.Series(grid.line_for_reward)].index
    grid_line_mapping = {i: l for i, l in enumerate(grid_line_index)}

    grid_trafo_index = grid.net.trafo.loc[pd.Series(grid.trafo_for_reward)].index
    grid_trafo_mapping = {i: t for i, t in enumerate(grid_trafo_index)}

    grid_trafo3w_index = grid.net.trafo3w.loc[pd.Series(grid.trafo3w_for_reward)].index
    grid_trafo3w_mapping = {i: t for i, t in enumerate(grid_trafo3w_index)}

    if grid.nminus1_definition is not None:
        nminus1_definition: NMinus1Definition = grid.nminus1_definition
    else:
        raise ValueError(
            "No nminus1 definition found. Please run the analysis with nminus1=True"
        )

    # get indices of failures
    failure_grid_indices = (
        grid.net.line.loc[pd.Series(nminus1_definition.line_mask)].index.tolist()
        + grid.net.trafo.loc[pd.Series(nminus1_definition.trafo_mask)].index.tolist()
        + grid.net.trafo3w.loc[
            pd.Series(nminus1_definition.trafo3w_mask)
        ].index.tolist()
    )

    # put asset type according to which position in the array it is
    failure_asset_types = (
        ["line"] * int(nminus1_definition.line_mask.sum())
        + ["trafo"] * int(nminus1_definition.trafo_mask.sum())
        + ["trafo3w"] * int(nminus1_definition.trafo3w_mask.sum())
    )
    df_failure_mapping = pd.DataFrame(
        {"grid_id": failure_grid_indices, "failure_asset_type": failure_asset_types}
    )

    results = []
    for (
        element_type,
        loading_per_failure_array,
        grid_index_mapping,
    ) in zip(
        ["line", "trafo", "trafo3w"],
        [
            line_loading_per_failure,
            trafo_loading_per_failure,
            trafo3w_loading_per_failure,
        ],
        [grid_line_mapping, grid_trafo_mapping, grid_trafo3w_mapping],
    ):
        # get indices of overloads
        overload_array = np.argwhere(loading_per_failure_array >= crit_threshold)
        for overload in overload_array:
            return_dict = {
                "vec_env": overload[0],
                "step": file.stem,
                "nminus1_failure_type": df_failure_mapping.loc[
                    overload[1], "failure_asset_type"
                ],
                "nminus1_failure_id": df_failure_mapping.loc[overload[1], "grid_id"],
                "overloaded_element_type": element_type,
                "overloaded_element_id": grid_index_mapping[overload[2]],
                "loading_value": loading_per_failure_array[
                    overload[0], overload[1], overload[2]
                ].item(),
            }
            results.append(return_dict)
    return results


def main(data_path: Path, nminus1: bool, dc: bool, n_procs: int, crit_threshold: float):
    """Takes all of the completed .npy files from the analysis produced by collect_powerflow_calculations.py
    and collects them into a csv file with the overloads.

    Args:
        data_path (Path): path to the grid situation
        nminus1 (bool): whether or not to run the analysis for n-1
        dc (bool): whether or not the analysis was run for DC
        n_procs (int): number of processes to use to collect results
        crit_threshold (float): threshold above which a loading_value is considered an overload

    Writes the completed file to data_path/powerflow_analysis/{path_suffix}/overloads.csv
    """
    grid = load_grid(data_path, nminus1=nminus1, dc=dc)
    path_suffix = "nminus1_dc" if dc else "nminus1_ac"
    analysis_path = data_path / "powerflow_analysis" / path_suffix
    list_files = list(analysis_path.glob("*.npy"))
    if not len(list_files):
        raise ValueError(
            f"No analysis files found in {analysis_path}. Try running collect_powerflow_calculations.py first."
        )
    results: Iterable[AsyncResult] = []
    if nminus1:
        with Pool(n_procs) as p:
            results = list(
                tqdm(
                    p.imap(
                        partial(
                            process_nminus1_observation_file,
                            grid=grid,
                            crit_threshold=crit_threshold,
                        ),
                        list_files,
                    )
                )
            )
    else:
        raise NotImplementedError("Not implemented yet")
    # flatten list of lists
    results = list(chain(results))
    df_results = pd.DataFrame(results)
    df_results.to_csv(analysis_path / "overloads.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path)  # path to grid data folder
    parser.add_argument("--dc", action="store_true")
    parser.add_argument("--n_procs", type=int, default=cpu_count())
    parser.add_argument("--crit_threshold", type=float, default=90.0)

    args = parser.parse_args()
    main(
        data_path=args.data_path,
        dc=args.dc,
        n_procs=args.n_procs,
        crit_threshold=args.crit_threshold,
    )
