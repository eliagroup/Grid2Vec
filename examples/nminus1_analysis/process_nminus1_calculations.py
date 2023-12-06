import argparse
import warnings
from functools import partial
from itertools import chain
from logging import getLogger
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from grid2vec.grid import Grid, NMinus1Definition, load_grid

warnings.simplefilter("ignore", category=FutureWarning)
logger = getLogger(__name__)


def process_nminus1_observation_file(
    file: Path, grid: Grid, crit_threshold: float = 90.0
) -> List[Dict[str, float | str]]:
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

    if grid.nminus1_definition is not None:
        nminus1_definition: NMinus1Definition = grid.nminus1_definition
    else:
        raise ValueError(
            "No nminus1 definition found. Please re-create the grid with a nminus1 definition."
        )

    # get mapping of nminus1 element ids to pandapower element ids for each element type
    nminus1_to_pandapower_element_mapping = {}
    for element_type in ["line", "trafo", "trafo3w"]:
        # index of the pandapower element ids that are relevant for the nminus1 analysis
        # by taking the True values from the nminus1 mask
        pandapower_nminus1_index = getattr(grid.net, element_type).index[
            getattr(nminus1_definition, f"{element_type}_mask")
        ]

        # create a simple dict mapping the nminus1 array element ids to the pandapower element ids
        nminus1_to_pandapower_element_mapping[element_type] = {
            i: e for i, e in enumerate(pandapower_nminus1_index)
        }

    results: List[Dict[str, float | str]] = []
    for element_type in ["line", "trafo", "trafo3w"]:
        # get relevant mapping created above
        pandapower_nminus1_element_mapping = nminus1_to_pandapower_element_mapping[
            element_type
        ]
        # get array of loading values for each failure
        loading_per_failure_array = obs[f"{element_type}_loading_per_failure"]

        # get the indices (failed element, overloaded element) of the overloads
        overload_array = np.argwhere(loading_per_failure_array >= crit_threshold)

        for overload in overload_array:
            failed_element_id = overload[0]
            overloaded_element_id = overload[1]

            nminus1_element_type = nminus1_definition[failed_element_id][0].name.lower()
            return_dict = {
                "chronic": file.parent.stem,
                "step": file.stem,
                "nminus1_failure_type": nminus1_element_type,
                "nminus1_failure_id": nminus1_to_pandapower_element_mapping[
                    nminus1_element_type
                ][failed_element_id],
                "overloaded_element_type": element_type,
                "overloaded_element_id": pandapower_nminus1_element_mapping[
                    overloaded_element_id
                ],
                "loading_value": loading_per_failure_array[
                    failed_element_id, overloaded_element_id
                ].item(),
            }
            results.append(return_dict)

    if not len(results):
        logger.warning(f"No overloads found in {file}")

    return results


def main(data_path: Path, dc: bool, n_procs: int, crit_threshold: float):
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
    grid = load_grid(data_path, nminus1=True, dc=dc)
    path_suffix = "nminus1_dc" if dc else "nminus1_ac"
    analysis_path = data_path / "powerflow_analysis" / path_suffix
    list_files = list(analysis_path.glob("*/*.npy"))

    if not len(list_files):
        raise ValueError(
            f"No analysis files found in {analysis_path}. Try running collect_powerflow_calculations.py first."
        )
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

    # flatten list of lists
    results = chain.from_iterable(results)  # type: ignore
    df_results = pd.DataFrame(results)
    df_results.to_csv(analysis_path / "nminus1_overloads.csv", index=False)


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
