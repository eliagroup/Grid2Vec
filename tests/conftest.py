import bz2
import json
import os
from copy import deepcopy
from typing import List

import grid2op
import numpy as np
import pandapower as pp
import pytest
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv

from grid2vec.grid import Grid, load_grid
from grid2vec.grid2op_compat import load_grid_grid2op


def compress_bz2(source_file: str) -> None:
    """Compress a file with bz2 and remove the original file

    Args:
        source_file (str): The file to compress
    """
    dest_path = source_file + ".bz2"
    with open(source_file, "rb") as source, bz2.BZ2File(dest_path, "wb") as dest:
        dest.writelines(source)
    os.remove(source_file)


def save_load_p(timestep_nets: List[pp.pandapowerNet], folder: str) -> None:
    """Saves the load_p values of a list of pandapower networks to a csv file

    Args:
        timestep_nets (List[pp.pandapowerNet]): The pandapower networks to save
        folder (str): Where to save to
    """
    base_net = timestep_nets[0]
    timestep_nets = timestep_nets[1:]

    # Export load real power
    load_p = base_net.load[["name", "p_mw"]].transpose()

    # Make the first row the header row
    load_p.columns = load_p.iloc[0]
    load_p.drop("name", inplace=True)

    # Append the value for the timestep
    for timestep_net in timestep_nets:
        load_p.loc[len(load_p)] = timestep_net.load["p_mw"].values

    os.makedirs(folder, exist_ok=True)
    load_p.to_csv(os.path.join(folder, "load_p.csv"), index=False, sep=";")

    # Compress with bz2
    compress_bz2(os.path.join(folder, "load_p.csv"))


def save_load_q(timestep_nets: List[pp.pandapowerNet], folder: str) -> None:
    """Saves the load_q values of a list of pandapower networks to a csv file

    Args:
        timestep_nets (List[pp.pandapowerNet]): The pandapower networks to save
        folder (str): Where to save to
    """
    base_net = timestep_nets[0]
    timestep_nets = timestep_nets[1:]

    # Export load reactive power
    load_q = base_net.load[["name", "q_mvar"]].transpose()

    # Make the first row the header row
    load_q.columns = load_q.iloc[0]
    load_q.drop("name", inplace=True)

    # Append the value for the timestep
    for timestep_net in timestep_nets:
        load_q.loc[len(load_q)] = timestep_net.load["q_mvar"].values

    os.makedirs(folder, exist_ok=True)
    load_q.to_csv(os.path.join(folder, "load_q.csv"), index=False, sep=";")

    # Compress with bz2
    compress_bz2(os.path.join(folder, "load_q.csv"))


def save_prod_p(timestep_nets: List[pp.pandapowerNet], folder: str) -> None:
    """Saves the prod_p values of a list of pandapower networks to a csv file

    Args:
        timestep_nets (List[pp.pandapowerNet]): The pandapower networks to save
        folder (str): Where to save to
    """
    base_net = timestep_nets[0]
    timestep_nets = timestep_nets[1:]

    # Export gen real power
    prod_p = base_net.gen[["name", "p_mw"]].transpose()

    # Make the first row the header row
    prod_p.columns = prod_p.iloc[0]
    prod_p.drop("name", inplace=True)

    # Append the value for the timestep
    for timestep_net in timestep_nets:
        prod_p.loc[len(prod_p)] = timestep_net.gen["p_mw"].values

    os.makedirs(folder, exist_ok=True)
    prod_p.to_csv(os.path.join(folder, "prod_p.csv"), index=False, sep=";")

    # Compress with bz2
    compress_bz2(os.path.join(folder, "prod_p.csv"))


def save_prod_v(timestep_nets: List[pp.pandapowerNet], folder: str) -> None:
    """Saves the prod_v values of a list of pandapower networks to a csv file

    Args:
        timestep_nets (List[pp.pandapowerNet]): The pandapower networks to save
        folder (str): Where to save to
    """
    base_net = timestep_nets[0]
    timestep_nets = timestep_nets[1:]

    # Export load real power
    # Convert per-unit voltages to actual voltages
    base_net.gen["grid2op_voltage"] = (
        base_net.bus.loc[base_net.gen.bus]["vn_kv"].values
        * base_net.gen["vm_pu"].values
    )

    prod_v = base_net.gen[["name", "grid2op_voltage"]].transpose()

    del base_net.gen["grid2op_voltage"]

    # Make the first row the header row
    prod_v.columns = prod_v.iloc[0]
    prod_v.drop("name", inplace=True)

    for timestep_net in timestep_nets:
        timestep_net.gen["grid2op_voltage"] = (
            timestep_net.bus.loc[timestep_net.gen.bus]["vn_kv"].values
            * timestep_net.gen["vm_pu"].values
        )

        prod_v.loc[len(prod_v)] = timestep_net.gen["grid2op_voltage"].values

    os.makedirs(folder, exist_ok=True)
    prod_v.to_csv(os.path.join(folder, "prod_v.csv"), index=False, sep=";")

    # Compress with bz2
    compress_bz2(os.path.join(folder, "prod_v.csv"))


def make_example_data(folder: str) -> None:
    """Build an example grid file which resembles the grid2op format but has more elements for testing"""
    net = pp.networks.mv_oberrhein()
    # Make some sgens to generators
    convert_sgens = list(
        np.random.choice(net.sgen.index, len(net.sgen) // 2, replace=False)
    )
    for idx in convert_sgens:
        sgen = net.sgen.loc[idx]
        pp.create_gen(
            net,
            bus=sgen.bus,
            name=sgen.name,
            p_mw=sgen.p_mw,
            vm_pu=1,
            sn_mva=sgen.sn_mva,
        )
        net.sgen.drop(idx, inplace=True)

    os.makedirs(folder, exist_ok=True)
    pp.to_json(net, os.path.join(folder, "grid.json"))

    # Use all lines for n-1
    line_for_nminus1 = np.ones(len(net.line), dtype=bool)
    np.save(os.path.join(folder, "line_for_nminus1.npy"), line_for_nminus1)

    # Randomly choose some lines to be part of the reward computation
    line_for_reward = np.random.rand(len(net.line)) > 0.5
    np.save(os.path.join(folder, "line_for_reward.npy"), line_for_reward)

    # Add some timestamps
    with open(os.path.join(folder, "start_datetime.info"), "w") as f:
        f.write("2021-02-03 00:30")
    with open(os.path.join(folder, "time_interval.info"), "w") as f:
        f.write("00:30")
    with open(os.path.join(folder, "default_crit_threshold.info"), "w") as f:
        f.write("0.5")

    # Extract the thermal limits from the net
    # Simplify the trafo limits to only the HV side
    thermal_limits = np.concatenate(
        [net.line["max_i_ka"], net.trafo["sn_mva"] / net.trafo["vn_hv_kv"]]
    )
    np.save(os.path.join(folder, "thermal_limits.npy"), thermal_limits)

    # Generate chronics
    # 2 chronics of 7 timesteps each are enough
    timestep_nets = [net]
    for _ in range(13):
        # Slightly change the loads and consumptions for each timestep (+- 1%)
        net_copy = deepcopy(net)
        net_copy.load["p_mw"] *= np.random.randn(len(net_copy.load)) * 0.01
        net_copy.load["q_mvar"] *= np.random.randn(len(net_copy.load)) * 0.01
        net_copy.gen["p_mw"] *= np.random.randn(len(net_copy.gen)) * 0.01

        # Check convergence
        pp.runpp(net_copy)
        timestep_nets.append(net_copy)

    save_load_q(timestep_nets[:7], os.path.join(folder, "chronics/0000"))
    save_load_p(timestep_nets[:7], os.path.join(folder, "chronics/0000"))
    save_prod_p(timestep_nets[:7], os.path.join(folder, "chronics/0000"))
    save_prod_v(timestep_nets[:7], os.path.join(folder, "chronics/0000"))

    save_load_q(timestep_nets[7:], os.path.join(folder, "chronics/0001"))
    save_load_p(timestep_nets[7:], os.path.join(folder, "chronics/0001"))
    save_prod_p(timestep_nets[7:], os.path.join(folder, "chronics/0001"))
    save_prod_v(timestep_nets[7:], os.path.join(folder, "chronics/0001"))


def enumerate_all_actions(env: BaseEnv) -> List[dict]:
    """Enumerates all possible actions for a given environment.

    Args:
        env (BaseEnv): The environment to enumerate actions for.

    Returns:
        List[dict]: A list of all possible actions in dict format
    """
    converter = IdToAct(env.action_space)
    converter.init_converter(
        set_line_status=True,
        change_line_status=False,
        set_topo_vect=True,
        change_topo_vect=False,
        change_bus_vect=False,
        redispatch=False,
        curtail=False,
        storage=False,
    )

    mapping = []
    for i in range(converter.n):
        action = converter.convert_act(i).as_serializable_dict()
        # Fixes a grid2op issue 527 action space sampling
        if len(action) == 1 and all([v == {} for v in action.values()]):
            action = {}
        mapping.append(action)
    return mapping


@pytest.fixture(scope="package")
def grid_folder() -> str:
    folder = os.path.join(os.path.dirname(__file__), "../test_data/testgrid/")
    if not os.path.exists(folder):
        make_example_data(folder)
    return folder


@pytest.fixture(scope="package")
def grid(grid_folder: str) -> Grid:
    grid = load_grid(grid_folder)
    return grid


@pytest.fixture
def net(grid_folder: str) -> pp.pandapowerNet:
    return pp.from_json(os.path.join(grid_folder, "grid.json"))


@pytest.fixture(scope="package")
def sandbox_grid() -> Grid:
    return load_grid_grid2op(
        "l2rpn_case14_sandbox", include_chronic_indices=list(range(10))
    )


@pytest.fixture(scope="package")
def sandbox_action_dump_file() -> str:
    dump_folder = os.path.join(
        os.path.dirname(__file__), "..", "test_data", "action_dumps"
    )
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

    dump_file = os.path.join(dump_folder, "sandbox_all.json")
    if not os.path.exists(dump_file):
        # Make sandbox dump
        all_actions = enumerate_all_actions(grid2op.make("l2rpn_case14_sandbox"))
        with open(dump_file, "w") as f:
            json.dump(all_actions, f)

    return dump_file
