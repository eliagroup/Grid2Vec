from gymnasium.spaces.dict import Dict as GymDict

from grid2vec.grid import Grid
from grid2vec.result_spec import find_spec
from grid2vec.spaces import (
    default_results_spec,
    get_action_space,
    get_observation_space,
)


def test_default_results_spec(grid: Grid) -> None:
    res_spec = default_results_spec(grid=grid, n_envs=2)
    spec = find_spec(res_spec, "gen_p")
    assert spec is not None
    assert spec.shape == (2, grid.n_gen)


def test_get_action_space(grid: Grid) -> None:
    action_space = get_action_space(grid)
    assert action_space["switch"].shape == (grid.n_switch_controllable,)
    assert action_space["line"].shape == (grid.n_line_controllable,)
    assert action_space["trafo"].shape == (grid.n_trafo_tap_controllable,)
    assert action_space["trafo3w"].shape == (grid.n_trafo3w_tap_controllable,)
    assert action_space["topo_vect"].shape == (grid.n_topo_vect_controllable,)


def test_get_observation_space(grid: Grid) -> None:
    grid = grid
    res_spec = default_results_spec(grid=grid)
    observation_space = get_observation_space(res_spec)
    assert isinstance(observation_space, GymDict)
    assert set(["loading_line", "gen_q"]).issubset(set(observation_space.keys()))
    assert set(["gen_p", "load_p", "load_q"]).issubset(set(observation_space.keys()))
    assert observation_space["gen_p"].shape == (grid.n_gen,)
