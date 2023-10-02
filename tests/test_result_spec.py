from dataclasses import asdict

import pandapower as pp

from grid2vec.result_spec import (
    describe_nminus1,
    describe_results,
    find_spec,
    set_env_dim,
    spec_to_jax,
)


def test_result_specs(net: pp.pandapowerNet) -> None:
    pp.runpp(net)

    res_spec = describe_results(
        n_envs=10,
        n_line=len(net.line),
        n_trafo=len(net.trafo),
        n_trafo3w=len(net.trafo3w),
        n_gen=len(net.gen),
        n_load=len(net.load),
    )

    assert hash(res_spec)

    for spec in res_spec:
        assert spec.shape[0] == 10
        assert len(spec.key)

        if spec.pp_res_table is None or spec.pp_res_key is None:
            continue

        # All specs present in net res_ tables
        table = getattr(net, spec.pp_res_table)
        assert spec.pp_res_key in table.keys()

        # All transformations work, correct shapes
        res = table[spec.pp_res_key].values
        assert res.shape == spec.shape[1:]
        transformed = spec.transformer(res)
        assert transformed.shape == res.shape

    nminus1_spec = describe_nminus1(
        n_envs=10,
        n_failures=12,
        n_line=len(net.line),
        n_trafo=len(net.trafo),
        n_trafo3w=len(net.trafo3w),
    )
    for spec in nminus1_spec:
        assert spec.shape[0:2] == (10, 12)

    jax_spec = spec_to_jax(res_spec)

    assert set(jax_spec.keys()) == set([spec.key for spec in res_spec])


def test_find_spec(net: pp.pandapowerNet) -> None:
    res_spec = describe_results(
        n_envs=10,
        n_line=len(net.line),
        n_trafo=len(net.trafo),
        n_trafo3w=len(net.trafo3w),
        n_gen=len(net.gen),
        n_load=len(net.load),
    )

    spec = find_spec(res_spec, "p_or_line")
    assert spec is not None
    assert spec.key == "p_or_line"

    spec = find_spec(res_spec, "gen_p")
    assert spec is not None
    assert spec.key == "gen_p"


def test_set_env_dim(net: pp.pandapowerNet) -> None:
    res_spec = describe_results(
        n_envs=10,
        n_line=len(net.line),
        n_trafo=len(net.trafo),
        n_trafo3w=len(net.trafo3w),
        n_gen=len(net.gen),
        n_load=len(net.load),
    )

    for spec in res_spec:
        assert spec.shape[0] == 10

    res_spec_new = set_env_dim(res_spec, 5)

    for spec_new, spec_old in zip(res_spec_new, res_spec):
        assert spec_new.shape[0] == 5

        for k, v in asdict(spec_old).items():
            if k != "shape":
                assert asdict(spec_new)[k] == v
