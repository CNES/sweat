# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from evaspa.edge import (
    DoubleLinearEdge,
    Edge,
    EdgeError,
    FlatEdge,
    FlatLinearEdge,
    LinearEdge,
    ParabolicEdge,
    ThresholdLinearEdge,
)


def setup_data(
    var_min: float,
    var_max: float,
    dry_c0: float,
    wet_c0: float,
    dry_c1: float,
    wet_c1: float,
    dry_c2: float = 0.0,
    wet_c2: float = 0.0,
    dry_cut: float = 0.0,
    wet_cut: float = 0.0,
    size: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data for tests
    """
    np.random.seed(0)

    def dry_edge(v: float) -> float:
        return dry_c2 * v * v + dry_c1 * v + dry_c0

    def wet_edge(v: float) -> float:
        return wet_c2 * v * v + wet_c1 * v + wet_c0

    df = pd.DataFrame(
        data={
            "var": np.random.uniform(
                low=var_min, high=var_max, size=size * size
            )
        }
    )
    df["lst"] = df.apply(
        lambda x: np.random.uniform(wet_edge(x["var"]), dry_edge(x["var"])),
        axis=1,
    )
    dry_lst_cut = dry_edge(dry_cut)
    wet_lst_cut = wet_edge(wet_cut)
    df.loc[(df["var"] <= dry_cut) & (df["lst"] > dry_lst_cut), "lst"] = (
        dry_lst_cut
    )
    df.loc[(df["var"] <= wet_cut) & (df["lst"] < wet_lst_cut), "lst"] = (
        wet_lst_cut
    )
    lst = df["lst"].to_numpy().reshape(size, -1)
    var = df["var"].to_numpy().reshape(size, -1)

    return var, lst


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param('{"position":"top"}', 330),
        pytest.param('{"position":"bottom"}', 300),
    ],
)
def test_flat_edge(config, expected) -> None:
    """
    Test FlatEdge
    """
    # Generate data
    var, lst = setup_data(
        var_min=0.0,
        var_max=0.6,
        dry_c0=330.0,
        dry_c1=-13.0,
        wet_c0=300,
        wet_c1=30,
    )
    edge = FlatEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.value, expected, atol=1.0)
    np.testing.assert_allclose(edge.get(0.0), expected, atol=1.0)
    np.testing.assert_allclose(
        edge.get(var), expected * np.ones_like(var), atol=1.0
    )


@pytest.mark.parametrize(
    "config",
    [
        pytest.param('{"position":"foo"}'),
    ],
)
def test_flat_edge_error(config) -> None:
    """
    Test FlatEdge with exception raising
    """
    with pytest.raises(ValidationError):
        FlatEdge.model_validate_json(config)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"size","interval_size":0.05,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"density","interval_nb":100,"percentile":[99,100],"selection":"min"}',
        '{"position":"top","interval_type":"size","interval_size":0.05,"percentile":[98,100],"selection":"median"}',
        '{"position":"top","interval_type":"density","percentile":[98,100]}',
    ],
)
def test_linear_edge(config) -> None:
    """
    Test LinearEdge
    """
    # Generate data
    var, lst = setup_data(
        var_min=0.0,
        var_max=0.6,
        dry_c0=330.0,
        dry_c1=-13.0,
        wet_c0=300,
        wet_c1=30,
        size=80,
    )
    edge = LinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs[1], 330, atol=1.0)
    np.testing.assert_allclose(edge.coeffs[0], -13, atol=1.0)
    np.testing.assert_allclose(edge.get(0.3), 330.0 - 13.0 * 0.3, atol=1.0)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"size","interval_nb":20,"selection":"max"}',
        '{"position":"top","interval_type":"foo","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"size","interval_size":0.5,"percentile":[98,100],"selection":"foo"}',
        '{"position":"top","interval_type":"size","interval_size":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[46,12],"selection":"median"}',
        '{"position":"bottom","interval_type":"density","interval_nb":20,"percentile":[-1,5],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[99,105],"selection":"median"}',
        '{"interval_type":"density","interval_nb":20,"percentile":[99,105],"selection":"median"}',
    ],
)
def test_linear_edge_error(config) -> None:
    """
    Test LinearEdge
    """
    with pytest.raises(ValidationError):
        LinearEdge.model_validate_json(config)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"density","interval_nb":100,"percentile":[99,100],"selection":"min"}',
        '{"position":"top","interval_type":"size","interval_size":0.01,"percentile":[99,100],"selection":"median"}',
    ],
)
def test_parabolic_edge(config) -> None:
    """
    Test LinearEdge
    """
    # Generate data
    var, lst = setup_data(
        var_min=0.0,
        var_max=0.6,
        dry_c0=330.0,
        dry_c1=-5.0,
        dry_c2=-10.0,
        wet_c0=300,
        wet_c1=30,
        wet_c2=15.0,
        size=80,
    )
    edge = ParabolicEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs[2], 330, atol=1.0)
    np.testing.assert_allclose(edge.coeffs[1], -5, atol=1.0)
    np.testing.assert_allclose(edge.coeffs[0], -10, atol=1.0)
    np.testing.assert_allclose(
        edge.get(0.3), 330.0 - 5.0 * 0.3 - 10.0 * 0.3 * 0.3, atol=1.0
    )


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"size","interval_nb":20,"selection":"max"}',
        '{"position":"top","interval_type":"foo","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"size","interval_size":0.5,"percentile":[98,100],"selection":"foo"}',
        '{"position":"top","interval_type":"size","interval_size":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[46,12],"selection":"median"}',
        '{"position":"bottom","interval_type":"density","interval_nb":20,"percentile":[-1,5],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[99,105],"selection":"median"}',
        '{"interval_type":"density","interval_nb":20,"percentile":[99,105],"selection":"median"}',
        '{"interval_type":"size","interval_nb":20,"selection":"max"}',
    ],
)
def test_parabolic_edge_error(config) -> None:
    """
    Test LinearEdge
    """
    with pytest.raises(ValidationError):
        ParabolicEdge.model_validate_json(config)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"density","interval_nb":100,"percentile":[99,100],"selection":"min"}',
        '{"position":"top","interval_type":"size","interval_size":0.05,"percentile":[98,100],"selection":"median"}',
    ],
)
def test_top_linear_edge_with_threshold(config) -> None:
    """
    Test LinearEdge
    """
    # Generate data
    var, lst = setup_data(
        var_min=0.0,
        var_max=0.6,
        dry_c0=330.0,
        dry_c1=-13.0,
        dry_c2=0.0,
        wet_c0=300,
        wet_c1=30,
        wet_c2=0.0,
        dry_cut=0.2,
        wet_cut=0.0,
    )
    edge = ThresholdLinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs[1], 330, atol=1.0)
    np.testing.assert_allclose(edge.coeffs[0], -13, atol=1.0)
    np.testing.assert_allclose(edge.get(0.3), 330.0 - 13.0 * 0.3, atol=1.0)
    np.testing.assert_allclose(edge.threshold, 0.2, atol=0.05)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"bottom","interval_type":"density","interval_nb":100,"percentile":[0,2],"selection":"min"}',
        '{"position":"bottom","interval_type":"size","interval_size":0.05,"percentile":[0,2],"selection":"median"}',
    ],
)
def test_bottom_linear_edge_with_threshold(config) -> None:
    """
    Test LinearEdge
    """
    # Generate data
    var, lst = setup_data(
        var_min=0.0,
        var_max=0.6,
        dry_c0=330.0,
        dry_c1=-13.0,
        dry_c2=0.0,
        wet_c0=300,
        wet_c1=30,
        wet_c2=0.0,
        dry_cut=0.0,
        wet_cut=0.2,
    )
    edge = ThresholdLinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs[1], 300, atol=1.0)
    np.testing.assert_allclose(edge.coeffs[0], 30, atol=1.0)
    np.testing.assert_allclose(edge.get(0.3), 300.0 + 30.0 * 0.3, atol=1.0)
    np.testing.assert_allclose(edge.threshold, 0.2, atol=0.05)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"size","interval_nb":20,"selection":"max"}',
        '{"position":"top","interval_type":"foo","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"size","interval_size":0.5,"percentile":[98,100],"selection":"foo"}',
        '{"position":"top","interval_type":"size","interval_size":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"bottom","interval_type":"density","interval_nb":20,"percentile":[46,12],"selection":"median"}',
        '{"position":"bottom","interval_type":"density","interval_nb":20,"percentile":[-1,5],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[99,105],"selection":"median"}',
        '{"interval_type":"density","interval_nb":20,"percentile":[99,100],"selection":"median"}',
    ],
)
def test_linear_edge_with_threshold_error(config) -> None:
    """
    Test LinearEdge
    """
    with pytest.raises(ValidationError):
        ThresholdLinearEdge.model_validate_json(config)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"size","interval_size":0.02,"percentile":[99,100],"selection":"max"}',
        '{"position":"top","interval_type":"size","interval_size":0.05,"percentile":[98,100],"selection":"median"}',
    ],
)
def test_double_linear_edge(config) -> None:
    """
    Test LinearEdge
    """
    # Generate data
    var1, lst1 = setup_data(
        var_min=0.0,
        var_max=0.3,
        dry_c0=325.0,
        dry_c1=37.0,
        dry_c2=0.0,
        wet_c0=310,
        wet_c1=-36,
        wet_c2=0.0,
        dry_cut=0.0,
        wet_cut=0.0,
    )
    var2, lst2 = setup_data(
        var_min=0.3,
        var_max=0.6,
        dry_c0=340.0,
        dry_c1=-13.0,
        dry_c2=0.0,
        wet_c0=290,
        wet_c1=30,
        wet_c2=0.0,
        dry_cut=0.0,
        wet_cut=0.0,
    )
    var = np.concatenate([var1, var2])
    lst = np.concatenate([lst1, lst2])
    edge = DoubleLinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs1[1], 325, atol=1.0)
    np.testing.assert_allclose(edge.coeffs1[0], 37, atol=1.0)
    np.testing.assert_allclose(edge.coeffs2[1], 340, atol=1.0)
    np.testing.assert_allclose(edge.coeffs2[0], -13, atol=1.0)
    np.testing.assert_allclose(edge.get(0.2), 325.0 + 37.0 * 0.2, atol=1.0)
    np.testing.assert_allclose(edge.get(0.4), 340.0 - 13.0 * 0.4, atol=1.0)
    np.testing.assert_allclose(edge.inflection, 0.3, atol=0.05)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"size","interval_nb":20,"selection":"max"}',
        '{"position":"top","interval_type":"foo","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"size","interval_size":0.5,"percentile":[98,100],"selection":"foo"}',
        '{"position":"top","interval_type":"size","interval_size":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"bottom","interval_type":"density","interval_nb":20,"percentile":[46,12],"selection":"median"}',
        '{"position":"bottom","interval_type":"density","interval_nb":20,"percentile":[-1,5],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[99,105],"selection":"median"}',
        '{"interval_type":"density","interval_nb":20,"percentile":[99,100],"selection":"median"}',
    ],
)
def test_double_linear_edge_error(config) -> None:
    """
    Test LinearEdge
    """
    with pytest.raises(ValidationError):
        DoubleLinearEdge.model_validate_json(config)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"size","interval_size":0.05,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"density","interval_nb":100,"percentile":[99,100],"selection":"min"}',
        '{"position":"top","interval_type":"size","interval_size":0.05,"percentile":[98,100],"selection":"median"}',
    ],
)
def test_flat_linear_edge(config) -> None:
    """
    Test LinearEdge
    """
    # Generate data
    var, lst = setup_data(
        var_min=0.0,
        var_max=0.6,
        dry_c0=330.0,
        dry_c1=-13.0,
        dry_c2=0.0,
        wet_c0=300,
        wet_c1=30,
        wet_c2=0.0,
        dry_cut=0.2,
        wet_cut=0.0,
    )
    edge = FlatLinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs2[1], 330, atol=1.0)
    np.testing.assert_allclose(edge.coeffs2[0], -13, atol=1.0)
    np.testing.assert_allclose(edge.get(0.3), 330.0 - 13.0 * 0.3, atol=1.0)
    np.testing.assert_allclose(edge.inflection, 0.2, atol=0.05)


@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","interval_type":"size","interval_nb":20,"selection":"max"}',
        '{"position":"top","interval_type":"foo","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"position":"top","interval_type":"size","interval_size":0.5,"percentile":[98,100],"selection":"foo"}',
        '{"position":"top","interval_type":"size","interval_size":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":0,"percentile":[98,100],"selection":"median"}',
        '{"position":"bottom","interval_type":"density","interval_nb":20,"percentile":[46,12],"selection":"median"}',
        '{"position":"bottom","interval_type":"density","interval_nb":20,"percentile":[-1,5],"selection":"median"}',
        '{"position":"top","interval_type":"density","interval_nb":20,"percentile":[99,105],"selection":"median"}',
        '{"interval_type":"density","interval_nb":20,"percentile":[99,100],"selection":"median"}',
    ],
)
def test_flat_linear_edge_error(config) -> None:
    """
    Test LinearEdge
    """
    with pytest.raises(ValidationError):
        FlatLinearEdge.model_validate_json(config)


@pytest.mark.parametrize(
    ("name", "config"),
    [
        pytest.param(
            "FlatEdge",
            {
                "position": "top",
            },
        ),
        pytest.param(
            "FlatEdge",
            {
                "position": "bottom",
            },
        ),
        pytest.param(
            "LinearEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": [98, 100],
                "selection": "max",
            },
        ),
        pytest.param(
            "LinearEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": [0, 4],
                "selection": "median",
            },
        ),
        pytest.param(
            "ParabolicEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": [98, 100],
                "selection": "median",
            },
        ),
        pytest.param(
            "ParabolicEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": [0, 4],
                "selection": "max",
            },
        ),
        pytest.param(
            "ThresholdLinearEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": [98, 100],
                "selection": "median",
            },
        ),
        pytest.param(
            "ThresholdLinearEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": [0, 4],
                "selection": "max",
            },
        ),
        pytest.param(
            "DoubleLinearEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": [98, 100],
                "selection": "median",
            },
        ),
        pytest.param(
            "DoubleLinearEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": [0, 4],
                "selection": "max",
            },
        ),
        pytest.param(
            "FlatLinearEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": [98, 100],
                "selection": "median",
            },
        ),
        pytest.param(
            "FlatLinearEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": [0, 4],
                "selection": "max",
            },
        ),
    ],
)
def test_create_edge(name, config) -> None:
    """
    Test LinearEdge
    """
    Edge.create(name, config)


@pytest.mark.parametrize(
    ("name", "config"),
    [
        pytest.param(
            "FlatEdge",
            {
                "selection": "foo",
            },
        ),
        pytest.param(
            "LinearEdge",
            {
                "interval_type": "size",
                "interval_nb": 20,
                "selection": "max",
            },
        ),
    ],
)
def test_create_edge_error(name, config) -> None:
    """
    Test LinearEdge
    """
    with pytest.raises(EdgeError):
        Edge.create(name, config)
