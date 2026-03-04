# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sweat.evaspa.edge import (
    DoubleLinearEdge,
    Edge,
    EdgeError,
    FlatEdge,
    FlatLinearEdge,
    FlatPercentileEdge,
    FlatRegressionEdge,
    LinearEdge,
    ParabolicEdge,
    ThresholdLinearEdge,
    compute_variable_percentile,
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
    law: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data for tests
    """
    np.random.seed(0)

    def dry_edge(v: float) -> float:
        return dry_c2 * v * v + dry_c1 * v + dry_c0

    def wet_edge(v: float) -> float:
        return wet_c2 * v * v + wet_c1 * v + wet_c0

    if law == "nonuniform":
        df = pd.DataFrame(
            data={
                "var": np.random.triangular(
                    left=var_min, mode=var_min, right=var_max, size=size * size
                )
            }
        )
    else:
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","percentile":1,"selection":"max"}',
        '{"position":"top","percentile":0.01,"selection":"median"}',
        '{"position":"top","percentile":0.01,"selection":"min"}',
        '{"position":"top","percentile":0.01,"selection":"mean"}',
        (
            '{"position":"top","percentile":0.01,"percentile_limit":100,'
            '"selection":"mean"}'
        ),
        '{"position":"top","nb_points":10, "selection":"median"}',
    ],
)
def test_flat_percentile_edge(config) -> None:
    """
    Test FlatPercentileEdge
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
    edge = FlatPercentileEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.value, 330, atol=1.0)
    np.testing.assert_allclose(edge.get(0.3), 330.0, atol=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        '{"position":"top","percentile":10,"selection":"foo"}',
        '{"position":"top","percentile":101,"selection":"median"}',
        '{"position":"bottom","percentile":-1,"selection":"median"}',
        (
            '{"position":"top","percentile":1,"nb_points":100,'
            '"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_limits":[10,0],'
            '"nb_points":100,"selection":"median"}'
        ),
        (
            '{"position":"top","percentile":1,"percentile_limit":-1,'
            '"selection":"median"}'
        ),
        '{"position":"top","nb_points":0, "selection":"mean"}',
        '{"position":"top", "selection":"mean"}',
        '{"percentile":1,"selection":"median"}',
    ],
)
def test_flat_percentile_edge_error(config) -> None:
    """
    Test FlatPercentileEdge with exception raising
    """
    with pytest.raises(ValidationError):
        FlatPercentileEdge.model_validate_json(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {
                "position": "top",
                "interval_type": "size",
                "interval_size": 0.25,
                "nb_points": 1,
                "selection": "median",
            },
            np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3]),
        ),
        pytest.param(
            {
                "position": "top",
                "interval_type": "size",
                "interval_nb": 10,
                "nb_points": 1,
                "selection": "median",
            },
            np.array([0, 1, 2, 2, 3, 5, 5, 6, 8, 9]),
        ),
        pytest.param(
            {
                "position": "top",
                "interval_type": "size",
                "interval_nb": 10,
                "interval_limits": [0.2, 0.8],
                "nb_points": 1,
                "selection": "median",
            },
            np.array([0, 1, 3, 8, 9, 9]),
        ),
        pytest.param(
            {
                "position": "top",
                "interval_type": "size",
                "interval_nb": 10,
                "interval_limits": ["percentile(20)", "percentile(88)"],
                "nb_points": 1,
                "selection": "median",
            },
            np.array([0, 1, 3, 8, 9, 9]),
        ),
        pytest.param(
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 5,
                "nb_points": 1,
                "selection": "median",
            },
            np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        ),
    ],
)
def test_get_intervals(config, expected) -> None:
    """
    Test method get_points in RegressionEdge
    """
    # Generate data
    var = np.array([0.0, 0.15, 0.21, 0.28, 0.35, 0.57, 0.58, 0.62, 0.81, 1.0])
    lst = np.ones(10)
    # Compute
    edge = LinearEdge.model_validate(config)
    df = edge._prepare(var, lst)  # noqa
    intervals = edge._get_intervals(df["var"])  # noqa
    # Check
    np.testing.assert_array_almost_equal(intervals, expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {
                "position": "top",
                "interval_type": "size",
                "interval_size": 0.5,
                "nb_points": 1,
                "selection": "median",
            },
            np.array([-0.65, -0.17, 0.28, 0.83]),
        ),
        pytest.param(
            {
                "position": "top",
                "interval_type": "size",
                "interval_nb": 10,
                "nb_points": 1,
                "selection": "median",
            },
            np.array(
                [-0.88, -0.65, -0.48, -0.3, -0.09, 0.1, 0.3, 0.5, 0.69, 1.0]
            ),
        ),
        pytest.param(
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 10,
                "nb_points": 1,
                "selection": "median",
            },
            np.array(
                [-0.32, -0.03, 0.15, 0.29, 0.41, 0.53, 0.66, 0.8, 0.98, 1.0]
            ),
        ),
    ],
)
def test_get_points(config, expected) -> None:
    """
    Test method get_points in RegressionEdge
    """
    # Generate data
    np.random.seed(0)
    var = np.random.normal(0.5, 0.5, 1000)
    var = np.clip(var, -1, 1)
    lst = np.ones(1000)
    # Compute
    edge = Edge.create("LinearEdge", config)
    points = edge.get_points(var, lst)  # type: ignore
    # Check
    np.testing.assert_array_almost_equal(points[0], expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":100,'
            '"percentile":1,"selection":"min"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":5,"percentile_limit":10,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"nb_points": 20, "selection":"median"}'
        ),
        '{"position":"top","interval_type":"density","percentile":2}',
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile_bounds":[50,10],'
            '"percentile_intervals":[100,1000],"selection":"median"}'
        ),
    ],
)
def test_linear_edge_with_variable_percentile(config) -> None:
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
        law="nonuniform",
    )
    edge = LinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs[1], 328, atol=1.0)
    np.testing.assert_allclose(edge.coeffs[0], -13, atol=1.0)
    np.testing.assert_allclose(edge.get(0.3), 328 - 13.0 * 0.3, atol=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","use_breakpoint":true}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","use_breakpoint":true,"fit_breakpoint":0.3}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","fit_breakpoint":0.3}'
        ),
    ],
)
def test_top_linear_edge_using_breakpoint(config) -> None:
    """
    Test LinearEdge (top) using breakpoint
    """
    # Generate data
    var1, lst1 = setup_data(
        var_min=0.0,
        var_max=0.3,
        dry_c0=325.0,
        dry_c1=37.0,
        dry_c2=0.0,
        wet_c0=290,
        wet_c1=70,
        wet_c2=0.0,
        dry_cut=0.0,
        wet_cut=0.0,
    )
    var2, lst2 = setup_data(
        var_min=0.3,
        var_max=0.5,
        dry_c0=348.0,
        dry_c1=-40.0,
        dry_c2=0.0,
        wet_c0=305,
        wet_c1=20,
        wet_c2=0.0,
        dry_cut=0.0,
        wet_cut=0.0,
    )
    var = np.concatenate([var1, var2])
    lst = np.concatenate([lst1, lst2])
    edge = LinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs[1], 348, atol=1.0)
    np.testing.assert_allclose(edge.coeffs[0], -40, atol=2.0)
    np.testing.assert_allclose(edge.get(0.3), 348 - 40.0 * 0.3, atol=1.0)
    assert edge.fit_breakpoint
    np.testing.assert_allclose(edge.fit_breakpoint, 0.3, atol=0.1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"bottom","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","use_breakpoint":true}'
        ),
        (
            '{"position":"bottom","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","use_breakpoint":true,"fit_breakpoint":0.3}'
        ),
        (
            '{"position":"bottom","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","fit_breakpoint":0.3}'
        ),
    ],
)
def test_bottom_linear_edge_using_breakpoint(config) -> None:
    """
    Test LinearEdge (bottom) using breakpoint
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
    edge = LinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    assert edge.fit_breakpoint
    np.testing.assert_allclose(edge.fit_breakpoint, 0.3, atol=0.1)
    np.testing.assert_allclose(edge.coeffs[1], 310, atol=2.0)
    np.testing.assert_allclose(edge.coeffs[0], -36, atol=3.0)
    np.testing.assert_allclose(edge.get(0.2), 310 - 36.0 * 0.2, atol=1.0)


@pytest.mark.unit
def test_linear_edge_with_slope_correction() -> None:
    """
    Test LinearEdge with slope correction enabled
    """
    # Generate data
    var, lst = setup_data(
        var_min=0.0,
        var_max=0.6,
        dry_c0=330.0,
        dry_c1=13.0,
        wet_c0=300,
        wet_c1=-30,
    )
    config_sup = (
        '{"position":"top","interval_type":"density","interval_nb":20,'
        '"percentile":2,"selection":"median","slope_correction":true}'
    )
    edge_sup = LinearEdge.model_validate_json(config_sup)
    edge_sup.fit(var, lst)
    np.testing.assert_allclose(edge_sup.coeffs[1], 333.3, atol=1.0)
    np.testing.assert_allclose(edge_sup.coeffs[0], 0.0, atol=0.01)
    config_inf = (
        '{"position":"bottom","interval_type":"density",'
        '"interval_nb":20,"percentile":2,"selection":"median",'
        '"slope_correction":true}'
    )
    edge_inf = LinearEdge.model_validate_json(config_inf)
    edge_inf.fit(var, lst)
    np.testing.assert_allclose(edge_inf.coeffs[1], 291.5, atol=1.0)
    np.testing.assert_allclose(edge_inf.coeffs[0], 0.0, atol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_nb":20,'
            '"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.1,'
            '"interval_nb":20,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"foo","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.5,'
            '"percentile":2,"selection":"foo"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_size":0.4,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":20,'
            '"percentile":112,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":-1,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"nb_points": 0, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":1,"nb_points": 10, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"selection":"median"}'
        ),
        (
            '{"interval_type":"density","interval_nb":20,"percentile":1,'
            '"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"interval_limits":["percentile(-1)","percentile(99)"],'
            '"nb_points": 10, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"interval_limits":[0,"percentile(99)"],"nb_points": 10, '
            '"selection":"median"}'
        ),
        (
            '{"position":"top","percentile":1,"nb_points":100,'
            '"selection":"median"}'
        ),
        (
            '{"position":"top","percentile":1,'
            '"percentile_bounds":[10,1],"selection":"median"}'
        ),
        (
            '{"position":"top","nb_points":100,'
            '"percentile_bounds":[10,1],"selection":"median"}'
        ),
        ('{"position":"top","percentile_bounds":[10,1],"selection":"median"}'),
        (
            '{"position":"top","percentile_bounds":[10,100],'
            '"percentile_intervals":[1000,100],"selection":"median"}'
        ),
        (
            '{"position":"top","percentile_bounds":[0,1],'
            '"percentile_intervals":[100,1000],"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_limits":[10,0],'
            '"nb_points":100,"selection":"median"}'
        ),
        (
            '{"position":"top","percentile":1,"percentile_limit":-1,'
            '"selection":"median"}'
        ),
    ],
)
def test_linear_edge_error(config) -> None:
    """
    Test LinearEdge with exception raising
    """
    with pytest.raises(ValidationError):
        LinearEdge.model_validate_json(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"density","interval_nb":100,'
            '"percentile":1,"selection":"min"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":1,"selection":"median"}'
        ),
    ],
)
def test_parabolic_edge(config) -> None:
    """
    Test ParabolicEdge
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_nb":20,'
            '"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"foo","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.5,'
            '"percentile":2,"selection":"foo"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":20,'
            '"percentile":101,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":-1,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.1,'
            '"interval_nb":20,"percentile":1,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"nb_points": 0, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":1,"nb_points": 10, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"selection":"median"}'
        ),
        (
            '{"interval_type":"density","interval_nb":20,"percentile":1,'
            '"selection":"median"}'
        ),
        ('{"interval_type":"size","interval_nb":20,"selection":"max"}'),
    ],
)
def test_parabolic_edge_error(config) -> None:
    """
    Test ParabolicEdge with exception raising
    """
    with pytest.raises(ValidationError):
        ParabolicEdge.model_validate_json(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"density","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":100,'
            '"percentile":1,"selection":"min"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median", "use_extremum":false}'
        ),
    ],
)
def test_top_linear_edge_with_threshold(config) -> None:
    """
    Test ThresholdLinearEdge (top)
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"bottom","interval_type":"density","interval_nb":100,'
            '"percentile":2,"selection":"min"}'
        ),
        (
            '{"position":"bottom","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","use_extremum":false}'
        ),
    ],
)
def test_bottom_linear_edge_with_threshold(config) -> None:
    """
    Test ThresholdLinearEdge (bottom)
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_nb":20,'
            '"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"foo","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.5,'
            '"percentile":2,"selection":"foo"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":101,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":-1,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"nb_points": 0, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":1,"nb_points": 10, "selection":"median"}'
        ),
        (
            '{"interval_type":"density","interval_nb":20,'
            '"percentile":1,"selection":"median"}'
        ),
    ],
)
def test_linear_edge_with_threshold_error(config) -> None:
    """
    Test ThresholdLinearEdge with exception raising
    """
    with pytest.raises(ValidationError):
        ThresholdLinearEdge.model_validate_json(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"density","interval_nb":100,'
            '"percentile":1,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","use_extremum":false}'
        ),
    ],
)
def test_double_linear_edge(config) -> None:
    """
    Test DoubleLinearEdge
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
    np.testing.assert_allclose(edge.coeffs1[1], 325, atol=3.0)
    np.testing.assert_allclose(edge.coeffs1[0], 37, atol=4.0)
    np.testing.assert_allclose(edge.coeffs2[1], 340, atol=3.0)
    np.testing.assert_allclose(edge.coeffs2[0], -13, atol=3.0)
    np.testing.assert_allclose(edge.get(0.2), 325.0 + 37.0 * 0.2, atol=1.0)
    np.testing.assert_allclose(edge.get(0.4), 340.0 - 13.0 * 0.4, atol=1.0)
    np.testing.assert_allclose(edge.fit_breakpoint, 0.3, atol=0.05)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_nb":20,'
            '"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"foo","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.5,'
            '"percentile":2,"selection":"foo"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":101,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":-1,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"nb_points": 0, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":1,"nb_points": 10, "selection":"median"}'
        ),
        (
            '{"interval_type":"density","interval_nb":20,'
            '"percentile":1,"selection":"median"}'
        ),
    ],
)
def test_double_linear_edge_error(config) -> None:
    """
    Test DoubleLinearEdge with exception raising
    """
    with pytest.raises(ValidationError):
        DoubleLinearEdge.model_validate_json(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":100,'
            '"percentile":1,"selection":"min"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.01,'
            '"percentile":2,"selection":"median","use_extremum":false}'
        ),
    ],
)
def test_flat_linear_edge(config) -> None:
    """
    Test FlatLinearEdge
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
    np.testing.assert_allclose(edge.fit_breakpoint, 0.2, atol=0.05)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_nb":20,'
            '"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"foo","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.5,'
            '"percentile":2,"selection":"foo"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":101,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":-1,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"nb_points": 0, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":1,"nb_points": 10, "selection":"median"}'
        ),
        (
            '{"interval_type":"density","interval_nb":20,"percentile":1,'
            '"selection":"median"}'
        ),
    ],
)
def test_flat_linear_edge_error(config) -> None:
    """
    Test FlatLinearEdge with exception raising
    """
    with pytest.raises(ValidationError):
        FlatLinearEdge.model_validate_json(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":100,'
            '"percentile":1,"selection":"min"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":2,"selection":"median"}'
        ),
        '{"position":"top","interval_type":"density","percentile":2}',
    ],
)
def test_flat_regression_edge(config) -> None:
    """
    Test FlatRegressionEdge
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
    edge = FlatRegressionEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeff, 330, atol=1.0)
    np.testing.assert_allclose(edge.get(0.3), 330.0, atol=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        (
            '{"position":"top","interval_type":"size","interval_nb":20,'
            '"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"foo","interval_nb":20,'
            '"percentile":2,"selection":"max"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.5,'
            '"percentile":2,"selection":"foo"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":0,'
            '"percentile":2,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"density","interval_nb":20,'
            '"percentile":101,"selection":"median"}'
        ),
        (
            '{"position":"bottom","interval_type":"density","interval_nb":20,'
            '"percentile":-1,"selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"nb_points": 0, "selection":"median"}'
        ),
        (
            '{"position":"top","interval_type":"size","interval_size":0.05,'
            '"percentile":1,"nb_points": 10, "selection":"median"}'
        ),
        (
            '{"interval_type":"density","interval_nb":20,'
            '"percentile":2,"selection":"median"}'
        ),
    ],
)
def test_flat_regression_edge_error(config) -> None:
    """
    Test FlatRegressionEdge with exception raising
    """
    with pytest.raises(ValidationError):
        FlatRegressionEdge.model_validate_json(config)


@pytest.mark.unit
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
                "percentile": 2,
                "selection": "max",
            },
        ),
        pytest.param(
            "LinearEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": 4,
                "selection": "median",
            },
        ),
        pytest.param(
            "ParabolicEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": 2,
                "selection": "median",
            },
        ),
        pytest.param(
            "ParabolicEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": 4,
                "selection": "max",
            },
        ),
        pytest.param(
            "ThresholdLinearEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": 2,
                "selection": "median",
            },
        ),
        pytest.param(
            "ThresholdLinearEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": 4,
                "selection": "max",
            },
        ),
        pytest.param(
            "DoubleLinearEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": 2,
                "selection": "median",
            },
        ),
        pytest.param(
            "DoubleLinearEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": 4,
                "selection": "max",
            },
        ),
        pytest.param(
            "FlatLinearEdge",
            {
                "position": "top",
                "interval_type": "density",
                "interval_nb": 20,
                "percentile": 2,
                "selection": "median",
            },
        ),
        pytest.param(
            "FlatLinearEdge",
            {
                "position": "bottom",
                "interval_type": "size",
                "interval_size": 0.05,
                "percentile": 4,
                "selection": "max",
            },
        ),
    ],
)
def test_create_edge(name, config) -> None:
    """
    Test edge creation
    """
    Edge.create(name, config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "config"),
    [
        pytest.param(
            "FlatEdge",
            {
                "position": "top",
                "selection": "foo",
            },
        ),
        pytest.param(
            "LinearEdge",
            {
                "position": "top",
                "interval_type": "size",
                "interval_nb": 20,
                "selection": "max",
            },
        ),
        pytest.param(
            "FooEdge",
            {
                "position": "top",
                "interval_type": "size",
                "interval_nb": 20,
                "selection": "max",
            },
        ),
    ],
)
def test_create_edge_error(name, config) -> None:
    """
    Test edge creation with exception raising
    """
    with pytest.raises(EdgeError):
        Edge.create(name, config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n", "n_sparse", "q_sparse", "n_dense", "q_dense", "expected"),
    [
        pytest.param(10, 100, 0.10, 1000000, 0.0001, 0.1),
        pytest.param(2000000, 100, 0.10, 1000000, 0.0001, 0.0001),
        pytest.param(1000, 100, 0.10, 1000000, 0.0001, 0.0178),
        pytest.param(10000, 100, 0.10, 1000000, 0.0001, 0.0032),
        pytest.param(1000, 500, 5, 100000, 0.1, 2.997),
        pytest.param(10000, 500, 5, 100000, 0.1, 0.547),
    ],
)
def test_compute_variable_percentile(
    n, n_sparse, q_sparse, n_dense, q_dense, expected
) -> None:
    """
    Test compute variable percentile
    """
    res = compute_variable_percentile(n, n_sparse, q_sparse, n_dense, q_dense)
    np.testing.assert_almost_equal(res, expected, decimal=3)
