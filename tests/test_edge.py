#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import pytest
import numpy as np
import pandas as pd

from pydantic import ValidationError

from evaspa.edge import FlatEdge, LinearEdge, create_edge, EdgeError


def setup_data(
    var_min: float,
    var_max: float,
    dry_c1: float,
    dry_c0: float,
    wet_c1: float,
    wet_c0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for tests
    """
    var = np.array([var_min, var_max])

    def dry_edge(v: float) -> float:
        return dry_c1 * v + dry_c0

    def wet_edge(v: float) -> float:
        return wet_c1 * v + wet_c0

    size = 125
    df = pd.DataFrame(
        data={"var": np.random.uniform(low=var_min, high=var_max, size=size * size)}
    )
    df["lst"] = df.apply(
        lambda x: np.random.uniform(wet_edge(x["var"]), dry_edge(x["var"])), axis=1
    )
    df["ef"] = df.apply(
        lambda x: (dry_edge(x["var"]) - x["lst"])
        / (dry_edge(x["var"]) - wet_edge(x["var"])),
        axis=1,
    )
    lst = df["lst"].to_numpy().reshape(size, -1)
    var = df["var"].to_numpy().reshape(size, -1)
    ef = df["ef"].to_numpy().reshape(size, -1)

    return var, lst, ef


@pytest.mark.parametrize(
    "config,expected",
    [
        pytest.param('{"selection":"max"}', 330),
        pytest.param('{"selection":"min"}', 300),
    ],
)
def test_flat_edge(config, expected) -> None:
    """
    Test FlatEdge
    """
    # Generate data
    var, lst, ef_ref = setup_data(
        var_min=0.0, var_max=0.6, dry_c0=330.0, dry_c1=-13.0, wet_c0=300, wet_c1=30
    )
    edge = FlatEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.value, expected, atol=1.0)
    np.testing.assert_allclose(edge.get(0.0), expected, atol=1.0)
    np.testing.assert_allclose(edge.get(var), expected * np.ones_like(var), atol=1.0)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param('{"selection":"foo"}'),
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
        '{"interval_type":"size","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"interval_type":"density","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"interval_type":"size","interval_nb":100,"percentile":[98,100],"selection":"max"}',
        '{"interval_type":"size","interval_nb":20,"percentile":[98,100],"selection":"median"}',
        '{"percentile":[98,100]}',
    ],
)
def test_linear_edge(config) -> None:
    """
    Test LinearEdge
    """
    # Generate data
    var, lst, ef_ref = setup_data(
        var_min=0.0, var_max=0.6, dry_c0=330.0, dry_c1=-13.0, wet_c0=300, wet_c1=30
    )
    edge = LinearEdge.model_validate_json(config)
    edge.fit(var, lst)
    np.testing.assert_allclose(edge.coeffs[1], 330, atol=1.0)
    np.testing.assert_allclose(edge.coeffs[0], -13, atol=1.0)
    np.testing.assert_allclose(edge.get(0.3), 330.0 - 13.0 * 0.3, atol=1.0)


@pytest.mark.parametrize(
    "config",
    [
        '{"interval_type":"size","interval_nb":20,"selection":"max"}',
        '{"interval_type":"foo","interval_nb":20,"percentile":[98,100],"selection":"max"}',
        '{"interval_type":"size","interval_nb":100,"percentile":[98,100],"selection":"foo"}',
        '{"interval_type":"size","interval_nb":0,"percentile":[98,100],"selection":"median"}',
        '{"interval_type":"size","interval_nb":20,"percentile":[46,12],"selection":"median"}',
        '{"interval_type":"size","interval_nb":20,"percentile":[-1,5],"selection":"median"}',
        '{"interval_type":"size","interval_nb":20,"percentile":[99,105],"selection":"median"}',
    ],
)
def test_linear_edge_error(config) -> None:
    """
    Test LinearEdge
    """
    with pytest.raises(ValidationError):
        LinearEdge.model_validate_json(config)


@pytest.mark.parametrize(
    "name,config",
    [
        pytest.param(
            "FlatEdge",
            {
                "selection": "max",
            },
        ),
        pytest.param(
            "FlatEdge",
            {
                "selection": "min",
            },
        ),
        pytest.param(
            "LinearEdge",
            {
                "interval_type": "size",
                "interval_nb": 20,
                "percentile": [98, 100],
                "selection": "max",
            },
        ),
        pytest.param(
            "LinearEdge",
            {
                "interval_type": "size",
                "interval_nb": 20,
                "percentile": [0, 4],
                "selection": "median",
            },
        ),
    ],
)
def test_create_edge(name, config) -> None:
    """
    Test LinearEdge
    """
    create_edge(name, config)


@pytest.mark.parametrize(
    "name,config",
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
        create_edge(name, config)
