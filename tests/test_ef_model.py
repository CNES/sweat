#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import pytest
import numpy as np
import pandas as pd

from evaspa.ef_model import create_efmodel


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
    "config",
    [
        {
            "dry_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [98, 100],
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [0, 2],
                    "selection": "median",
                },
            },
        },
    ],
)
def test_create_model(config) -> None:
    """
    Test LinearEdge
    """
    # Generate data
    var, lst, ef_ref = setup_data(
        var_min=0.0, var_max=0.6, dry_c0=330.0, dry_c1=-13.0, wet_c0=300, wet_c1=30
    )
    model = create_efmodel(config)
    model.fit(var, lst)
