#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pandas as pd

import evaspa.ef as efm


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


def test_efmodel1() -> None:
    """
    Test EF model 1
    """
    assert efm.EFModel1.name == "EF_1"
    # Parameters
    albedo_min = 0.0
    albedo_max = 0.6
    dry_c1 = -13.0
    dry_c0 = 330.0
    wet_c1 = 30
    wet_c0 = 300.0
    # Generate data
    albedo, lst, ef_ref = setup_data(
        albedo_min, albedo_max, dry_c1, dry_c0, wet_c1, wet_c0
    )
    # Run model
    model = efm.EFModel1(var=albedo, lst=lst, nb_intervals=20, percentile=2)
    ef = model.compute_ef(albedo, lst)
    # Validation
    np.testing.assert_allclose(model.dry_coeffs, (dry_c1, dry_c0), rtol=0.1)
    np.testing.assert_allclose(model.wet_coeffs, (wet_c1, wet_c0), rtol=0.1)
    np.testing.assert_allclose(ef, ef_ref, atol=0.07)
