# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from sweat.evaspa.merging import (
    MergingConfig,
    MergingMethod,
    UncertaintyMethod,
    merge,
    merge_to_dataset,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {"merging_method": "mean"},
            {
                "merging_method": "mean",
                "uncertainty_method": "std",
            },
        ),
        pytest.param(
            {"merging_method": "median"},
            {
                "merging_method": "median",
                "uncertainty_method": "nmad",
            },
        ),
        pytest.param(
            {
                "merging_method": "mean",
                "uncertainty_method": "interquartile",
            },
            {
                "merging_method": "mean",
                "uncertainty_method": "interquartile",
            },
        ),
    ],
)
def test_check_merging_config(config, expected) -> None:
    """
    Test check merging config
    """
    cfg = MergingConfig.model_validate(config)
    assert expected == cfg.model_dump(mode="json")


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"merging_method": "foo"},
        {"merging_method": "mean", "foo": "foo"},
        {"uncertainty_method": "std"},
        {"merging_method": "mean", "uncertainty_method": "foo"},
    ],
)
def test_check_merging_config_error(config) -> None:
    """
    Test check merging config with error
    """
    with pytest.raises(ValidationError):
        MergingConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "merging_method",
        "uncertainty_method",
        "expected_value",
        "expected_uncertainty",
    ),
    [
        pytest.param(
            MergingMethod.MEAN,
            UncertaintyMethod.STD,
            np.array([7.78]),
            np.array([6.18]),
        ),
        pytest.param(
            MergingMethod.MEDIAN,
            UncertaintyMethod.NMAD,
            np.array([7.3]),
            np.array([4.89]),
        ),
        pytest.param(
            MergingMethod.MEAN,
            UncertaintyMethod.INTERQUARTILE,
            np.array([7.78]),
            np.array([6.0]),
        ),
        pytest.param(
            MergingMethod.MEDIAN,
            UncertaintyMethod.INTERQUARTILE,
            np.array([7.3]),
            np.array([6.0]),
        ),
    ],
)
def test_merge(
    merging_method, uncertainty_method, expected_value, expected_uncertainty
) -> None:
    """:
    Test merge function
    """
    values = np.array([[2], [8], [10], [5.2], [7.3], [4], [21], [0.5], [12.0]])
    data = xr.Dataset(
        data_vars={f"model{i}": (["x"], v) for i, v in enumerate(values)},
        coords={
            "x": ("x", np.array([1])),
        },
        attrs={"description": "EF models"},
    )
    merged, uncertainty = merge(
        data,
        merging_method=merging_method,
        uncertainty_method=uncertainty_method,
    )
    assert merged
    assert uncertainty
    np.testing.assert_almost_equal(merged.values, expected_value, decimal=2)
    np.testing.assert_almost_equal(
        uncertainty.values, expected_uncertainty, decimal=2
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "merging_method",
        "uncertainty_method",
        "expected_value",
        "expected_uncertainty",
    ),
    [
        pytest.param(
            MergingMethod.MEAN,
            UncertaintyMethod.STD,
            np.array([7.78, np.nan, 7.78]),
            np.array([6.18, np.nan, 6.18]),
        ),
        pytest.param(
            MergingMethod.MEDIAN,
            UncertaintyMethod.NMAD,
            np.array([7.3, np.nan, 7.3]),
            np.array([4.89, np.nan, 4.89]),
        ),
        pytest.param(
            MergingMethod.MEAN,
            UncertaintyMethod.INTERQUARTILE,
            np.array([7.78, np.nan, 7.78]),
            np.array([6.0, np.nan, 6.0]),
        ),
        pytest.param(
            MergingMethod.MEDIAN,
            UncertaintyMethod.INTERQUARTILE,
            np.array([7.3, np.nan, 7.3]),
            np.array([6.0, np.nan, 6.0]),
        ),
    ],
)
def test_merge_with_nan(
    merging_method, uncertainty_method, expected_value, expected_uncertainty
) -> None:
    """:
    Test merge function with nan
    """
    values = np.array(
        [
            [2, np.nan, 2],
            [8, 8, 8],
            [10, 10, 10],
            [5.2, 5.2, 5.2],
            [7.3, 7.3, 7.3],
            [4, 4, 4],
            [21, 21, 21],
            [0.5, 0.5, 0.5],
            [12.0, 12.0, 12.0],
        ]
    )
    data = xr.Dataset(
        data_vars={f"model{i}": (["x"], v) for i, v in enumerate(values)},
        coords={
            "x": ("x", np.array([0, 1, 2])),
        },
        attrs={"description": "EF models"},
    )
    merged, uncertainty = merge(
        data,
        merging_method=merging_method,
        uncertainty_method=uncertainty_method,
    )
    np.testing.assert_almost_equal(merged.values, expected_value, decimal=2)
    np.testing.assert_almost_equal(
        uncertainty.values, expected_uncertainty, decimal=2
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("merging_method", "uncertainty_method", "expected"),
    [
        pytest.param(MergingMethod.MEAN, UncertaintyMethod.STD, (0.4, 0.15)),
        pytest.param(MergingMethod.MEDIAN, UncertaintyMethod.NMAD, (0.4, 0.15)),
        pytest.param(
            MergingMethod.MEAN, UncertaintyMethod.INTERQUARTILE, (0.4, 0.105)
        ),
    ],
)
def test_merge_to_dataset(merging_method, uncertainty_method, expected) -> None:
    """
    Test merge_to_dataset function
    """
    data = xr.Dataset(
        data_vars={
            "model1": (["y", "x"], np.random.normal(0.5, 0.1, (100, 100))),
            "model2": (["y", "x"], np.random.normal(0.3, 0.1, (100, 100))),
        },
        coords={
            "y": ("y", np.linspace(0, 99, num=100)),
            "x": ("x", np.linspace(0, 99, num=100)),
        },
        attrs={"description": "EF models"},
    )
    merged = merge_to_dataset(
        data,
        merging_method=merging_method,
        uncertainty_method=uncertainty_method,
        name="merged",
    )
    np.testing.assert_almost_equal(
        merged["merged"].mean(), expected[0], decimal=2
    )
    np.testing.assert_almost_equal(
        merged["uncertainty_merged"].mean(), expected[1], decimal=2
    )
