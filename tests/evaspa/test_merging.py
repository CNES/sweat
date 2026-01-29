# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pytest
import xarray as xr

from sweat.evaspa.merging import MergeMethod, merge, merge_to_dataset


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        pytest.param(MergeMethod.MEAN, (0.4, 0.15)),
        pytest.param(MergeMethod.MEDIAN, (0.4, 0.15)),
    ],
)
def test_merge(method, expected) -> None:
    """:
    Test merge function
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
    merged, uncertainty = merge(data, method=method)
    np.testing.assert_almost_equal(merged.mean(), expected[0], decimal=2)
    np.testing.assert_almost_equal(uncertainty.mean(), expected[1], decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "values1", "values2", "expected_value", "expected_uncertainty"),
    [
        pytest.param(
            MergeMethod.MEAN,
            np.array([1, 2, 2, 1, np.nan]),
            np.array([2, 1, 1, 2, 1]),
            np.array([1.5, 1.5, 1.5, 1.5, np.nan]),
            np.array([0.71, 0.71, 0.71, 0.71, np.nan]),
        ),
        pytest.param(
            MergeMethod.MEDIAN,
            np.array([1, 2, 2, 1, np.nan]),
            np.array([2, 1, 1, 2, 1]),
            np.array([1.5, 1.5, 1.5, 1.5, np.nan]),
            np.array([0.74, 0.74, 0.74, 0.74, np.nan]),
        ),
    ],
)
def test_merge_with_nan(
    method, values1, values2, expected_value, expected_uncertainty
) -> None:
    """:
    Test merge function
    """
    data = xr.Dataset(
        data_vars={
            "model1": (["x"], values1),
            "model2": (["x"], values2),
        },
        coords={
            "x": ("x", np.arange(len(values1))),
        },
        attrs={"description": "EF models"},
    )
    merged, uncertainty = merge(data, method=method)
    np.testing.assert_almost_equal(merged, expected_value, decimal=2)
    np.testing.assert_almost_equal(uncertainty, expected_uncertainty, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        pytest.param(MergeMethod.MEAN, (0.4, 0.15)),
        pytest.param(MergeMethod.MEDIAN, (0.4, 0.15)),
    ],
)
def test_merge_to_dataset(method, expected) -> None:
    """
    Test merge function
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
    merged = merge_to_dataset(data, method=method, name="merged")
    np.testing.assert_almost_equal(
        merged["merged"].mean(), expected[0], decimal=2
    )
    np.testing.assert_almost_equal(
        merged["uncertainty_merged"].mean(), expected[1], decimal=2
    )
