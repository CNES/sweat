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
