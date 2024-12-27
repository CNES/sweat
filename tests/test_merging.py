#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import pytest
import numpy as np
import xarray as xr

from evaspa.merging import merge, MergeMethod, merge_to_dataset


@pytest.mark.parametrize(
    "method,expected",
    [
        pytest.param(MergeMethod.MEAN, 0.4),
        pytest.param(MergeMethod.MEAN, 0.4),
    ],
)
def test_merge(method, expected) -> None:
    """
    Test merge function
    """
    data = xr.Dataset(
        data_vars=dict(
            model1=(["y", "x"], np.random.normal(0.5, 0.1, (100, 100))),
            model2=(["y", "x"], np.random.normal(0.3, 0.1, (100, 100))),
        ),
        coords=dict(
            y=("y", np.linspace(0, 99, num=100)),
            x=("x", np.linspace(0, 99, num=100)),
        ),
        attrs=dict(description="EF models"),
    )
    merged = merge(data, method=method)
    np.testing.assert_almost_equal(merged.mean(), expected, decimal=1)


@pytest.mark.parametrize(
    "method,expected",
    [
        pytest.param(MergeMethod.MEAN, 0.4),
        pytest.param(MergeMethod.MEAN, 0.4),
    ],
)
def test_merge_to_dataset(method, expected) -> None:
    """
    Test merge function
    """
    data = xr.Dataset(
        data_vars=dict(
            model1=(["y", "x"], np.random.normal(0.5, 0.1, (100, 100))),
            model2=(["y", "x"], np.random.normal(0.3, 0.1, (100, 100))),
        ),
        coords=dict(
            y=("y", np.linspace(0, 99, num=100)),
            x=("x", np.linspace(0, 99, num=100)),
        ),
        attrs=dict(description="EF models"),
    )
    merged = merge_to_dataset(data, method=method)
    np.testing.assert_almost_equal(merged.merged.mean(), expected, decimal=1)
