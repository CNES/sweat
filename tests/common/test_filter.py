# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pytest
import xarray as xr

from evaspa.common.filter import (
    FilterConfig,
    FilterParams,
    determine_valid_pixels,
    mask,
)


@pytest.mark.parametrize(
    ("data", "values", "invert", "expected"),
    [
        pytest.param(
            [10, 0, 10, 5, 20], 0, False, [False, True, False, False, False]
        ),
        pytest.param(
            [10, 0, 10, 5, 20], [0, 10], False, [True, True, True, False, False]
        ),
        pytest.param(
            [10, 0, 10, 5, 20], [0, 10], True, [False, False, False, True, True]
        ),
    ],
)
def test_mask(data, values, invert, expected):
    """
    Test mask function
    """
    res = mask(data, values=values, invert=invert)
    np.testing.assert_equal(res, expected)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param(
            {
                "cloud": "cloud",
                "water": None,
                "qa": None,
                "zones": None,
                "cover": None,
            },
            [0, 1, 1, 1, 0],
        ),
        pytest.param(
            {
                "cloud": "cloud",
                "water": "water",
                "qa": None,
                "zones": None,
                "cover": None,
            },
            [0, 1, 0, 1, 0],
        ),
        pytest.param(
            {
                "cloud": "cloud",
                "water": "water",
                "qa": "qa",
                "zones": "zones",
                "cover": "cover",
            },
            [0, 0, 0, 1, 0],
        ),
        pytest.param(
            {
                "cloud": None,
                "water": None,
                "qa": None,
                "zones": "zones",
                "cover": "cover",
            },
            [1, 0, 1, 1, 1],
        ),
    ],
)
def test_determine_valid_pixels(params, expected) -> None:
    """
    Test filter method
    """
    # Generate data
    data = xr.Dataset(
        data_vars={
            "lst": (["x"], np.random.random(size=(5))),
            "cloud": (["x"], [1, 0, 0, 0, 1]),
            "water": (["x"], [0, 0, 1, 0, 1]),
            "qa": (["x"], [1, 0, 0, 0, 0]),
            "zones": (["x"], [0, 1, 0, 0, 0]),
            "cover": (["x"], [10, 20, 10, 10, 10]),
        },
        coords={
            "x": ("x", np.linspace(0, 5, num=5)),
        },
        attrs={"description": "Test data."},
    )
    valid = determine_valid_pixels(data, **params)
    np.testing.assert_equal(valid.data, expected)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {"cloud": 1},
            [1, 0, 0, 0, 1],
        ),
        pytest.param(
            {"cloud": 0, "cover": [20, 60]},
            [0, 1, 1, 1, 0],
        ),
    ],
)
def test_determine_valid_pixels_config(config, expected) -> None:
    """
    Test filter method (config option)
    """
    # Generate data
    data = xr.Dataset(
        data_vars={
            "lst": (["x"], np.random.random(size=(5))),
            "cloud": (["x"], [1, 0, 0, 0, 1]),
            "water": (["x"], [0, 0, 1, 0, 1]),
            "qa": (["x"], [1, 0, 0, 0, 0]),
            "zones": (["x"], [0, 1, 0, 0, 0]),
            "cover": (["x"], [60, 20, 20, 20, 60]),
        },
        coords={
            "x": ("x", np.linspace(0, 5, num=5)),
        },
        attrs={"description": "Test data."},
    )
    valid = determine_valid_pixels(
        data, cloud="cloud", cover="cover", config=config
    )
    np.testing.assert_equal(valid.data, expected)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"cloud": 1},
        {"cloud": 1, "zones": 1},
    ],
)
def test_filterparams(config) -> None:
    """
    Test FilterParams
    """
    assert FilterParams.model_validate(config)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"cloud": "cloud_mask"},
        {"cloud": "cloud_mask", "config": {"cloud": 1}},
    ],
)
def test_filterconfig(config) -> None:
    """
    Test FilterConfig
    """
    assert FilterConfig.model_validate(config)
