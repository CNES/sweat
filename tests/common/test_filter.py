# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from pydantic import ValidationError

from evaspa.common.filter import (
    FilterConfig,
    FilteringConfig,
    FilterParams,
    apply_condition,
    determine_valid_pixels,
    eval_condition,
    filter_valid_pixels,
    mask,
)


def convert(arr: npt.NDArray) -> xr.DataArray:
    return xr.DataArray(
        arr,
        coords={"y": np.arange(arr.shape[0]), "x": np.arange(arr.shape[1])},
        dims=["y", "x"],
    )


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            "cloud": {"op": "==", "value": 1},
        },
        {
            "lst": {
                "and": [{"op": ">", "value": 291}, {"op": "<", "value": 298}]
            }
        },
        {
            "lulc": {
                "or": [{"op": "==", "value": 10}, {"op": "==", "value": 20}],
            }
        },
    ],
)
def test_filteringconfig(config) -> None:
    """
    Test FilteringConfig
    """
    assert FilteringConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        "foo",
        {
            "cloud": {"op": "foo", "value": 1},
        },
        {
            "cloud": {"op": "=="},
        },
    ],
)
def test_filteringconfig_error(config) -> None:
    """
    Test FilteringConfig with error
    """
    with pytest.raises(ValidationError):
        FilteringConfig(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "cond", "expected"),
    [
        pytest.param(
            np.array([[1, 0], [1, 1]]),
            {"op": "==", "value": 1},
            np.array([[True, False], [True, True]]),
        ),
        pytest.param(
            np.array([[1, 0], [1, 1]]),
            {"op": "!=", "value": 1},
            np.array([[False, True], [False, False]]),
        ),
        pytest.param(
            np.array([[1, 0], [1, 1]]),
            {"op": ">", "value": 0.5},
            np.array([[True, False], [True, True]]),
        ),
        pytest.param(
            np.array([[1, 0.5], [1, 1]]),
            {"op": ">=", "value": 0.5},
            np.array([[True, True], [True, True]]),
        ),
        pytest.param(
            np.array([[1, 0], [1, 1]]),
            {"op": "<", "value": 0.5},
            np.array([[False, True], [False, False]]),
        ),
        pytest.param(
            np.array([[1, 0.5], [1, 1]]),
            {"op": "<=", "value": 0.5},
            np.array([[False, True], [False, False]]),
        ),
    ],
)
def test_eval_condition(entry, cond, expected) -> None:
    """
    Test eval_condition function
    """
    data = convert(entry)
    res = eval_condition(data, cond)
    ref = convert(expected)
    xr.testing.assert_equal(res, ref)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "cond", "expected"),
    [
        pytest.param(
            np.array([[10, 5], [8, 7]]),
            {"op": ">", "value": 6},
            np.array([[True, False], [True, True]]),
        ),
        pytest.param(
            np.array([[10, 5], [8, 7]]),
            {"and": [{"op": ">", "value": 4}, {"op": "<", "value": 8}]},
            np.array([[False, True], [False, True]]),
        ),
        pytest.param(
            np.array([[10, 5], [8, 7]]),
            {"or": [{"op": "==", "value": 10}, {"op": "==", "value": 8}]},
            np.array([[True, False], [True, False]]),
        ),
    ],
)
def test_apply_condition(entry, cond, expected) -> None:
    """
    Test eval_condition function
    """
    data = convert(entry)
    res = apply_condition(data, cond)
    ref = convert(expected)
    xr.testing.assert_equal(res, ref)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {
                "cloud": {"op": "!=", "value": 1},
            },
            np.array([[True, False], [True, True]]),
        ),
        pytest.param(
            {},
            np.array([[True, True], [True, True]]),
        ),
        pytest.param(
            {
                "lst": {
                    "and": [
                        {"op": ">", "value": 291},
                        {"op": "<", "value": 298},
                    ]
                }
            },
            np.array([[True, False], [False, True]]),
        ),
        pytest.param(
            {
                "cloud": {"op": "!=", "value": 1},
                "lulc": {
                    "or": [
                        {"op": "==", "value": 10},
                        {"op": "==", "value": 20},
                    ],
                },
            },
            np.array([[True, False], [False, True]]),
        ),
    ],
)
def test_filter_valid_pixels(config, expected) -> None:
    """
    Test function for filtering valid pixels
    """
    data = xr.Dataset(
        data_vars={
            "lst": (["y", "x"], np.array([[295, 300], [290, 295]])),
            "cloud": (["y", "x"], np.array([[0, 1], [0, 0]])),
            "lulc": (["y", "x"], np.array([[10, 10], [30, 20]])),
        },
        coords={
            "y": ("y", np.array([0, 1])),
            "x": ("x", np.array([0, 1])),
        },
        attrs={"description": "Test data"},
    )
    res = filter_valid_pixels(data, config)
    ref = convert(expected)
    xr.testing.assert_equal(res, ref)


@pytest.mark.unit
def test_filter_valid_pixels_with_warnings(caplog) -> None:
    """
    Test function for filtering valid pixels with warnings
    """
    config = {
        "cloud": {"op": "!=", "value": 1},
    }
    data = xr.Dataset(
        data_vars={
            "lst": (["y", "x"], np.array([[295, 300], [290, 295]])),
        },
        coords={
            "y": ("y", np.array([0, 1])),
            "x": ("x", np.array([0, 1])),
        },
        attrs={"description": "Test data"},
    )
    expected = np.array([[True, True], [True, True]])
    res = filter_valid_pixels(data, config)
    ref = convert(expected)
    xr.testing.assert_equal(res, ref)
    msg = "Variable cloud not found"
    assert msg in caplog.text


@pytest.mark.unit
def test_filter_valid_pixels_exc() -> None:
    """
    Test function for filtering valid pixels (with exception)
    """
    # Create empty dataset
    empty_ds = xr.Dataset()
    config = {
        "cloud": {"op": "!=", "value": 1},
    }
    with pytest.raises(
        ValueError, match="Dataset is empty, filtering not possible"
    ):
        filter_valid_pixels(empty_ds, config)
