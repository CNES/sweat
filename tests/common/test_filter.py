# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from pydantic import ValidationError

from evaspa.common import filter
from evaspa.common.constant import FLAGS_TYPE


def convert_dataarray(arr: npt.NDArray) -> xr.DataArray:
    return xr.DataArray(
        arr,
        coords={"y": np.arange(arr.shape[0]), "x": np.arange(arr.shape[1])},
        dims=["y", "x"],
    )


def convert_dataset(variables: dict) -> xr.Dataset:
    # Get the first item
    shape = next(iter(variables.items()))[1].shape
    return xr.Dataset(
        data_vars={k: (["y", "x"], v) for k, v in variables.items()},
        coords={
            "y": ("y", np.arange(shape[0])),
            "x": ("x", np.arange(shape[1])),
        },
        attrs={"description": "Test data"},
    )


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
    assert filter.FilteringConfig.model_validate(config)


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
        filter.FilteringConfig(config)


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
    data = convert_dataarray(entry)
    res = filter.eval_condition(data, cond)
    ref = convert_dataarray(expected)
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
    data = convert_dataarray(entry)
    res = filter.apply_condition(data, cond)
    ref = convert_dataarray(expected)
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
def test_detect_valid_pixels(config, expected) -> None:
    """
    Test function for detecting valid pixels
    """
    data = convert_dataset(
        {
            "lst": np.array([[295, 300], [290, 295]]),
            "cloud": np.array([[0, 1], [0, 0]]),
            "lulc": np.array([[10, 10], [30, 20]]),
        }
    )
    res = filter.detect_valid_pixels(data, config)
    ref = convert_dataarray(expected)
    xr.testing.assert_equal(res, ref)


@pytest.mark.unit
def test_detect_valid_pixels_with_warnings(caplog) -> None:
    """
    Test function for detecting valid pixels with warnings
    """
    config = {
        "cloud": {"op": "!=", "value": 1},
    }
    data = convert_dataset(
        {
            "lst": np.array([[295, 300], [290, 295]]),
        }
    )
    expected = np.array([[True, True], [True, True]])
    res = filter.detect_valid_pixels(data, config)
    ref = convert_dataarray(expected)
    xr.testing.assert_equal(res, ref)
    msg = "Variable cloud not found"
    assert msg in caplog.text


@pytest.mark.unit
def test_detect_valid_pixels_exc() -> None:
    """
    Test function for detecting valid pixels (with exception)
    """
    # Create empty dataset
    empty_ds = xr.Dataset()
    config = {
        "cloud": {"op": "!=", "value": 1},
    }
    with pytest.raises(
        ValueError, match="Dataset is empty, filtering not possible"
    ):
        filter.detect_valid_pixels(empty_ds, config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        pytest.param(
            "all",
            np.array([[False, True], [True, False]]),
        ),
        pytest.param(
            [],
            np.array([[False, False], [False, False]]),
        ),
        pytest.param(
            ["lst"],
            np.array([[False, True], [False, False]]),
        ),
        pytest.param(
            ["lst", "albedo"],
            np.array([[False, True], [True, False]]),
        ),
    ],
)
def test_detect_nan_pixels(variables, expected):
    """
    Test function for detect nan pixels
    """
    data = convert_dataset(
        {
            "lst": np.array([[293, np.nan], [293, 293]]),
            "albedo": np.array([[0.5, 0.5], [np.nan, 0.5]]),
        }
    )
    res = filter.detect_nan_pixels(data, variables=variables)
    ref = convert_dataarray(expected)
    xr.testing.assert_equal(res, ref)


@pytest.mark.unit
def test_detect_nan_pixels_with_warnings(caplog) -> None:
    """
    Test function for detecting valid pixels with warnings
    """
    variables = ["lst", "fcover"]
    data = convert_dataset(
        {
            "lst": np.array([[295, 300], [290, 295]]),
        }
    )
    expected = np.array([[False, False], [False, False]])
    res = filter.detect_nan_pixels(data, variables)
    ref = convert_dataarray(expected)
    xr.testing.assert_equal(res, ref)
    msg = "Variable fcover not found"
    assert msg in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data", "variables", "expected"),
    [
        pytest.param(
            xr.Dataset(
                data_vars={
                    "lst": (["y", "x"], np.array([[295, 300], [290, 295]])),
                },
                coords={
                    "y": ("y", np.array([0, 1])),
                    "x": ("x", np.array([0, 1])),
                },
                attrs={"description": "Test data"},
            ),
            "foo",
            "Only 'all' can be provided",
        ),
        pytest.param(
            xr.Dataset(),
            "all",
            "Dataset is empty, not possible to detect nan pixels",
        ),
    ],
)
def test_detect_nan_pixels_exc(data, variables, expected) -> None:
    """
    Test function for detecting valid pixels (with exception)
    """
    # Create empty dataset
    with pytest.raises(ValueError, match=expected):
        filter.detect_nan_pixels(data, variables)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data", "nan_config", "valid_config", "valid_expected", "flags_expected"),
    [
        pytest.param(
            {
                "lst": np.array(
                    [[295, 300, 300], [298, 300, 297], [290, 295, 295]]
                ),
                "albedo": np.array(
                    [
                        [0.3, 0.32, 0.2],
                        [0.4, 0.3, 0.3],
                        [0.25, 0.4, 0.3],
                    ]
                ),
            },
            None,
            None,
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=FLAGS_TYPE),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=FLAGS_TYPE),
        ),
        pytest.param(
            {
                "lst": np.array(
                    [[295, 300, np.nan], [298, 300, np.nan], [290, 295, 295]]
                ),
                "albedo": np.array(
                    [
                        [0.3, 0.32, 0.2],
                        [0.4, 0.3, 0.3],
                        [0.25, 0.4, 0.3],
                    ]
                ),
            },
            ["lst"],
            None,
            np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=FLAGS_TYPE),
            np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=FLAGS_TYPE),
        ),
        pytest.param(
            {
                "lst": np.array(
                    [[295, 300, np.nan], [298, 300, np.nan], [290, 295, 295]]
                ),
                "albedo": np.array(
                    [
                        [0.3, 0.32, 0.2],
                        [0.4, 0.3, 0.3],
                        [0.25, 0.4, 0.3],
                    ]
                ),
            },
            None,
            {
                "albedo": {
                    "op": "<",
                    "value": 0.4,
                }
            },
            np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1]], dtype=FLAGS_TYPE),
            np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=FLAGS_TYPE),
        ),
        pytest.param(
            {
                "lst": np.array(
                    [[295, 300, np.nan], [np.nan, 300, np.nan], [290, 295, 295]]
                ),
                "albedo": np.array(
                    [
                        [0.3, 0.32, 0.2],
                        [0.4, 0.3, 0.3],
                        [0.25, 0.4, 0.3],
                    ]
                ),
            },
            ["lst"],
            {
                "albedo": {
                    "op": "<",
                    "value": 0.4,
                }
            },
            np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=FLAGS_TYPE),
            np.array([[0, 0, 1], [3, 0, 1], [0, 2, 0]], dtype=FLAGS_TYPE),
        ),
        pytest.param(
            {
                "lst": np.array(
                    [[295, 300, np.nan], [298, 300, np.nan], [290, 295, 295]]
                ),
                "albedo": np.array(
                    [
                        [0.4, 0.32, 0.2],
                        [0.4, 0.3, 0.3],
                        [0.25, 0.4, 0.3],
                    ]
                ),
                "valid": np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]]),
                "flags": np.array([[4, 0, 0], [0, 0, 8], [0, 0, 0]]),
            },
            ["lst"],
            {
                "albedo": {
                    "op": "<",
                    "value": 0.4,
                }
            },
            np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=FLAGS_TYPE),
            np.array([[6, 0, 1], [2, 0, 9], [0, 2, 0]], dtype=FLAGS_TYPE),
        ),
    ],
)
def test_find_valid_pixels(
    data, nan_config, valid_config, valid_expected, flags_expected
):
    """
    Test function for detect nan pixels
    """
    data = convert_dataset(data)
    valid, flags = filter.find_valid_pixels(data, nan_config, valid_config)
    valid_ref = convert_dataarray(valid_expected)
    flags_ref = convert_dataarray(flags_expected)
    xr.testing.assert_equal(valid, valid_ref)
    xr.testing.assert_equal(flags, flags_ref)
