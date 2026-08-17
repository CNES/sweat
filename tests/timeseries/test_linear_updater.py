# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from sweat.timeseries.linear_updater import LinearUpdater, LinearUpdaterParams
from sweat.timeseries.status_handler import (
    DISTANCE_MAX,
    DISTANCE_TYPE,
    STATE_TYPE,
    VALIDITY_FLAGS_TYPE,
    State,
)
from sweat.timeseries.types import TimeSeriesVar as TSVar


def create_dataarray(arr, dates, y, x):
    """
    Create a DataArray from an array
    """
    return xr.DataArray(
        np.tile(arr[np.newaxis, np.newaxis, :], (len(y), len(x), 1)),
        dims=["y", "x", "time"],
        coords={
            "time": dates,
            "x": x,
            "y": y,
        },
    )


def setup_data(
    window: int, size_y: int, size_x: int
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Setup data
    """
    assert window > 5
    # Create dimensions
    today = pd.Timestamp.now()
    dates = pd.date_range(end=today, periods=window, freq="D").date
    x = np.arange(size_x)
    y = np.arange(size_y)
    shape = (size_y, size_x, window)

    # Create input dataset
    data = xr.Dataset(
        data_vars={
            TSVar.ET.value: (["y", "x", "time"], np.full(shape, np.nan)),
            TSVar.RADIATION.value: (["y", "x", "time"], np.ones(shape)),
            TSVar.UPDATED.value: (
                ["y", "x", "time"],
                np.zeros(shape).astype(bool),
            ),
            TSVar.STATE.value: (
                ["y", "x", "time"],
                np.full(shape, State.NODATA.value).astype(STATE_TYPE),
            ),
            TSVar.DISTANCE.value: (
                ["y", "x", "time"],
                np.full(shape, DISTANCE_MAX).astype(DISTANCE_TYPE),
            ),
            TSVar.VALIDITY_FLAGS.value: (
                ["y", "x", "time"],
                np.zeros(shape).astype(VALIDITY_FLAGS_TYPE),
            ),
        },
        coords={
            "time": dates,
            "x": x,
            "y": y,
        },
    )
    acquisition_dates = [2, 5]
    feed_dates = dates[acquisition_dates]
    feed_et = np.array([7, 10])
    valid_et = np.array([0, 1])
    # Create an xarray dataset
    feed = xr.Dataset(
        {
            TSVar.ET.value: create_dataarray(feed_et, feed_dates, y, x),
            TSVar.VALID.value: create_dataarray(valid_et, feed_dates, y, x),
        },
        coords={
            "time": feed_dates,
            "x": x,
            "y": y,
        },
    )
    return data, feed


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config"),
    [
        {},
        {"parallel": True},
        {"parallel": True, "num_workers": 12},
        {"parallel": False},
    ],
)
def test_updaterconfig(config) -> None:
    """
    Test UpdaterConfig
    """
    assert LinearUpdaterParams.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param({"foo": True}, ValidationError),
        pytest.param({"num_workers": 0}, ValidationError),
    ],
)
def test_updaterconfig_error(config, error) -> None:
    """
    Test UpdaterConfig with error
    """
    with pytest.raises(error):
        LinearUpdaterParams.model_validate(config)


@pytest.mark.unit
def test_linear_updater() -> None:
    """
    Test Linear Updater
    """
    assert LinearUpdater()


@pytest.mark.unit
def test_requested_variables() -> None:
    """
    Test requested variables function
    """
    assert LinearUpdater.variables == [TSVar.RADIATION.value]


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "et",
        "state",
        "distance",
        "feed_et",
        "feed_valid",
        "et_expected",
        "state_expected",
        "distance_expected",
        "updated_expected",
    ),
    [
        pytest.param(
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([7, 7, 7, 7, 7, 7, 7], dtype=STATE_TYPE),
            np.array([255, 255, 255, 255, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([6.0, 4.5]),
            np.array([True, True], dtype=bool),
            np.array([np.nan, np.nan, np.nan, 6.0, np.nan, np.nan, 4.5]),
            np.array([7, 7, 7, 0, 7, 7, 0], dtype=STATE_TYPE),
            np.array([255, 255, 255, 0, 255, 255, 0], dtype=DISTANCE_TYPE),
            np.array([False, False, False, True, False, False, True]),
            id="two_acquisitions",
        ),
        pytest.param(
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([7, 7, 7, 7, 7, 7, 7], dtype=STATE_TYPE),
            np.array([255, 255, 255, 255, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([6.0, 4.5]),
            np.array([True, False], dtype=bool),
            np.array([np.nan, np.nan, np.nan, 6.0, np.nan, np.nan, np.nan]),
            np.array([7, 7, 7, 0, 7, 7, 7], dtype=STATE_TYPE),
            np.array([255, 255, 255, 0, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([False, False, False, True, False, False, False]),
            id="one_acquisition",
        ),
        pytest.param(
            np.array([6.0, 5.5, 5.0, 4.5, 4.5, 4.5, np.nan]),
            np.array([1, 1, 1, 0, 3, 3, 7], dtype=STATE_TYPE),
            np.array([4, 4, 4, 0, 1, 2, 255], dtype=DISTANCE_TYPE),
            np.array([4.5, 6.0]),
            np.array([True, True], dtype=bool),
            np.array([6.0, 5.5, 5.0, 4.5, 4.5, 4.5, 6.0]),
            np.array([1, 1, 1, 0, 3, 3, 0], dtype=STATE_TYPE),
            np.array([4, 4, 4, 0, 1, 2, 0], dtype=DISTANCE_TYPE),
            np.array([False, False, False, True, False, False, True]),
            id="new_acquisition",
        ),
        pytest.param(
            np.array([6.0, 5.5, 5.0, 4.5, 4.5, 4.5, 4.5]),
            np.array([1, 1, 1, 0, 3, 3, 3], dtype=STATE_TYPE),
            np.array([4, 4, 4, 0, 1, 2, 3], dtype=DISTANCE_TYPE),
            np.array([4.5, 6.0]),
            np.array([False, True], dtype=bool),
            np.array([6.0, 5.5, 5.0, 4.5, 4.5, 4.5, 6.0]),
            np.array([1, 1, 1, 0, 3, 3, 0], dtype=STATE_TYPE),
            np.array([4, 4, 4, 0, 1, 2, 0], dtype=DISTANCE_TYPE),
            np.array([False, False, False, False, False, False, True]),
            id="replace_by_new_acquisition",
        ),
        pytest.param(
            np.array([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]),
            np.array([3, 3, 3, 3, 3, 3, 3], dtype=STATE_TYPE),
            np.array([2, 3, 4, 5, 6, 7, 8], dtype=DISTANCE_TYPE),
            np.array([4.5, 6.0]),
            np.array([False, False], dtype=bool),
            np.array([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]),
            np.array([3, 3, 3, 3, 3, 3, 3], dtype=STATE_TYPE),
            np.array([2, 3, 4, 5, 6, 7, 8], dtype=DISTANCE_TYPE),
            np.array([False, False, False, False, False, False, False]),
            id="no_valid_acquisition",
        ),
    ],
)
def test_update_with_new_acquisitions(
    et,
    state,
    distance,
    feed_et,
    feed_valid,
    et_expected,
    state_expected,
    distance_expected,
    updated_expected,
) -> None:
    """
    Test function for updating with new acquisitions
    """
    window = 7
    today = pd.Timestamp.now()
    dates = pd.date_range(end=today, periods=window, freq="D").date

    # Create dimensions
    size_x = 5
    size_y = 10
    x = np.arange(size_x)
    y = np.arange(size_y)

    # Create input dataset
    data = xr.Dataset(
        data_vars={
            TSVar.ET.value: create_dataarray(et, dates, y, x),
            TSVar.RADIATION.value: create_dataarray(
                np.ones(window), dates, y, x
            ),
            TSVar.UPDATED.value: create_dataarray(
                np.zeros(window), dates, y, x
            ),
            TSVar.STATE.value: create_dataarray(state, dates, y, x),
            TSVar.DISTANCE.value: create_dataarray(distance, dates, y, x),
            TSVar.VALIDITY_FLAGS.value: create_dataarray(
                np.zeros(window), dates, y, x
            ),
        },
        coords={
            "time": dates,
            "x": x,
            "y": y,
        },
    )
    # Create an xarray dataset
    acquisition_dates = [3, 6]
    feed_dates = dates[acquisition_dates]
    feed = xr.Dataset(
        {
            TSVar.ET.value: create_dataarray(feed_et, feed_dates, y, x),
            TSVar.VALID.value: create_dataarray(feed_valid, feed_dates, y, x),
        },
        coords={
            "time": feed_dates,
            "x": x,
            "y": y,
        },
    )
    # Update
    linear_updater = LinearUpdater()
    new_et, new_state, new_updated, new_distance = (
        linear_updater.update_with_new_acquisitions(data, feed)
    )
    # Validate
    ref_et = create_dataarray(et_expected, dates, y, x)
    ref_state = create_dataarray(state_expected, dates, y, x)
    ref_distance = create_dataarray(distance_expected, dates, y, x)
    ref_updated = create_dataarray(updated_expected, dates, y, x)
    xr.testing.assert_allclose(new_et, ref_et)
    xr.testing.assert_equal(new_state, ref_state)
    xr.testing.assert_equal(new_updated, ref_updated)
    xr.testing.assert_equal(new_distance, ref_distance)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "et",
        "state",
        "distance",
        "et_expected",
        "state_expected",
        "distance_expected",
        "updated_expected",
    ),
    [
        pytest.param(
            np.array([np.nan, np.nan, np.nan, 6.0, np.nan, np.nan, 4.5]),
            np.array([7, 7, 7, 0, 7, 7, 0], dtype=STATE_TYPE),
            np.array([255, 255, 255, 0, 255, 255, 0], dtype=DISTANCE_TYPE),
            np.array([6.0, 6.0, 6.0, 6.0, 5.5, 5.0, 4.5]),
            np.array([3, 3, 3, 0, 1, 1, 0], dtype=STATE_TYPE),
            np.array([3, 2, 1, 0, 3, 3, 0], dtype=DISTANCE_TYPE),
            np.array([True, True, True, False, True, True, False]),
            id="two_acquisitions",
        ),
        pytest.param(
            np.array([np.nan, np.nan, np.nan, 6.0, np.nan, np.nan, np.nan]),
            np.array([7, 7, 7, 0, 7, 7, 7], dtype=STATE_TYPE),
            np.array([255, 255, 255, 0, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
            np.array([3, 3, 3, 0, 2, 2, 2], dtype=STATE_TYPE),
            np.array([3, 2, 1, 0, 1, 2, 3], dtype=DISTANCE_TYPE),
            np.array([True, True, True, False, True, True, True]),
            id="one_acquisition",
        ),
        pytest.param(
            np.array([2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 4.5]),
            np.array([2, 7, 7, 7, 7, 7, 0], dtype=STATE_TYPE),
            np.array([2, 255, 255, 255, 255, 255, 0], dtype=DISTANCE_TYPE),
            np.array([2.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]),
            np.array([2, 3, 3, 3, 3, 3, 0], dtype=STATE_TYPE),
            np.array([2, 5, 4, 3, 2, 1, 0], dtype=DISTANCE_TYPE),
            np.array([False, True, True, True, True, True, False]),
            id="one_acquisition_with_existing_data",
        ),
        pytest.param(
            np.array([5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([2, 7, 7, 7, 7, 7, 7], dtype=STATE_TYPE),
            np.array([6, 255, 255, 255, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([2, 6, 6, 6, 6, 6, 6], dtype=STATE_TYPE),
            np.array([6, 255, 255, 255, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([False, True, True, True, True, True, True]),
            id="no_acquisition",
        ),
    ],
)
def test_update_with_acquisitions(
    et,
    state,
    distance,
    et_expected,
    state_expected,
    distance_expected,
    updated_expected,
) -> None:
    """
    Test function for updating with acquisitions
    """
    # Create input
    window = 7
    today = pd.Timestamp.now()
    dates = pd.date_range(end=today, periods=window, freq="D").date
    size_x = 5
    size_y = 10
    x = np.arange(size_x)
    y = np.arange(size_y)
    input_et = create_dataarray(et, dates, y, x)
    input_radiation = create_dataarray(np.ones(window), dates, y, x)
    input_state = create_dataarray(state, dates, y, x)
    input_distance = create_dataarray(distance, dates, y, x)
    # Update
    linear_updater = LinearUpdater()
    new_et, new_updated, new_state, new_distance = (
        linear_updater.update_with_acquisitions(
            et=input_et,
            radiation=input_radiation,
            state=input_state,
            distance=input_distance,
        )
    )
    # Validate
    ref_et = create_dataarray(et_expected, dates, y, x)
    ref_state = create_dataarray(state_expected, dates, y, x)
    ref_distance = create_dataarray(distance_expected, dates, y, x)
    ref_updated = create_dataarray(updated_expected, dates, y, x)
    xr.testing.assert_allclose(new_et, ref_et)
    xr.testing.assert_equal(new_state, ref_state)
    xr.testing.assert_equal(new_updated, ref_updated)
    xr.testing.assert_equal(new_distance, ref_distance)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "et",
        "state",
        "distance",
        "et_expected",
        "state_expected",
        "distance_expected",
        "updated_expected",
    ),
    [
        pytest.param(
            np.array([6.0, 6.0, 6.0, 6.0, 5.5, 5.0, 4.5]),
            np.array([3, 3, 3, 0, 1, 1, 0], dtype=STATE_TYPE),
            np.array([3, 2, 1, 0, 3, 3, 0], dtype=DISTANCE_TYPE),
            np.array([6.0, 6.0, 6.0, 6.0, 5.5, 5.0, 4.5]),
            np.array([3, 3, 3, 0, 1, 1, 0], dtype=STATE_TYPE),
            np.array([3, 2, 1, 0, 3, 3, 0], dtype=DISTANCE_TYPE),
            np.array([False, False, False, False, False, False, False]),
            id="with_acquisitions",
        ),
        pytest.param(
            np.array([5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([2, 6, 6, 6, 6, 6, 6], dtype=STATE_TYPE),
            np.array([6, 255, 255, 255, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
            np.array([2, 2, 2, 2, 2, 2, 2], dtype=STATE_TYPE),
            np.array([6, 7, 8, 9, 10, 11, 12], dtype=DISTANCE_TYPE),
            np.array([False, True, True, True, True, True, True]),
            id="forward_extrapolation",
        ),
        pytest.param(
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 11]),
            np.array([6, 6, 6, 6, 6, 6, 3], dtype=STATE_TYPE),
            np.array([255, 255, 255, 255, 255, 255, 12], dtype=DISTANCE_TYPE),
            np.array([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]),
            np.array([3, 3, 3, 3, 3, 3, 3], dtype=STATE_TYPE),
            np.array([18, 17, 16, 15, 14, 13, 12], dtype=DISTANCE_TYPE),
            np.array([True, True, True, True, True, True, False]),
            id="backward_extrapolation",
        ),
        pytest.param(
            np.array([2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0]),
            np.array([2, 6, 6, 6, 6, 6, 3], dtype=STATE_TYPE),
            np.array([2, 255, 255, 255, 255, 255, 2], dtype=DISTANCE_TYPE),
            np.array([2.0, 2.0, 2.0, 2.0, 8.0, 8.0, 8.0]),
            np.array([2, 2, 2, 2, 3, 3, 3], dtype=STATE_TYPE),
            np.array([2, 3, 4, 5, 4, 3, 2], dtype=DISTANCE_TYPE),
            np.array([False, True, True, True, True, True, False]),
            id="forward_backward_extrapolation",
        ),
        pytest.param(
            np.array([2.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]),
            np.array([2, 3, 3, 3, 3, 3, 0], dtype=STATE_TYPE),
            np.array([2, 5, 4, 3, 2, 1, 0], dtype=DISTANCE_TYPE),
            np.array([2.0, 2.0, 4.5, 4.5, 4.5, 4.5, 4.5]),
            np.array([2, 2, 3, 3, 3, 3, 0], dtype=STATE_TYPE),
            np.array([2, 3, 4, 3, 2, 1, 0], dtype=DISTANCE_TYPE),
            np.array([False, True, False, False, False, False, False]),
            id="extrapolation_with_acquisition",
        ),
    ],
)
def test_update_with_extrapolation_only(
    et,
    state,
    distance,
    et_expected,
    state_expected,
    distance_expected,
    updated_expected,
) -> None:
    """
    Test function for updating with extrapolation only
    """
    # Create input
    window = 7
    today = pd.Timestamp.now()
    dates = pd.date_range(end=today, periods=window, freq="D").date
    size_x = 5
    size_y = 10
    x = np.arange(size_x)
    y = np.arange(size_y)
    input_et = create_dataarray(et, dates, y, x)
    input_radiation = create_dataarray(np.ones(window), dates, y, x)
    input_state = create_dataarray(state, dates, y, x)
    input_distance = create_dataarray(distance, dates, y, x)
    # Update
    linear_updater = LinearUpdater()
    new_et, new_updated, new_state, new_distance = (
        linear_updater.update_with_extrapolation_only(
            et=input_et,
            radiation=input_radiation,
            state=input_state,
            distance=input_distance,
        )
    )
    # Validate
    ref_et = create_dataarray(et_expected, dates, y, x)
    ref_state = create_dataarray(state_expected, dates, y, x)
    ref_distance = create_dataarray(distance_expected, dates, y, x)
    ref_updated = create_dataarray(updated_expected, dates, y, x)
    xr.testing.assert_allclose(new_et, ref_et)
    xr.testing.assert_equal(new_state, ref_state)
    xr.testing.assert_equal(new_updated, ref_updated)
    xr.testing.assert_equal(new_distance, ref_distance)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "et",
        "state",
        "distance",
        "et_expected",
        "state_expected",
        "distance_expected",
        "updated_expected",
    ),
    [
        pytest.param(
            np.array([np.nan, np.nan, np.nan, 6.0, np.nan, np.nan, 4.5]),
            np.array([7, 7, 7, 0, 7, 7, 0], dtype=STATE_TYPE),
            np.array([255, 255, 255, 0, 255, 255, 0], dtype=DISTANCE_TYPE),
            np.array([6.0, 6.0, 6.0, 6.0, 5.5, 5.0, 4.5]),
            np.array([3, 3, 3, 0, 1, 1, 0], dtype=STATE_TYPE),
            np.array([3, 2, 1, 0, 3, 3, 0], dtype=DISTANCE_TYPE),
            np.array([True, True, True, False, True, True, False]),
            id="two_acquisitions",
        ),
        pytest.param(
            np.array([np.nan, np.nan, np.nan, 6.0, np.nan, np.nan, np.nan]),
            np.array([7, 7, 7, 0, 7, 7, 7], dtype=STATE_TYPE),
            np.array([255, 255, 255, 0, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
            np.array([3, 3, 3, 0, 2, 2, 2], dtype=STATE_TYPE),
            np.array([3, 2, 1, 0, 1, 2, 3], dtype=DISTANCE_TYPE),
            np.array([True, True, True, False, True, True, True]),
            id="one_acquisition",
        ),
        pytest.param(
            np.array([2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 4.5]),
            np.array([2, 7, 7, 7, 7, 7, 0], dtype=STATE_TYPE),
            np.array([2, 255, 255, 255, 255, 255, 0], dtype=DISTANCE_TYPE),
            np.array([2.0, 2.0, 4.5, 4.5, 4.5, 4.5, 4.5]),
            np.array([2, 2, 3, 3, 3, 3, 0], dtype=STATE_TYPE),
            np.array([2, 3, 4, 3, 2, 1, 0], dtype=DISTANCE_TYPE),
            np.array([False, True, True, True, True, True, False]),
            id="extrapolation_with_acquisition",
        ),
        pytest.param(
            np.array([5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([2, 7, 7, 7, 7, 7, 7], dtype=STATE_TYPE),
            np.array([6, 255, 255, 255, 255, 255, 255], dtype=DISTANCE_TYPE),
            np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
            np.array([2, 2, 2, 2, 2, 2, 2], dtype=STATE_TYPE),
            np.array([6, 7, 8, 9, 10, 11, 12], dtype=DISTANCE_TYPE),
            np.array([False, True, True, True, True, True, True]),
            id="forward_extrapolation",
        ),
        pytest.param(
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 11]),
            np.array([7, 7, 7, 7, 7, 7, 3], dtype=STATE_TYPE),
            np.array([255, 255, 255, 255, 255, 255, 12], dtype=DISTANCE_TYPE),
            np.array([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]),
            np.array([3, 3, 3, 3, 3, 3, 3], dtype=STATE_TYPE),
            np.array([18, 17, 16, 15, 14, 13, 12], dtype=DISTANCE_TYPE),
            np.array([True, True, True, True, True, True, False]),
            id="backward_extrapolation",
        ),
        pytest.param(
            np.array([2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0]),
            np.array([2, 7, 7, 7, 7, 7, 3], dtype=STATE_TYPE),
            np.array([2, 255, 255, 255, 255, 255, 2], dtype=DISTANCE_TYPE),
            np.array([2.0, 2.0, 2.0, 2.0, 8.0, 8.0, 8.0]),
            np.array([2, 2, 2, 2, 3, 3, 3], dtype=STATE_TYPE),
            np.array([2, 3, 4, 5, 4, 3, 2], dtype=DISTANCE_TYPE),
            np.array([False, True, True, True, True, True, False]),
            id="forward_backward_extrapolation",
        ),
    ],
)
def test_update_time_series(
    et,
    state,
    distance,
    et_expected,
    state_expected,
    distance_expected,
    updated_expected,
) -> None:
    """
    Test function for updating with acquisitions
    """
    # Create input
    window = 7
    today = pd.Timestamp.now()
    dates = pd.date_range(end=today, periods=window, freq="D").date
    size_x = 5
    size_y = 10
    x = np.arange(size_x)
    y = np.arange(size_y)
    # Create input dataset
    data = xr.Dataset(
        data_vars={
            TSVar.ET.value: create_dataarray(et, dates, y, x),
            TSVar.RADIATION.value: create_dataarray(
                np.ones(window), dates, y, x
            ),
            TSVar.UPDATED.value: create_dataarray(
                np.zeros(window), dates, y, x
            ),
            TSVar.STATE.value: create_dataarray(state, dates, y, x),
            TSVar.DISTANCE.value: create_dataarray(distance, dates, y, x),
            TSVar.VALIDITY_FLAGS.value: create_dataarray(
                np.zeros(window), dates, y, x
            ),
        },
        coords={
            "time": dates,
            "x": x,
            "y": y,
        },
    )
    # Update
    linear_updater = LinearUpdater()
    updated_data = linear_updater.update_time_series(data)
    # Validate
    ref_et = create_dataarray(et_expected, dates, y, x)
    ref_state = create_dataarray(state_expected, dates, y, x)
    ref_distance = create_dataarray(distance_expected, dates, y, x)
    ref_updated = create_dataarray(updated_expected, dates, y, x)
    xr.testing.assert_allclose(updated_data[TSVar.ET.value], ref_et)
    xr.testing.assert_equal(updated_data[TSVar.STATE.value], ref_state)
    xr.testing.assert_equal(updated_data[TSVar.UPDATED.value], ref_updated)
    xr.testing.assert_equal(updated_data[TSVar.DISTANCE.value], ref_distance)


@pytest.mark.functional
def test_update():
    """
    Test update function
    """
    # Setup data
    data, feed = setup_data(7, 100, 100)
    # Update
    linear_updater = LinearUpdater()
    updated_data = linear_updater.update(data, feed)
    assert updated_data
    assert updated_data.sizes == {"y": 100, "x": 100, "time": 7}
    assert updated_data.attrs == data.attrs


@pytest.mark.functional
def test_update_no_feed():
    """
    Test update function (without new acquisition)
    """
    # Setup data
    data, _ = setup_data(7, 100, 100)
    # Update
    linear_updater = LinearUpdater()
    updated_data = linear_updater.update(data)
    assert updated_data
    assert updated_data.sizes == {"y": 100, "x": 100, "time": 7}
    assert updated_data.attrs == data.attrs


@pytest.mark.functional
def test_update_parallel():
    """
    Test update function
    """
    # Setup data
    data, feed = setup_data(7, 500, 1000)
    # Update
    linear_updater = LinearUpdater(parallel=True)
    updated_data = linear_updater.update(data, feed)
    assert updated_data
    assert updated_data.sizes == {"y": 500, "x": 1000, "time": 7}
    assert updated_data.attrs == data.attrs
