# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import datetime as dt

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from evaspa.timeseries.constant import TimeSeriesVar as TSVar
from evaspa.timeseries.linear_updater import LinearUpdater, LinearUpdaterParams
from evaspa.timeseries.status_handler import (
    STATUS_TYPE,
    ProcessingMode,
    RadiationMode,
)


def setup_data(
    size: int,
    a_ratio: float,
    b_ratio: float,
    a_radiation: float = 24,
    b_radiation: float = 200,
    status: npt.NDArray | None = None,
) -> xr.Dataset:
    """
    Create an xarray dataset
    """
    # Windows size
    window_size = size
    # Date time series
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    # Generate an array of consecutive dates
    dates = np.array(pd.date_range(end=today, periods=window_size).to_list())
    # Create radiation time series
    radiation_time_series = a_radiation * np.arange(window_size) + b_radiation
    # Create ET time series
    ratio_time_series = a_ratio * np.arange(window_size) + b_ratio
    et_time_series = ratio_time_series * radiation_time_series
    # Create status
    if status is None:
        status = np.zeros_like(et_time_series, dtype=STATUS_TYPE)

    return xr.Dataset(
        {
            TSVar.ET.value: (
                [TSVar.TIME.value],
                et_time_series,
            ),
            TSVar.RADIATION.value: (
                [TSVar.TIME.value],
                radiation_time_series,
            ),
            TSVar.FLAGS.value: (
                [TSVar.TIME.value],
                status,
            ),
        },
        coords={
            TSVar.TIME.value: dates,
        },
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"strict_mode": True, "radiation_mode": "EXTERNAL"},
        {"strict_mode": True, "radiation_mode": 1},
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
        pytest.param({"foo": True, "radiation_mode": 1}, ValidationError),
        pytest.param({"foo": True, "radiation_mode": "foo"}, ValidationError),
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
    assert LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.THEORITICAL
    )


@pytest.mark.unit
def test_requested_variables() -> None:
    """
    Test requested variables function
    """
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.THEORITICAL
    )
    assert linear_updater.get_requested_variables() == [TSVar.RADIATION.value]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "strict_mode", "expected"),
    [
        pytest.param(2, True, -1),
        pytest.param(2, False, 1),
        pytest.param(6, True, 3),
        pytest.param(6, False, 3),
        pytest.param(0, True, -1),
        pytest.param(0, False, -1),
    ],
)
def test_find_previous(index, strict_mode, expected) -> None:
    """
    Test find previous function
    """
    status_arr = np.array(
        [
            0b000000000001000,
            0b000000011000010,
            0b000000011000010,
            0b000000000000000,
            0b000000001000011,
            0b000000010000011,
            0b000000000000001,
        ],
        dtype=STATUS_TYPE,
    )
    linear_updater = LinearUpdater(
        strict_mode=strict_mode, radiation_mode=RadiationMode.THEORITICAL
    )
    prev_index = linear_updater.find_previous(status_arr, index)
    assert prev_index == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "strict_mode", "expected"),
    [
        pytest.param(0, True, 3),
        pytest.param(0, False, 3),
        pytest.param(2, True, 3),
        pytest.param(2, False, 3),
        pytest.param(4, True, -1),
        pytest.param(4, False, 6),
        pytest.param(6, True, -1),
        pytest.param(6, False, -1),
    ],
)
def test_find_next(index, strict_mode, expected) -> None:
    """
    Test find next function
    """
    status_arr = np.array(
        [
            0b000000000001000,
            0b000000011000010,
            0b000000011000010,
            0b000000000000000,
            0b000000001000011,
            0b000000010000011,
            0b000000000000010,
        ],
        dtype=STATUS_TYPE,
    )
    linear_updater = LinearUpdater(
        strict_mode=strict_mode, radiation_mode=RadiationMode.THEORITICAL
    )
    prev_index = linear_updater.find_next(status_arr, index)
    assert prev_index == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "prev_index", "next_index", "expected_value", "expected_mode"),
    [
        pytest.param(2, 0, 3, 1.736, ProcessingMode.NOMINAL),
        pytest.param(5, 4, 6, 3.2, ProcessingMode.DEGRADATED),
    ],
)
def test_interpolate(
    index, prev_index, next_index, expected_value, expected_mode
) -> None:
    """
    Test interpolate function
    """
    ts = setup_data(
        size=7,
        a_ratio=0.001,
        b_ratio=0.005,
        status=np.array([0, 0, 0, 0, 0, 32, 0], dtype=STATUS_TYPE),
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.EXTERNAL
    )
    value, mode = linear_updater.interpolate(index, prev_index, next_index, ts)
    np.testing.assert_almost_equal(value, expected_value)
    np.testing.assert_equal(mode, expected_mode)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "prev_index", "expected_value", "expected_mode"),
    [
        pytest.param(2, 1, 1.488, ProcessingMode.NOMINAL),
        pytest.param(5, 4, 2.88, ProcessingMode.DEGRADATED),
    ],
)
def test_extrapolate(index, prev_index, expected_value, expected_mode) -> None:
    """
    Test extrapolate function
    """
    ts = setup_data(
        size=7,
        a_ratio=0.001,
        b_ratio=0.005,
        status=np.array([32, 32, 32, 32, 0, 0, 32], dtype=STATUS_TYPE),
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.THEORITICAL
    )
    value, mode = linear_updater.extrapolate(index, prev_index, ts)
    np.testing.assert_almost_equal(value, expected_value)
    np.testing.assert_equal(mode, expected_mode)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "next_index", "expected_value", "expected_mode"),
    [
        pytest.param(1, 2, 1.568, ProcessingMode.NOMINAL),
        pytest.param(5, 6, 3.52, ProcessingMode.DEGRADATED),
    ],
)
def test_backward_extrapolate(
    index, next_index, expected_value, expected_mode
) -> None:
    """
    Test interpolate function
    """
    ts = setup_data(
        size=7,
        a_ratio=0.001,
        b_ratio=0.005,
        status=np.array([0, 0, 0, 0, 0, 32, 0], dtype=STATUS_TYPE),
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.EXTERNAL
    )
    value, mode = linear_updater.backward_extrapolate(index, next_index, ts)
    np.testing.assert_almost_equal(value, expected_value)
    np.testing.assert_equal(mode, expected_mode)


@pytest.mark.unit
def test_update_new_acquisition() -> None:
    """
    Test function for upadting new acquisitions
    """
    # Window size
    window_size = 7
    # Date time series
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    # Generate an array of consecutive dates
    dates = np.array(pd.date_range(end=today, periods=window_size).to_list())
    ts = setup_data(
        size=7,
        a_ratio=0,
        b_ratio=0.006,
        a_radiation=0.0,
        b_radiation=200,
        status=np.array([0, 2, 2, 0, 3, 1, 1], dtype=STATUS_TYPE),
    )
    # Create an xarray dataset
    acquisition_dates = [0, 3, 6]
    feed = xr.Dataset(
        {
            TSVar.ET.value: (
                [TSVar.TIME.value],
                np.array([2.0, 2.1, 2.2]),
            ),
            TSVar.VALID.value: (
                [TSVar.TIME.value],
                np.array([0, 1, 1]),
            ),
        },
        coords={
            TSVar.TIME.value: dates[acquisition_dates],
        },
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.THEORITICAL
    )
    linear_updater.update_new_acquisition(ts, feed)
    np.testing.assert_almost_equal(
        ts[TSVar.ET.value].data, np.array([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 2.2])
    )
    np.testing.assert_almost_equal(
        ts[TSVar.FLAGS.value].data, np.array([0, 2, 2, 0, 3, 1, 16])
    )


@pytest.mark.unit
def test_update_with_new_acquisitions() -> None:
    """
    Test function for upadting new acquisitions
    """
    # Window size
    window_size = 7
    # Date time series
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    # Generate an array of consecutive dates
    dates = np.array(pd.date_range(end=today, periods=window_size).to_list())
    ts = setup_data(
        size=7,
        a_ratio=0,
        b_ratio=0.006,
        a_radiation=0.0,
        b_radiation=200,
        status=np.array([0, 2, 2, 0, 3, 1, 1], dtype=STATUS_TYPE),
    )
    # Create an xarray dataset
    acquisition_dates = [0, 3, 6]
    feed = xr.Dataset(
        {
            TSVar.ET.value: (
                [TSVar.TIME.value],
                np.array([2.0, 2.1, 2.2]),
            ),
            TSVar.VALID.value: (
                [TSVar.TIME.value],
                np.array([0, 1, 1]),
            ),
        },
        coords={
            TSVar.TIME.value: dates[acquisition_dates],
        },
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.THEORITICAL
    )
    linear_updater.update_with_new_acquisitions(ts, feed)
    # np.testing.assert_almost_equal(
    #    ts[TSVar.ET.value].data, np.array([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 2.2])
    # )
    # np.testing.assert_almost_equal(
    #    ts[TSVar.FLAGS.value].data, np.array([0, 2, 2, 0, 3, 1, 16])
    # )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "status_arr", "expected_value", "expected_status"),
    [
        pytest.param(
            4,
            np.array([1, 0, 1, 1, 1, 0, 1], dtype=STATUS_TYPE),
            2.0,
            STATUS_TYPE(530),
        ),
        pytest.param(
            6,
            np.array([1, 0, 1, 1, 1, 0, 1], dtype=STATUS_TYPE),
            2.2,
            STATUS_TYPE(147),
        ),
        pytest.param(
            0,
            np.array([1, 0, 1, 1, 1, 0, 1], dtype=STATUS_TYPE),
            1.4,
            STATUS_TYPE(148),
        ),
        pytest.param(
            6,
            np.array([5, 5, 5, 5, 5, 5, 1], dtype=STATUS_TYPE),
            np.nan,
            STATUS_TYPE(29),
        ),
    ],
)
def test_update_nodata(index, status_arr, expected_value, expected_status):
    """
    Test update nodata function
    """
    # Setup data
    ts = setup_data(
        size=7,
        a_ratio=0.001,
        b_ratio=0.006,
        a_radiation=0.0,
        b_radiation=200,
        status=status_arr,
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.EXTERNAL
    )
    status = ts[TSVar.FLAGS.value].isel({TSVar.TIME.value: index}).item()
    new_value, new_status = linear_updater.update_nodata(index, status, ts)
    np.testing.assert_almost_equal(new_value, expected_value)
    np.testing.assert_equal(new_status, expected_status)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "status_arr", "expected_value", "expected_status"),
    [
        pytest.param(
            2,
            np.array([0, 642, 642, 0, 642, 0, 1], dtype=STATUS_TYPE),
            1.6,
            STATUS_TYPE(402),
        ),
        pytest.param(
            2,
            np.array([0, 386, 394, 0, 1, 1, 1], dtype=STATUS_TYPE),
            1.6,
            STATUS_TYPE(402),
        ),
        pytest.param(
            2,
            np.array([0, 386, 386, 0, 1, 1, 1], dtype=STATUS_TYPE),
            1.6,
            STATUS_TYPE(386),
        ),
    ],
)
def test_update_interpolated_data(
    index, status_arr, expected_value, expected_status
):
    """
    Test update interpolated data function
    """
    # Setup data
    ts = setup_data(
        size=7,
        a_ratio=0.001,
        b_ratio=0.006,
        a_radiation=0.0,
        b_radiation=200,
        status=status_arr,
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.EXTERNAL
    )
    status = ts[TSVar.FLAGS.value].isel({TSVar.TIME.value: index}).item()
    new_value, new_status = linear_updater.update_interpolated_data(
        index, status, ts
    )
    np.testing.assert_almost_equal(new_value, expected_value)
    np.testing.assert_equal(new_status, expected_status)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "status_arr", "expected_value", "expected_status"),
    [
        pytest.param(
            2,
            np.array([645, 517, 388, 0, 1, 0, 1], dtype=STATUS_TYPE),
            1.8,
            STATUS_TYPE(148),
        ),
        pytest.param(
            2,
            np.array([389, 261, 140, 0, 1, 1, 1], dtype=STATUS_TYPE),
            1.8,
            STATUS_TYPE(148),
        ),
        pytest.param(
            4,
            np.array([1, 1, 1, 0, 139, 1, 1], dtype=STATUS_TYPE),
            1.8,
            STATUS_TYPE(147),
        ),
        pytest.param(
            4,
            np.array([1, 1, 1, 0, 139, 1, 0], dtype=STATUS_TYPE),
            2.0,
            STATUS_TYPE(402),
        ),
        pytest.param(
            4,
            np.array([1, 1, 1, 0, 131, 1, 1], dtype=STATUS_TYPE),
            2.0,
            STATUS_TYPE(131),
        ),
    ],
)
def test_update_extrapolated_data(
    index, status_arr, expected_value, expected_status
):
    """
    Test update extrapolated data function
    """
    # Setup data
    ts = setup_data(
        size=7,
        a_ratio=0.001,
        b_ratio=0.006,
        a_radiation=0.0,
        b_radiation=200,
        status=status_arr,
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.EXTERNAL
    )
    status = ts[TSVar.FLAGS.value].isel({TSVar.TIME.value: index}).item()
    new_value, new_status = linear_updater.update_extrapolated_data(
        index, status, ts
    )
    np.testing.assert_almost_equal(new_value, expected_value)
    np.testing.assert_equal(new_status, expected_status)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_arr", "expected_values", "expected_status"),
    [
        pytest.param(
            np.array([260, 140, 0, 386, 1, 16, 1], dtype=STATUS_TYPE),
            np.array([1.2, 1.6, 1.6, 1.8, 2.0, 2.2, 2.2]),
            np.array([260, 148, 0, 386, 402, 16, 147], dtype=STATUS_TYPE),
        ),
    ],
)
def test_update_time_series(status_arr, expected_values, expected_status):
    """
    Test update time series function
    """
    # Setup data
    ts = setup_data(
        size=7,
        a_ratio=0.001,
        b_ratio=0.006,
        a_radiation=0.0,
        b_radiation=200,
        status=status_arr,
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.EXTERNAL
    )
    updated_ts = linear_updater.update_time_series(ts)
    np.testing.assert_almost_equal(
        updated_ts[TSVar.ET.value].values, expected_values
    )
    np.testing.assert_equal(
        updated_ts[TSVar.FLAGS.value].values, expected_status
    )


@pytest.mark.functional
def test_update():
    """
    Test update interpolated data function
    """
    # Setup data
    window_size = 7
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    dates = np.array(pd.date_range(end=today, periods=window_size).to_list())
    ts = xr.Dataset(
        {
            TSVar.ET.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [1.2, 1.4, 1.6, 1.8, 1.8, 1.8, np.nan],
                                [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, np.nan],
                            ]
                        ]
                    ),
                    (2, 1, 0),
                ),
            ),
            TSVar.RADIATION.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [200, 200, 200, 200, 200, 200, 200],
                                [200, 200, 200, 200, 200, 200, 200],
                            ]
                        ]
                    ),
                    (2, 1, 0),
                ),
            ),
            TSVar.FLAGS.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [0, 386, 386, 0, 131, 132, 1],
                                [132, 0, 131, 259, 387, 515, 1],
                            ]
                        ],
                        dtype=STATUS_TYPE,
                    ),
                    (2, 1, 0),
                ),
            ),
        },
        coords={
            "time": dates,
            "x": np.arange(1),
            "y": np.arange(2),
        },
    )
    acquisition_dates = [1, 3, 5, 6]
    feed_dates = dates[acquisition_dates]
    feed = xr.Dataset(
        {
            TSVar.ET.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [np.nan, 1.8, np.nan, 2.4],
                                [1.4, np.nan, 2.2, np.nan],
                            ]
                        ]
                    ),
                    (2, 1, 0),
                ),
            ),
            TSVar.VALID.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array([[[0, 1, 0, 1], [1, 0, 1, 0]]]), (2, 1, 0)
                ),
            ),
        },
        coords={
            "time": feed_dates,
            "x": np.arange(1),
            "y": np.arange(2),
        },
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.EXTERNAL
    )
    updated_ts = linear_updater.update(ts, feed)
    ref = xr.Dataset(
        {
            TSVar.ET.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
                                [1.4, 1.4, 1.6, 1.8, 2.0, 2.2, 2.2],
                            ]
                        ]
                    ),
                    (2, 1, 0),
                ),
            ),
            TSVar.FLAGS.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [0, 386, 386, 0, 402, 402, 16],
                                [132, 0, 530, 530, 530, 16, 147],
                            ]
                        ],
                        dtype=STATUS_TYPE,
                    ),
                    (2, 1, 0),
                ),
            ),
        },
        coords={
            "time": dates,
            "x": np.arange(1),
            "y": np.arange(2),
        },
    )
    xr.testing.assert_allclose(updated_ts[TSVar.ET.value], ref[TSVar.ET.value])
    xr.testing.assert_equal(
        updated_ts[TSVar.FLAGS.value], ref[TSVar.FLAGS.value]
    )


@pytest.mark.functional
def test_update_no_feed():
    """
    Test update interpolated data function
    """
    # Setup data
    window_size = 7
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    dates = np.array(pd.date_range(end=today, periods=window_size).to_list())
    ts = xr.Dataset(
        {
            TSVar.ET.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [1.2, 1.4, 1.6, 1.8, 1.8, 1.8, np.nan],
                                [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, np.nan],
                            ]
                        ]
                    ),
                    (2, 1, 0),
                ),
            ),
            TSVar.RADIATION.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [200, 200, 200, 200, 200, 200, 200],
                                [200, 200, 200, 200, 200, 200, 200],
                            ]
                        ]
                    ),
                    (2, 1, 0),
                ),
            ),
            TSVar.FLAGS.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [0, 386, 386, 0, 131, 259, 1],
                                [132, 0, 131, 259, 387, 515, 1],
                            ]
                        ],
                        dtype=STATUS_TYPE,
                    ),
                    (2, 1, 0),
                ),
            ),
        },
        coords={
            "time": dates,
            "x": np.arange(1),
            "y": np.arange(2),
        },
    )
    linear_updater = LinearUpdater(
        strict_mode=True, radiation_mode=RadiationMode.EXTERNAL
    )
    updated_ts = linear_updater.update(ts, feed=None)
    ref = xr.Dataset(
        {
            TSVar.ET.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [1.2, 1.4, 1.6, 1.8, 1.8, 1.8, 1.8],
                                [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],
                            ]
                        ]
                    ),
                    (2, 1, 0),
                ),
            ),
            TSVar.FLAGS.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [0, 386, 386, 0, 131, 259, 403],
                                [132, 0, 131, 259, 387, 515, 659],
                            ]
                        ],
                        dtype=STATUS_TYPE,
                    ),
                    (2, 1, 0),
                ),
            ),
        },
        coords={
            "time": dates,
            "x": np.arange(1),
            "y": np.arange(2),
        },
    )
    xr.testing.assert_allclose(updated_ts[TSVar.ET.value], ref[TSVar.ET.value])
    xr.testing.assert_equal(
        updated_ts[TSVar.FLAGS.value], ref[TSVar.FLAGS.value]
    )
