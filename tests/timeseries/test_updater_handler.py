# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from sweat.timeseries import updater_handler
from sweat.timeseries.constant import TimeSeriesVar as TSVar
from sweat.timeseries.status_handler import STATUS_TYPE


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"method": "linear"},
        {
            "method": "linear",
            "params": {"strict_mode": True, "radiation_mode": "EXTERNAL"},
        },
    ],
)
def test_updaterconfig(config) -> None:
    """
    Test UpdaterConfig
    """
    assert updater_handler.UpdaterConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param("foo", ValidationError),
        pytest.param({"method": "foo"}, ValidationError),
        pytest.param({"method": "api"}, ValueError),
        pytest.param(
            {"method": "linear", "params": {"foo": "foo"}}, ValidationError
        ),
    ],
)
def test_updaterconfig_error(config, error) -> None:
    """
    Test UpdaterConfig with error
    """
    with pytest.raises(error):
        updater_handler.UpdaterConfig.model_validate(config)


@pytest.mark.functional
def test_create() -> None:
    """
    Test create function
    """
    config = {
        "method": "linear",
    }
    updater_config = updater_handler.UpdaterConfig.model_validate(config)
    assert updater_handler.create(
        method=updater_config.method, params=updater_config.params
    )


@pytest.mark.functional
def test_run() -> None:
    """
    Test run function
    """
    # Setup data
    window_size = 7
    today = dt.datetime.now(tz=dt.UTC).date()
    dates = np.array(pd.date_range(end=today, periods=window_size).to_list())
    ts = xr.Dataset(
        {
            TSVar.ET.value: (
                ["x", "y", "time"],
                np.array(
                    [
                        [
                            [1.2, 1.4, 1.6, 1.8, 1.8, 1.8, np.nan],
                            [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, np.nan],
                        ]
                    ]
                ),
            ),
            TSVar.RADIATION.value: (
                ["x", "y", "time"],
                np.array(
                    [
                        [
                            [200, 200, 200, 200, 200, 200, 200],
                            [200, 200, 200, 200, 200, 200, 200],
                        ]
                    ]
                ),
            ),
            TSVar.FLAGS.value: (
                ["x", "y", "time"],
                np.array(
                    [
                        [
                            [0, 386, 386, 0, 131, 132, 1],
                            [132, 0, 131, 259, 387, 515, 1],
                        ]
                    ],
                    dtype=STATUS_TYPE,
                ),
            ),
        },
        coords={
            "time": np.array(
                pd.date_range(end=today, periods=window_size).to_list()
            ),
            "x": np.arange(1),
            "y": np.arange(2),
        },
    )
    ts = ts.transpose("time", "x", "y")
    acquisition_dates = [1, 3, 5, 6]
    feed_dates = dates[acquisition_dates]
    feed = xr.Dataset(
        {
            TSVar.ET.value: (
                ["x", "y", "time"],
                np.array(
                    [[[np.nan, 1.8, np.nan, 2.4], [1.4, np.nan, 2.2, np.nan]]]
                ),
            ),
            TSVar.VALID.value: (
                ["x", "y", "time"],
                np.array([[[0, 1, 0, 1], [1, 0, 1, 0]]]),
            ),
        },
        coords={
            "time": feed_dates,
            "x": np.arange(1),
            "y": np.arange(2),
        },
    )
    feed = feed.transpose("time", "x", "y")
    # Setup config
    config = {
        "method": "linear",
    }
    # Run
    assert updater_handler.run(ts, feed=feed, config=config)
