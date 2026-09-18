# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from sweat.timeseries import updater_handler
from sweat.timeseries.status_handler import (
    INIT_STATUS,
    STATUS_TYPE,
)
from sweat.timeseries.types import TimeSeriesVar as TSVar


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"method": "linear"},
        {
            "method": "linear",
            "params": {},
        },
        {
            "method": "linear",
            "params": {"parallel": True},
        },
        {
            "method": "linear",
            "params": {"parallel": True, "num_workers": 8},
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
@pytest.mark.parametrize(
    "config",
    [
        {"method": "linear"},
        {
            "method": "linear",
            "params": {},
        },
        {
            "method": "linear",
            "params": {"parallel": True},
        },
        {
            "method": "linear",
            "params": {"parallel": True, "num_workers": 8},
        },
    ],
)
def test_create(config) -> None:
    """
    Test create function
    """
    updater_config = updater_handler.UpdaterConfig.model_validate(config)
    assert updater_handler.create(
        method=updater_config.method, params=updater_config.params
    )


@pytest.mark.functional
@pytest.mark.parametrize(
    "config",
    [
        {
            "method": "linear",
            "params": {"parallel": False},
        },
        {
            "method": "linear",
            "params": {"parallel": True},
        },
    ],
)
def test_run(config) -> None:
    """
    Test run function
    """
    # Generate 7 dates until now
    window = 7
    today = pd.Timestamp.now()
    dates = pd.date_range(end=today, periods=window, freq="D").date

    # Create dimensions
    size_x = 600
    size_y = 400
    x = np.arange(size_x)
    y = np.arange(size_y)

    shape = (size_y, size_x, window)

    # Create data variables
    et_data = np.full(shape, np.nan)
    radiation_data = np.ones(shape)
    flags_data = np.full(shape, INIT_STATUS, dtype=STATUS_TYPE)

    # Create the dataset
    ts = xr.Dataset(
        data_vars={
            TSVar.ET.value: (["y", "x", "time"], et_data),
            TSVar.RADIATION.value: (["y", "x", "time"], radiation_data),
            TSVar.FLAGS.value: (["y", "x", "time"], flags_data),
        },
        coords={
            "time": dates,
            "x": x,
            "y": y,
        },
    )
    # Select first date randomly from first 4 dates (leaves room for gap)
    first_idx = np.random.randint(0, 4)
    # Select second date from at least 3 positions after first
    second_idx = np.random.randint(first_idx + 3, len(dates))

    acquisition_dates = sorted([dates[first_idx], dates[second_idx]])

    # Create data variables
    et_feed = np.random.uniform(low=1.0, high=6.0, size=(size_y, size_x, 2))
    valid_feed = np.random.randint(0, 2, size=(size_y, size_x, 2))

    # Create the dataset
    feed = xr.Dataset(
        data_vars={
            TSVar.ET.value: (["y", "x", "time"], et_feed),
            TSVar.VALID.value: (["y", "x", "time"], valid_feed),
        },
        coords={
            "time": acquisition_dates,
            "x": x,
            "y": y,
        },
    )
    updated_ts = updater_handler.run(ts, feed=feed, config=config)
    assert updated_ts.attrs == ts.attrs
