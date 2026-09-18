# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales


import datetime as dt
import json
import os

import pandas as pd
import pytest

from sweat.timeseries.timeseries_handler import create_config, window_generator


@pytest.mark.unit
def test_window_generator_1_day_shift() -> None:
    """
    Test window_generator with 1-day shift
    Trivial case
    """
    min_date_str = "2023-03-01"
    max_date_str = "2023-03-12"
    min_date_dt = dt.datetime.strptime(min_date_str, "%Y-%m-%d")  # noqa: DTZ007
    max_date_dt = dt.datetime.strptime(max_date_str, "%Y-%m-%d")  # noqa: DTZ007
    expected_windows = [
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 1, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 2, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 3, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 4, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 5, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 6, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 7, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 2, 0, 0), dt.datetime(2023, 3, 8, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 3, 0, 0), dt.datetime(2023, 3, 9, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 4, 0, 0), dt.datetime(2023, 3, 10, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 5, 0, 0), dt.datetime(2023, 3, 11, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 6, 0, 0), dt.datetime(2023, 3, 12, 0, 0)),  # noqa: DTZ001
    ]
    iterations = 0
    for windows in window_generator(min_date_dt, max_date_dt, 7, 1):
        assert expected_windows[iterations] == windows
        iterations += 1
    assert iterations == len(expected_windows)


@pytest.mark.unit
def test_window_generator_3_day_shift() -> None:
    """
    Test window_generator with a 3-day shift
    Potential problematic case:
    the final window has fewer days than the shift size
    """
    min_date_str = "2023-03-01"
    max_date_str = "2023-03-12"
    min_date_dt = dt.datetime.strptime(min_date_str, "%Y-%m-%d")  # noqa: DTZ007
    max_date_dt = dt.datetime.strptime(max_date_str, "%Y-%m-%d")  # noqa: DTZ007
    expected_windows = [
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 1, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 4, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 1, 0, 0), dt.datetime(2023, 3, 7, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 4, 0, 0), dt.datetime(2023, 3, 10, 0, 0)),  # noqa: DTZ001
        (dt.datetime(2023, 3, 7, 0, 0), dt.datetime(2023, 3, 12, 0, 0)),  # noqa: DTZ001
    ]
    iterations = 0
    for windows in window_generator(min_date_dt, max_date_dt, 7, 3):
        assert expected_windows[iterations] == windows
        iterations += 1
    assert iterations == len(expected_windows)


@pytest.mark.unit
def test_create_config(tmp_path) -> None:
    """
    Test create_config
    """
    test_data_dir = tmp_path / "config"
    min_date_str = "2025-08-23"
    max_date_str = "2025-08-28"
    min_date_dt = dt.datetime.strptime(min_date_str, "%Y-%m-%d")  # noqa: DTZ007
    max_date_dt = dt.datetime.strptime(max_date_str, "%Y-%m-%d")  # noqa: DTZ007
    radiation_dir = os.path.join("tests", "data", "timeseries")
    et_time_series_dir = os.path.join("tests", "data", "timeseries")
    et_single_date_dir = os.path.join("tests", "data", "timeseries")
    dem = os.path.join("tests", "data", "timeseries", "dem.tif")
    debug = {"profile": True}
    config = create_config(
        window_start=min_date_dt,
        window_end=max_date_dt,
        et_single_date_dir=et_single_date_dir,
        radiation_dir=radiation_dir,
        et_time_series_dir=et_time_series_dir,
        dem=dem,
        params={},
        debug=debug,
        verbose=True,
        config_dir=str(test_data_dir),
    )
    # Check if dictionary contains the expected dates
    expected_dates = pd.date_range(
        min_date_dt, freq="1D", end=max_date_dt
    ).to_list()
    assert expected_dates == config.input.dates
    # Check that all listed paths point to existing files
    for et_single_date_path in config.input.et_single_date:
        assert os.path.isfile(et_single_date_path)
    for radiation_path in config.input.radiation:
        assert os.path.isfile(radiation_path)
    for et_time_series_path in config.input.et_time_series:
        assert os.path.isfile(et_time_series_path)
    # Check if the .json file exists
    date_min = min_date_dt.strftime("%Y%m%d")
    date_max = max_date_dt.strftime("%Y%m%d")
    json_filename = f"config_{date_min}_{date_max}.json"
    json_path = os.path.join(test_data_dir, json_filename)
    assert os.path.isfile(json_path)
    # Check that the .json file matches the dictionary
    with open(json_path) as json_file:
        dict_json = json.load(json_file)
    assert json.loads(config.model_dump_json()) == dict_json
