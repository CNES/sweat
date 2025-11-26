# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import datetime as dt
import json
import os
import shutil

import pytest
import rioxarray as rio  # noqa: F401

from sweat.timeseries.config import check_config_timeseries
from sweat.timeseries.timeseries_handler import create_config, window_generator


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """
    Create temperory directory for all the tests in the module
    """
    dir_path = tmp_path_factory.mktemp("test_data")
    yield dir_path
    # Cleanup after all tests in the module
    shutil.rmtree(dir_path)


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
def test_create_config(test_data_dir) -> None:
    """
    Test create_config
    """
    min_date_str = "2025-08-23"
    max_date_str = "2025-08-28"
    min_date_dt = dt.datetime.strptime(min_date_str, "%Y-%m-%d")  # noqa: DTZ007
    max_date_dt = dt.datetime.strptime(max_date_str, "%Y-%m-%d")  # noqa: DTZ007
    expected_dates = [
        "2025-08-23",
        "2025-08-24",
        "2025-08-25",
        "2025-08-26",
        "2025-08-27",
        "2025-08-28",
    ]
    radiation_dir = os.path.join("tests", "data", "timeseries")
    et_time_series_dir = radiation_dir
    et_single_date_dir = radiation_dir
    config_dict = create_config(
        min_date_dt,
        max_date_dt,
        et_single_date_dir,
        radiation_dir,
        et_time_series_dir,
        test_data_dir,
        True,
    )
    # Check if the dictionary contains the configuration parameters
    check_config_timeseries(config_dict)
    # Check if dictionary contains the expected dates
    assert expected_dates == config_dict["input"]["dates"]
    # Check that all listed paths point to existing files
    for et_single_date_path in config_dict["input"]["et_single_date"]:
        assert os.path.isfile(et_single_date_path)
    for radiation_path in config_dict["input"]["radiation"]:
        assert os.path.isfile(radiation_path)
    for et_time_series_path in config_dict["input"]["et_time_series"]:
        assert os.path.isfile(et_time_series_path)
    # Check if the .json file exists
    date_min = min_date_dt.strftime("%Y%m%d")
    date_max = max_date_dt.strftime("%Y%m%d")
    json_filename = f"config_{date_min}-{date_max}.json"
    json_path = os.path.join(test_data_dir, json_filename)
    assert os.path.isfile(json_path)
    # Check that the .json file matches the dictionary
    with open(json_path) as json_file:
        dict_json = json.load(json_file)
    assert config_dict == dict_json
