# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
from pathlib import Path

import pytest

import sweat.timeseries.config as cfg
from sweat.__about__ import __version__


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {
            "stack": {"et_single_date_filtering": {}},
            "update": {
                "method": "linear",
                "params": {"parallel": True, "num_workers": 8},
            },
        },
    ],
)
def test_timeseries_params_config(config) -> None:
    """
    Test TimeSeriesParamsConfig
    """
    assert cfg.TimeSeriesParamsConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            "verbose": True,
            "profile": True,
        },
        {
            "verbose": True,
            "profile": True,
            "config_verbose": True,
        },
        {
            "config_verbose": True,
        },
    ],
)
def test_timeseries_debugging_config(config) -> None:
    """
    Test TimeSeriesDebuggingConfig
    """
    assert cfg.TimeSeriesDebuggingConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "version",
    [None, "0"],
)
def test_timeseries_input_file(version, tmp_path) -> None:
    """
    Test TimeseriesInputFile
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {
            "dates": ["2025-08-23", "2025-08-24", "2025-08-25"],
            "et_time_series": [
                os.path.join(
                    "tests", "data", "timeseries", "et_time_series_20250823.tif"
                ),
                os.path.join(
                    "tests", "data", "timeseries", "et_time_series_20250824.tif"
                ),
            ],
            "radiation": [
                os.path.join(
                    "tests", "data", "timeseries", "radiation_20250823.tif"
                ),
                os.path.join(
                    "tests", "data", "timeseries", "radiation_20250824.tif"
                ),
            ],
            "et_single_date": [
                os.path.join(
                    "tests", "data", "timeseries", "et_single_date_20250824.tif"
                ),
            ],
            "dem": os.path.join("tests", "data", "timeseries", "dem.tif"),
        },
        "output": {"path": str(d)},
        "params": {},
    }
    if version is not None:
        config["version"] = version
    entry = cfg.TimeSeriesInputFile.model_validate(config)
    assert entry.version == __version__


@pytest.mark.unit
def test_check_config_timeseries(tmp_path) -> None:
    """
    Test check method for timeseries
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {
            "dates": ["2025-08-23", "2025-08-24", "2025-08-25"],
            "et_time_series": [
                os.path.join(
                    "tests", "data", "timeseries", "et_time_series_20250823.tif"
                ),
                os.path.join(
                    "tests", "data", "timeseries", "et_time_series_20250824.tif"
                ),
            ],
            "radiation": [
                os.path.join(
                    "tests", "data", "timeseries", "radiation_20250823.tif"
                ),
                os.path.join(
                    "tests", "data", "timeseries", "radiation_20250824.tif"
                ),
            ],
            "et_single_date": [
                os.path.join(
                    "tests", "data", "timeseries", "et_single_date_20250824.tif"
                ),
            ],
            "dem": os.path.join("tests", "data", "timeseries", "dem.tif"),
        },
        "output": {"path": str(d)},
        "params": {},
    }
    assert cfg.check_config_timeseries(config)


@pytest.mark.unit
def test_check_config_timeseries_debug(tmp_path) -> None:
    """
    Test check method for timeseries with debug section
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {
            "dates": ["2025-08-23", "2025-08-24", "2025-08-25"],
            "et_time_series": [
                os.path.join(
                    "tests", "data", "timeseries", "et_time_series_20250823.tif"
                ),
                os.path.join(
                    "tests", "data", "timeseries", "et_time_series_20250824.tif"
                ),
            ],
            "radiation": [
                os.path.join(
                    "tests", "data", "timeseries", "radiation_20250823.tif"
                ),
                os.path.join(
                    "tests", "data", "timeseries", "radiation_20250824.tif"
                ),
            ],
            "et_single_date": [
                os.path.join(
                    "tests", "data", "timeseries", "et_single_date_20250824.tif"
                ),
            ],
            "dem": os.path.join("tests", "data", "timeseries", "dem.tif"),
        },
        "output": {"path": str(d)},
        "params": {},
        "debug": {"profile": True, "verbose": True},
    }
    assert cfg.check_config_timeseries(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "version",
    [None, "0"],
)
def test_window_timeseries_input_file(version, tmp_path) -> None:
    """
    Test version in InputFile for timeseries over a period
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {
            "dates": ["2025-08-23", "2025-08-24", "2025-08-25"],
            "et_time_series": [
                os.path.join(
                    "tests", "data", "timeseries", "et_time_series_20250823.tif"
                ),
                os.path.join(
                    "tests", "data", "timeseries", "et_time_series_20250824.tif"
                ),
            ],
            "radiation": [
                os.path.join(
                    "tests", "data", "timeseries", "radiation_20250823.tif"
                ),
                os.path.join(
                    "tests", "data", "timeseries", "radiation_20250824.tif"
                ),
            ],
            "et_single_date": [
                os.path.join(
                    "tests", "data", "timeseries", "et_single_date_20250824.tif"
                ),
            ],
            "dem": os.path.join("tests", "data", "timeseries", "dem.tif"),
        },
        "output": {"path": str(d)},
        "params": {},
    }
    if version is not None:
        config["version"] = version
    entry = cfg.TimeSeriesInputFile.model_validate(config)
    assert entry.version == __version__


@pytest.mark.unit
def test_check_config_window_timeseries(tmp_path) -> None:
    """
    Test check method for timeseries over a period
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {
            "period_start": "2025-08-24",
            "period_end": "2025-08-29",
            "window": 5,
            "shift": 2,
            "radiation_dir": os.path.join("tests", "data", "timeseries"),
            "et_single_date_dir": os.path.join("tests", "data", "timeseries"),
        },
        "output": {"path": str(d)},
        "params": {},
    }
    assert cfg.check_config_window_timeseries(config)


@pytest.mark.unit
def test_check_config_window_timeseries_debug(tmp_path) -> None:
    """
    Test check method with debug section for timeseries over a period
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {
            "period_start": "2025-08-24",
            "period_end": "2025-08-29",
            "window": 5,
            "shift": 2,
            "radiation_dir": os.path.join("tests", "data", "timeseries"),
            "et_single_date_dir": os.path.join("tests", "data", "timeseries"),
        },
        "output": {"path": str(d)},
        "params": {},
        "debug": {"config_verbose": True},
    }
    assert cfg.check_config_window_timeseries(config)
