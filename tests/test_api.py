# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import datetime as dt
import os
import shutil

import geopandas as gpd
import pytest

from sweat import api
from sweat.evaspa import tiling


def get_data_path() -> str:
    """
    Get data path
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.functional
def test_generate_tiles() -> None:
    """
    Test methods for regroup tiles
    """
    data_path = get_data_path()
    tiles, adjs = api.generate_tiles(os.path.join(data_path, "roi.gpkg"))
    assert len(tiles) == 8
    assert len(adjs) == 8
    tiles, adjs = api.generate_tiles(
        os.path.join(data_path, "roi.gpkg"), orbit_id=110
    )
    assert tiles.empty
    assert adjs.empty
    with pytest.raises(ValueError, match="Orbit ID must be between 0 and 114"):
        tiles, adjs = api.generate_tiles(
            os.path.join(data_path, "roi.gpkg"), orbit_id=210
        )


@pytest.mark.functional
def test_regroup() -> None:
    """
    Test methods for regroup tiles
    """
    data_path = get_data_path()
    tiles = gpd.read_file(os.path.join(data_path, "roi_tiles.gpkg"))
    land = gpd.read_file(os.path.join(data_path, "land_italy.gpkg"))
    adjs = tiling.generate_adjacents(tiles, land)
    group = api.regroup_tiles(tiles, adjs, threshold=100)
    assert len(group) == 9
    group = api.regroup_tiles(tiles, adjs, threshold=1000000)
    assert len(group) == 5


@pytest.mark.functional
@pytest.mark.parametrize(
    "entry",
    [
        {
            "path": "tests/data/modis_test.tif",
            "date": "2018-05-16T10:00:00-00:00",
        },
        {
            "path": "tests/data/modis_test_geo.tif",
            "date": "2018-05-16T10:00:00-00:00",
        },
        {
            "path": "tests/data/modis_test_dem.tif",
            "date": "2018-05-16T10:00:00-00:00",
        },
        {
            "path": "tests/data/modis_test_full.tif",
            "date": "2018-05-16T10:00:00-00:00",
        },
        {
            "path": "tests/data/modis_dir",
            "date": "2018-05-16T10:00:00-00:00",
        },
    ],
)
def test_read_input_data(entry: dict[str, str]) -> None:
    """
    Test read input data
    """
    res = api.read_input_data(entry)
    assert res


@pytest.mark.functional
@pytest.mark.parametrize(
    ("entry", "params", "debug"),
    [
        pytest.param(
            {
                "path": "tests/data/modis_test_geo.tif",
                "date": "2018-05-16T10:00:00-00:00",
            },
            {
                "ef": {
                    "models": "default_evaspa",
                },
            },
            None,
        ),
        pytest.param(
            {
                "path": "tests/data/modis_test_geo.tif",
                "date": "2018-05-16T10:00:00-00:00",
            },
            {
                "filtering": {},
                "ef": {
                    "check": {"threshold": 0.02},
                    "models": [
                        {
                            "name": "model1",
                            "dry_edge": {
                                "type": "LinearEdge",
                                "config": {
                                    "interval_type": "density",
                                    "interval_nb": 20,
                                    "percentile": [98, 100],
                                    "selection": "median",
                                },
                            },
                            "wet_edge": {
                                "type": "LinearEdge",
                                "config": {
                                    "interval_type": "density",
                                    "interval_nb": 20,
                                    "percentile": [0, 2],
                                    "selection": "median",
                                },
                            },
                            "var": "albedo",
                        },
                        {
                            "name": "model2",
                            "dry_edge": {
                                "type": "LinearEdge",
                                "config": {
                                    "interval_type": "density",
                                    "interval_nb": 20,
                                    "percentile": [95, 100],
                                    "selection": "median",
                                },
                            },
                            "wet_edge": {
                                "type": "FlatEdge",
                            },
                            "var": "albedo",
                        },
                    ],
                    "options": {"selection": False, "merging": "mean"},
                },
                "seb": {
                    "use_topo": True,
                    "models": ["kustas"],
                    "merging": "mean",
                },
                "daily": {"use_topo": True, "method": "toa"},
            },
            None,
        ),
        pytest.param(
            {
                "path": "tests/data/modis_test_dem.tif",
                "date": "2018-05-16T10:00:00-00:00",
            },
            {
                "filtering": {},
                "ef": {
                    "check": {"threshold": 0.02},
                    "models": [
                        {
                            "name": "model1",
                            "dry_edge": {
                                "type": "LinearEdge",
                                "config": {
                                    "interval_type": "density",
                                    "interval_nb": 20,
                                    "percentile": [98, 100],
                                    "selection": "median",
                                },
                            },
                            "wet_edge": {
                                "type": "LinearEdge",
                                "config": {
                                    "interval_type": "density",
                                    "interval_nb": 20,
                                    "percentile": [0, 2],
                                    "selection": "median",
                                },
                            },
                            "var": "albedo",
                        },
                        {
                            "name": "model2",
                            "dry_edge": {
                                "type": "LinearEdge",
                                "config": {
                                    "interval_type": "density",
                                    "interval_nb": 20,
                                    "percentile": [95, 100],
                                    "selection": "median",
                                },
                            },
                            "wet_edge": {
                                "type": "FlatEdge",
                            },
                            "var": "albedo",
                        },
                    ],
                    "options": {"selection": False, "merging": "mean"},
                },
                "seb": {
                    "use_topo": True,
                    "models": ["kustas"],
                    "merging": "mean",
                },
                "daily": {"use_topo": True, "method": "toa"},
            },
            {
                "path": "out/debug",
                "profile": True,
                "verbose": False,
            },
        ),
    ],
)
def test_run_evaspa(entry, params, debug) -> None:
    """
    Test run EVASPA
    """
    data = api.read_input_data(entry)
    res = api.run_evaspa(data, params, debug)
    assert res


@pytest.mark.functional
@pytest.mark.parametrize(
    ("entry", "params", "debug"),
    [
        pytest.param(
            {
                "path": "tests/data/modis_test_full.tif",
                "date": "2018-05-16T10:00:00-00:00",
            },
            {},
            None,
        ),
        pytest.param(
            {
                "path": "tests/data/modis_test_full.tif",
                "date": "2018-05-16T10:00:00-00:00",
            },
            {
                "prepare": {"use_topo": True},
                "filtering": {"tdp": {"op": ">=", "value": -30.0}},
                "stic": {"threshold": 0.01, "nb_steps": 15},
                "daily": {"method": "toa", "use_topo": True},
            },
            None,
        ),
        pytest.param(
            {
                "path": "tests/data/modis_test_full.tif",
                "date": "2018-05-16T10:00:00-00:00",
            },
            {
                "prepare": {"use_topo": True},
                "filtering": {"tdp": {"op": ">=", "value": -30.0}},
                "stic": {"threshold": 0.01, "nb_steps": 15},
                "daily": {"method": "toa", "use_topo": True},
            },
            {
                "path": "out/debug",
                "profile": True,
                "verbose": False,
            },
        ),
    ],
)
def test_run_stic(entry, params, debug) -> None:
    """
    Test run EVASPA
    """
    data = api.read_input_data(entry)
    res = api.run_stic(data, params, debug)
    assert res


@pytest.mark.functional
@pytest.mark.parametrize(
    ("entry", "params", "debug"),
    [
        pytest.param(
            {
                "dates": [
                    "2025-08-23",
                    "2025-08-24",
                    "2025-08-25",
                    "2025-08-26",
                    "2025-08-27",
                    "2025-08-28",
                    "2025-08-29",
                ],
                "et_time_series": [
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250823.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250824.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250825.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250826.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250827.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250828.tif",
                    ),
                ],
                "radiation": [
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250823.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250824.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250825.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250826.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250827.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250828.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250829.tif"
                    ),
                ],
                "et_single_date": [
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250824.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250826.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250828.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250829.tif",
                    ),
                ],
            },
            {},
            None,
        ),
        pytest.param(
            {
                "dates": [
                    "2025-08-23",
                    "2025-08-24",
                    "2025-08-25",
                    "2025-08-26",
                    "2025-08-27",
                    "2025-08-28",
                    "2025-08-29",
                ],
                "et_time_series": [
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250823.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250824.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250825.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250826.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250827.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250828.tif",
                    ),
                ],
                "radiation": [
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250823.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250824.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250825.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250826.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250827.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250828.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250829.tif"
                    ),
                ],
                "et_single_date": [
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250824.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250826.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250828.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250829.tif",
                    ),
                ],
                "dem": os.path.join(
                    "tests",
                    "data",
                    "timeseries",
                    "dem.tif",
                ),
            },
            {
                "stack": {"et_single_date_filtering": {}},
                "update": {
                    "method": "linear",
                    "params": {"strict_mode": True, "radiation_mode": 0},
                },
            },
            None,
        ),
        pytest.param(
            {
                "dates": [
                    "2025-08-23",
                    "2025-08-24",
                    "2025-08-25",
                    "2025-08-26",
                    "2025-08-27",
                    "2025-08-28",
                    "2025-08-29",
                ],
                "et_time_series": [
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250823.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250824.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250825.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250826.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250827.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_time_series_20250828.tif",
                    ),
                ],
                "radiation": [
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250823.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250824.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250825.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250826.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250827.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250828.tif"
                    ),
                    os.path.join(
                        "tests", "data", "timeseries", "radiation_20250829.tif"
                    ),
                ],
                "et_single_date": [
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250824.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250826.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250828.tif",
                    ),
                    os.path.join(
                        "tests",
                        "data",
                        "timeseries",
                        "et_single_date_20250829.tif",
                    ),
                ],
            },
            {},
            {
                "path": "out/debug",
                "profile": True,
                "verbose": False,
            },
        ),
    ],
)
def test_run_timeseries(entry, params, debug) -> None:
    """
    Test run EVASPA
    """
    et_ts, radiation_ts, et_sd, dem = api.read_ts_input_data(entry)
    res = api.run_timeseries(et_ts, radiation_ts, et_sd, dem, params, debug)
    assert res


@pytest.mark.functional
@pytest.mark.parametrize(
    ("window", "shift", "params", "debug", "verbose"),
    [pytest.param(7, 1, {}, {}, True)],
)
def test_run_window_time_series(
    window, shift, params, debug, verbose, tmp_path
) -> None:
    """
    Test run_window_time_series
    """
    test_data_dir = str(tmp_path)
    # Copy only radiation et_single_date .tif files in the test directory
    data_dir = os.path.join(get_data_path(), "timeseries")
    for filename in os.listdir(data_dir):
        if filename.startswith(("radiation", "et_single_date")):
            shutil.copy(os.path.join(data_dir, filename), test_data_dir)
    # Run window time series
    api.run_window_time_series(
        period_start=dt.datetime.strptime("2025-08-23", "%Y-%m-%d").replace(
            tzinfo=dt.UTC
        ),
        period_end=dt.datetime.strptime("2025-08-29", "%Y-%m-%d").replace(
            tzinfo=dt.UTC
        ),
        et_single_date_dir=test_data_dir,
        radiation_dir=test_data_dir,
        et_time_series_dir=test_data_dir,
        window=window,
        shift=shift,
        params=params,
        debug=debug,
        verbose=verbose,
        config_dir=test_data_dir,
    )
    # Check if all the et_time_series .tif files have been created
    expected_files = [
        "et_time_series_20250823.tif",
        "et_time_series_20250824.tif",
        "et_time_series_20250825.tif",
        "et_time_series_20250826.tif",
        "et_time_series_20250827.tif",
        "et_time_series_20250828.tif",
        "et_time_series_20250829.tif",
    ]
    for et_time_series_file in expected_files:
        assert et_time_series_file in os.listdir(test_data_dir)
