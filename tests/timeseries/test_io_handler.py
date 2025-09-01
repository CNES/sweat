# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import datetime as dt
import glob
import os
import shutil

import affine
import numpy as np
import pandas as pd
import pytest
import rioxarray as rio  # noqa: F401
import xarray as xr
from pyproj import CRS

from evaspa.common.io import write_dataset
from evaspa.timeseries import io_handler as ioh
from evaspa.timeseries import status_handler as sh
from evaspa.timeseries.constant import TimeSeriesVar as TSVar

# TODO: run a setup for create data, check if it is possible


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """
    Create temperory directory for all the tests in the module
    """
    dir_path = tmp_path_factory.mktemp("test_data")
    yield dir_path
    # Cleanup after all tests in the module
    shutil.rmtree(dir_path)


# This will run once before any tests in the module
@pytest.fixture(scope="module", autouse=True)
def setup_test_data(test_data_dir):
    """
    Setup data test
    """
    # Generate test data here
    # Setup data
    # Define parameters
    window_size = 7  # Time
    x_size = 1
    y_size = 2
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    dates = np.array(pd.date_range(end=today, periods=window_size).to_list())
    # Define transform: (origin_x, origin_y), pixel size = 0.1 degree
    transform = affine.Affine.translation(0, 40) * affine.Affine.scale(
        0.1, -0.1
    )
    # Generate x/y coordinates from transform
    x_coords = [(transform * (i, 0))[0] for i in range(x_size)]  # x: lon
    y_coords = [(transform * (0, j))[1] for j in range(y_size)]  # y: lat
    # Create a dataset
    et_ts = xr.Dataset(
        {
            TSVar.ET.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [1.2, 1.4, 1.6, 1.8, 1.8, 1.8],
                                [1.4, 1.4, 1.4, 1.4, 1.4, 1.4],
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
                                [0, 386, 386, 0, 131, 275],
                                [132, 0, 131, 259, 387, 531],
                            ]
                        ],
                        dtype=sh.STATUS_TYPE,
                    ),
                    (2, 1, 0),
                ),
            ),
        },
        coords={
            "time": dates[:-1],
            "x": x_coords,
            "y": y_coords,
        },
    )
    # Add CRS and transform metadata (compatible with rioxarray)
    et_ts = et_ts.rio.write_crs(CRS(4236))
    et_ts = et_ts.rio.write_transform(transform)
    # Create a radiation dataset
    radiation_ts = xr.Dataset(
        {
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
        },
        coords={
            "time": dates,
            "x": x_coords,
            "y": y_coords,
        },
    )
    # Add CRS and transform metadata (compatible with rioxarray)
    radiation_ts = radiation_ts.rio.write_crs(CRS(4236))
    radiation_ts = radiation_ts.rio.write_transform(transform)
    # Create a dataset for ET signe dates
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
            TSVar.FLAGS.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array([[[0, 1, 0, 1], [1, 0, 1, 0]]]), (2, 1, 0)
                ),
            ),
        },
        coords={
            "time": feed_dates,
            "x": x_coords,
            "y": y_coords,
        },
    )
    # Add CRS and transform metadata (compatible with rioxarray)
    feed = feed.rio.write_crs(CRS(4236))
    feed = feed.rio.write_transform(transform)
    # Create a DEM dataset
    dem = xr.Dataset(
        {
            TSVar.HEIGHT.value: (
                ["y", "x"],
                np.array([[10.0], [10.0]]),
            ),
            TSVar.SLOPE.value: (
                ["y", "x"],
                np.array([[5.0], [5.0]]),
            ),
            TSVar.ASPECT.value: (
                ["y", "x"],
                np.array([[10.0], [10.0]]),
            ),
        },
        coords={
            "x": x_coords,
            "y": y_coords,
        },
    )
    # Add CRS and transform metadata (compatible with rioxarray)
    dem = dem.rio.write_crs(CRS(4236))
    dem = dem.rio.write_transform(transform)
    # Write data
    output_dir = str(test_data_dir)
    ioh.write_timeseries(
        et_ts,
        root_name="et_time_series",
        directory=output_dir,
    )
    ioh.write_timeseries(
        radiation_ts, root_name="radiation", directory=output_dir
    )
    ioh.write_timeseries(feed, root_name="et_single_date", directory=output_dir)
    write_dataset(dem, filename="dem.tif", directory=output_dir)


def get_list_files(output_dir) -> tuple[list[str], list[str], list[str], str]:
    """
    Generate lists of file paths
    """
    return (
        [
            os.path.join(output_dir, file)
            for file in glob.glob("et_time_series_*.tif", root_dir=output_dir)
        ],
        [
            os.path.join(output_dir, file)
            for file in glob.glob("radiation_*.tif", root_dir=output_dir)
        ],
        [
            os.path.join(output_dir, file)
            for file in glob.glob("et_single_date_*.tif", root_dir=output_dir)
        ],
        os.path.join(output_dir, "dem.tif"),
    )


@pytest.mark.unit
def test_write_timeseries(test_data_dir) -> None:
    """
    Test write_timeseries
    """
    et_ts_files, radiation_files, et_files, dem = get_list_files(
        str(test_data_dir)
    )
    assert len(et_ts_files) > 0
    assert len(radiation_files) > 0
    assert len(et_files) > 0
    for file in et_ts_files:
        assert os.path.exists(file)
    for file in radiation_files:
        assert os.path.exists(file)
    for file in et_files:
        assert os.path.exists(file)
    assert os.path.exists(dem)


@pytest.mark.unit
@pytest.mark.parametrize(
    "keys",
    [
        ["dates", "et_time_series", "radiation", "et_single_date", "dem"],
        ["et_time_series", "radiation", "et_single_date"],
        ["et_time_series", "radiation"],
        ["et_time_series"],
        ["dates", "et_time_series", "dem"],
    ],
)
def test_timeseries_input_config(keys, test_data_dir) -> None:
    """
    Test TimeSeriesInputConfig
    """
    et_ts_files, radiation_files, et_files, dem_file = get_list_files(
        str(test_data_dir)
    )
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    dates = pd.date_range(end=today, periods=7).to_list()
    # Convert to strings in desired format, e.g., "YYYY-MM-DD"
    date_strings = [date.strftime("%Y-%m-%d") for date in dates]
    full_config = {
        "dates": date_strings,
        "et_time_series": et_ts_files,
        "radiation": radiation_files,
        "et_single_date": et_files,
        "dem": dem_file,
    }
    # Create the subdictionary
    config = {k: full_config[k] for k in keys if k in full_config}
    assert ioh.TimeSeriesInputConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param(
            {
                "et_time_series": ["foo"],
            },
            OSError,
        ),
    ],
)
def test_timeseries_input_config_error(config, error) -> None:
    """
    Test TimeSeriesInputConfig with error
    """
    with pytest.raises(error):
        ioh.TimeSeriesInputConfig.model_validate(config)


@pytest.mark.unit
def test_extract_date_from_filename(test_data_dir) -> None:
    """
    Test function for extracting date of filename
    """
    et_ts_files, _, _, _ = get_list_files(str(test_data_dir))
    date = ioh.extract_date_from_filename(sorted(et_ts_files)[-1])
    today = pd.to_datetime(
        (dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=1)).strftime(
            "%Y%m%d"
        ),
        format="%Y%m%d",
    )
    assert date == today


@pytest.mark.unit
def test_read_et_time_series(test_data_dir) -> None:
    """
    Test function for reading et time series files
    """
    et_ts_files, _, _, _ = get_list_files(str(test_data_dir))
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    dates = pd.date_range(end=today, periods=7).to_list()
    ts = ioh.read_et_time_series(et_ts_files, dates)
    assert ts
    assert ts.sizes[TSVar.TIME.value] == 7
    assert ts.sizes["x"] == 1
    assert ts.sizes["y"] == 2
    np.testing.assert_almost_equal(
        ts[TSVar.ET.value].values[:, 0, 0],
        np.array([1.2, 1.4, 1.6, 1.8, 1.8, 1.8, np.nan]),
    )
    np.testing.assert_almost_equal(
        ts[TSVar.FLAGS.value].values[:, 1, 0],
        np.array([132, 0, 131, 259, 387, 515, 1], dtype=sh.STATUS_TYPE),
    )


@pytest.mark.unit
def test_read_et_time_series_without_dates(test_data_dir) -> None:
    """
    Test function for reading et time series files (without dates)
    """
    et_ts_files, _, _, _ = get_list_files(str(test_data_dir))
    ts = ioh.read_et_time_series(et_ts_files, dates=None)
    assert ts
    assert ts.sizes[TSVar.TIME.value] == 6
    assert ts.sizes["x"] == 1
    assert ts.sizes["y"] == 2
    np.testing.assert_almost_equal(
        ts[TSVar.ET.value].values[:, 0, 0],
        np.array([1.2, 1.4, 1.6, 1.8, 1.8, 1.8]),
    )
    np.testing.assert_almost_equal(
        ts[TSVar.FLAGS.value].values[:, 1, 0],
        np.array([132, 0, 131, 259, 387, 515], dtype=sh.STATUS_TYPE),
    )


@pytest.mark.unit
def test_read_radiation_time_series(test_data_dir) -> None:
    """
    Test function for reading radiation time series files
    """
    _, ts_files, _, _ = get_list_files(str(test_data_dir))
    ts = ioh.read_radiation_time_series(ts_files)
    assert ts
    assert ts.sizes[TSVar.TIME.value] == 7
    assert ts.sizes["x"] == 1
    assert ts.sizes["y"] == 2
    np.testing.assert_almost_equal(
        ts[TSVar.RADIATION.value].values[:, 0, 0],
        np.array([200, 200, 200, 200, 200, 200, 200]),
    )


@pytest.mark.unit
def test_read_et_single_date(test_data_dir) -> None:
    """
    Test function for reading et sibgle date files
    """
    _, _, files, _ = get_list_files(str(test_data_dir))
    ts = ioh.read_et_single_date(files)
    assert ts
    assert ts.sizes[TSVar.TIME.value] == 4
    assert ts.sizes["x"] == 1
    assert ts.sizes["y"] == 2
    np.testing.assert_almost_equal(
        ts[TSVar.ET.value].values[:, 0, 0],
        np.array([np.nan, 1.8, np.nan, 2.4]),
    )
    np.testing.assert_almost_equal(
        ts[TSVar.FLAGS.value].values[:, 1, 0],
        np.array([1, 0, 1, 0], dtype=sh.STATUS_TYPE),
    )


@pytest.mark.functional
@pytest.mark.parametrize(
    "keys",
    [
        ["dates", "et_time_series", "radiation", "et_single_date", "dem"],
        ["et_time_series", "radiation", "et_single_date"],
        ["et_time_series", "radiation"],
        ["et_time_series"],
        ["dates", "et_time_series", "dem"],
    ],
)
def test_read_input(keys, test_data_dir) -> None:
    """
    Test TimeSeriesInputConfig
    """
    et_ts_files, radiation_files, et_files, dem_file = get_list_files(
        str(test_data_dir)
    )
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    dates = pd.date_range(end=today, periods=7).to_list()
    # Convert to strings in desired format, e.g., "YYYY-MM-DD"
    date_strings = [date.strftime("%Y-%m-%d") for date in dates]
    full_config = {
        "dates": date_strings,
        "et_time_series": et_ts_files,
        "radiation": radiation_files,
        "et_single_date": et_files,
        "dem": dem_file,
    }
    # Create the subdictionary
    config = {k: full_config[k] for k in keys if k in full_config}
    et_ts, radiation_ts, et_sd, dem = ioh.read_input(config)
    assert et_ts
    if "radiation" in config:
        assert radiation_ts
    else:
        assert radiation_ts is None
    if "et_single_date" in config:
        assert et_sd
    else:
        assert et_sd is None
    if "dem" in config:
        assert dem
    else:
        assert dem is None
