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
from pydantic import ValidationError
from pyproj import CRS

from sweat.common.io import read_data_from_file, write_dataset
from sweat.timeseries import io_handler as ioh
from sweat.timeseries import stack_handler as sth
from sweat.timeseries import status_handler as sh
from sweat.timeseries.constant import TimeSeriesVar as TSVar

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
    window_size = 7  # Time
    x_size = 1
    y_size = 2
    today = dt.datetime.now(tz=dt.UTC).date()
    dates = np.array(pd.date_range(end=today, periods=window_size).to_list())
    # Define transform: (origin_x, origin_y), pixel size = 0.1 degree
    transform = affine.Affine.translation(0, 40) * affine.Affine.scale(
        0.1, -0.1
    )
    # Generate x/y coordinates from transform
    x_coords = [(transform * (i, 0))[0] for i in range(x_size)]  # x: lon
    y_coords = [(transform * (0, j))[1] for j in range(y_size)]  # y: lat
    # Create ET time series data
    et_ts = xr.Dataset(
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
            TSVar.FLAGS.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [0, 386, 386, 0, 131, 275, 1],
                                [132, 0, 131, 259, 387, 531, 1],
                            ]
                        ],
                        dtype=sh.STATUS_TYPE,
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
    et_ts = et_ts.rio.write_crs(CRS(4236))
    et_ts = et_ts.rio.write_transform(transform)
    # Create radiation time series
    radiation_ts = xr.Dataset(
        {
            TSVar.RADIATION.value: (
                ["time", "y", "x"],
                np.transpose(
                    np.array(
                        [
                            [
                                [
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                ],
                                [
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                    3.5e7,
                                ],
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
    # Create ET single dates
    acquisition_dates = [1, 3, 5, 6]
    et_sd = xr.Dataset(
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
                    np.array([[[0, 0, 0, 0], [0, 1, 0, 1]]]), (2, 1, 0)
                ),
            ),
        },
        coords={
            "time": dates[acquisition_dates],
            "x": x_coords,
            "y": y_coords,
        },
    )
    # Add CRS and transform metadata (compatible with rioxarray)
    et_sd = et_sd.rio.write_crs(CRS(4236))
    et_sd = et_sd.rio.write_transform(transform)
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
    ioh.write_timeseries(
        et_sd, root_name="et_single_date", directory=output_dir
    )
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
@pytest.mark.parametrize(
    "config",
    [
        {
            "et_single_date_filtering": {
                "flags": {"op": "==", "value": 0},
            }
        }
    ],
)
def test_timeseries_stack_config(config) -> None:
    """
    Test TimeSeriesStackConfig
    """
    assert sth.TimeSeriesStackConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param(
            {
                "foo": {
                    "flags": {"op": "==", "value": 0},
                }
            },
            ValidationError,
        ),
        pytest.param(
            {"et_single_date_filtering": "foo"},
            ValidationError,
        ),
    ],
)
def test_timeseries_stack_config_error(config, error) -> None:
    """
    Test TimeSeriesInputConfig with error
    """
    with pytest.raises(error):
        sth.TimeSeriesStackConfig.model_validate(config)


@pytest.mark.unit
def test_fill_radiation_missing() -> None:
    """
    Test function for filling radiation data at missing date
    """
    data = xr.DataArray(
        data=np.array([[np.nan], [np.nan]]),
        dims=["y", "x"],
        coords={
            "time": pd.to_datetime("2025-08-23"),
            "x": np.array([0.0]),
            "y": np.array([40.0, 39.9]),
        },
        attrs={
            "crs": CRS(4326),
            "affine": affine.Affine(0.1, 0.0, -0.05, 0.0, -0.1, 40.05),
        },
    )
    filled_data = sth.fill_radiation_missing(data, dem=None)
    np.testing.assert_allclose(
        filled_data.values, np.array([[35275114.436], [35297798.513]])
    )


@pytest.mark.unit
def test_fill_radiation_missing_with_dem() -> None:
    """
    Test function for filling radiation data at missing date
    with DEM data
    """
    data = xr.DataArray(
        data=np.array([[np.nan], [np.nan]]),
        dims=["y", "x"],
        coords={
            "time": pd.to_datetime("2025-08-23"),
            "x": np.array([0.0]),
            "y": np.array([40.0, 39.9]),
        },
        attrs={
            "crs": CRS(4326),
            "affine": affine.Affine(0.1, 0.0, -0.05, 0.0, -0.1, 40.05),
        },
    )
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
            "x": np.array([0.0]),
            "y": np.array([40.0, 39.9]),
        },
        attrs={
            "crs": CRS(4326),
            "affine": affine.Affine(0.1, 0.0, -0.05, 0.0, -0.1, 40.05),
        },
    )
    filled_data = sth.fill_radiation_missing(data, dem)
    np.testing.assert_allclose(
        filled_data.values, np.array([[34050036.737489], [34077366.434024]])
    )


@pytest.mark.unit
def test_stack_time_series(test_data_dir) -> None:
    """
    Test function for reading et time series files
    """
    et_ts_files, radiation_files, _, _ = get_list_files(str(test_data_dir))
    et_ts = ioh.read_et_time_series(et_ts_files, None)
    radiation_ts = ioh.read_radiation_time_series(radiation_files)
    ts = sth.stack_time_series(et_ts, radiation_ts, dem=None)
    assert ts
    assert ts.sizes[TSVar.TIME.value] == 7
    assert ts.sizes["x"] == 1
    assert ts.sizes["y"] == 2
    assert list(ts.data_vars) == [
        TSVar.ET.value,
        TSVar.FLAGS.value,
        TSVar.RADIATION.value,
    ]
    assert not ts[TSVar.RADIATION.value].isnull().all().item()


@pytest.mark.unit
def test_stack_time_series_with_missing_dates(test_data_dir) -> None:
    """
    Test function for reading et time series files
    """
    et_ts_files, radiation_files, _, _ = get_list_files(str(test_data_dir))
    # Create radiation time series with missing files
    radiation_dates = [0, 1, 2, 3, 5]
    radiation_files_with_missing_dates = [
        radiation_files[i] for i in radiation_dates
    ]
    et_ts = ioh.read_et_time_series(et_ts_files, None)
    radiation_ts = ioh.read_radiation_time_series(
        radiation_files_with_missing_dates
    )
    ts = sth.stack_time_series(et_ts, radiation_ts, dem=None)
    assert ts
    assert ts.sizes[TSVar.TIME.value] == 7
    assert ts.sizes["x"] == 1
    assert ts.sizes["y"] == 2
    assert list(ts.data_vars) == [
        TSVar.ET.value,
        TSVar.FLAGS.value,
        TSVar.RADIATION.value,
    ]
    assert not ts[TSVar.RADIATION.value].isnull().all().item()


@pytest.mark.functional
def test_run(test_data_dir) -> None:
    """
    Test run function for preparing time series
    """
    et_ts_files, radiation_files, et_sd_files, dem_file = get_list_files(
        str(test_data_dir)
    )
    et_ts = ioh.read_et_time_series(et_ts_files, None)
    radiation_ts = ioh.read_radiation_time_series(radiation_files)
    et_sd = ioh.read_et_single_date(et_sd_files)
    dem = read_data_from_file(dem_file)
    et_single_date_filtering = {
        "flags": {"op": "==", "value": 0},
    }
    ts, feed = sth.run(
        et_ts,
        radiation_ts,
        et_sd,
        dem,
        et_single_date_filtering=et_single_date_filtering,
    )
    assert ts
    assert feed
