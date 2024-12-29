# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from pyproj import CRS

from evaspa import io


def get_test_data_dir() -> str:
    """
    Get directory path for test data
    """
    if os.environ["EVASPA_TEST_DATA_PATH"] is None:
        msg = "Variable EVASPA_TEST_DATA_PATH must be set"
        raise ValueError(msg)
    return os.environ["EVASPA_TEST_DATA_PATH"]


@pytest.mark.requires_test_data
def test_read_data_from_file() -> None:
    """
    Test read data from a file
    """
    input_path = os.path.join(get_test_data_dir(), "Landsat_20230327_28PCA.tif")
    xarr = io.read_data_from_file(input_path)
    assert xarr.sizes["x"] == 1832
    assert xarr.sizes["y"] == 1832
    assert set(xarr.data_vars) == {
        "qa",
        "ndvi",
        "rld",
        "emis",
        "red",
        "blue",
        "cloud",
        "lai",
        "albedo",
        "green",
        "lst",
        "water",
        "rsd",
        "nir",
    }


@pytest.mark.requires_test_data
def test_read_data() -> None:
    """
    Test read data
    """
    input_path = os.path.join(get_test_data_dir(), "Landsat_20230327_28PCA")
    xarr = io.read_data(input_path)
    assert xarr.sizes["x"] == 1832
    assert xarr.sizes["y"] == 1832
    assert set(xarr.data_vars) == {
        "qa",
        "ndvi",
        "rld",
        "emis",
        "red",
        "blue",
        "cloud",
        "lai",
        "albedo",
        "green",
        "lst",
        "water",
        "rsd",
        "nir",
    }


def setup_data(georef: bool = False) -> xr.Dataset:
    """
    Create test data
    """
    georef = True
    lat = np.linspace(42.0, 43.0, num=11)
    lon = np.linspace(0.6, 1.4, num=9)
    crs = CRS(4326)
    transform = rio.transform.from_bounds(0.5, 42.0, 1.5, 43.0, 11, 11)
    data = xr.Dataset(
        data_vars={
            "band1": (
                ["lat", "lon"],
                np.random.uniform(low=0, high=10, size=(11, 9)),
            ),
            "band2": (
                ["lat", "lon"],
                np.random.uniform(low=0, high=1, size=(11, 9)),
            ),
        },
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
        },
        attrs={"description": "Test data"},
    )
    if georef:
        data = data.assign_attrs({"crs": crs, "transform": transform})
    return data


@pytest.mark.parametrize(
    ("georef", "separated", "expected"),
    [
        pytest.param(True, False, 1),
        pytest.param(True, True, 2),
        pytest.param(False, False, 1),
        pytest.param(False, True, 2),
    ],
)
def test_write_data(georef, separated, expected, tmp_path) -> None:
    """
    Test write data
    """
    d = tmp_path / "test_io"
    d.mkdir()
    data = setup_data(georef)
    io.write_dataset(data, "test.tif", d, separated)
    files = list(d.glob("**/*.tif"))
    assert len(files) == expected
