# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from pydantic import ValidationError
from pyproj import CRS

from evaspa.common import io


@pytest.mark.unit
def test_read_data_from_file() -> None:
    """
    Test read data from a file
    """
    input_path = os.path.join("tests", "data", "modis_test_full.tif")
    xarr = io.read_data_from_file(input_path)
    assert xarr.sizes["x"] == 145
    assert xarr.sizes["y"] == 145
    assert set(xarr.data_vars) == {
        "ta",
        "tdp",
        "rld",
        "emis",
        "lai",
        "albedo",
        "lst",
        "rsd",
        "height",
        "aspect",
        "slope",
        "fcover",
        "ndvi",
    }


@pytest.mark.unit
def test_read_data() -> None:
    """
    Test read data
    """
    input_path = os.path.join("tests", "data", "modis_dir")
    xarr = io.read_data(input_path)
    assert xarr.sizes["x"] == 145
    assert xarr.sizes["y"] == 145
    assert set(xarr.data_vars) == {
        "ta",
        "tdp",
        "rld",
        "emis",
        "lai",
        "albedo",
        "lst",
        "rsd",
        "height",
        "aspect",
        "slope",
        "fcover",
        "ndvi",
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


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        {"path": "tests/data/modis_test.tif"},
        {
            "path": "tests/data/modis_test.tif",
            "date": "2018-05-16T10:00:00-00:00",
        },
    ],
)
def test_inputconfig(entry) -> None:
    """
    Test InputConfig
    """
    io.InputConfig.model_validate(entry)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "exception"),
    [
        pytest.param({"foo": "foo"}, pytest.raises(ValidationError)),
        pytest.param(
            {"path": "foo"}, pytest.raises(OSError, match="Path not found")
        ),
        pytest.param(
            {"path": "tests/data/modis_test.tif", "date": "foo"},
            pytest.raises(ValidationError),
        ),
    ],
)
def test_inputconfig_exc(entry, exception) -> None:
    """
    Test InputConfig with error
    """
    with exception:
        io.InputConfig.model_validate(entry)


@pytest.mark.unit
def test_outputconfig(tmp_path) -> None:
    """
    Test OutputConfig
    """
    d = Path(tmp_path) / "out"
    output = {"path": str(d)}
    io.OutputConfig.model_validate(output)
    assert d.exists()
