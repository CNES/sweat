# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from pydantic import ValidationError
from pyproj import CRS

from sweat.common import io


@pytest.mark.unit
def test_read_data_from_file() -> None:
    """
    Test read data from a file
    """
    input_path = os.path.join("tests", "data", "data_test_20230303T112730.tif")
    xarr = io.read_data_from_file(input_path)
    assert xarr.sizes["x"] == 131
    assert xarr.sizes["y"] == 134
    assert len(xarr) == 26
    assert sorted(cast(list[str], xarr.data_vars)) == sorted(
        [
            "albedo",
            "aspect",
            "blue",
            "cloud",
            "daily_msg",
            "emis",
            "fcover",
            "fdiff_msg",
            "green",
            "height",
            "lai",
            "lst",
            "ndvi",
            "nir",
            "qa",
            "red",
            "rld_era5",
            "rld_msg",
            "rsd_era5",
            "rsd_msg",
            "slope",
            "swir",
            "swir2",
            "ta",
            "tdp",
            "water",
        ]
    )


@pytest.mark.unit
def test_read_data() -> None:
    """
    Test read data
    """
    input_path = os.path.join("tests", "data", "data_dir")
    xarr = io.read_data(input_path)
    assert xarr.sizes["x"] == 131
    assert xarr.sizes["y"] == 134
    assert len(xarr) == 26
    assert sorted(cast(list[str], xarr.data_vars)) == sorted(
        [
            "albedo",
            "aspect",
            "blue",
            "cloud",
            "daily_msg",
            "emis",
            "fcover",
            "fdiff_msg",
            "green",
            "height",
            "lai",
            "lst",
            "ndvi",
            "nir",
            "qa",
            "red",
            "rld_era5",
            "rld_msg",
            "rsd_era5",
            "rsd_msg",
            "slope",
            "swir1",
            "swir2",
            "ta",
            "tdp",
            "water",
        ]
    )


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
        {
            "path": "tests/data/data_test_20230303T112730.tif",
        },
        {
            "path": "tests/data/data_test_20230303T112730.tif",
            "date": "2023-03-03T11:27:30-00:00",
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
            {"path": "tests/data/data_test_20230303T112730.tif", "date": "foo"},
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
    config = io.OutputConfig.model_validate(output)
    assert d.exists()
    assert config.fmt == "json"


@pytest.mark.unit
@pytest.mark.parametrize(
    "fmt",
    ["json", "yaml", "yml"],
)
def test_outputconfig_valid_extension(fmt, tmp_path) -> None:
    """
    Test OutputConfig with valid extensions
    """
    d = Path(tmp_path) / "out"
    output = {"path": str(d), "fmt": fmt}
    config = io.OutputConfig.model_validate(output)
    assert config.fmt == fmt


@pytest.mark.unit
@pytest.mark.parametrize("fmt", [".json", ".txt", "invalid", ""])
def test_outputconfig_invalid_extension(tmp_path, fmt: str) -> None:
    """
    Test OutputConfig rejects invalid extensions
    """
    d = Path(tmp_path) / "out"
    output = {"path": str(d), "fmt": fmt}
    with pytest.raises(ValidationError):
        io.OutputConfig.model_validate(output)
