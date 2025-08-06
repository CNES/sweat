# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
import rasterio as rio
import rasterio.warp
import xarray as xr
from pyproj import CRS
from sensorsio.utils import bb_transform

from evaspa.common import daily


def setup_dataset(
    variables: list[str],
    date: dt.datetime,
    min_value: float = 0.0,
    max_value: float = 1.0,
    size: tuple[int, int] = (100, 200),
    epsg: int = 4326,
    wgs84_bounds: tuple[float, float, float, float] = (1.0, 43.0, 2.0, 44.0),
    seed: int = 0,
) -> xr.Dataset:
    """
    Create a rondom dataset
    """
    assert len(variables) > 0
    np.random.seed(seed)
    # Coordinate generation
    crs = CRS(epsg)
    bounds = bb_transform(
        str(CRS(4326)), str(crs), rio.coords.BoundingBox(*wgs84_bounds)
    )
    # BoundingBox(xmin, ymin, xmax, ymax)
    x = np.linspace(
        start=bounds.left,
        stop=bounds.right,
        num=size[1],
        dtype=np.float32,
    )
    y = np.linspace(
        start=bounds.bottom,
        stop=bounds.top,
        num=size[0],
        dtype=np.float32,
    )
    # Data generation
    data_vars = {}
    for var in variables:
        data_vars[var] = (
            ["y", "x"],
            np.random.uniform(low=min_value, high=max_value, size=size),
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "y": ("y", y),
            "x": ("x", x),
        },
        attrs={"description": "Test data", "crs": crs, "date": date},
    )


@pytest.mark.functional
@pytest.mark.parametrize(
    ("use_topo"),
    [
        False,
        True,
    ],
)
def test_toa_daily_estimate(use_topo):
    """
    Test toa daily estimate function
    """
    data = setup_dataset(
        ["var", "height", "slope", "aspect"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.timezone.utc),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=105,
        min_value=95,
        size=(200, 100),
    )
    dem = None
    if use_topo:
        dem = data[["height", "slope", "aspect"]]
    daily.toa_daily_estimate(data[["var"]], date=data.attrs["date"], dem=dem)


@pytest.mark.functional
def test_toa_daily_estimate_missing_dem_data(caplog):
    """
    Test toa daily estimate function (missing DEM data)
    """
    caplog.clear()
    data = setup_dataset(
        ["var", "aspect"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.timezone.utc),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=105,
        min_value=95,
        size=(200, 100),
    )
    dem = data[["aspect"]]
    daily.toa_daily_estimate(data, date=data.attrs["date"], dem=dem)
    assert (
        "No DEM information (aspect or slope) to compute topographic corrections. Topographic corrections are disabled."
        in caplog.text
    )


@pytest.mark.functional
def test_toa_daily_estimate_mising_crs():
    """
    Test toa daily estimate function (missing CRS data)
    """
    data = setup_dataset(
        ["var"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.timezone.utc),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=105,
        min_value=95,
        size=(200, 100),
    )
    data.attrs["crs"] = None
    with pytest.raises(
        ValueError,
        match="Impossible to compute TOA extrapolation because CRS is missing in the metadata",
    ):
        daily.toa_daily_estimate(data, date=data.attrs["date"])


@pytest.mark.unit
def test_extrapolate_unknown_method(caplog):
    """
    Test extrapolation function with an unknown method
    """
    data = setup_dataset(
        ["var"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.timezone.utc),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=105,
        min_value=95,
        size=(200, 100),
    )
    res = daily.extrapolate_at_daily_scale(data, method="foo")
    assert "Extrapolation method foo unknown" in caplog.text
    assert data.attrs == res.attrs
    np.testing.assert_array_equal(res["var"], np.nan * np.ones((200, 100)))


@pytest.mark.unit
def test_extrapolate_unavailable_variables(caplog):
    """
    Test extrapolation function with unavailable variables
    """
    data = setup_dataset(
        ["var"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.timezone.utc),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=105,
        min_value=95,
        size=(100, 100),
    )
    daily.extrapolate_at_daily_scale(
        data, variables=["var", "foo"], method="foo"
    )
    assert (
        "Variables ['foo'] not available for daily extrapolation" in caplog.text
    )


@pytest.mark.unit
def test_extrapolate_toa_missing_date():
    """
    Test toa extrapolation function with missing date
    """
    data = setup_dataset(
        ["var"],
        None,
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=105,
        min_value=95,
        size=(100, 100),
    )
    data.attrs["crs"] = None
    with pytest.raises(
        ValueError,
        match="Impossible to extrapolate because the date is missing in metadata",
    ):
        daily.extrapolate_at_daily_scale(data, method="toa")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variables", "method", "use_topo", "expected"),
    [
        pytest.param(["var1"], "toa", False, 1),
        pytest.param(["var1", "var2", "var3"], "toa", True, 3),
        pytest.param(None, "toa", True, 3),
    ],
)
def test_extrapolate_at_daily_scale(variables, method, use_topo, expected):
    """
    Test extrapolation function at daily scale
    """
    data = setup_dataset(
        ["var1", "var2", "var3", "height", "slope", "aspect"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.timezone.utc),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=90,
        min_value=0,
        size=(200, 100),
    )
    dem = None
    if use_topo:
        dem = data[["height", "slope", "aspect"]]
    res = daily.extrapolate_at_daily_scale(
        data[["var1", "var2", "var3"]],
        variables=variables,
        method=method,
        dem=dem,
        use_topo=use_topo,
    )
    assert len(res.data_vars) == expected


@pytest.mark.unit
def test_extrapolate_at_daily_scale_without_dem(caplog):
    """
    Test extrapolation function at daily scale
    """
    caplog.clear()
    data = setup_dataset(
        ["var1", "var2", "var3"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.timezone.utc),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=90,
        min_value=0,
        size=(200, 100),
    )
    daily.extrapolate_at_daily_scale(
        data, variables=None, method="toa", dem=None, use_topo=True
    )
    assert (
        "No DEM information to compute topographic corrections. Topographic corrections are disabled."
        in caplog.text
    )
