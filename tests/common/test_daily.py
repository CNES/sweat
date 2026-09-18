# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pytest
import rasterio as rio
import rasterio.warp
import xarray as xr
from pydantic import ValidationError
from pyproj import CRS
from sensorsio.utils import bb_transform

from sweat.common import daily


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
    Create a random dataset
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"method": "toa"},
        {"method": "geo"},
        {"method": "toa", "use_topo": True},
        {"method": "geo", "name": "msg"},
    ],
)
def test_dailyconfig(config) -> None:
    """
    Test DailyConfig
    """
    assert daily.DailyConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"method": "foo"},
        {"method": "toa", "foo": True},
    ],
)
def test_dailyconfig_error(config) -> None:
    """
    Test DailyConfig with exception raising
    """
    with pytest.raises(ValidationError):
        daily.DailyConfig.model_validate_json(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "msg_expected", "config_expected"),
    [
        (
            {"method": "geo", "use_topo": True},
            "The 'use_topo' parameter is not used with daily method='geo'",
            {"method": "geo", "name": None},
        ),
        (
            {"method": "toa", "name": "msg"},
            "The 'name' parameter is not used with daily method='toa'",
            {"method": "toa", "use_topo": False},
        ),
    ],
)
def test_dailyconfig_consistency(
    config, msg_expected, config_expected, caplog
) -> None:
    """
    Test DailyConfig with consistency checking between parameters
    """
    caplog.set_level(logging.WARNING)
    res = daily.DailyConfig.model_validate(config)
    assert msg_expected in caplog.text
    assert res.model_dump() == config_expected


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
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.UTC),
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
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.UTC),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=105,
        min_value=95,
        size=(200, 100),
    )
    dem = data[["aspect"]]
    daily.toa_daily_estimate(data, date=data.attrs["date"], dem=dem)
    assert (
        "No DEM information (aspect or slope) to compute topographic "
        "corrections. Topographic corrections are disabled."
    ) in caplog.text


@pytest.mark.functional
def test_toa_daily_estimate_mising_crs():
    """
    Test toa daily estimate function (missing CRS data)
    """
    data = setup_dataset(
        ["var"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.UTC),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=105,
        min_value=95,
        size=(200, 100),
    )
    data.attrs["crs"] = None
    with pytest.raises(
        ValueError,
        match=(
            "Impossible to compute TOA extrapolation "
            "because CRS is missing in the metadata"
        ),
    ):
        daily.toa_daily_estimate(data, date=data.attrs["date"])


@pytest.mark.unit
def test_extrapolate_unknown_method(caplog):
    """
    Test extrapolation function with an unknown method
    """
    data = setup_dataset(
        ["var"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.UTC),
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
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.UTC),
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
        match=(
            "Impossible to extrapolate because the date is missing in metadata"
        ),
    ):
        daily.extrapolate_at_daily_scale(data, method="toa")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variables", "method", "use_topo", "name", "expected"),
    [
        pytest.param(["var1"], "toa", False, None, 3, id="selection_toa"),
        pytest.param(
            ["var1", "var2", "var3"],
            "toa",
            True,
            None,
            5,
            id="all_toa_use_topo",
        ),
        pytest.param(None, "toa", True, None, 5, id="none_toa_use_topo"),
        pytest.param(
            ["var1", "var2", "var3"],
            "geo",
            False,
            None,
            5,
            id="all_geo_default",
        ),
        pytest.param(
            ["var1", "var2", "var3"], "geo", False, "msg", 5, id="all_geo_msg"
        ),
    ],
)
def test_extrapolate_at_daily_scale(
    variables, method, use_topo, name, expected
):
    """
    Test extrapolation function at daily scale
    """
    data = setup_dataset(
        [
            "var1",
            "var2",
            "var3",
            "height",
            "slope",
            "aspect",
            "daily_rsd",
            "rsd",
            "daily_msg",
            "rsd_msg",
        ],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.UTC),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=90,
        min_value=0,
        size=(200, 100),
    )
    res = daily.extrapolate_at_daily_scale(
        data[["var1", "var2", "var3"]],
        variables=variables,
        method=method,
        extra=data,
        use_topo=use_topo,
        name=name,
    )
    assert len(res.data_vars) == expected


@pytest.mark.unit
def test_extrapolate_at_daily_scale_without_dem(caplog):
    """
    Test extrapolation function at daily scale (toa method)
    """
    caplog.clear()
    data = setup_dataset(
        ["var1", "var2", "var3"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.UTC),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=90,
        min_value=0,
        size=(200, 100),
    )
    daily.extrapolate_at_daily_scale(
        data, variables=None, method="toa", extra=None, use_topo=True
    )
    assert (
        "No DEM information to compute topographic corrections. "
        "Topographic corrections are disabled." in caplog.text
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "name", "msg_expected"),
    [
        pytest.param(
            False,
            None,
            "Impossible to extrapolate because no geostationary is providing",
            id="no_data",
        ),
        pytest.param(
            True,
            None,
            r"Geostationary data \(rsd or/and daily_rsd\) is missing",
            id="missing_default_data",
        ),
        pytest.param(
            True,
            "foo",
            r"Geostationary data \(rsd_foo or/and daily_foo\) is missing",
            id="missing_data",
        ),
    ],
)
def test_extrapolate_at_daily_scale_without_data(extra, name, msg_expected):
    """
    Test extrapolation function at daily scale with exception raising
    """
    data = setup_dataset(
        ["var1", "var2", "var3", "daily_msg", "rsd_msg"],
        dt.datetime(2025, 1, 9, 11, 30, 00, tzinfo=dt.UTC),
        wgs84_bounds=(1.0, 43.0, 2.0, 44.0),
        epsg=32631,
        max_value=90,
        min_value=0,
        size=(200, 100),
    )
    with pytest.raises(ValueError, match=msg_expected):
        daily.extrapolate_at_daily_scale(
            data,
            variables=None,
            method="geo",
            extra=data if extra else None,
            use_topo=False,
            name=name,
        )
