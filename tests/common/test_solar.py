# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
import rasterio as rio
import rasterio.warp
import xarray as xr
from dateutil import tz
from pyproj import CRS
from sensorsio.utils import bb_transform

from evaspa.common import solar


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


@pytest.mark.unit
def test_to_latlon():
    """
    Test to_latlon function
    """
    x = np.linspace(739122, 740122, num=10)
    y = np.linspace(5064871, 5074871, num=20)
    x_grid, y_grid = np.meshgrid(x, y)
    lat, lon = solar._to_lonlat(crs=CRS(32631), x=x_grid, y=y_grid)  # noqa: SLF001
    assert x_grid.shape == lon.shape
    assert y_grid.shape == lat.shape


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "lat", "lon", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 1, 12, 16, 00, 00, tzinfo=dt.timezone.utc),
            42,
            0,
            dt.datetime(
                2025, 1, 12, 17, 00, 00, tzinfo=tz.gettz("Europe/Madrid")
            ),
        ),
        pytest.param(
            dt.datetime(2025, 1, 12, 16, 00, 00, tzinfo=dt.timezone.utc),
            42,
            -80,
            dt.datetime(
                2025, 1, 12, 11, 00, 00, tzinfo=tz.gettz("America/New_York")
            ),
        ),
    ],
)
def test_to_localtime(date, lat, lon, expected):
    """
    Test to_localtime function
    """
    res = solar._to_localtime(date, np.deg2rad(lat), np.deg2rad(lon))  # noqa: SLF001
    assert res == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "expected"),
    [
        pytest.param(dt.date(2025, 6, 25), 3.012),
        pytest.param(dt.date(2024, 6, 24), 3.004),
    ],
)
def test_day_angle(date, expected):
    """
    Test day angle function function
    """
    res = solar._day_angle(date)  # noqa: SLF001
    np.testing.assert_almost_equal(res, expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 25, 12, 0, 0, tzinfo=dt.timezone.utc), 3.012
        ),
        pytest.param(
            dt.datetime(2024, 6, 24, 12, 0, 0, tzinfo=dt.timezone.utc), 3.004
        ),
        pytest.param(
            dt.datetime(2024, 6, 24, 10, 0, 0, tzinfo=dt.timezone.utc), 3.003
        ),
    ],
)
def test_fractional_year_angle(date, expected):
    """
    Test fractional year angle function function
    """
    res = solar._fractional_year_angle(date)  # noqa: SLF001
    np.testing.assert_almost_equal(res, expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "expected"),
    [
        pytest.param(dt.date(2025, 1, 8), -6.70),
        pytest.param(dt.date(2025, 6, 5), 1.72),
        pytest.param(
            dt.datetime(2025, 6, 25, 2, 50, 0, tzinfo=dt.timezone.utc), -2.28
        ),
    ],
)
def test_equation_of_time_milne(date, expected):
    """
    Test equation of time function
    """
    tc = solar._equation_of_time_milne(date)  # noqa: SLF001
    np.testing.assert_almost_equal(tc, expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 25, 6, 50, 0, tzinfo=dt.timezone.utc), -2.15
        ),
    ],
)
def test_equation_of_time_noaa(date, expected):
    """
    Test equation of time function
    """
    tc = solar._equation_of_time_noaa(date)  # noqa: SLF001
    np.testing.assert_almost_equal(tc, expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "lon", "lat", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 4, 10, 30, 00, tzinfo=dt.timezone.utc),
            1.0,
            43.1,
            -21.03,
        ),
        pytest.param(
            dt.datetime(2025, 12, 10, 17, 50, 00, tzinfo=dt.timezone.utc),
            -74.0,
            40.5,
            15.11,
        ),
    ],
)
def test_hour_angle(date, lon, lat, expected):
    """
    Test hour angle function
    """
    hour_angle = solar._hour_angle(date, np.deg2rad(lon), np.deg2rad(lat))  # noqa: SLF001
    np.testing.assert_almost_equal(np.rad2deg(hour_angle), expected, decimal=2)


@pytest.mark.unit
def test_declination_angle():
    """
    Test declination angle function
    """
    dates = np.array(
        [dt.date(2025, 1, 1) + dt.timedelta(days=d) for d in range(365)]
    )
    res = np.array([solar._declination_angle(date) for date in dates])  # noqa: SLF001
    expected = np.deg2rad(
        23.45 * np.sin(2 * np.pi / 365.0 * (np.arange(1, 366) + 284))
    )
    np.testing.assert_array_almost_equal(res, expected, decimal=0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "expected"),
    [
        pytest.param(dt.date(2025, 1, 1), 1.03505),
        pytest.param(dt.date(2024, 4, 1), 1.00082),
    ],
)
def test_sun_earth_distance(date, expected):
    """
    Test Sun-Earth distance function
    """
    res = solar._sun_earth_distance(date)  # noqa: SLF001
    np.testing.assert_approx_equal(res, expected, significant=4)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "lat", "expected"),
    [
        pytest.param(dt.date(2025, 3, 21), 45.0, -1.57),
        pytest.param(dt.date(2025, 1, 1), 70.0, 0.0),
        pytest.param(dt.date(2025, 6, 29), 70.0, -np.pi),
        pytest.param(
            dt.date(2025, 1, 10),
            np.array([[30.0, 70.0], [-70.0, 0.0]]),
            np.array([[-1.33, -0.0], [-np.pi, -1.57]]),
        ),
    ],
)
def test_sunrise_angle(date, lat, expected):
    """
    Test sunrise angle function
    """
    res = solar._sunrise_angle(date, np.deg2rad(lat))  # noqa: SLF001
    np.testing.assert_array_almost_equal(res, expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 2, 4, 11, 0, 0, tzinfo=dt.timezone.utc),
            1.0,
            45.0,
            None,
            (63.1, 161.24),
        ),
        pytest.param(
            dt.datetime(2025, 2, 4, 11, 0, 0, tzinfo=dt.timezone.utc),
            1.0,
            45.0,
            CRS(4326),
            (63.1, 161.24),
        ),
    ],
)
def test_compute_sun_angles(date, x, y, crs, expected):
    """
    Test sun angles function
    """
    sza, saa = solar.compute_sun_angles(date, x, y, crs)
    np.testing.assert_approx_equal(sza, expected[0], significant=2)
    np.testing.assert_approx_equal(saa, expected[1], significant=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "sza", "saa", "slope", "aspect", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
            0,
            0,
            0,
            0,
            1322.49,
        ),
        pytest.param(
            dt.datetime(2025, 6, 21, 17, 0, 0, tzinfo=dt.timezone.utc),
            0,
            90,
            0,
            0,
            1322.49,
        ),
        pytest.param(
            dt.datetime(2025, 6, 21, 17, 0, 0, tzinfo=dt.timezone.utc),
            30,
            90,
            0,
            0,
            1145.31,
        ),
        pytest.param(
            dt.datetime(2025, 6, 21, 17, 0, 0, tzinfo=dt.timezone.utc),
            30,
            90,
            20,
            90,
            1302.4,
        ),
        pytest.param(
            dt.datetime(2025, 6, 21, 17, 0, 0, tzinfo=dt.timezone.utc),
            30,
            0,
            60,
            180,
            0,
        ),
    ],
)
def test_compute_toa_solar_radiation_from_angles(
    date, sza, saa, slope, aspect, expected
):
    """
    Test function for computing toa solar radiation from sun angles
    """
    toa = solar.compute_toa_solar_radiation_from_sun_angles(
        date, sza, saa, slope, aspect
    )
    np.testing.assert_allclose(toa, expected, atol=1.0, rtol=0.1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "slope", "aspect", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 4, 17, 0, 0, tzinfo=dt.timezone.utc),
            0.0,
            30.0,
            CRS(4326),
            0,
            0,
            516.74,
        ),
        pytest.param(
            dt.datetime(2025, 6, 4, 17, 0, 0, tzinfo=dt.timezone.utc),
            0.0,
            30.0,
            CRS(4326),
            20,
            270,
            894.08,
        ),
    ],
)
def test_compute_toa_solar_radiation(date, x, y, crs, slope, aspect, expected):
    """
    Test function for computing toa solar radiation for a given position
    """
    toa = solar.compute_toa_solar_radiation(date, x, y, crs, slope, aspect)
    np.testing.assert_approx_equal(toa, expected, significant=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "slope", "aspect", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 4, 17, 0, 0, tzinfo=dt.timezone.utc),
            0.0,
            30.0,
            CRS(4326),
            0,
            0,
            516.74,
        ),
        pytest.param(
            dt.datetime(2025, 6, 4, 17, 0, 0, tzinfo=dt.timezone.utc),
            0.0,
            30.0,
            CRS(4326),
            20,
            270,
            894.08,
        ),
    ],
)
def test_compute_toa_solar_radiation_from_hour_angle(
    date, x, y, crs, slope, aspect, expected
):
    """
    Test function for computing toa solar radiation for a given position with hour angle
    """
    toa = solar.compute_toa_solar_radiation_from_hour_angle(
        date, x, y, crs, slope, aspect
    )
    np.testing.assert_approx_equal(toa, expected, significant=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 2, 4, 11, 0, 0, tzinfo=dt.timezone.utc),
            1,
            45,
            CRS(4326),
            15214985.74,
        ),
    ],
)
def test_toa_daily_irradiance(date, x, y, crs, expected):
    """
    Test function for compiting toa daily irradiance
    """
    toa_daily = solar._toa_daily_irradiance(date, x, y, crs)  # noqa: SLF001
    np.testing.assert_approx_equal(toa_daily, expected, significant=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "slope", "aspect", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 2, 4, 11, 0, 0, tzinfo=dt.timezone.utc),
            1,
            45,
            CRS(4326),
            0,
            0,
            14881235.786,
        ),
    ],
)
def test_compute_daily_toa_solar_radiation(
    date, x, y, crs, slope, aspect, expected
):
    """
    Test function for computing toa daily solar radiation for a given position
    """
    toa = solar.compute_daily_toa_solar_radiation(
        date, x, y, crs, slope, aspect
    )
    np.testing.assert_approx_equal(toa, expected, significant=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "slope", "aspect", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 2, 4, 11, 0, 0, tzinfo=dt.timezone.utc),
            1,
            45,
            CRS(4326),
            0,
            0,
            14881235.786,
        ),
    ],
)
def test_compute_daily_toa_solar_radiation_from_hour_angle(
    date, x, y, crs, slope, aspect, expected
):
    """
    Test function for computing toa daily solar radiation for a given position (from hour angle)
    """
    toa = solar.compute_daily_toa_solar_radiation_from_hour_angle(
        date, x, y, crs, slope, aspect
    )
    np.testing.assert_approx_equal(toa, expected, significant=2)


@pytest.mark.unit
def test_compute_diffuse_fraction():
    """
    Test function for computing toa daily solar radiation for a given position
    """
    date = dt.datetime(2025, 2, 4, 11, 0, 0, tzinfo=dt.timezone.utc)
    data = setup_dataset(
        variables=["rsd"],
        date=date,
        size=(1, 1),
        min_value=1200,
        max_value=1200,
    )
    fdiff = solar.compute_diffuse_fraction(date, data["rsd"])
    np.testing.assert_almost_equal(fdiff.data, 0.24, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.timezone.utc),
            -74,
            40,
            None,
            25416.05,
        ),
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.timezone.utc),
            -74,
            40,
            4326,
            25416.05,
        ),
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.timezone.utc),
            585360,
            4428236,
            32618,
            25416.05,
        ),
    ],
)
def test_convert_to_local_time(date, x, y, crs, expected):
    """
    Test function for converting to local time
    """
    res = solar.convert_to_local_time(date, x, y, crs)
    np.testing.assert_almost_equal(res, expected, decimal=0)
