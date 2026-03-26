# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions for sun angles computation, top of
atmosphere solar radiation
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import xarray as xr
from pyproj import CRS, Transformer
from rasterio import warp
from timezonefinder import TimezoneFinder

from sweat.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Constant
SOLAR_FLUX = 1367  # W.m-2


def _to_lonlat(
    crs: CRS, x: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Convert UTM grid to lon/lat grids
    """
    size = np.array(x).shape
    lon, lat = warp.transform(
        crs, CRS(4326), np.array(x).flatten(), np.array(y).flatten()
    )
    return np.array(lon).reshape(size), np.array(lat).reshape(size)


def _to_localtime(date: dt.datetime, lat: float, lon: float) -> dt.datetime:
    """
    Convert UTC time to local time

    Parameters
    ----------
    date: dt.datetime
       Date
    lat: float
       Latitude (in radians)
    lon: float
       Longitude (in radians)

    Returns
    -------
    day_angle : np.array
        Day angle
    """
    # if necessary
    if date.tzinfo is None:
        msg = "Date with timezone, use UTC timezone"
        logger.warning(msg)
        date = date.replace(tzinfo=ZoneInfo("UTC"))
    # Get time zone
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=np.rad2deg(lon), lat=np.rad2deg(lat))
    # Convert to new time zone
    return date.astimezone(ZoneInfo(str(tz)))


def _is_leap_year(year) -> bool:
    """
    Determine whether a year is a leap year.

    Parameters
    ----------
    year: int
       Year

    Returns
    -------
    is_leap_year : bool
    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _to_date_of_year(date: dt.datetime) -> int:
    """
    Function to convert datetime to day of year
    """
    return date.timetuple().tm_yday


def _day_angle(date: dt.datetime, offset: int = 1) -> float:
    """
    Compute the day angle for the Earth's orbit around the Sun.

    Parameters
    ----------
    date: dt.datetime
       Date
    offset: int
       Offset

    Returns
    -------
    day_angle : np.array
        Day angle (in radians)
    """
    day = _to_date_of_year(date)
    days_in_year = 366 if _is_leap_year(date.year) else 365
    return (2.0 * np.pi / days_in_year) * (day - offset)


def _fractional_year_angle(date: dt.datetime) -> float:
    """
    Compute the fractional year angle for the Earth's orbit around the Sun.

    Parameters
    ----------
    date: dt.datetime
       Date

    Returns
    -------
    fractional_year_angle : np.array
        Fractional year angle (in radians)
    """
    day = _to_date_of_year(date)
    days_in_year = 366 if _is_leap_year(date.year) else 365
    return (2.0 * np.pi / days_in_year) * (day - 1 + (date.hour - 12) / 24)


def _equation_of_time_milne(date: dt.datetime) -> float:
    """
    Compute equation of time

    Notes
    -----------
    The method used comes from
    R. M. Milne, Note on the Equation, of Time,
    The Mathematical Gazette, vol. 10, no. 155,
    pp. 372 - 375, 1921.
    See PVCDROM a website by Solar Power Lab at Arizona State
    University (ASU)
    http://www.pveducation.org/pvcdrom/2-properties-sunlight/solar-time

    Parameters
    ----------
    date : dt.datetime
        Date

    Returns
    -------
    equation_of_time : float
        Time correction factor (in minutes)
    """
    # day angle relative to Vernal Equinox, typically March 22 (day number 81)
    bday = _day_angle(date, offset=81)
    return float(
        9.87 * np.sin(2.0 * bday) - 7.53 * np.cos(bday) - 1.5 * np.sin(bday)
    )


def _equation_of_time_noaa(date: dt.datetime) -> float:
    """
    Compute equation of time

    Notes
    -----------
    The method used comes from NOAA
    https://gml.noaa.gov/grad/solcalc/solareqns.PDF

    Parameters
    ----------
    date : dt.datetime
        Date

    Returns
    -------
    equation_of_time : float
        Time correction factor (in minutes)
    """
    frac_year = _fractional_year_angle(date)
    return float(
        229.18
        * (
            0.000075
            + 0.001868 * np.cos(frac_year)
            - 0.032077 * np.sin(frac_year)
            - 0.014615 * np.cos(2 * frac_year)
            - 0.040849 * np.sin(2 * frac_year)
        )
    )


def _hour_angle(
    time: dt.datetime, lon: npt.ArrayLike, lat: npt.ArrayLike
) -> npt.NDArray:
    """
    Compute hour angle in local solar time.

    Notes
    -----
    Zero correspond to local solar noon.

    Parameters
    ----------
    time : dt.datetime
        Date time
    lon : np.array_like (in radians)
        Longitude
    lat : np.array_like
        Latitude (in radians)

    Returns
    -------
    hour_angle: np.array
        Hour angle (in radians)
    """
    # times must be localized
    if time.tzinfo is None:
        msg = "Time must be localized"
        raise ValueError(msg)
    # convert to np.array
    lat_rad = np.array(lat)
    lon_rad = np.array(lon)
    # Local time
    local_time = _to_localtime(time, lat_rad.mean(), lon_rad.mean())
    # local standard meridian time
    delta_utc = local_time.utcoffset().total_seconds() // 3600  # type: ignore
    lstm = 15 * delta_utc
    # time correction factor (in minutes)
    tc = 4 * (np.rad2deg(lon_rad) - lstm) + _equation_of_time_milne(time)
    # time after local midnight
    midnight = dt.datetime.combine(
        local_time.date(), dt.time(0), tzinfo=local_time.tzinfo
    )
    delta_time = local_time - midnight
    # local solar time (in hours)
    f = np.vectorize(
        lambda x: (
            (delta_time + dt.timedelta(hours=x) / 60.0).total_seconds() / 3600
        )
    )
    lst = f(tc)
    return np.deg2rad(15.0 * (lst - 12.0))


def convert_to_local_time(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
) -> npt.NDArray:
    """
    Convert time from UTM to local solar time (in seconds)

    Notes
    -----
    See https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time

    Parameters
    ----------
    date: np.array_like
        List of dates
    x : np.array_like
        X coordinate / Longitude (in degrees)
    y : np.array_like
        Y coordinate / Latitude (in degrees)
    crs : pyproj.CRS
        Coordinate Reference System

    Returns
    -------
    local_time: np.array
        Local solar time
    """
    # Convert to lat/lon
    if crs is None:
        crs = CRS(4326)
    x_grid, y_grid = np.meshgrid(x, y)
    if np.isscalar(x) and np.isscalar(y):
        x_grid = x  # type: ignore
        y_grid = y  # type: ignore
    # Convert to lat/lon
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    # Apply transformation to the grid
    lon, lat = transformer.transform(x_grid, y_grid)
    # Local time
    local_time = _to_localtime(
        date, np.deg2rad(np.mean(lat)), np.deg2rad(np.mean(lon))
    )
    # local standard meridian time
    delta_utc = local_time.utcoffset().total_seconds() // 3600  # type: ignore
    lstm = 15 * delta_utc
    # time correction factor (in minutes)
    tc = 4 * (lon - lstm) + _equation_of_time_milne(date)
    # time after local midnight
    midnight = dt.datetime.combine(
        local_time.date(), dt.time(0), tzinfo=local_time.tzinfo
    )
    delta_time = local_time - midnight
    # local solar time (in hours)
    f = np.vectorize(
        lambda x: (
            (delta_time + dt.timedelta(hours=x) / 60.0).total_seconds() / 3600
        )
    )
    time_ls = f(tc)
    return time_ls * 3600


def _sun_zenith_angle(
    lat: npt.ArrayLike, hour_angle: npt.ArrayLike, decl_angle: npt.ArrayLike
) -> npt.NDArray:
    """
    Compute the sun zenith angle with an analytical
    expression based on spherical troginometry.

    Notes
    -----
    .. See `PVCDROM: Elevation Angle
       <https://www.pveducation.org/pvcdrom/properties-of-sunlight/
       elevation-angle>`_

    Parameters
    ----------
    lat : np.array_like
        Latitude (in radians)
    hour_angle : np.array_like
        Hour angle (in radians)
    decl_angle : np.array_like
        Declination angle (in radians)

    Returns
    -------
    sza : np.array
        Solar zenith angle (in degrees)
    """
    return np.arccos(
        np.sin(decl_angle) * np.sin(lat)
        + np.cos(decl_angle) * np.cos(lat) * np.cos(hour_angle)
    )


def _sun_azimuth_angle(
    sza: npt.ArrayLike,
    lat: npt.ArrayLike,
    hour_angle: npt.ArrayLike,
    decl_angle: npt.ArrayLike,
) -> npt.NDArray:
    """
    Compute the sun azimuth angle with an analytical
    expression based on spherical troginometry.

    Notes
    -----
    .. See `PVCDROM: Azimuth Angle
       <https://www.pveducation.org/pvcdrom/properties-of-sunlight/
       azimuth-angle>`_

    Parameters
    ----------
    sza : np.array_like
        Sun Zenith Angle (in radians)
    lat : np.array_like
        Latitude (in radians)
    hour_angle : np.array_like
        Hour angle (in radians)
    decl_angle : np.array_like
        Declination angle (in radians)

    Returns
    -------
    saa : np.array
        Solar azimuth angle (in radians)
    """
    elevation_angle = np.pi / 2 - np.array(sza)
    cos = (
        np.sin(decl_angle) * np.cos(lat)
        - np.cos(decl_angle) * np.sin(lat) * np.cos(hour_angle)
    ) / np.cos(elevation_angle)
    cos = np.where(cos > 1.0, 1.0, cos)
    cos = np.where(cos < -1.0, -1.0, cos)
    saa = np.arccos(cos)
    return np.where(np.array(hour_angle) > 0, 2 * np.pi - saa, saa)


def _declination_angle(date: dt.datetime) -> float:
    """
    Compute declination in radians
    """
    # Day angle
    gamma = _day_angle(date)
    # Solar declinaison given by Spencer, J. W. (1971).
    # Fourier series representation of the position
    # of the sun. Search, 2(5), 172-172.
    return (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )


def _sun_earth_distance(date: dt.datetime) -> float:
    """
    Compute the Sun-Earth distance factor at the given day.

    Notes
    -----
    The Sun-Earth distance factor is given by
    Spencer, J. W. (1971). Fourier series representation
    of the position of the sun. Search, 2(5), 172-172.

    Parameters
    ----------
    date: dt.datetime
        Date

    Returns
    -------
    factor: float
        Sun-Earth distance factor
    """
    gamma = _day_angle(date)
    return (
        1.00011
        + 0.034221 * np.cos(gamma)
        + 0.00128 * np.sin(gamma)
        + 0.000719 * np.cos(2 * gamma)
        + 0.000077 * np.sin(2 * gamma)
    )


def _sunrise_angle(date: dt.datetime, lat: npt.ArrayLike) -> npt.NDArray:
    """
    Estimate sunrise angle (in radians)

    Parameters
    ----------
    date: dt.datetime
        Date
    lat : np.array_like
        Latitude (in radians)

    Returns
    -------
    sunrise: np.array
        Sunrise time from midnight (in hours)
    """
    decl_angle = _declination_angle(date)
    cos = -np.sin(lat) * np.sin(decl_angle) / np.cos(lat) / np.cos(decl_angle)

    def func(cos_angle: float) -> float:
        if cos_angle > 1:
            # The sun never rises on this location (on the specified date)
            return 0.0
        if cos_angle < -1:
            # The sun never sets on this location (on the specified date)
            return np.pi
        return np.arccos(cos_angle)

    vfunc = np.vectorize(func)
    return -vfunc(cos)


def compute_sun_angles(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the sun zenith angle with an analytical
    expression based on spherical troginometry.

    Notes
    -----
    .. See `PVCDROM: Elevation Angle
       <https://www.pveducation.org/pvcdrom/properties-of-sunlight/
       elevation-angle>`_

    Parameters
    ----------
    date: dt.datetime
        Date
    x : np.array_like
        X coordinate / Longitude (in degrees)
    y : np.array_like
        Y coordinate / Latitude (in degrees)
    crs : pyproj.CRS
        Coordinate Reference System

    Returns
    -------
    sza : np.array
        Solar zenith angle (in degrees)
    saa : np.array
        Solar azimuth angle (in degrees)
    """
    if crs is None:
        crs = CRS(4326)
    x_grid, y_grid = np.meshgrid(x, y)
    if np.isscalar(x) and np.isscalar(y):
        x_grid = x  # type: ignore
        y_grid = y  # type: ignore
    lon, lat = _to_lonlat(crs, x_grid, y_grid)  # type: ignore
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    hour_angle = _hour_angle(date, lon_rad, lat_rad)
    decl_angle = _declination_angle(date)
    sza_rad = _sun_zenith_angle(lat_rad, hour_angle, decl_angle)
    saa_rad = _sun_azimuth_angle(sza_rad, lat_rad, hour_angle, decl_angle)
    return np.rad2deg(sza_rad), np.rad2deg(saa_rad)


def compute_toa_solar_radiation_from_sun_angles(
    date: dt.datetime,
    sza: npt.ArrayLike,
    saa: npt.ArrayLike,
    slope: npt.ArrayLike | None = None,
    aspect: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Compute the instant solar radiation (no cloud, no atmosphere)
    for a given position to strictly horizontal land surfaces.

    Notes
    -----
    The formula is given by
    E_TOA = E0 x v(day) x cos(theta)
    with
        - E0 is the solar constant corresponding to the
        average TOA flux over the whole year
        - v(day) is the Sun-Earth distance factor at the day
        of observation
        - cos(theta) is the cosine of the solar incidence angle
        relative to the normal to the land surface. If slope and
        aspect are provided, they are taken into account.
    See Richard G. Allen, Ricardo Trezza, Masahiro Tasumi,
    Analytical integrated functions for daily solar radiation on slopes,
    Agricultural and Forest Meteorology,
    Volume 139, Issues 1-2, 2006,

    Parameters
    ----------
    date: dt.datetime
        Date
    sza: np.array_like
        Sun Zenith angle (in degrees)
    saa: np.array_like
        Sun Azimuth Angle
    slope: np.array_like
        Slope
    aspect: np.array_like
        Aspect

    Returns
    -------
    toa_irradiance: np.array
        TOA instant irradiance (W.m-2)
    """
    # Convert in radians
    sza_rad = np.deg2rad(sza)
    saa_rad = np.deg2rad(saa)
    # Slope and aspect
    slope_rad = np.zeros_like(sza_rad) if slope is None else np.deg2rad(slope)
    aspect_rad = (
        np.zeros_like(sza_rad) if aspect is None else np.deg2rad(aspect)
    )

    cos_theta = np.sin(sza_rad) * np.sin(slope_rad) * np.cos(
        aspect_rad - saa_rad
    ) + np.cos(sza_rad) * np.cos(slope_rad)
    cos_theta = np.where(cos_theta < 0, 0, cos_theta)
    return SOLAR_FLUX * _sun_earth_distance(date) * cos_theta


def compute_toa_solar_radiation_from_hour_angle(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
    slope: npt.ArrayLike | None = None,
    aspect: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Compute the instant solar radiation (no cloud, no atmosphere)
    for a given position to strictly horizontal land surfaces.

    Notes
    -----
    The formula is given by
    E_TOA = E0 x v(day) x cos(theta)
    with
        - E0 is the solar constant corresponding to the
        average TOA flux over the whole year
        - v(day) is the Sun-Earth distance factor at the day
        of observation
        - cos(theta) is the cosine of the solar incidence angle
        relative to the normal to the land surface. If slope and
        aspect are provided, they are taken into account.
        cos(theta) = sin(delta)sin(lat)cos(slope)
                - sin(delta)cos(lat)sin(slope)cos(aspect)
                + cos(delta)cos(lat)cos(slope)cos(w)
                + cos(delta)sin(lat)sin(slope)cos(aspect)cos(w)
                + cos(delta)sin(aspect)sin(slope)sin(w)
    where delta is the declination of the earth (positive during
    northern hemisphere summer), lat is the latitude of the
    pixel (positive for the northern hemisphere and negative
    for the southern hemisphere), slope is the surface slope, where
    slope = 0 for horizontal and slope = pi/2 radians for vertical slope
    (slope is always positive and represents the slope in any
    direction), and aspect is the surface aspect angle, where aspect = 0
    for slopes oriented due south, aspect = -pi/2 radians for
    slopes oriented due east, aspect = +pi/2 radians for slopes
    oriented due west and aspect = +/-pi radians for slopes
    oriented due north. Parameter w is the hour angle,
    where w = 0 at solar noon, w is negative in morning
    and w is positive in afternoon.
    See Richard G. Allen, Ricardo Trezza, Masahiro Tasumi,
    Analytical integrated functions for daily solar radiation on slopes,
    Agricultural and Forest Meteorology,
    Volume 139, Issues 1-2, 2006,

    Parameters
    ----------
    date: dt.datetime
        Date
    x: np.array_like
        X coordinates
    y: np.array_like
        Y coordinates
    crs: pyproj.CRS
        Corrdinate Reference System
    slope: np.array_like
        Slope
    aspect: np.array_like
        Aspect

    Returns
    -------
    toa_irradiance: np.array
        TOA instant irradiance (W.m-2)
    """
    # Convert in radians
    if crs is None:
        crs = CRS(4326)
    x_grid, y_grid = np.meshgrid(x, y)
    if np.isscalar(x) and np.isscalar(y):
        x_grid = x  # type: ignore
        y_grid = y  # type: ignore
    lon, lat = _to_lonlat(crs, x_grid, y_grid)  # type: ignore
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    # Hour ange
    hour_angle = _hour_angle(date, lon_rad, lat_rad)
    # Declination angle
    decl_angle = _declination_angle(date)
    # Slope and aspect
    slope_rad = np.zeros_like(lat_rad) if slope is None else np.deg2rad(slope)
    aspect_rad = (
        np.zeros_like(lat_rad) if aspect is None else np.deg2rad(aspect)
    )
    aspect_rad -= np.pi  # convention for aspect

    cos_theta = (
        np.sin(decl_angle) * np.sin(lat_rad) * np.cos(slope_rad)
        - np.sin(decl_angle)
        * np.cos(lat_rad)
        * np.sin(slope_rad)
        * np.cos(aspect_rad)
        + np.cos(decl_angle)
        * np.cos(lat_rad)
        * np.cos(slope_rad)
        * np.cos(hour_angle)
        + np.cos(decl_angle)
        * np.sin(lat_rad)
        * np.sin(slope_rad)
        * np.cos(aspect_rad)
        * np.cos(hour_angle)
        + np.cos(decl_angle)
        * np.sin(aspect_rad)
        * np.sin(slope_rad)
        * np.sin(hour_angle)
    )
    cos_theta = np.where(cos_theta < 0, 0, cos_theta)
    return SOLAR_FLUX * _sun_earth_distance(date) * cos_theta


def compute_toa_solar_radiation(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
    slope: npt.ArrayLike | None = None,
    aspect: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Compute the instant solar radiation (no cloud, no atmosphere)
    for a given position to strictly horizontal land surfaces.

    Notes
    -----
    The formula is given by
    E_TOA = E0 x v(day) x cos(theta)
    with
        - E0 is the solar constant corresponding to the
        average TOA flux over the whole year
        - v(day) is the Sun-Earth distance factor at the day
        of observation
        - cos(theta) is the cosine of the solar incidence angle
        relative to the normal to the land surface. If slope and
        aspect are provided, they are taken into account.
    See Richard G. Allen, Ricardo Trezza, Masahiro Tasumi,
    Analytical integrated functions for daily solar radiation on slopes,
    Agricultural and Forest Meteorology,
    Volume 139, Issues 1-2, 2006

    Parameters
    ----------
    date: dt.datetime
        Date
    x: np.array_like
        X coordinates
    y: np.array_like
        Y coordinates
    crs: pyproj.CRS
        Corrdinate Reference System
    slope: np.array_like
        Slope
    aspect: np.array_like
        Aspect

    Returns
    -------
    toa_irradiance: np.array
        TOA instant irradiance (W.m-2)
    """
    # Convert to lat/lon
    sza, saa = compute_sun_angles(date, x, y, crs)
    return compute_toa_solar_radiation_from_sun_angles(
        date, sza, saa, slope, aspect
    )


def _toa_daily_irradiance(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
) -> npt.NDArray:
    """
    Compute the daily solar radiation (no atmosphere, no cloud)
    for a given position to strictly horizontal land surfaces.

    Notes
    -----
    The formula is given by
    Richard G. Allen, Ricardo Trezza, Masahiro Tasumi,
    Analytical integrated functions for daily solar radiation on slopes,
    Agricultural and Forest Meteorology,
    Volume 139, Issues 1-2, 2006,

    Parameters
    ----------
    date: np.array_like
        List of dates
    x : np.array_like
        X coordinate / Longitude (in degrees)
    y : np.array_like
        Y coordinate / Latitude (in degrees)
    crs : pyproj.CRS
        Coordinate Reference System

    Returns
    -------
    toa_irradiance: float
        TOA daily irradiance (J.m-2.day-1)
    """
    # Convert to lat/lon
    if crs is None:
        crs = CRS(4326)
    x_grid, y_grid = np.meshgrid(x, y)
    if np.isscalar(x) and np.isscalar(y):
        x_grid = x  # type: ignore
        y_grid = y  # type: ignore
    _, lat = _to_lonlat(crs, x_grid, y_grid)  # type: ignore
    # convert angle in radians
    lat = np.deg2rad(lat)
    # Solar declinaison
    delta = _declination_angle(date)
    # Sunrise hour angle
    cosh0 = -np.tan(lat) * np.tan(delta)
    if cosh0 > 1:
        # The sun never rises on this location (on the specified date)
        cosh0 = 0.0
    if cosh0 < -1:
        # The sun never sets on this location (on the specified date)
        cosh0 = np.pi
    h0 = np.arccos(-np.tan(lat) * np.tan(delta))
    return (
        24.0
        * 3600
        * SOLAR_FLUX
        * _sun_earth_distance(date)
        / np.pi
        * (
            h0 * np.sin(lat) * np.sin(delta)
            + np.cos(lat) * np.cos(delta) * np.sin(h0)
        )
    )


def compute_daily_toa_solar_radiation_from_hour_angle(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
    slope: npt.ArrayLike | None = None,
    aspect: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Compute the instant solar radiation (no cloud, no atmosphere)
    for a given position for inclined surfaces having specified slope
    and aspect.

    Notes
    -----
    For a specific time, the formula is given by
    E_TOA = E0 x v(day) x cos(theta)
    with
        - E0 is the solar constant corresponding to the
        average TOA flux over the whole year
        - v(day) is the Sun-Earth distance factor at the day
        of observation
        - cos(theta) is the cosine of the solar incidence angle
        relative to the normal to the land surface. If slope and
        aspect are provided, they are taken into account.
        cos(theta) = sin(delta)sin(lat)cos(slope)
                - sin(delta)cos(lat)sin(slope)cos(aspect)
                + cos(delta)cos(lat)cos(slope)cos(w)
                + cos(delta)sin(lat)sin(slope)cos(aspect)cos(w)
                + cos(delta)sin(aspect)sin(slope)sin(w)
    where delta is the declination of the earth (positive during
    northern hemisphere summer), lat is the latitude of the
    pixel (positive for the northern hemisphere and negative
    for the southern hemisphere), slope is the surface slope, where
    slope = 0 for horizontal and slope = pi/2 radians for vertical slope
    (slope is always positive and represents the slope in any
    direction), and aspect is the surface aspect angle, where aspect = 0
    for slopes oriented due south, aspect = -pi/2 radians for
    slopes oriented due east, aspect = +pi/2 radians for slopes
    oriented due west and aspect = +/-pi radians for slopes
    oriented due north. Parameter w is the hour angle,
    where w = 0 at solar noon, w is negative in morning
    and w is positive in afternoon.
    The formula is intergarted between sunrise and sunset angles.
    See Richard G. Allen, Ricardo Trezza, Masahiro Tasumi,
    Analytical integrated functions for daily solar radiation on slopes,
    Agricultural and Forest Meteorology,
    Volume 139, Issues 1-2, 2006

    Parameters
    ----------
    date: np.array_like
        List of dates
    x : np.array_like
        X coordinate / Longitude (in degrees)
    y : np.array_like
        Y coordinate / Latitude (in degrees)
    crs : pyproj.CRS
        Coordinate Reference System
    slope: np.array_like
        Slope
    aspect: np.array_like
        Aspect

    Returns
    -------
    toa_irradiance: float
        TOA daily irradiance (J.m-2.day-1)
    """
    # Convert to lat/lon
    if crs is None:
        crs = CRS(4326)
    x_grid, y_grid = np.meshgrid(x, y)
    if np.isscalar(x) and np.isscalar(y):
        x_grid = x  # type: ignore
        y_grid = y  # type: ignore
    lon, lat = _to_lonlat(crs, x_grid, y_grid)  # type: ignore
    # convert angle in radians
    lat_rad = np.deg2rad(lat)
    # Slope and aspect
    slope_rad = np.zeros_like(lat) if slope is None else np.deg2rad(slope)
    aspect_rad = np.zeros_like(lat) if aspect is None else np.deg2rad(aspect)
    aspect_rad -= np.pi  # convention for aspect
    # Solar declinaison
    delta = _declination_angle(date)
    # Sunrise and sunset angles (in radians)
    sunrise_angle = _sunrise_angle(date, lat_rad)
    # Integration with 15 minutes interval (15° per hour)
    hstep = np.deg2rad(15 / 4)  # 15 minutes
    nbsteps = np.max(np.ceil(-2.0 * sunrise_angle / hstep).astype(int))
    hstep = -2.0 * sunrise_angle / (nbsteps - 1)
    toa_daily = np.zeros_like(lat_rad)
    h1 = sunrise_angle.copy()
    h2 = sunrise_angle.copy() + hstep
    for _ in np.arange(nbsteps - 2):
        # factor integral cos(theta) between h1 and h2
        cos_theta = (
            np.sin(delta) * np.sin(lat_rad) * np.cos(slope_rad) * hstep
            - np.sin(delta)
            * np.cos(lat_rad)
            * np.sin(slope_rad)
            * np.cos(aspect_rad)
            * hstep
            + np.cos(delta)
            * np.cos(lat_rad)
            * np.cos(slope_rad)
            * (np.sin(h2) - np.sin(h1))
            + np.cos(delta)
            * np.sin(lat_rad)
            * np.sin(slope_rad)
            * np.cos(aspect_rad)
            * (np.sin(h2) - np.sin(h1))
            - np.cos(delta)
            * np.sin(aspect_rad)
            * np.sin(slope_rad)
            * (np.cos(h2) - np.cos(h1))
        )
        toa_daily += np.where(cos_theta < 0, 0, cos_theta)
        h1 += hstep
        h2 += hstep
    return (
        toa_daily * 12.0 * 3600 * SOLAR_FLUX * _sun_earth_distance(date) / np.pi
    )


def compute_daily_toa_solar_radiation(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
    slope: npt.ArrayLike | None = None,
    aspect: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Compute the instant solar radiation (no cloud, no atmosphere)
    for a given position for inclined surfaces having specified slope
    and aspect.

    Notes
    -----
    For a specific time, the formula is given by
    E_TOA = E0 x v(day) x cos(theta)
    with
        - E0 is the solar constant corresponding to the
        average TOA flux over the whole year
        - v(day) is the Sun-Earth distance factor at the day
        of observation
        - cos(theta) is the cosine of the solar incidence angle
        relative to the normal to the land surface. If slope and
        aspect are provided, they are taken into account.
    cos(theta) = cos(sza)cos(slope)
                + sin(sza)sin(slope)cos(aspect-saa)
    where delta is the declination of the earth (positive during
    northern hemisphere summer), lat is the latitude of the
    pixel (positive for the northern hemisphere and negative
    for the southern hemisphere), slope is the surface slope, where
    slope = 0 for horizontal and slope = pi/2 radians for vertical slope
    (slope is always positive and represents the slope in any
    direction), and aspect is the surface aspect angle, where for
    north hemisphere aspect = pi for slopes oriented due south,
    aspect = pi/2 radians for slopes oriented due east,
    aspect = -pi/2 radians for slopes
    oriented due west and aspect = 0 radians for slopes
    oriented due north. Parameter w is the hour angle,
    where w = 0 at solar noon, w is negative in morning
    and w is positive in afternoon.
    The formula is intergarted between sunrise and sunset angles.
    See Richard G. Allen, Ricardo Trezza, Masahiro Tasumi,
    Analytical integrated functions for daily solar radiation on slopes,
    Agricultural and Forest Meteorology,
    Volume 139, Issues 1-2, 2006

    Parameters
    ----------
    date: np.array_like
        List of dates
    x : np.array_like
        X coordinate / Longitude (in degrees)
    y : np.array_like
        Y coordinate / Latitude (in degrees)
    crs : pyproj.CRS
        Coordinate Reference System
    slope: np.array_like
        Slope
    aspect: np.array_like
        Aspect

    Returns
    -------
    toa_irradiance: float
        TOA daily irradiance (J.m-2.day-1)
    """
    # Convert to lat/lon
    if crs is None:
        crs = CRS(4326)
    x_grid, y_grid = np.meshgrid(x, y)
    if np.isscalar(x) and np.isscalar(y):
        x_grid = x  # type: ignore
        y_grid = y  # type: ignore
    _, lat = _to_lonlat(crs, x_grid, y_grid)  # type: ignore
    lat_rad = np.deg2rad(lat)
    # Slope and aspect
    slope_rad = np.zeros_like(lat) if slope is None else np.deg2rad(slope)
    aspect_rad = np.zeros_like(lat) if aspect is None else np.deg2rad(aspect)
    # Solar declinaison
    delta = _declination_angle(date)
    # Sunrise angle (in radians)
    sunrise_angle = np.min(_sunrise_angle(date, lat_rad))
    # Integration with 15 minutes interval (15° per hour)
    hstep = np.deg2rad(15 / 4)  # 15 minutes
    nbsteps = int(np.ceil(-2 * sunrise_angle / hstep))
    hsteps = np.linspace(
        sunrise_angle, -sunrise_angle, num=nbsteps, endpoint=True
    )

    def _partial(h):
        sza_rad = _sun_zenith_angle(lat_rad, h, delta)
        saa_rad = _sun_azimuth_angle(sza_rad, lat_rad, h, delta)
        cos_theta = np.sin(sza_rad) * np.sin(slope_rad) * np.cos(
            aspect_rad - saa_rad
        ) + np.cos(sza_rad) * np.cos(slope_rad)
        return np.where(cos_theta < 0, 0, cos_theta)

    toa_daily = _partial(hsteps[0]) / 2.0
    for h in hsteps[1:-1]:
        toa_daily += _partial(h)
    toa_daily += _partial(hsteps[-1]) / 2.0
    return (
        toa_daily
        * 12.0
        * 3600
        * SOLAR_FLUX
        * _sun_earth_distance(date)
        / np.pi
        * hstep
    )


def _to_fdiff(rsd: float, r0: float, sza: float) -> float:
    """
    Compute fraction of diffuse radiation.

    Notes
    -----
    The computation is based on
    C.J.T. Spitters, H.A.J.M. Toussaint, J. Goudriaan, Separating
    the diffuse and direct component of global radiation and
    its implications for modeling canopy photosynthesis Part I.
    Components of incoming radiation, Agricultural and Forest Meteorology,
    Volume 38, Issues 1-3, 1986.

    Parameters
    ----------
    rsd: float
        Global radiation
    r0: float
        Theoritical radiation
    sza: float
        Sun Zenith Angle (in degrees)

    Returns
    -------
    fdiff: np.array
        Fraction of diffuse radiation
    """
    ratio = rsd / r0
    if ratio <= 0.22:  # noqa:PLR2004
        return 1.0
    elif ratio <= 0.35:  # noqa:PLR2004, RET505
        return 1.0 - 6.4 * (ratio - 0.22) * (ratio - 0.22)
    l_value = 0.847 - 1.61 * np.sin(sza) + 1.04 * np.sin(sza) ** 2
    k_value = (1.47 - l_value) / 1.66
    if ratio <= k_value:
        return 1.47 - 1.66 * ratio
    return l_value


def compute_diffuse_fraction(
    date: dt.datetime,
    rsd: xr.DataArray,
    sza: xr.DataArray | None = None,
    saa: xr.DataArray | None = None,
    crs: CRS | None = None,
) -> xr.DataArray:
    """
    This method computes the fraction of diffuse radiation
    from the global radiation.

    Notes
    -----
    We use the relationship between
    the fraction of diffuse radiation (Rdiff) compared to
    global radiation data (Rsd) and the fraction of global
    radiation data (Rsd) compared to  theoretical radiation (R0),
    as recommended by de Jon (1980) for hourly radiation.

    C.J.T. Spitters, H.A.J.M. Toussaint, J. Goudriaan, Separating
    the diffuse and direct component of global radiation and
    its implications for modeling canopy photosynthesis Part I.
    Components of incoming radiation, Agricultural and Forest Meteorology,
    Volume 38, Issues 1-3, 1986.

    Parameters
    ----------
    date: dt.datetime
        Date
    rsd: xr.DataArray
        Global radiation data
    sza: xr.DataArray
        Sun Zenith Angle (in degrees)
    saa: xr.DataArray
        Sun Azimuth Angle (in degrees)
    crs: CRS
        Coordinate Reference System

    Returns
    -------
    fdiff: np.array
        Fraction of diffuse radiation
    """
    if crs is None:
        crs = CRS(4326)
    # Sun zenith angle
    if sza is None or saa is None:
        sza_arr, saa_arr = compute_sun_angles(
            date, x=rsd.coords["x"], y=rsd.coords["y"], crs=crs
        )
    else:
        sza_arr = sza.data
        saa_arr = saa.data
    # Theoritical radiation
    r0 = compute_toa_solar_radiation_from_sun_angles(
        date=date, sza=sza_arr, saa=saa_arr
    )
    # Compute
    fdiff_func = np.vectorize(_to_fdiff)
    fdiff = fdiff_func(rsd.data, r0, np.deg2rad(sza_arr))
    return rsd.copy(data=fdiff)
