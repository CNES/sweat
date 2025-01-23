# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import rasterio as rio
import xarray as xr  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field
from pyproj import CRS
from timezonefinder import TimezoneFinder

from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Constant
SOLAR_FLUX = 1367  # W.m-2


class DailyConfig(BaseModel):
    """
    Daily config for ET processing
    """

    model_config = ConfigDict(extra="forbid")

    method: str = Field(default="toa")


def _to_lonlat(
    crs: CRS, x: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Convert UTM grid to lon/lat grids
    """
    size = np.array(x).shape
    lon, lat = rio.warp.transform(
        crs, CRS(4326), np.array(x).flatten(), np.array(y).flatten()
    )
    return np.array(lon).reshape(size), np.array(lat).reshape(size)


def _to_localtime(date: dt.datetime, lat: float, lon: float) -> dt.datetime:
    """
    Convert UTC time to local time
    """
    # if necessary
    if date.tzinfo is None:
        msg = "Date with timezone, use UTC timezone"
        logger.warning(msg)
        date = date.replace(tzinfo=ZoneInfo("UTC"))
    # Get time zone
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=lon, lat=lat)
    # Convert to new time zone
    return date.astimezone(ZoneInfo(str(tz)))


def _day_angle(day: npt.ArrayLike, offset: int = 1) -> npt.NDArray:
    """
    Description
    -----------
    Compute the day angle for the Earth's orbit around the Sun.

    Parameters
    ----------
    day: np.array_like
       Day of the year ranging form 1 to 365 or 366

    Returns
    -------
    day_angle : np.array
        Day angle
    """
    return (2.0 * np.pi / 365.0) * (np.array(day) - offset)


def _to_dateofyear(date: dt.datetime) -> int:
    """
    Function to convert datetime to day of year
    """
    return date.timetuple().tm_yday


def _equation_of_time(day: int) -> float:
    """
    Description
    -----------
    Equation of time from
    R. M. Milne, Note on the Equation, of Time,
    The Mathematical Gazette, vol. 10, no. 155,
    pp. 372 - 375, 1921.
    See PVCDROM a website by Solar Power Lab at Arizona State
    University (ASU)
    http://www.pveducation.org/pvcdrom/2-properties-sunlight/solar-time

    Parameters
    ----------
    day : int
        Day of the year

    Returns
    -------
    equation_of_time : float
        Time correction factor (in minutes)
    """
    # day angle relative to Vernal Equinox, typically March 22 (day number 81)
    bday = _day_angle(day, offset=81)
    return float(
        9.87 * np.sin(2.0 * bday) - 7.53 * np.cos(bday) - 1.5 * np.sin(bday)
    )


def _hour_angle(
    time: dt.datetime, lon: npt.ArrayLike, lat: npt.ArrayLike
) -> npt.NDArray:
    """
    Description
    -----------
    Compute hour angle in local solar time.
    Zero correspond to local solar noon.

    Parameters
    ----------
    time : dt.datetime
        Date time
    lon : np.array_like
        Longitude (in degrees)
    lat : np.array_like
        Latitude (in degrees)

    Returns
    -------
    hour_angle: np.array
        Hour angle (in degrees)
    """
    # times must be localized
    if time.tzinfo is None:
        msg = "Time must be localized"
        raise ValueError(msg)
    # convert to np.array
    lat_arr = np.array(lat)
    lon_arr = np.array(lon)
    # Local time
    local_time = _to_localtime(time, lat_arr.mean(), lon_arr.mean())
    # local standard meridian time
    delta_utc = local_time.utcoffset().total_seconds() // 3600  # type: ignore
    lstm = 15 * delta_utc
    # time correction factor (in minutes)
    tc = 4 * (lon_arr - lstm) + _equation_of_time(_to_dateofyear(time))
    # time after local midnight
    midnight = dt.datetime.combine(
        local_time.date(), dt.time(0), tzinfo=local_time.tzinfo
    )
    delta_time = local_time - midnight
    # local solar time (in hours)
    f = np.vectorize(
        lambda x: (delta_time + dt.timedelta(hours=x) / 60.0).total_seconds()
        / 3600
    )
    lst = f(tc)
    return 15.0 * (lst - 12.0)


def _sun_zenith_angle(
    lat: npt.ArrayLike, hour_angle: npt.ArrayLike, decl_angle: npt.ArrayLike
) -> npt.NDArray:
    """
    Compute the sun zenith angle with an analytical
    expression based on spherical troginometry.

    .. See `PVCDROM: Elevation Angle
       <https://www.pveducation.org/pvcdrom/properties-of-sunlight/
       elevation-angle>`_

    Parameters
    ----------
    lat : np.array_like
        Latitude (in degrees)
    hour_angle : np.array_like
        Hour angle (in degrees)
    decl_angle : np.array_like
        Declination angle (in degrees)

    Returns
    -------
    sza : np.array
        Solar zenith angle (in degrees)
    """
    lat_rad = np.deg2rad(np.array(lat))
    hour_angle_rad = np.deg2rad(np.array(hour_angle))
    decl_angle_rad = np.deg2rad(np.array(decl_angle))
    return np.rad2deg(
        np.arccos(
            np.sin(decl_angle_rad) * np.sin(lat_rad)
            + np.cos(decl_angle_rad) * np.cos(lat_rad) * np.cos(hour_angle_rad)
        )
    )


def _declination_angle(day: npt.ArrayLike) -> npt.NDArray:
    """
    Compute declination
    """
    # Day angle
    gamma = _day_angle(day)
    # Solar declinaison given by Spencer, J. W. (1971).
    # Fourier series representation of the position
    # of the sun. Search, 2(5), 172-172.
    return np.rad2deg(
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )


def _sun_earth_distance(day: npt.ArrayLike) -> npt.NDArray:
    """
    Description
    -----------
    Compute the Sun-Earth distance factor at the given day.
    The Sun-Earth distance factor is given by
    Spencer, J. W. (1971). Fourier series representation
    of the position of the sun. Search, 2(5), 172-172.

    Parameters
    ----------
    day: int
        Day of the year, ranging from 1 on January 1st to
        365 or 366 on December 31th.

    Return
    ------
    factor: float
        Sun-Earth distance factor
    """
    gamma = _day_angle(day)
    return (
        1.00011
        + 0.034221 * np.cos(gamma)
        + 0.00128 * np.sin(gamma)
        + 0.000719 * np.cos(2 * gamma)
        + 0.000077 * np.sin(2 * gamma)
    )


def _sunrise(day: int, lat: float):
    """
    Description
    -----------
    Estimate sunrise time.

    Parameters
    ----------
    day : int
        Day of the year
    lat : float
        Latitude (in degrees)

    Returns
    -------
    sunrise: float
        Sunrise time from midnight (in hours)
    """
    lat_rad = np.deg2rad(lat)
    decl_angle_rad = np.deg2rad(_declination_angle(day))
    tc = _equation_of_time(day)
    return (
        12.0
        - 1
        / 15.0
        * np.rad2deg(
            np.arccos(
                -np.sin(lat_rad)
                * np.sin(decl_angle_rad)
                / np.cos(lat_rad)
                / np.cos(decl_angle_rad)
            )
        )
        - tc / 60.0
    )


def _sunset(day: int, lat: float):
    """
    Description
    -----------
    Estimate sunrise time.

    Parameters
    ----------
    day : int
        Day of the year
    lat : float
        Latitude (in degrees)

    Returns
    -------
    sunrise: float
        Sunrise time from midnight (in hours)
    """
    lat_rad = np.deg2rad(lat)
    decl_angle_rad = np.deg2rad(_declination_angle(day))
    tc = _equation_of_time(day)
    return (
        12.0
        + 1
        / 15.0
        * np.rad2deg(
            np.arccos(
                -np.sin(lat_rad)
                * np.sin(decl_angle_rad)
                / np.cos(lat_rad)
                / np.cos(decl_angle_rad)
            )
        )
        - tc / 60.0
    )


def toa_instant_irradiance(
    day: npt.ArrayLike, sza: npt.ArrayLike
) -> npt.NDArray:
    """
    Description
    -----------
    Compute the instantaneous irradiance
    at the top of atmosphere.
    E_TOA = E0 x v(day) x cos(SZA)
    with
      - E0 is the solar constant corresponding to the
      average TOA flux over the whole year
      - v(day) is the Sun-Earth distance factor at the day
      of observation
      - SZA is the Sun Zenith Angle

    Parameters
    ----------
    day: np.array_like
        Day of the year representing by an integer,
        ranging from 1 on January 1st to
        365 or 366 on December 31th.
    sza: np.array_like
        Sun Zenith angle (in degrees)

    Return
    ------
    toa_irradiance: np.array
        TOA instant irradiance (W.m-2)
    """
    return (
        SOLAR_FLUX
        * _sun_earth_distance(np.array(day))
        * np.cos(np.deg2rad(np.array(sza)))
    )


def toa_daily_irradiance(day: npt.ArrayLike, lat: npt.ArrayLike) -> npt.NDArray:
    """
    Description
    -----------
    Compute the daily solar irradiance at the top
    of atmposhere for a given position (lat,lon).
    The formula is given by
    Mousavi Maleki, S.A.; Hizam, H.; Gomes, C.
    Estimation of Hourly, Daily and Monthly Global Solar
    Radiation on Inclined Surfaces: Models Re-Visited.
    Energies 2017, 10, 134.

    Parameters
    ----------
    day: int
        Day of the year, ranging from 1 on January 1st to
        365 or 366 on December 31th.
    lat: float
        Latitude

    Return
    ------
    toa_irradiance: float
        TOA daily irradiance (J.m-2.day-1)
    """
    # convert angle in radians
    lat_rad = np.deg2rad(lat)
    # Solar declinaison
    delta = np.deg2rad(_declination_angle(day))
    # Sunrise hour angle
    h0 = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    return (
        24.0
        * 3600
        * SOLAR_FLUX
        * _sun_earth_distance(day)
        / np.pi
        * (
            h0 * np.sin(lat_rad) * np.sin(delta)
            + np.cos(lat_rad) * np.cos(delta * np.sin(h0))
        )
    )


def toa_daily_estimate(data: xr.Dataset, date: dt.datetime) -> xr.Dataset:
    """
    Description
    -----------
    The instantaneous value is transformed
    into a daily value by considering a scaling factor.
    This facor is equal to the ratio between daily downwelling
    shortwave radiation and between instant downwelling shortwave
    radiation.
    In this method the shortwave radiation is estimate at
    top-of-atmosphere.

    Parameters
    ----------
    data: xr.Dataset
        Instantaneous data
    date: dt.datetime
        Date time

    Returns
    -------
    daily: xr.DataArray
        Daily extrapolated data
    """
    # Get day of year
    day = _to_dateofyear(date)
    # Get lon/lat coordinates
    crs = data.attrs.get("crs", None)
    if crs is None:
        msg = "Impossible to compute TOA extrapolation because CRS is missing in the metadata"
        raise ValueError(msg)
    if crs == CRS(4326):
        msg = "Data already in lat/lon"
        logger.debug(msg)
        lon, lat = np.meshgrid(data.coords["x"], data.coords["y"])
    else:
        msg = "Conversion to lat/lon"
        logger.debug(msg)
        x, y = np.meshgrid(data.coords["x"], data.coords["y"])
        lon, lat = _to_lonlat(crs, x, y)
    # Get sun zenith angle
    if "sza" not in data.data_vars:
        msg = (
            "Sun zenith angle is missing, use of an analytical formula instead"
        )
        logger.warning(msg)
        hour_angle = _hour_angle(date, lon, lat)
        decl_angle = _declination_angle(day)
        sza = _sun_zenith_angle(lat, hour_angle, decl_angle)
    else:
        sza = data["sza"].data
    # Initiate dataset
    daily = data.copy(data=None)
    daily.attrs = data.attrs.copy()
    # Compute TOA solar radiation
    toa_daily = toa_daily_irradiance(day, lat)
    toa_inst = toa_instant_irradiance(day, sza)
    # Compute ratio
    for var in daily.data_vars:
        daily[var] = data[var] * toa_daily / toa_inst
    return daily


def extrapolate_at_daily_scale(
    data: xr.Dataset, variables: list[str] | None = None, method: str = "toa"
) -> xr.Dataset:
    """
    Description
    -----------
    The instantaneous value of ET obtained previously is transformed
    in a daily value by considering a scaling factor and the ratio between
    instant downwelling shortwave radiation and daily downwelling
    shortwave radiation.
    The ratio used depend on the method chosen.
    Only data in the variable list is extrapolated.
    If the list is empty, all data are extrapolated.

    Parameters
    ----------
    data: xr.Dataset
        Instantaneous data
    variables: list[str]
        List of variables to extrapolate.
    method: str
        Method used for extrapolation (default: toa)

    Returns
    -------
    daily: xr.DataArray
        Daily extrapolated data
    """
    # Data selection
    if variables is None:
        keep = list(data.data_vars)
    else:
        keep = list(filter(lambda x: x in data.data_vars, variables))
        not_keep = list(filter(lambda x: x not in data.data_vars, variables))
        if len(not_keep) != 0:
            msg = f"Variables {not_keep} not available for daily extrapolation"
            logger.warning(msg)
    msg = f"Daily extrapolation performed on {keep}"
    logger.debug(msg)
    # Extrapolation
    if method.lower() == "toa":
        if data.attrs.get("date", None) is None:
            msg = "Impossible to extrapolate because the date is missing in metadata"
            raise ValueError(msg)
        daily = toa_daily_estimate(data=data[keep], date=data.attrs["date"])
    else:
        msg = f"Extrapolation method {method} unknown"
        logger.error(msg)
        daily = data.copy(data=None)
        daily.attrs = data.attrs.copy()
        for var in keep:
            daily[var].data = np.nan * np.ones_like(daily["var"].data)
    return daily
