# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for flux computations
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import numpy.typing as npt
import xarray as xr
from pyproj import CRS
from scipy.constants import c, h, k, pi

from evaspa.common.solar import compute_diffuse_fraction, compute_sun_angles
from evaspa.debugging import register_debugging
from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)
# Stefan-Boltzmann constant
CST_SB = ((2 * pi**5) * (k**4)) / (15 * (c**2) * (h**3))
# Latent heat vaporization constant
LATENT_HEAT_VAPORIZATION = 2.45e6  # J.kg-2


def correct_direct_radiation(
    rsd: npt.ArrayLike,
    sza: npt.ArrayLike,
    saa: npt.ArrayLike,
    slope: npt.ArrayLike,
    aspect: npt.ArrayLike,
) -> npt.NDArray:
    """
    Correct instant direct downward shortwave radiation
    for arbitrary terrains and sun positions.

    Notes
    -----
    The instant direct downward shortwave radiation has been computed
    by taking cos(SZA) as the cosine of the solar incidence angle
    relative to the normal to the land surface.
    The objective is to take into account slope and aspect
    impact on direct shortwave radiation.

    RSD_corr = RSD x cos(i) / cos(sza)
    cos(i) = cos(slope)cos(sza) + sin(slope)sin(sza)cos(aspect-saa)

    G. E. Liston and K. Elder, “A meteorological distribution system
    for high-resolution terrestrial modeling (MicroMet),” vol. 7,
    no. 2, pp. 217-234, 2006, Journal of Hydrometeorology

    Parameters
    ----------
    rsd: np.array_like
        Direct downward shortwave radiation
    sza: np.array_like
        Sun Zenith Angle
    saa: np.array_like
        Sun Azimuth Angle
    slope: np.array_like
        Slope
    aspect: np.array_like
        Aspect

    Returns
    -------
    rsd_corr: np.array
        Direct downward shortwave radiation corrected with topography
    """
    # convert deg to rad
    slope_rad = np.deg2rad(slope)
    aspect_rad = np.deg2rad(aspect)
    sza_rad = np.deg2rad(sza)
    saa_rad = np.deg2rad(saa)
    cos_i = np.cos(slope_rad) * np.cos(sza_rad) + np.sin(slope_rad) * np.sin(
        sza_rad
    ) * np.cos(aspect_rad - saa_rad)
    cos_i = np.where(cos_i < 0, 0, cos_i)
    return np.array(rsd) * cos_i / np.cos(sza_rad)


def correct_shortwave_radiation(
    rsd: xr.DataArray,
    slope: xr.DataArray,
    aspect: xr.DataArray,
    fdiff: xr.DataArray | None = None,
    date: dt.datetime | None = None,
    sza: xr.DataArray | None = None,
    saa: xr.DataArray | None = None,
    crs: CRS | None = None,
) -> xr.DataArray:
    """
    This method corrects instant direct downward shortwave radiation
    for arbitrary terrains and sun positions.

    Notes
    -----
    The instant direct downward shortwave radiation has been computed
    by taking cos(SZA) as the cosine of the solar incidence angle
    relative to the normal to the land surface.
    The objective is to take into account slope and aspect
    impact on direct shortwave radiation.

    G. E. Liston and K. Elder, “A meteorological distribution system
    for high-resolution terrestrial modeling (MicroMet),” vol. 7,
    no. 2, pp. 217-234, 2006, Journal of Hydrometeorology

    Parameters
    ----------
    rsd: xr.DataArray
        Global radiation data
    slope: xr.DataArray
        Slope
    aspect: xr.DataArray
        Aspect
    fdiff: xr.DataArray
        Diffuse Fraction
    date: dt.datetime
        Date
    sza: xr.DataArray
        Sun Zenith Angle
    saa: xr.DataArray
        Sun Azimuth Angle
    crs: CRS
        Coordinate Reference System

    Returns
    -------
    rsd_corr: xr.DataArray
        Corrected shortwave radiation
    """
    if crs is None:
        crs = CRS(4326)
    # Sun zenith angle
    if sza is None or saa is None:
        logger.warning("Sun angle not present, theoretical calculation done")
        if date is None:
            msg = "Date is missing"
            raise ValueError(msg)
        sza_arr, saa_arr = compute_sun_angles(
            date, x=rsd.coords["x"], y=rsd.coords["y"], crs=crs
        )
        sza = rsd.copy(data=sza_arr)
        saa = rsd.copy(data=saa_arr)
    if fdiff is None:
        msg = (
            "No diffuse fraction data for shortwave radiation. "
            "Use theoritical equation."
        )
        logger.warning(msg)
        if date is None:
            msg = "Date is missing"
            raise ValueError(msg)
        fdiff = compute_diffuse_fraction(date, rsd, sza, saa, crs)
    rsd_diff = rsd * fdiff
    rsd_direct = rsd.copy(
        data=correct_direct_radiation(
            rsd.data - rsd_diff.data,
            sza.data,
            saa.data,
            slope.data,
            aspect.data,
        )
    )
    return rsd_direct + rsd_diff


def compute_rn(
    lst: npt.ArrayLike,
    emis: npt.ArrayLike,
    albedo: npt.ArrayLike,
    rsd: npt.ArrayLike,
    rld: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute net radiation Rn

    Parameters
    ----------
    lst : np.array_like
        Land surface temperature
    emis : np.array_like
        Land surface emissivity
    albedo : np.array_like
        Land surface albedo
    rsd : np.array_like
        Downward Shortwave Radiation
    rld : np.array_like
        Downward Longwave Radiation

    Returns
    -------
    rn: np.array
        Net radiation
    """
    ln = np.array(emis) * (np.array(rld) - CST_SB * (np.array(lst) ** 4))
    return (1 - np.array(albedo)) * np.array(rsd) + ln, ln


@register_debugging
def create_net_radiation(
    data: xr.Dataset, use_topo: bool = False
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Compute net radiation dataset and longwave net radiation
    dataset.

    Notes
    -----
    The net radiation is computed from land surface temperature,
    land surface emissivity, albedo and downward longwave and
    shortwave radiations.
    If several downward shortwave and longwace radiations are
    available, the dataset will contain several net radiation
    estimation.

    Parameters
    ----------
    data : xr.Dataset
        Data containing (LST, emissivity, albedo and downward
        shortwave and longwave radiation)

    Returns
    -------
    rn: xr.Dataset
        Net radiation dataset
    ln: xr.Dataset
        Longwave net radiation dataset
    """
    # Check DEM availability if topographic corrections are requested
    if use_topo and (
        "aspect" not in data.data_vars or "slope" not in data.data_vars
    ):
        logger.warning(
            "No DEM information (aspect or slope) to compute topographic corrections. Topographic corrections are disabled."
        )
        use_topo = False

    # Check if several rsd/rld are available
    rsd_data: list[str] = [str(v) for v in data.data_vars if "rsd" in str(v)]
    if len(rsd_data) == 0:
        msg = "No RSD data available"
        raise ValueError(msg)

    msg = f"RSD data available: {rsd_data}"
    logger.debug(msg)
    rld_data: list[str] = [str.replace(v, "rsd", "rld", 1) for v in rsd_data]
    for v in rld_data:
        if v not in data.data_vars:
            msg = (
                f"No RLD data ({v}) associated to RSD"
                f" data ({str.replace(v, 'rld', 'rsd')})"
            )
            raise ValueError(msg)
    msg = f"RLD data available: {rld_data}"
    logger.debug(msg)
    # Compute rn and ln for all rsd/rld available
    rn = {}
    ln = {}
    for rsd_name, rld_name in zip(rsd_data, rld_data, strict=False):
        rsd = data[rsd_name]
        rld = data[rld_name]
        if use_topo:
            # Correct RSD with topographic corrections
            # Get diffuse fraction
            fdiff_name = str.replace(rsd_name, "rsd", "fdiff", 1)
            fdiff = data.get(fdiff_name, None)
            rsd = correct_shortwave_radiation(
                rsd,
                data["slope"],
                data["aspect"],
                fdiff,
                data.attrs.get("date", None),
                data.get("sza", None),
                data.get("sza", None),
                data.attrs.get("crs", None),
            )
        # Compute Rn
        rn_name = str.replace(rsd_name, "rsd", "rn", 1)
        ln_name = str.replace(rsd_name, "rsd", "ln", 1)
        rn_data, ln_data = compute_rn(
            lst=data["lst"],
            emis=data["emis"],
            albedo=data["albedo"],
            rsd=rsd,
            rld=rld,
        )
        rn[rn_name] = xr.DataArray(
            data=rn_data,
            dims=data.dims,
            coords=data.coords.copy(),
        )
        ln[ln_name] = xr.DataArray(
            data=ln_data,
            dims=data.dims,
            coords=data.coords.copy(),
        )
    # Attributes
    attrs = {
        "crs": data.attrs.get("crs", None),
        "transform": data.attrs.get("transform", None),
    }
    return (
        xr.Dataset(data_vars=rn, coords=data.coords.copy(), attrs=attrs),
        xr.Dataset(data_vars=ln, coords=data.coords.copy(), attrs=attrs),
    )


def compute_et_from_le(
    le: npt.ArrayLike, temperature: float | None = None
) -> npt.NDArray:
    """
    Compute ET in mm from LE in W (J.m-2).
    ET = LE / L with L is the latent heat vaoprization of water.

    Parameters
    ----------
    le : np.array_like
        Latent heat flux
    temperature : float
        Temperature

    Returns
    -------
    et: np.array
        Evapotranspiration
    """
    if temperature is None:
        latent_heat = LATENT_HEAT_VAPORIZATION
    else:
        msg = "The variation of latent heat of vaporization of water with temperature is not implemented yet."
        logger.warning(msg)
        latent_heat = LATENT_HEAT_VAPORIZATION
    return np.array(le) / latent_heat
