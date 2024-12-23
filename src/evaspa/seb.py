#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.constants import c, h, k, pi
from pydantic import BaseModel, ConfigDict, Field

from .logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Stefan-Boltzmann constat
CST_SB = ((2 * pi**5) * (k**4)) / (15 * (c**2) * (h**3))

DEFAULT_G_MODELS = ["kustas"]


class SEBConfig(BaseModel):
    """
    SEB config for ET processing
    """

    model_config = ConfigDict(extra="forbid")

    use_topo: bool = Field(default=False)
    g_models: list[str] = Field(default=DEFAULT_G_MODELS)


def compute_rn(
    lst: npt.ArrayLike,
    emis: npt.ArrayLike,
    albedo: npt.ArrayLike,
    rsd: npt.ArrayLike,
    rld: npt.ArrayLike,
) -> npt.NDArray:
    """
    Description
    -----------
    Compute net radiation Rn

    Parameters
    ----------
    lst : np.array_like
        Land surface temperature
    lst : np.array_like
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
    return (
        (1 - np.array(albedo)) * np.array(rsd)
        - np.array(emis) * CST_SB * (np.array(lst) ** 4)
        + np.array(emis) * np.array(rld)
    )


def create_net_radiation(data: xr.Dataset) -> xr.Dataset:
    """
    Description
    -----------
    Compute net radiation dataset.
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
    """
    # Check if several rsd/rld are available
    rsd_data: list[str] = [str(v) for v in data.data_vars if "rsd" in str(v)]
    if len(rsd_data) == 0:
        raise ValueError("No RSD data available")
    rld_data: list[str] = [str.replace(v, "rsd", "rld", 1) for v in rsd_data]
    for v in rld_data:
        if v not in data.data_vars:
            raise ValueError(
                f"No RLD data ({v}) associated to RSD"
                f" data ({str.replace(v,'rld','rsd')})"
            )
    # Compute rn for all rsd/rld available
    rn = {}
    for rsd, rld in zip(rsd_data, rld_data):
        name = str.replace(rsd, "rsd", "rn", 1)
        rn[name] = xr.DataArray(
            data=compute_rn(
                data["lst"], data["emis"], data["albedo"], data[rsd], data[rld]
            ),
            dims=data.dims,
            coords=data.coords.copy(),
        )
    # Attributes
    attrs = {
        "crs": data.attrs.get("crs", None),
        "transform": data.attrs.get("transform", None),
    }
    return xr.Dataset(data_vars=rn, coords=data.coords.copy(), attrs=attrs)


def compute_g_kustas(
    rn: npt.ArrayLike, ndvi: npt.ArrayLike, c1: float = 0.4, c2: float = 0.33
) -> npt.NDArray:
    """
    Description
    -----------
    Compute G flux from Kustas et al. 1993
    G/Rn = c1 - c2 x NDVI

    Kustas W.P., Daughtry C.S.T. and Oevelen P.J.V., 1993.
    Analytical Treatment of the Relationships between
    Soil Heat Flux/Net Radiation Ratio and Vegetation Indices.
    Remote Sensing of Environment, 46, 319-330.

    Parameters
    ----------
    rn : np.array_like
        Net radiation
    ndvi: np.array_like
        NDVI
    c1: float
        c1 parameter (default = 0.4)
    c2: float
        c2 parameter (default = 0.33)

    Returns
    -------
    g_flux: np.array
        Ground heat flux
    """
    return np.array(rn) * (c1 - (c2 * np.array(ndvi)))


def compute_g_su(
    rn: npt.ArrayLike, fcover: npt.ArrayLike, c1: float = 0.315, c2: float = 0.05
) -> npt.NDArray:
    """
    Description
    -----------
    Compute G flux from Su 2002
    G/Rn = c2 + (1 - fcover) x (c1 - c2)

    Su Z., 2002. The surface energy balance system
    (SEBS) for estimation of turbulent fluxes.
    Hydrol. Earth Syst. Sci., 6, 85–99.

    Parameters
    ----------
    rn : np.array_like
        Net radiation
    fcover: np.array_like
        Fcover
    c1: float
        c1 parameter (default = 0.4)
    c2: float
        c2 parameter (default = 0.33)

    Returns
    -------
    g_flux: np.array
        Ground heat flux
    """
    return np.array(rn) * (c2 + (1.0 - np.array(fcover)) * (c1 - c2))


def compute_g_choudhury(
    rn: npt.ArrayLike, lai: npt.ArrayLike, c1: float = 0.3, c2: float = 0.5
) -> npt.NDArray:
    """
    Description
    -----------
    Compute G flux from Choudhury et al. 1987
    G/Rn = c1 x exp(-c2 x lai)

    Choudhury, B.J., Idso, S.B., & Reginato, R.J. (1987).
    Analysis of an empirical model for soil heat flux under
    a growing wheat crop for estimating evaporation
    by an infrared temperature based energy balance equation.
    Agricultural and Forest Meteorology, 39, 283-297

    Parameters
    ----------
    rn : np.array_like
        Net radiation
    fcover: np.array_like
        Fcover
    c1: float
        c1 parameter (default = 0.4)
    c2: float
        c2 parameter (default = 0.33)

    Returns
    -------
    g_flux: np.array
        Ground heat flux
    """
    return np.array(rn) * c1 * np.exp(-c2 * np.array(lai))


def create_gflux(
    data: xr.Dataset,
    rn: xr.Dataset,
    g_models: list[str] = ["kustas", "su", "choudhury"],
) -> xr.Dataset:
    """
    Description
    -----------
    Compute G flux dataset using several methods.

    Parameters
    ----------
    data : xr.Dataset
        Data containing (LST, emissivity, albedo and downward
        shortwave and longwave radiation)

    Returns
    -------
    rn: xr.Dataset
        Net radiation dataset
    """
    g = {}
    for var in rn.data_vars:
        for model in g_models:
            if model == "kustas":
                g[f"{var}_kustas"] = xr.DataArray(
                    data=compute_g_kustas(rn[var], data["ndvi"]),
                    dims=data.dims,
                    coords=data.coords.copy(),
                )
            elif model == "su":
                g[f"{var}_su"] = xr.DataArray(
                    data=compute_g_su(rn[var], data["fcover"]),
                    dims=data.dims,
                    coords=data.coords.copy(),
                )
            elif model == "choudhury":
                g[f"{var}_choudhury"] = xr.DataArray(
                    data=compute_g_choudhury(rn[var], data["lai"]),
                    dims=data.dims,
                    coords=data.coords.copy(),
                )
            else:
                logger.warning("Method unknown for G flux: {method}")
    # Attributes
    attrs = {
        "crs": data.attrs.get("crs", None),
        "transform": data.attrs.get("transform", None),
    }
    return xr.Dataset(data_vars=g, coords=data.coords.copy(), attrs=attrs)


def compute_le(
    ef: npt.ArrayLike, rn: npt.ArrayLike, gflux: npt.ArrayLike
) -> npt.NDArray:
    """
    Description
    -----------
    Compute latent heat flux based on the following formula
    LE = EF x (Rn - G)

    Parameters
    ----------
    ef : np.array_like
        Evaporative fraction
    rn: np.array_like
        Net radiation
    gflux: np.array_like
        Ground heat flux

    Returns
    -------
    le: np.array
        Latent heat flux
    """
    return np.array(ef) * (np.array(rn) - np.array(gflux))


def create_le(ef: xr.Dataset, rn: xr.Dataset, gflux: xr.Dataset) -> xr.Dataset:
    """
    Description
    -----------
    Create latent heat flux dataset for all EF models,
    all Rn models and all G models.
    For each combination of models, LE is computed with
    the following formula
    LE = EF x (Rn - G)

    Parameters
    ----------
    ef : xr.Dataset
        Evaporative fraction dataset
    rn: xr.Dataset
        Net radiation dataset
    gflux: xr.Dataset
        Ground heat flux dataset

    Returns
    -------
    le: xr.Dataset
        Latent heat flux dataset
    """
    vars = {}
    for ef_model in ef.data_vars:
        for rn_model in rn.data_vars:
            for g_model in gflux.data_vars:
                name = f"{ef_model}_{rn_model}_{g_model}"
                vars[name] = xr.DataArray(
                    data=compute_le(ef[ef_model], rn[rn_model], gflux[g_model]),
                    dims=ef.dims,
                    coords=ef.coords.copy(),
                )

    # Attributes
    attrs = {
        "crs": ef.attrs.get("crs", None),
        "transform": ef.attrs.get("transform", None),
    }
    return xr.Dataset(data_vars=vars, coords=ef.coords.copy(), attrs=attrs)


def run(
    data: xr.Dataset, ef: xr.Dataset, use_topo=False, g_models=DEFAULT_G_MODELS
) -> xr.Dataset:
    """
    Description
    -----------
    Compute latent heat flux dataset for all EF models.
    First all Rn models and all G models are computed.
    Then, for each combination of models, LE is computed with
    the following formula
    LE = EF x (Rn - G)

    Parameters
    ----------
    rn: xr.Dataset
        Net radiation dataset
    ef : xr.Dataset
        Evaporative fraction dataset
    use_topo: bool
        Topography to take into account
    g_models: list[str]
        List of G models

    Returns
    -------
    le: xr.Dataset
        Latent heat flux dataset
    """
    rn_xr = create_net_radiation(data)
    g_xr = create_gflux(data, rn_xr, g_models=g_models)
    return create_le(ef, rn_xr, g_xr)
