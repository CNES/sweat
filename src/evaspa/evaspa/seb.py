# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

from enum import Enum

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from evaspa.common.constant import FLAGS_TYPE, ETVar
from evaspa.common.flux import (
    compute_et_from_le,
    compute_rn,
    correct_shortwave_radiation,
)
from evaspa.debugging import register_debugging
from evaspa.evaspa.merging import MergeMethod, merge_to_dataset
from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


class RatioModel(Enum):
    """List of G/Rn ratio models"""

    KUSTAS = "kustas"
    SU = "su"
    CHOUDHURY = "choudhury"


DEFAULT_MODELS = [RatioModel.KUSTAS]
ALL_MODELS = [RatioModel.KUSTAS, RatioModel.SU, RatioModel.CHOUDHURY]


class SEBConfig(BaseModel):
    """
    SEB config for ET processing
    """

    model_config = ConfigDict(extra="forbid")

    use_topo: bool = Field(default=False)
    models: list[RatioModel] = Field(default=DEFAULT_MODELS)
    merging: MergeMethod = Field(default=MergeMethod.MEDIAN)


@register_debugging
def create_net_radiation(
    data: xr.Dataset, use_topo: bool = False
) -> xr.Dataset:
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
    # Compute rn for all rsd/rld available
    rn = {}
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
        name = str.replace(rsd_name, "rsd", "rn", 1)
        rn[name] = xr.DataArray(
            data=compute_rn(
                lst=data["lst"],
                emis=data["emis"],
                albedo=data["albedo"],
                rsd=rsd,
                rld=rld,
            )[0],
            dims=data.dims,
            coords=data.coords.copy(),
        )
    # Attributes
    attrs = {
        "crs": data.attrs.get("crs", None),
        "transform": data.attrs.get("transform", None),
    }
    return xr.Dataset(data_vars=rn, coords=data.coords.copy(), attrs=attrs)


def _ratio_from_kustas(
    ndvi: npt.ArrayLike, c1: float = 0.4, c2: float = 0.33
) -> npt.NDArray:
    """
    Description
    -----------
    Compute G/Rn ratio from Kustas et al. 1993
    G/Rn = c1 - c2 x NDVI

    Kustas W.P., Daughtry C.S.T. and Oevelen P.J.V., 1993.
    Analytical Treatment of the Relationships between
    Soil Heat Flux/Net Radiation Ratio and Vegetation Indices.
    Remote Sensing of Environment, 46, 319-330.

    Parameters
    ----------
    ndvi: np.array_like
        NDVI
    c1: float
        c1 parameter (default = 0.4)
    c2: float
        c2 parameter (default = 0.33)

    Returns
    -------
    ratio: np.array
        Ground heat flux / Net radiation
    """
    return c1 - (c2 * np.array(ndvi))


def _ratio_from_su(
    fcover: npt.ArrayLike,
    c1: float = 0.315,
    c2: float = 0.05,
) -> npt.NDArray:
    """
    Description
    -----------
    Compute G/Rn ratio from Su 2002
    G/Rn = c2 + (1 - fcover) x (c1 - c2)

    Su Z., 2002. The surface energy balance system
    (SEBS) for estimation of turbulent fluxes.
    Hydrol. Earth Syst. Sci., 6, 85-99.

    Parameters
    ----------
    fcover: np.array_like
        Fcover
    c1: float
        c1 parameter (default = 0.315)
    c2: float
        c2 parameter (default = 0.05)

    Returns
    -------
    ratio: np.array
        Ground heat flux / Net radiation
    """
    return c2 + (1.0 - np.array(fcover)) * (c1 - c2)


def _ratio_from_choudhury(
    lai: npt.ArrayLike, c1: float = 0.3, c2: float = 0.5
) -> npt.NDArray:
    """
    Description
    -----------
    Compute G/Rn ratio from Choudhury et al. 1987
    G/Rn = c1 x exp(-c2 x lai)

    Choudhury, B.J., Idso, S.B., & Reginato, R.J. (1987).
    Analysis of an empirical model for soil heat flux under
    a growing wheat crop for estimating evaporation
    by an infrared temperature based energy balance equation.
    Agricultural and Forest Meteorology, 39, 283-297

    Parameters
    ----------
    fcover: np.array_like
        Fcover
    c1: float
        c1 parameter (default = 0.3)
    c2: float
        c2 parameter (default = 0.5)

    Returns
    -------
    ratio: np.array
        Ground heat flux / Net radiation
    """
    return c1 * np.exp(-c2 * np.array(lai))


@register_debugging
def create_ratio(
    data: xr.Dataset,
    models: list[RatioModel] = DEFAULT_MODELS,
) -> xr.Dataset:
    """
    Description
    -----------
    Compute G/Rn ratio dataset using several models.

    Parameters
    ----------
    data : xr.Dataset
        Data containing (LST, emissivity, albedo and downward
        shortwave and longwave radiation)
    models: list[str]
        List of ratio models

    Returns
    -------
    ratio: xr.Dataset
        Ground heat flux / Net radiation
    """

    ratio = {}
    for model in models:
        # Check if model is available
        try:
            name = RatioModel(model).name
        except ValueError:
            msg = f"Unknown model for G/Rn ratio: {model}"
            logger.warning(msg)
            continue
        # Compute ratio for the model
        try:
            if name == RatioModel.KUSTAS.name:
                ratio[model.value] = xr.DataArray(
                    data=_ratio_from_kustas(data["ndvi"]),
                    dims=data.dims,
                    coords=data.coords.copy(),
                )
            elif name == RatioModel.SU.name:
                ratio[model.value] = xr.DataArray(
                    data=_ratio_from_su(data["fcover"]),
                    dims=data.dims,
                    coords=data.coords.copy(),
                )
            elif name == RatioModel.CHOUDHURY.name:
                ratio[model.value] = xr.DataArray(
                    data=_ratio_from_choudhury(data["lai"]),
                    dims=data.dims,
                    coords=data.coords.copy(),
                )
        except KeyError as e:
            msg = f"Data missing for {model.value} model: {e}"
            logger.warning(msg)
    # Attributes
    attrs = {
        "crs": data.attrs.get("crs", None),
        "transform": data.attrs.get("transform", None),
    }
    return xr.Dataset(data_vars=ratio, coords=data.coords.copy(), attrs=attrs)


def _compute_le(
    ef: npt.ArrayLike, rn: npt.ArrayLike, ratio: npt.ArrayLike
) -> npt.NDArray:
    """
    Description
    -----------
    Compute latent heat flux based on the following formula
    LE = EF x (Rn - G) = EF x Rn (1 - ratio)

    Parameters
    ----------
    ef : np.array_like
        Evaporative fraction
    rn: np.array_like
        Net radiation
    ratio: np.array_like
        Ground heat flux / net radiation ratio

    Returns
    -------
    le: np.array
        Latent heat flux
    """
    return np.array(ef) * np.array(rn) * (1.0 - np.array(ratio))


def create_le(ef: xr.Dataset, rn: xr.Dataset, ratio: xr.Dataset) -> xr.Dataset:
    """
    Description
    -----------
    Create latent heat flux dataset for all EF models,
    all Rn models and all G/Rn ratio models.
    For each combination of models, LE is computed with
    the following formula
    LE = EF x (Rn - G) = EF x Rn x (1-ratio)

    Parameters
    ----------
    ef : xr.Dataset
        Evaporative fraction dataset
    rn: xr.Dataset
        Net radiation dataset
    ratio: np.array_like
        Ground heat flux / net radiation ratio

    Returns
    -------
    le: xr.Dataset
        Latent heat flux dataset
    """
    data_vars = {}
    for ef_model in ef.data_vars:
        for rn_model in rn.data_vars:
            for ratio_model in ratio.data_vars:
                name = f"{ef_model}_{rn_model}_{ratio_model}"
                data_vars[name] = ef[ef_model].copy(
                    data=_compute_le(
                        ef[ef_model].data,
                        rn[rn_model].data,
                        ratio[ratio_model].data,
                    )
                )
    # Attributes
    attrs = {
        "crs": ef.attrs.get("crs", None),
        "transform": ef.attrs.get("transform", None),
    }
    return xr.Dataset(data_vars, attrs=attrs)


@register_debugging
def run(
    data: xr.Dataset,
    ef: xr.Dataset,
    use_topo=False,
    models=DEFAULT_MODELS,
    merging: MergeMethod = MergeMethod.MEAN,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Description
    -----------
    Compute latent heat flux dataset for all EF models.
    First all Rn models and all G/Rn ratio models are computed.
    Then, for each combination of models, LE is computed with
    the following formula
    LE = EF x (Rn - G) = EF x RN x (1 - ratio)

    Parameters
    ----------
    rn: xr.Dataset
        Net radiation dataset
    ef : xr.Dataset
        Evaporative fraction dataset
    use_topo: bool
        Topography to take into account
    models: list[str]
        List of G models
    merging : MergeMethod
        Method used for merging

    Returns
    -------
    le: xr.Dataset
        Latent heat flux dataset
    le_merged : xr. Dataset
        Latent heat flux merged
    """
    if len(ef.data_vars) == 0:
        msg = "EF dataset empty"
        raise ValueError(msg)
    # Get valid and flags
    if ETVar.VALID.value in ef.data_vars:
        valid = data[ETVar.VALID.value]
    else:
        valid = xr.ones_like(
            next(iter(ef.data_vars.values())), dtype=FLAGS_TYPE
        )
    if ETVar.FLAGS.value in data.data_vars:
        flags = data[ETVar.FLAGS.value]
    else:
        flags = xr.zeros_like(
            next(iter(ef.data_vars.values())), dtype=FLAGS_TYPE
        )
    # Compute net radiation
    rn_xr = create_net_radiation(data, use_topo=use_topo)

    if len(rn_xr.data_vars) == 0:
        msg = "Radiation dataset empty"
        raise ValueError(msg)
    # Compute G flux
    ratio_xr = create_ratio(data, models=models)
    if len(ratio_xr.data_vars) == 0:
        msg = "G/Rn ratio dataset empty"
        raise ValueError(msg)
    # Compute latent heat flux
    le_xr = create_le(
        ef.drop_vars([ETVar.VALID.value, ETVar.FLAGS.value], errors="ignore"),
        rn_xr,
        ratio_xr,
    )
    merged_xr = merge_to_dataset(le_xr, method=merging, name="le")
    merged_xr["et"] = merged_xr["le"].copy(
        data=compute_et_from_le(merged_xr["le"])
    )
    le_xr[ETVar.VALID.value] = valid
    le_xr[ETVar.FLAGS.value] = flags
    merged_xr[ETVar.VALID.value] = valid
    merged_xr[ETVar.FLAGS.value] = flags
    return le_xr, merged_xr
