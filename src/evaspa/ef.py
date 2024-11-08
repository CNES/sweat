#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
import numpy.typing as npt
import xarray as xr


from pydantic import BaseModel, ValidationError

from .edge import Edge, EdgeError, EdgeConfig


class MergeMethod(Enum):
    """Method for merge EF mdoels"""

    MEAN = "mean"


class EFModelConfig(BaseModel):
    """
    Configuration of a EF model
    """

    name: str
    dry_edge: EdgeConfig
    wet_edge: EdgeConfig
    var: str


class EFOptionsConfig(BaseModel):
    """
    Options for EF processing
    """

    selection: bool
    merging: MergeMethod
    keep: bool


class EFConfig(BaseModel):
    """
    Configuration for EF processing
    """

    models: list[EFModelConfig]
    options: EFOptionsConfig


class EFConfigError(Exception):
    """Exception in EF Model configuration"""


class EFModelError(Exception):
    """Exception in EF model creation"""


@dataclass(frozen=True)
class EFModel:
    """Class for evaporative fraction model"""

    name: str
    wet_edge: Edge
    dry_edge: Edge
    var: str

    def fit(self, data: xr.Dataset, mask: str | None = None) -> None:
        """
        Description
        -----------
        Method to estimate dry and wet edges.
        data is a xarray.Dataset which must conatin "lst"
        and "var" as data varaibles.

        Parameters
        ----------
        data : xr.Datset
            Dataset containing the data
        mask : str
            Name of the data variables used for masking
        """
        if self.var not in data.data_vars:
            raise EFModelError(f"{self.var} not in the dataset")
        if "lst" not in data.data_vars:
            raise EFModelError("lst not in the dataset")
        if mask is not None:
            assert mask in data.data_vars
            data_masked = data.where(data[mask], drop=True)
        else:
            data_masked = data
        self.wet_edge.fit(data_masked[self.var], data_masked["lst"])
        self.dry_edge.fit(data_masked[self.var], data_masked["lst"])

    def tdry(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute dry temperature

        Parameters
        ----------
        var : np.array_like
            Data

        Returns
        -------
        tdry: np.array
            Temperature for dry edge
        """
        return self.dry_edge.get(var)

    def twet(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute wet temperature

        Parameters
        ----------
        var : np.array_like
            Data

        Returns
        -------
        twet: np.array
            Temperature for wet edge
        """
        return self.wet_edge.get(var)

    def compute(self, data: xr.Dataset, mask: str | None = None) -> xr.DataArray:
        """
        Description
        -----------
        Method to compute evaporative fraction

        Parameters
        ----------
        data : xr.Datset
            Dataset containing the data
        mask : str
            Name of the data variables used for masking
        var : np.array_like
            Data

        Returns
        -------
        ef: np.array
            Evaporative Fraction
        """
        if self.var not in data.data_vars:
            raise EFModelError(f"{self.var} not in the dataset")
        if "lst" not in data.data_vars:
            raise EFModelError("lst not in the dataset")
        if mask is not None:
            assert mask in data.data_vars
            data_masked = data.where(data[mask], drop=True)
        else:
            data_masked = data
        ef = (self.tdry(data_masked[self.var]) - np.array(data_masked["lst"])) / (
            self.tdry(data_masked[self.var]) - self.twet(data_masked[self.var])
        )
        ef = np.where(ef > 1, 1, ef)
        ef = np.where(ef < 0, 0, ef)
        return xr.DataArray(
            data=ef,
            dims=data.dims,
            coords=data.coords.copy(),
            attrs=self.to_dict(),
        )

    def to_dict(self) -> dict:
        """
        Return an dictionary
        """
        return {
            "name": self.name,
            "dry_edge": {
                "type": self.dry_edge.__class__.__name__,
                "config": self.dry_edge.to_dict(),
            },
            "wet_edge": {
                "type": self.wet_edge.__class__.__name__,
                "config": self.wet_edge.to_dict(),
            },
            "var": self.var,
        }

    def __str__(self) -> str:
        """
        String conversion
        """
        return (
            f"Model({self.name},dry_edge={self.dry_edge.__class__},"
            f"wet_edge={self.wet_edge.__class__},var={self.var})"
        )

    def __repr__(self) -> str:
        """
        For print method
        """
        return (
            f"Model: {self.name}\n"
            f" - dry edge = {self.dry_edge}\n"
            f" - wet edge = {self.wet_edge}\n"
            f" - var = {self.var}"
        )

    @classmethod
    def check(cls, config: dict) -> EFModelConfig:
        """
        Description
        -----------
        Check an EF model configuration

        Parameters
        ----------
        config : dict
            Configuration for EF model

        Returns
        -------
        efconfig: EFModelConfig
            Validated configuration for EF model
        """
        try:
            efconfig = EFModelConfig.model_validate(config)
        except ValidationError as e:
            raise EFConfigError("Error in model configuration") from e
        return efconfig

    @classmethod
    def create(cls, config: dict) -> EFModel:
        """
        Description
        -----------
        Create an EF model from a configuration.
        Evaporative fraction EF represents the ratio of latent heat flux
        to available energy is computed from the position of
        the surface temperature value respectively
        to the dry edge and the wet edges.
        A EF model is defined by:
        - a name
        - the variable to consider: for instance, albedo or fcover
        - the method to compute the dry edge
        - the method to compute the wet edge

        Parameters
        ----------
        config : dict
            Configuration for EF model

        Returns
        -------
        model: EFModel
            Validated EF model
        """
        # Read configuration
        efconfig = cls.check(config)

        # Dry edge
        try:
            dry_edge = Edge.create(efconfig.dry_edge.type, efconfig.dry_edge.config)
        except EdgeError as e:
            raise EFModelError("Error in dry edge creation") from e

        # Wet edge
        try:
            wet_edge = Edge.create(efconfig.wet_edge.type, efconfig.wet_edge.config)
        except EdgeError as e:
            raise EFModelError("Error in wet edge creation") from e

        return cls(
            name=efconfig.name, dry_edge=dry_edge, wet_edge=wet_edge, var=efconfig.var
        )


def check_variability(
    lst: npt.ArrayLike, mask: npt.ArrayLike | None = None, threshold: float = 2.0
) -> bool:
    """
    Description
    -----------
    Check variability of Land Surface Temperature
    To be applicable, EVASPA requires a certain
    variability in the input data (surface temperature,
    albedo, fraction cover) in order to be able to
    define the edges correctly.
    If variability is insufficient, this method returns False.

    Parameters
    ----------
    lst : np.array_like
        Land surface temperature
    mask : np.array_like
        Mask (valid not 0)
    threshold : float
        Threshold value to check variablity

    Returns
    -------
    check: bool
    """
    t = np.array(lst)
    if mask is None:
        m = np.ones_like(t)
    else:
        m = np.array(mask)
        assert t.shape == m.shape
    t = np.where(np.array(m) > 0, t, np.nan)
    # Compute the minimum and maximum values
    # of the relative difference from land surface temperature
    # and its median
    t_median = np.nanmedian(t)
    if abs(t_median) < 1.0:
        t_median = 1.0
    if ((np.nanmax(t) - np.nanmin(t)) / t_median) <= threshold:
        return False
    return True


def initialize(config: dict) -> tuple[list[EFModel], dict[str, Any]]:
    """
    Description
    -----------
    Initialize a list of EF models and processing options from a
    configuration

    Parameters
    ----------
    config: dict
        Configuration

    Returns
    -------
    return: list[EFModel],dict
        List of EF models and configuration
    """
    try:
        efconfig = EFConfig.model_validate(config)
    except ValidationError as e:
        raise EFConfigError("Error in EF configuration") from e
    models = [EFModel.create(cfg.model_dump()) for cfg in efconfig.models]
    return (models, efconfig.options.model_dump())


def compute(models: list[EFModel], data: xr.Dataset) -> xr.Dataset:
    """
    Description
    -----------
    Compute evaporative fraction of a list of EF models

    Parameters
    ----------
    models : list[EFModel]
        List of EF models
    data : xr. Dataset
        Data

    Returns
    -------
    ef: xr.Dataset
        Evaporative fraction
    """
    # TODO handle mask
    ef = {}
    for m in models:
        m.fit(data)
        ef[m.name] = m.compute(data)
    return xr.Dataset(ef)


def select(ef: xr.Dataset) -> xr.Dataset:
    """
    Description
    -----------
    Select evaporative fraction from a list
    TODO: Implement selection

    Parameters
    ----------
    ef : xr. Dataset
        Evaporative Fraction

    Returns
    -------
    return: xr.Dataset
        Selected evaporative fraction
    """
    return ef


def merge(
    ef: xr.Dataset, keep: bool = False, method: MergeMethod = MergeMethod.MEAN
) -> xr.Dataset:
    """
    Description
    -----------
    Merge evaporative fraction from a list

    Parameters
    ----------
    ef : xr. Dataset
        Evaporative Fraction Data

    Returns
    -------
    return: xr.Dataset
        Merged evaporative fraction
    """
    if method == MergeMethod.MEAN:
        ef_merged = ef.to_array(dim="new").mean("new")
    else:
        raise EFModel("Merge method unknown")
    if keep:
        return ef.assign(ef=ef_merged)
    else:
        return xr.Dataset(
            data_vars=dict(ef=ef_merged), coords=ef.coords.copy(), attrs=ef.attrs.copy()
        )


def run(
    models: list[EFModel],
    data: xr.Dataset,
    selection: bool = False,
    keep: bool = False,
    merging: MergeMethod = MergeMethod.MEAN,
) -> xr.Dataset:
    """
    Description
    -----------
    Compute evaporative fraction using models.

    Parameters
    ----------
    models : list[EFModel]
        List of EF models
    data : xr. Dataset
        Data
    selection : bool
        Flag to apply selection
    keep : bool
        Flag to keep intermediate computations
    merging : MergeMethod
        Method used for merging

    Returns
    -------
    ef : xr. Dataset
        Evaporative fraction
    """
    # TODO Handle mask
    ef = compute(models, data)
    if selection:
        ef = select(ef)
    ef = merge(ef, keep=keep, method=merging)
    return ef
