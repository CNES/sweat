#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt
import xarray as xr

from pydantic import BaseModel, ValidationError

from .edge import Edge, create_edge, EdgeError, EdgeConfig


class MergeMethod(Enum):
    """Method for merge EF mdoels"""

    AVERAGE = "average"


class EFModelConfig(BaseModel):
    """
    Configuration of a EF model
    """

    name: str
    dry_edge: EdgeConfig
    wet_edge: EdgeConfig
    var: str


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
        assert self.var in data.data_vars
        assert "lst" in data.data_vars
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
        return: np.array
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
        return: np.array
        """
        return self.wet_edge.get(var)

    def compute(self, data: xr.Dataset, mask: str | None = None) -> npt.NDArray:
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
        return: np.array
        """
        assert self.var in data.data_vars
        assert "lst" in data.data_vars
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
        return ef

    def to_json(self) -> dict:
        """
        Return an dictionary
        """
        return {
            "name": self.name,
            "dry_edge": {
                "type": self.dry_edge.__class__,
                "config": self.dry_edge.model_dump_json(),
            },
            "wet_edge": {
                "type": self.wet_edge.__class__,
                "config": self.wet_edge.model_dump_json(),
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


def check_efmodel(config: dict) -> EFModelConfig:
    """
    Description
    -----------
    Check an EF model configuration

    Parameters
    ----------
    config : dict
        Configuration for EF model
    return: EFModelConfig
    """
    try:
        efconfig = EFModelConfig.model_validate(config)
    except ValidationError as e:
        raise EFConfigError("Error in model configuration") from e
    return efconfig


def create_efmodel(config: dict) -> EFModel:
    """
    Description
    -----------
    Create an EF model from a configuration

    Parameters
    ----------
    config : dict
        Configuration for EF model
    return: EFModel
    """
    # Read configuration
    efconfig = check_efmodel(config)

    # Dry edge
    try:
        dry_edge = create_edge(efconfig.dry_edge.type, efconfig.dry_edge.config)
    except EdgeError as e:
        raise EFModelError("Error in dry edge creation") from e

    # Wet edge
    try:
        wet_edge = create_edge(efconfig.wet_edge.type, efconfig.wet_edge.config)
    except EdgeError as e:
        raise EFModelError("Error in wet edge creation") from e

    return EFModel(
        name=efconfig.name, dry_edge=dry_edge, wet_edge=wet_edge, var=efconfig.var
    )


def check_variability(lst: npt.ArrayLike) -> bool:
    """
    Description
    -----------
    Check variability of Land Surface Temperature

    Parameters
    ----------
    lst : np.array_like
        Land surface temperature
    return: bool
    """
    return True


def initialize(config: dict):
    """
    Description
    -----------
    Check variability of Land Surface Temperature

    Parameters
    ----------
    lst : np.array_like
        Land surface temperature
    return: bool
    """
    pass
