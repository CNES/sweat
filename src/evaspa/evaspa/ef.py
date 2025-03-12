# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from typing_extensions import Self

from evaspa.debugging import register_debugging
from evaspa.evaspa.edge import Edge, EdgeConfig, EdgeError
from evaspa.evaspa.merging import MergeMethod, merge_to_dataset
from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


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

    model_config = ConfigDict(extra="forbid")

    selection: bool = Field(default=False)
    merging: MergeMethod = Field(default=MergeMethod.MEDIAN)


class EFCheckConfig(BaseModel):
    """
    Check variablity for EF processing
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.02)


def update_efconfig(v: Any) -> list[EFModel]:
    """
    Description
    -----------
    Update EF models with a configuration file
    stored in conf directory

    Parameters
    ----------
    v: Any
       Field

    Returns
    -------
    models: List[EFModel]
        List if EF Models
    """
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "conf",
    )
    if isinstance(v, str):
        filename = os.path.join(config_path, f"{v}.json")
        if not os.path.isfile(filename):
            msg = f"No config file: {filename}"
            raise OSError(msg)
        # Read config file
        with open(filename) as json_file:
            conf = json.load(json_file)
        return conf["models"]
    return v


def get_available_configuration() -> list[str]:
    """
    Description
    -----------
    Return the names of available configuration
    for preconfigured EF models

    Returns
    -------
    config_list: list[str]
        Names of available configuration
    """
    config_path = Path(os.path.dirname(os.path.abspath(__file__))) / "conf"
    filenames = config_path.glob("*.json")
    return [filename.stem for filename in filenames]


EFModels = Annotated[list[EFModelConfig], BeforeValidator(update_efconfig)]


class EFConfig(BaseModel):
    """
    Configuration for EF processing
    """

    model_config = ConfigDict(extra="forbid")

    models: EFModels
    options: EFOptionsConfig = Field(default=EFOptionsConfig())
    check: EFCheckConfig = Field(default=EFCheckConfig())

    @model_validator(mode="after")
    def check_model_name(self) -> Self:
        """
        Check that all models have a different name
        """
        names = [model.name for model in self.models]
        duplicates = [k for k, v in Counter(names).items() if v > 1]
        if len(duplicates) > 0:
            msg = f"All EF models must a different name: {duplicates}"
            raise EFConfigError(msg)
        return self


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
            msg = f"{self.var} not in the dataset"
            raise EFModelError(msg)
        if "lst" not in data.data_vars:
            msg = "lst not in the dataset"
            raise EFModelError(msg)
        if mask is not None:
            if mask not in data.data_vars:
                msg = f"No mask {mask}"
                raise ValueError(msg)
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

    def compute(
        self, data: xr.Dataset, mask: str | None = None
    ) -> xr.DataArray:
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
            msg = f"{self.var} not in the dataset"
            raise EFModelError(msg)
        if "lst" not in data.data_vars:
            msg = "lst not in the dataset"
            raise EFModelError(msg)
        if mask is not None:
            if mask not in data.data_vars:
                msg = f"No mask {mask}"
                raise ValueError(msg)
            data_masked = data.where(data[mask], drop=True)
        else:
            data_masked = data
        ef = (
            self.tdry(data_masked[self.var]) - np.array(data_masked["lst"])
        ) / (
            self.tdry(data_masked[self.var]) - self.twet(data_masked[self.var])
        )
        ef = np.where(ef > 1, 1, ef)
        ef = np.where(ef < 0, 0, ef)
        return data["lst"].copy(data=ef).assign_attrs(self.to_dict())

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

    def __repr__(self) -> str:
        """
        String conversion method
        """
        return (
            f"Model({self.name},dry_edge={self.dry_edge.__class__.__name__},"
            f"wet_edge={self.wet_edge.__class__.__name__},var={self.var})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
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
            msg = "Error in model configuration"
            raise EFConfigError(msg) from e
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
            dry_edge = Edge.create(
                efconfig.dry_edge.type,
                efconfig.dry_edge.config | {"position": "top"},
            )
        except EdgeError as e:
            msg = f"Error in dry edge creation for model {efconfig.name}"
            raise EFModelError(msg) from e
        # Wet edge
        try:
            wet_edge = Edge.create(
                efconfig.wet_edge.type,
                efconfig.wet_edge.config | {"position": "bottom"},
            )
        except EdgeError as e:
            msg = f"Error in wet edge creation for model {efconfig.name}"
            raise EFModelError(msg) from e

        return cls(
            name=efconfig.name,
            dry_edge=dry_edge,
            wet_edge=wet_edge,
            var=efconfig.var,
        )


def check_variability(
    lst: npt.ArrayLike,
    mask: npt.ArrayLike | None = None,
    threshold: float = 2.0,
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
        if t.shape != m.shape:
            msg = "LST and mask are not the same size"
            raise ValueError(msg)
    t = np.where(np.array(m) > 0, t, np.nan)
    # Compute the minimum and maximum values
    # of the relative difference from land surface temperature
    # and its median
    t_median = np.nanmedian(t)
    if abs(t_median) < 1.0:
        t_median = 1.0
    if ((np.nanmax(t) - np.nanmin(t)) / t_median) <= threshold:
        value = (np.nanmax(t) - np.nanmin(t)) / t_median
        msg = f"Value = {value} / Threshold = {threshold}"
        logger.debug(msg)
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
        msg = "Error in EF configuration"
        raise EFConfigError(msg) from e
    models = [EFModel.create(cfg.model_dump()) for cfg in efconfig.models]
    return (models, efconfig.options.model_dump())


def compute(
    models: list[EFModel], data: xr.Dataset, mask: str | None = None
) -> xr.Dataset:
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
    # TODO: handle mask
    ef = {}
    for m in models:
        m.fit(data, mask)
        ef[m.name] = m.compute(data, mask)
    return xr.Dataset(ef, coords=data.coords.copy(), attrs=data.attrs.copy())


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


@register_debugging
def run(
    models: list[EFModel],
    data: xr.Dataset,
    mask: str | None = None,
    selection: bool = False,
    merging: MergeMethod = MergeMethod.MEAN,
) -> tuple[xr.Dataset, xr.Dataset]:
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
        Evaporative fraction for all the models
    ef_merged : xr. Dataset
        Evaporative fraction merged
    """
    # TODO: Handle mask
    ef = compute(models, data, mask)
    if selection:
        ef = select(ef)
    return ef, merge_to_dataset(ef, method=merging, name="ef")
