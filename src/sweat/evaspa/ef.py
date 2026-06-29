# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for Evaporative Fraction model management
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Self

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

from sweat.common.constant import (
    MSK_INPUT_FILTERED,
    MSK_INPUT_FILTERED_DURING_PROCESSING,
)
from sweat.common.filter import FilteringConfig, find_valid_pixels
from sweat.common.types import ETVar
from sweat.debugging import register_debugging
from sweat.evaspa.edge import Edge, EdgeConfig, EdgeError
from sweat.evaspa.merging import (
    MergingConfig,
    MergingMethod,
    UncertaintyMethod,
    merge_to_dataset,
)
from sweat.logging import LoggerManager

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

    filtering: FilteringConfig = Field(default=FilteringConfig({}))
    selection: bool = Field(default=False)
    merging: MergingConfig = Field(
        default=MergingConfig(
            merging_method=MergingMethod.MEDIAN,
            uncertainty_method=UncertaintyMethod.INTERQUARTILE,
        )
    )


class EFCheckConfig(BaseModel):
    """
    Check variability for EF processing
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.02)


def update_efconfig(v: Any) -> list[EFModel]:
    """
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
            msg = f"No EVASPA configuration file: {filename}"
            raise OSError(msg)
        # Read config file
        with open(filename) as json_file:
            conf = json.load(json_file)
        return conf["models"]
    return v


def get_available_configuration() -> list[str]:
    """
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

    def fit(self, data: xr.Dataset, mask: xr.DataArray | None = None) -> None:
        """
        Method to estimate dry and wet edges.
        data is a xarray.Dataset which must contain "lst"
        and "var" as data variables.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing the data
        mask : xr.DataArray
            Data used for masking
        """
        if self.var not in data.data_vars:
            msg = f"{self.var} not in the dataset"
            raise EFModelError(msg)
        if ETVar.LST.value not in data.data_vars:
            msg = "lst not in the dataset"
            raise EFModelError(msg)
        data_masked = data.where(mask) if mask is not None else data
        self.wet_edge.fit(data_masked[self.var], data_masked[ETVar.LST.value])
        self.dry_edge.fit(data_masked[self.var], data_masked[ETVar.LST.value])

    def tdry(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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
        self,
        data: xr.Dataset,
        mask: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """
        Method to compute evaporative fraction

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing the data
        mask : xr.DataArray
            Data used for masking
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
        if ETVar.LST.value not in data.data_vars:
            msg = "lst not in the dataset"
            raise EFModelError(msg)
        data_masked = data.where(mask) if mask is not None else data
        ef = (
            self.tdry(data_masked[self.var])
            - np.array(data_masked[ETVar.LST.value])
        ) / (
            self.tdry(data_masked[self.var]) - self.twet(data_masked[self.var])
        )
        ef = np.clip(ef, 0, 1)
        return data[ETVar.LST.value].copy(data=ef).assign_attrs(self.to_dict())

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
        Create an EF model from a configuration.

        Notes
        -----
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
    Check variability of Land Surface Temperature

    Notes
    -----
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
        Threshold value to check variability

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
    return (models, efconfig.options.model_dump(by_alias=True))


def get_variables_from_models(models: list[EFModel]) -> list[str]:
    """
    Get variables required for EF models

    Parameters
    ----------
    models : list[EFModel]
        List of EF models

    Returns
    -------
    variables: list[str]
        List of variables required in EF models
    """
    variables = [ETVar.LST.value]
    if len(models) > 0:
        variables += [m.var for m in models]
    return list(set(variables))


def compute(
    models: list[EFModel],
    data: xr.Dataset,
    mask: xr.DataArray | None = None,
    model_mask: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Compute evaporative fraction of a list of EF models

    Parameters
    ----------
    models : list[EFModel]
        List of EF models
    data : xr. Dataset
        Data
    mask : xr. DataArray
        Mask used for EF computation
    model_mask : xr.DataArray
        Mask used for edge computation,
        if not provided mask for EF computation is used

    Returns
    -------
    ef: xr.Dataset
        Evaporative fraction
    """
    if mask is not None and model_mask is None:
        model_mask = mask
    ef = {}
    for m in models:
        m.fit(data, mask=model_mask)
        ef[m.name] = m.compute(data, mask=mask)
    return xr.Dataset(ef, coords=data.coords.copy(), attrs=data.attrs.copy())


def select(ef: xr.Dataset) -> xr.Dataset:
    """
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
    filtering: dict | None = None,
    selection: bool = False,
    merging: dict | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
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
    merging : dict
        Configuration used for merging

    Returns
    -------
    ef : xr. Dataset
        Evaporative fraction for all the models
    ef_merged : xr. Dataset
        Evaporative fraction merged
    """
    # Check if the variables required by the models are present in the data
    variables = get_variables_from_models(models)
    for v in variables:
        if v not in data.data_vars:
            msg = f"Variable {v} is missing to compute EF from EF models"
            raise KeyError(msg)
    mask = None
    if ETVar.VALID.value in data.data_vars:
        mask = data[ETVar.VALID.value]
    # Filter data for EF models
    model_mask, model_flags = find_valid_pixels(
        data,
        nan_config=variables,
        valid_config=filtering,
    )
    # Compute EF models
    ef = compute(models, data, mask=mask, model_mask=model_mask)
    # Select EF models
    if selection:
        ef = select(ef)
    # Merge EF models
    if merging is None:
        merging_config = MergingConfig().model_dump()
    else:
        merging_config = MergingConfig.model_validate(merging).model_dump()
    merged = merge_to_dataset(ef, name=ETVar.EF.value, **merging_config)
    # Propagate masks
    if ETVar.FLAGS.value in data.data_vars:
        flags = data[ETVar.FLAGS.value]
    if ETVar.VALID.value in data.data_vars:
        valid = data[ETVar.VALID.value]
    if ETVar.FLAGS.value not in data.data_vars or (
        ETVar.VALID.value not in data.data_vars
    ):
        valid, flags = find_valid_pixels(
            data,
            nan_config=[ETVar.LST.value],
            valid_config=None,
        )
    # Identify pixel filtered just for edge computation
    flags = xr.where(
        ((model_flags & MSK_INPUT_FILTERED) != 0)
        & ((flags & (1 << 0)) == 0)
        & ((flags & (1 << 1)) == 0),
        flags | MSK_INPUT_FILTERED_DURING_PROCESSING,
        flags,
    )
    ef[ETVar.VALID.value] = valid
    ef[ETVar.FLAGS.value] = flags
    merged[ETVar.VALID.value] = valid
    merged[ETVar.FLAGS.value] = flags
    return ef, merged
