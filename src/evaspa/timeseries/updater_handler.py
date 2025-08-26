# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for time series update
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from evaspa.timeseries.abstract_updater import Updater
from evaspa.timeseries.linear_updater import LinearUpdater, LinearUpdaterParams


class UpdateMethod(str, Enum):
    """
    List of methods for interpolation/extrapolation
    """

    linear = "linear"
    api = "api"  # Antecedent prepricipitation index


class UpdaterConfig(BaseModel):
    """
    Configuration for parameters for time series computation
    """

    model_config = ConfigDict(extra="forbid")

    method: UpdateMethod = Field(default=UpdateMethod.linear)
    params: dict[str, Any] = Field(default={})

    @field_validator("method")
    @classmethod
    def check_method(cls, value: UpdateMethod):
        if value == UpdateMethod.api:
            msg = (
                "Antecedent precipitation index for"
                " time series update is not implemented yet."
            )
            raise ValueError(msg)
        return value

    # Replace `params` with typed submodel during validation
    @model_validator(mode="after")
    def validate_params(self) -> UpdaterConfig:
        if self.method == UpdateMethod.linear:
            params_cfg = LinearUpdaterParams.model_validate(self.params)
            self.params = params_cfg.model_dump()
        else:
            msg = f"Unsupported method: {self.method}"
            raise ValueError(msg)
        return self


def create(method=UpdateMethod, params=dict) -> Updater:
    """
    Factory
    """
    if method == UpdateMethod.linear:
        cfg = LinearUpdaterParams.model_validate(params)
        return LinearUpdater(**cfg.model_dump())
    if method == UpdateMethod.api:
        msg = (
            "Antecedent precipitation index for"
            " time series update is not implemented yet."
        )
    # Raise error
    msg = f"Unsupported method: {method.value}"
    raise ValueError(msg)


def run(data: xr.Dataset, feed: xr.Dataset, config=dict):
    """
    Run update
    """
    updater_config = UpdaterConfig.model_validate(config)
    # Create updater
    updater = create(method=updater_config.method, params=updater_config.params)
    # Update time series
    return updater.update(data, feed=feed)
