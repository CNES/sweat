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

import sweat.timeseries.status_handler as sh
from sweat.timeseries.abstract_updater import Updater
from sweat.timeseries.linear_updater import LinearUpdater, LinearUpdaterParams
from sweat.timeseries.types import TimeSeriesVar as TSVar


class UpdateMethod(str, Enum):
    """
    List of methods for interpolation/extrapolation
    """

    linear = "linear"
    api = "api"  # Antecedent precipitation index


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

    # Replace `params` with typed derived model during validation
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
    Factory to create a updater instance

    Parameters
    ----------
    method : UpdateMethod
        Type of updater to create
    params : dict
        Parameters used to configure the updater

    Returns
    -------
    updater : Updater
        Updater instance
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


def run(data: xr.Dataset, feed: xr.Dataset | None = None, config=dict):
    """
    Run update

    Parameters
    ----------
    data : xr.Dataset
        Data to update
    feed : xr.Dataset
        Data corresponding new acquisitions
    config : dict
        Configuration for update the time series

    Returns
    -------
    updated : xr.Dataset
        Updated data
    """
    updater_config = UpdaterConfig.model_validate(config)
    # Create updater
    updater = create(method=updater_config.method, params=updater_config.params)
    # Decode status
    (
        data[TSVar.UPDATED.value],
        data[TSVar.STATE.value],
        data[TSVar.DISTANCE.value],
        data[TSVar.VALIDITY_FLAGS.value],
    ) = sh.decode_status(data[TSVar.FLAGS.value])
    data = data.drop_vars(TSVar.FLAGS.value)
    # Update time series
    updated_data = updater.update(data, feed=feed)
    # Processing failed
    updated_data[TSVar.VALIDITY_FLAGS.value] = sh.set_bit(
        updated_data[TSVar.VALIDITY_FLAGS.value],
        sh.PROCESSING_BIT_POSITION,
        updated_data[TSVar.ET.value].isnull(),
        force_zero=False,
    )
    # Encode status
    updated_data[TSVar.FLAGS.value] = sh.encode_status(
        updated_data[TSVar.UPDATED.value],
        updated_data[TSVar.STATE.value],
        updated_data[TSVar.DISTANCE.value],
        updated_data[TSVar.VALIDITY_FLAGS.value],
    )
    return updated_data.drop_vars(
        [
            TSVar.UPDATED.value,
            TSVar.STATE.value,
            TSVar.DISTANCE.value,
            TSVar.VALIDITY_FLAGS.value,
        ]
    )
