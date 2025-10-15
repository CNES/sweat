# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for time series configuration management
"""

from __future__ import annotations

import os

from packaging.version import Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sweat.__about__ import __version__
from sweat.common.io import OutputConfig
from sweat.debugging import DebuggingConfig
from sweat.logging import LoggerManager
from sweat.timeseries.io_handler import TimeSeriesInputConfig
from sweat.timeseries.stack_handler import TimeSeriesStackConfig
from sweat.timeseries.updater_handler import UpdaterConfig

logger = LoggerManager.get_logger(__name__)


class TimeSeriesParamsConfig(BaseModel):
    """
    Class describing the format of the input file
    """

    model_config = ConfigDict(extra="forbid")

    stack: TimeSeriesStackConfig = Field(default=TimeSeriesStackConfig())
    update: UpdaterConfig = Field(default=UpdaterConfig())


class TimeSeriesInputFile(BaseModel):
    """
    Class describing the format of the input file
    """

    model_config = ConfigDict(extra="forbid")

    input: TimeSeriesInputConfig
    output: OutputConfig
    params: TimeSeriesParamsConfig = Field(default=TimeSeriesParamsConfig())
    debug: DebuggingConfig = Field(default=DebuggingConfig())
    version: str = Field(default=str(__version__))

    @field_validator("version")
    @classmethod
    def update_version(cls, v: str) -> str:
        """ "
        Update version in configuration
        """
        if Version(v) > Version(__version__):
            msg = "File generated with a newer version"
            logger.warning(msg)
        return str(__version__)

    @model_validator(mode="after")
    def update_debug(self):
        """
        Update debug path with output path
        """
        self.debug.path = os.path.join(self.output.path, "debug")
        # Create debug if necessary
        if self.debug.verbose:
            os.makedirs(self.debug.path, exist_ok=True)
        return self


def check_config_timeseries(config: dict) -> dict:
    """
    Check configuration

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration parameters

    Returns
    -------
    checked_config: dict
        Checked dictionary containing the configuration parameters
    """
    cfg = TimeSeriesInputFile.model_validate(config)
    return cfg.model_dump(mode="json")
