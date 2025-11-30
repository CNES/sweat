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
from sweat.timeseries.io_handler import (
    TimeSeriesInputConfig,
    WindowTimeSeriesInputConfig,
)
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
    Class describing the format of the input file for timeseries run
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


class TimeSeriesDebuggingConfig(DebuggingConfig):
    """
    Debugging configuration for timeseries
    """

    model_config = ConfigDict(extra="forbid")

    config_verbose: bool = Field(default=False)
    config_dir: str = Field(default=os.path.join(os.getcwd(), "config_dir"))

    def get_debug_config(self) -> dict:
        """
        Get the debugging configuration
        """
        return self.model_dump(exclude={"config_verbose", "config_dir"})

    def get_timeseries_config(self) -> dict:
        """
        Get the configuration related to timeseries
        """
        return self.model_dump(include={"config_verbose", "config_dir"})


class WindowTimeSeriesInputFile(BaseModel):
    """
    Class describing the format of the input file for timeseries run
    over a window
    """

    model_config = ConfigDict(extra="forbid")

    input: WindowTimeSeriesInputConfig
    output: OutputConfig
    params: TimeSeriesParamsConfig = Field(default=TimeSeriesParamsConfig())
    debug: TimeSeriesDebuggingConfig = Field(
        default=TimeSeriesDebuggingConfig()
    )
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
    def check(self):
        """
        Check consistency between all variables
        """
        # Output directory
        if self.input.et_time_series_dir is not None:
            if self.input.et_time_series_dir != self.output.path:
                msg = (
                    "if ET timeseries directory is set, "
                    "it must be identical to output path"
                )
                raise ValueError(msg)
        else:
            self.input.et_time_series_dir = self.output.path
        # Debug variables
        self.debug.path = os.path.join(self.output.path, "debug")
        self.debug.config_dir = os.path.join(self.output.path, "config_dir")
        # Create debug if necessary
        if self.debug.verbose:
            os.makedirs(self.debug.path, exist_ok=True)
        if self.debug.config_verbose:
            os.makedirs(self.debug.config_dir, exist_ok=True)
        return self


def check_config_timeseries(config: dict) -> dict:
    """
    Check configuration for timeseries

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


def check_config_window_timeseries(config: dict) -> dict:
    """
    Check configuration for timeseries over a window

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration parameters

    Returns
    -------
    checked_config: dict
        Checked dictionary containing the configuration parameters
    """
    cfg = WindowTimeSeriesInputFile.model_validate(config)
    return cfg.model_dump(mode="json")
