# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing debugging functions
"""

from __future__ import annotations

import functools
import json
import os
import time

from pydantic import BaseModel, ConfigDict, Field
from xarray import DataArray, Dataset

from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


REGISTERED_FUNCTIONS = {}


class DebuggingConfig(BaseModel):
    """
    Configuration for debugging
    """

    model_config = ConfigDict(extra="forbid")

    profile: bool = Field(default=False)
    verbose: bool = Field(default=False)
    path: str = Field(default=os.path.join(os.getcwd(), "debug"))


class _DebugDecorator:
    """
    Class decorator used for profiling and
    for debugging
    """

    def __init__(self, func, profile: bool = False, verbose: bool = False):
        functools.update_wrapper(self, func)
        self.func = func
        self.out = os.getcwd()
        self.name = func.__name__
        self.module = func.__module__
        self.profile = profile
        self.verbose = verbose

    def configure(self, profile: bool, verbose: bool, path: str):
        self.profile = profile
        self.verbose = verbose
        self.out = path

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = self.func(*args, **kwargs)
        end_time = time.perf_counter()
        if self.profile:
            msg = f"Function {self.module}.{self.name} executed in {(end_time - start_time):.4f}s"
            logger.info(msg)
        if self.verbose and result is not None:
            msg = f"Write results from {self.module}.{self.name}"
            logger.debug(msg)
            if isinstance(result, Dataset | DataArray):
                filename = f"{self.module}_{self.name}.nc"
                self._to_netcdf(result, filename)
            elif isinstance(result, tuple):
                for i, sub_result in enumerate(result):
                    if isinstance(sub_result, Dataset | DataArray):
                        filename = f"{self.module}_{self.name}_{i}.nc"
                        self._to_netcdf(sub_result, filename)
            else:
                msg = "Unable to write results"
                logger.debug(msg)

        return result

    def _to_netcdf(self, xarr: DataArray | Dataset, filename: str):
        """
        Write result to netcdf
        """
        msg = f"Writing {filename}"
        logger.debug(msg)
        data = xarr.copy()
        for key, value in data.attrs.items():
            if value is None:
                data.attrs[key] = "none"
            elif isinstance(value, dict):
                data.attrs[key] = json.dumps(value)
            else:
                data.attrs[key] = str(value)
        if isinstance(data, Dataset):
            for var in data.data_vars:
                for key, value in data[var].attrs.items():
                    if value is None:
                        data[var].attrs[key] = "none"
                    elif isinstance(value, dict):
                        data[var].attrs[key] = json.dumps(value)
                    else:
                        data[var].attrs[key] = str(value)
        data.to_netcdf(os.path.join(self.out, filename))


def debugging(profile: bool, verbose: bool):
    """
    Register a function
    """

    def wrapper_debugging(func):
        return _DebugDecorator(func, profile, verbose)

    return wrapper_debugging


def register_debugging(func):
    """
    Register a function for debugging
    """
    decorated_func = _DebugDecorator(func, profile=False, verbose=False)
    REGISTERED_FUNCTIONS[f"{func.__module__}.{func.__name__}"] = decorated_func
    return decorated_func


def get_registered_functions():
    return REGISTERED_FUNCTIONS


def configure_debugging(profile: bool, verbose: bool, path: str | None = None):
    """
    Configure debug mode for all registered functions
    """
    if path is None:
        path = os.getcwd()
    msg = f"Debugging configuration: profile={profile}, verbose={verbose}, path={path}"
    logger.debug(msg)
    for func in REGISTERED_FUNCTIONS.values():
        func.configure(profile, verbose, path)
