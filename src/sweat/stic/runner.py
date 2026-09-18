# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing methods for running STIC models
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from sweat.stic.registry import (
    DEFAULT_VERSION,
    MODEL_REGISTRY,
)

TYPES = Literal["float32", "float64"]


def check_model(data: xr.Dataset, version: str | None) -> bool:
    """
    Check if all variables are available to run the model

    Parameters
    ----------
    data: xr.Dataset
        Data
    version: str


    Returns
    -------
    checked: bool
        Return True if the dataset contains the required variables
    """
    if version is None:
        version = DEFAULT_VERSION
    if version not in MODEL_REGISTRY:
        msg = f"Unknown run_stic_model version: {version}"
        raise ValueError(msg)

    spec = MODEL_REGISTRY[version]

    missing = [v for v in spec.inputs if v not in data]
    if missing:
        msg = f"Missing variables: {missing}"
        raise ValueError(msg)
    return True


def run_model(
    data: xr.Dataset,
    valid: npt.ArrayLike | None = None,
    threshold: float = 0.01,
    nb_steps: int = 15,
    version: str | None = None,
    precision: TYPES = "float32",
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    """
    Run STIC model on a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    valid: npt.ArrayLike
        Valid mask
    threshold: float
        Threshold value
    nb_steps: int
        Max number of iterations
    version: str
        Model version
    precision: str
        Precision used for computation (float32 or float64)

    Returns
    -------
    results: tuple[np.array,np.array,np.array]
        Results of the model
    """
    if version is None:
        version = DEFAULT_VERSION
    if version not in MODEL_REGISTRY:
        msg = f"Unknown run_stic_model version: {version}"
        raise ValueError(msg)
    dtype = np.dtype(precision)

    spec = MODEL_REGISTRY[version]

    missing = [v for v in spec.inputs if v not in data]
    if missing:
        msg = f"Missing variables: {missing}"
        raise ValueError(msg)

    inputs = [data[v].data.astype(dtype) for v in spec.inputs]
    if valid is None:
        valid = np.ones_like(inputs[0], dtype=np.int64)
    return spec.raster_func(
        *inputs,
        valid=np.array(valid).astype(np.int64),
        threshold=threshold,
        nb_steps=nb_steps,
    )


def run_batch_model(
    data: pd.DataFrame,
    threshold: float = 0.01,
    nb_steps: int = 15,
    version: str | None = None,
    debug: bool = False,
    mapping: bool = True,
    precision: TYPES = "float32",
) -> npt.NDArray:
    """
    Run STIC model on a dataframe

    Parameters
    ----------
    data: pd.DataFrame
        Data
    threshold: float
        Threshold value
    nb_steps: int
        Max number of iterations
    version: str
        Model version
    debug: bool
        Use debug mode
    mapping: bool
        Used mapping for variables
    precision: str
        Precision used for computation (float32 or float64)

    Returns
    -------
    results: tuple[np.array,np.array,np.array]
        Results of the model
    """
    if version is None:
        version = DEFAULT_VERSION
    if version not in MODEL_REGISTRY:
        msg = f"Unknown run_batch_stic_model version: {version}"
        raise ValueError(msg)
    dtype = np.dtype(precision)

    spec = MODEL_REGISTRY[version]
    col_inputs = list(spec.inputs.keys())
    if mapping:
        col_inputs = list(spec.inputs.values())

    missing = [v for v in col_inputs if v not in data.columns]
    if missing:
        msg = f"Missing variables: {missing}"
        raise ValueError(msg)

    # Convert to numpy array
    inputs = np.asarray(data[col_inputs].to_numpy(), order="C", dtype=dtype)
    inputs.setflags(write=True)

    # Return un dataframe
    return spec.batch_func(
        inputs, threshold=threshold, nb_steps=nb_steps, debug=debug
    )


def run_batch_init_model(
    data: pd.DataFrame,
    version: str | None = None,
    debug: bool = False,
    mapping: bool = True,
    precision: TYPES = "float32",
) -> npt.NDArray:
    """
    Run STIC model initialization on a dataframe

    Parameters
    ----------
    data: pd.DataFrame
        Data
    version: str
        Model version
    debug: bool
        Use debug mode
    mapping: bool
        Used mapping for variables
    precision: str
        Precision used for computation (float32 or float64)

    Returns
    -------
    results: tuple[np.array,np.array,np.array]
        Results of the model
    """
    if version is None:
        version = DEFAULT_VERSION
    if version not in MODEL_REGISTRY:
        msg = f"Unknown run_batch_stic_model version: {version}"
        raise ValueError(msg)
    dtype = np.dtype(precision)

    spec = MODEL_REGISTRY[version]
    col_inputs = list(spec.inputs.keys())
    if mapping:
        col_inputs = list(spec.inputs.values())

    missing = [v for v in col_inputs if v not in data.columns]
    if missing:
        msg = f"Missing variables: {missing}"
        raise ValueError(msg)

    # Convert to numpy array
    inputs = np.asarray(data[col_inputs].to_numpy(), order="C", dtype=dtype)
    inputs.setflags(write=True)

    # Return un dataframe
    return spec.init_func(inputs, debug=debug)
