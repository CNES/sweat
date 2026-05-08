# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing methods for running STIC models
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from sweat.stic.registry import (
    DEFAULT_VERSION,
    MODEL_REGISTRY,
)


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
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Run STIC model on a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    threshold: float
        Threshold value
    nb_steps: int
        Max number of iterations
    version: str
        Model version

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

    spec = MODEL_REGISTRY[version]

    missing = [v for v in spec.inputs if v not in data]
    if missing:
        msg = f"Missing variables: {missing}"
        raise ValueError(msg)

    inputs = [data[v].data.astype(np.float32) for v in spec.inputs]
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

    spec = MODEL_REGISTRY[version]
    col_inputs = list(spec.inputs.keys())
    if mapping:
        col_inputs = list(spec.inputs.values())

    missing = [v for v in col_inputs if v not in data.columns]
    if missing:
        msg = f"Missing variables: {missing}"
        raise ValueError(msg)

    inputs = data[col_inputs].to_numpy()
    inputs = inputs.astype(np.float32)

    # Return un dataframe
    return spec.batch_func(
        inputs, threshold=threshold, nb_steps=nb_steps, debug=debug
    )


def run_batch_init_model(
    data: pd.DataFrame,
    version: str | None = None,
    debug: bool = False,
    mapping: bool = True,
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

    spec = MODEL_REGISTRY[version]
    col_inputs = list(spec.inputs.keys())
    if mapping:
        col_inputs = list(spec.inputs.values())

    missing = [v for v in col_inputs if v not in data.columns]
    if missing:
        msg = f"Missing variables: {missing}"
        raise ValueError(msg)

    inputs = data[col_inputs].to_numpy()
    inputs = inputs.astype(np.float32)
    # Return un dataframe
    return spec.init_func(inputs, debug=debug)
