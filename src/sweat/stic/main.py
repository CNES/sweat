# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing the API for STIC
"""

import numpy as np
import xarray as xr

from sweat.common.constant import (
    FLAGS_TYPE,
    ETVar,
)
from sweat.common.flux import compute_et_from_le, create_net_radiation
from sweat.logging import LoggerManager
from sweat.stic.convert import (
    convert_kelvin_to_celsius,
    convert_to_local_time,
    convert_to_rh,
)
from sweat.stic.registry import DEFAULT_VERSION
from sweat.stic.runner import run_model

logger = LoggerManager.get_logger(__name__)

# if bit 1 activated : The pixel is invalid : STIC not converged
MSK_STIC_NOT_CONVERGED = 1 << 3


def prepare(
    data: xr.Dataset,
    use_topo: bool = False,
    selected_radiation: str | None = None,
) -> xr.Dataset:
    """
    Prepare data for STIC

    The following steps are performed:

    - Compute LST in celsius
    - Compute relative humidity
    - Compute local time
    - Mask pixel if nan in input data
    - Create flags mask

    Parameters
    ----------
    data: xr.Dataset
        Data
    use_topo: bool
        Activate topographic correction
    selected_radiation: str
        Select provider for radiation

    Returns
    -------
    res: tuple[float]
        Output arrays
    """
    # Check
    checked_vars = [
        ETVar.LST,
        ETVar.ALBEDO,
        ETVar.TEMPERATURE,
        ETVar.DEWPOINT_TEMPERATURE,
        ETVar.FCOVER,
        ETVar.LAI,
        ETVar.EMISSIVITY,
    ]
    for var in checked_vars:
        if var.value not in data.data_vars:
            msg = f"Variable {var.value} is missing in the dataset"
            raise KeyError(msg)
    # Copy
    new_data = data.copy()
    # Converting temperature from Kelvin to Celsius degree
    new_data[ETVar.LST.value] = xr.apply_ufunc(
        convert_kelvin_to_celsius, data[ETVar.LST.value]
    )
    new_data[ETVar.TEMPERATURE.value] = xr.apply_ufunc(
        convert_kelvin_to_celsius, data[ETVar.TEMPERATURE.value]
    )
    new_data[ETVar.DEWPOINT_TEMPERATURE.value] = xr.apply_ufunc(
        convert_kelvin_to_celsius, data[ETVar.DEWPOINT_TEMPERATURE.value]
    )
    # Converting to relative humidity percentage
    new_data[ETVar.RH.value] = xr.apply_ufunc(
        convert_to_rh,
        new_data[ETVar.TEMPERATURE.value],
        new_data[ETVar.DEWPOINT_TEMPERATURE.value],
    )
    # Convert to local time
    new_data[ETVar.LOCAL_TIME.value] = xr.apply_ufunc(
        convert_to_local_time,
        new_data.attrs["date"],
        new_data.coords["x"],
        new_data.coords["y"],
        new_data.attrs["crs"],
    )
    # Compute net radiation
    rn_xr, ln_xr = create_net_radiation(data, use_topo=use_topo)
    if selected_radiation is not None:
        rn_xr = rn_xr[selected_radiation]
        ln_xr = ln_xr[selected_radiation]
    else:
        rn_xr = next(iter(rn_xr.data_vars.values()))
        ln_xr = next(iter(ln_xr.data_vars.values()))
    new_data[ETVar.NET_RADIATION.value] = rn_xr
    new_data[ETVar.LONGWAVE_NET_RADIATION.value] = ln_xr

    return new_data


def run(
    data: xr.Dataset,
    threshold: float = 0.01,
    nb_steps: int = 15,
    version: str | None = None,
) -> xr.Dataset:
    """
    Run STIC

    Parameters
    ----------
    data: xr.Dataset
        Data
    threshold: float
        Threshold value (default=0.01)
    nb_steps: int
        Maximum of iteration number (default = 15)
    version: str
        Model version

    Returns
    -------
    res: tuple[float]
        Output arrays
    """
    if version is None:
        version = DEFAULT_VERSION
    # Get valid and flags
    if ETVar.VALID.value in data.data_vars:
        valid_arr = data[ETVar.VALID.value].data.astype(np.int64)
    else:
        valid_arr = np.ones_like(data[ETVar.LST.value].data, dtype=np.int64)
    if ETVar.FLAGS.value in data.data_vars:
        flags_arr = data[ETVar.FLAGS.value].data
    else:
        flags_arr = np.ones_like(data[ETVar.LST.value].data, dtype=FLAGS_TYPE)

    # Run STIC main loop
    le_arr, ef_arr, converged_arr = run_model(
        data=data,
        valid=valid_arr,
        threshold=threshold,
        nb_steps=nb_steps,
        version=version,
    )
    converged_arr = converged_arr.astype(np.int64)
    et_arr = compute_et_from_le(le_arr)
    # Process non-converged pixels
    le_arr[converged_arr == 0] = np.nan
    et_arr[converged_arr == 0] = np.nan
    ef_arr[converged_arr == 0] = np.nan
    flags_arr = np.where(
        converged_arr == 0,
        flags_arr | MSK_STIC_NOT_CONVERGED,
        flags_arr,
    )
    valid_arr = valid_arr & converged_arr
    dims = data[ETVar.LST.value].dims
    return xr.Dataset(
        data_vars={
            ETVar.LE.value: (dims, le_arr),
            ETVar.ET.value: (dims, et_arr),
            ETVar.EF.value: (dims, ef_arr),
            ETVar.VALID.value: (dims, valid_arr.astype(FLAGS_TYPE)),
            ETVar.FLAGS.value: (dims, flags_arr.astype(FLAGS_TYPE)),
        },
        coords=data.coords,
        attrs=data.attrs.copy(),
    )
