# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing the API for STIC
"""

import numpy as np
import xarray as xr

from sweat.common import utils
from sweat.common.constant import (
    FLAGS_TYPE,
    MSK_PROCESSING_FAILED,
)
from sweat.common.flux import (
    compute_et_from_le,
    create_net_radiation,
    get_radiation_variables,
)
from sweat.common.types import ETVar
from sweat.logging import LoggerManager
from sweat.stic.convert import (
    convert_kelvin_to_celsius,
    convert_to_local_time,
    convert_to_rh,
)
from sweat.stic.registry import DEFAULT_VERSION, MODEL_REGISTRY
from sweat.stic.runner import run_model

logger = LoggerManager.get_logger(__name__)


def prepare(
    data: xr.Dataset,
    use_topo: bool = False,
    selected_radiation: str | None = None,
    version: str | None = None,
) -> xr.Dataset:
    """
    Prepare data for STIC

    Notes
    -----
    The following steps are performed:

    - Compute LST in celsius
    - Compute relative humidity
    - Compute local time
    - Mask pixel if nan in input data
    - Create flags mask
    - Compute vegetation indices required by the model if necessary

    Parameters
    ----------
    data: xr.Dataset
        Data
    use_topo: bool
        Activate topographic correction
    selected_radiation: str
        Select provider for radiation
    version: str
        Model version

    Returns
    -------
    prepared_data: tuple[float]
        Output arrays
    """
    # Default version if needed
    if version is None:
        version = DEFAULT_VERSION
    # Check version
    if version not in MODEL_REGISTRY:
        msg = f"Unknown run_stic_model version: {version}"
        raise ValueError(msg)
    spec = MODEL_REGISTRY[version]
    # Select radiation
    rsd_data, _ = get_radiation_variables(data)
    if selected_radiation is not None:
        if f"{ETVar.RSD.value}_{selected_radiation}" not in rsd_data:
            msg = f"Radiation data {selected_radiation} is missing"
            raise ValueError(msg)
        rsd_name = f"{ETVar.RSD.value}_{selected_radiation}"
    elif ETVar.RSD.value in data.data_vars:
        rsd_name = ETVar.RSD.value
    else:
        rsd_name = rsd_data[0]
    rld_name = str.replace(rsd_name, ETVar.RSD.value, ETVar.RLD.value, 1)
    fdiff_name = str.replace(rsd_name, ETVar.RSD.value, ETVar.FDIFF.value, 1)
    radiation_vars = [
        v for v in [rsd_name, rld_name, fdiff_name] if v in data.data_vars
    ]
    # Add DEM data, if available
    dem_vars = [
        d
        for d in data.data_vars
        if d in [ETVar.HEIGHT.value, ETVar.SLOPE.value, ETVar.ASPECT.value]
    ]
    # Check variables
    checked_vars = spec.required_inputs
    # Replace radiation
    checked_vars = [
        rsd_name if v == ETVar.RSD.value else v for v in checked_vars
    ]
    checked_vars = [
        rld_name if v == ETVar.RLD.value else v for v in checked_vars
    ]
    for var in checked_vars:
        if var not in data.data_vars:
            msg = f"Variable {var} is missing in the dataset"
            raise KeyError(msg)
    # Copy
    prepared_data = data.copy()
    # Extract dimensions, shape, coordinates
    dims = prepared_data[ETVar.LST.value].dims
    coords = prepared_data[ETVar.LST.value].coords
    # Converting temperature from Kelvin to Celsius degree
    prepared_data[ETVar.LST.value] = xr.DataArray(
        convert_kelvin_to_celsius(data[ETVar.LST.value]),
        dims=dims,
        coords=coords,
    )
    prepared_data[ETVar.TEMPERATURE.value] = xr.DataArray(
        convert_kelvin_to_celsius(data[ETVar.TEMPERATURE.value]),
        dims=dims,
        coords=coords,
    )
    prepared_data[ETVar.DEWPOINT_TEMPERATURE.value] = xr.DataArray(
        convert_kelvin_to_celsius(data[ETVar.DEWPOINT_TEMPERATURE.value]),
        dims=dims,
        coords=coords,
    )
    # Converting to relative humidity percentage
    prepared_data[ETVar.RH.value] = xr.DataArray(
        convert_to_rh(
            t2m=prepared_data[ETVar.TEMPERATURE.value],
            d2m=prepared_data[ETVar.DEWPOINT_TEMPERATURE.value],
        ),
        dims=dims,
        coords=coords,
    )
    # Convert to local time
    row_names = ["y", "lat", "latitude"]
    col_names = ["x", "lon", "longitude"]
    row = next((name for name in row_names if name in coords), None)
    col = next((name for name in col_names if name in coords), None)
    if row is None or col is None:
        msg = "Coordinate unknown for local time conversion"
        raise ValueError(msg)
    local_time = convert_to_local_time(
        prepared_data.attrs["date"],
        prepared_data.coords[col],
        prepared_data.coords[row],
        prepared_data.attrs["crs"],
    )
    prepared_data[ETVar.LOCAL_TIME.value] = ((row, col), local_time)
    # Compute net radiation
    rn_xr, ln_xr = create_net_radiation(
        data[
            [
                ETVar.LST.value,
                ETVar.ALBEDO.value,
                ETVar.EMISSIVITY.value,
                *radiation_vars,
                *dem_vars,
            ]
        ],
        use_topo=use_topo,
    )
    rn_xr = next(iter(rn_xr.data_vars.values()))
    ln_xr = next(iter(ln_xr.data_vars.values()))
    prepared_data[ETVar.NET_RADIATION.value] = rn_xr
    prepared_data[ETVar.LONGWAVE_NET_RADIATION.value] = ln_xr
    # Compute vegetation indices
    if (
        ETVar.NDVI.value in spec.inputs
        and ETVar.NDVI.value not in prepared_data.data_vars
    ):
        prepared_data[ETVar.NDVI.value] = xr.DataArray(
            utils.compute_ndvi(
                nir=prepared_data[ETVar.NIR.value],
                red=prepared_data[ETVar.RED.value],
            ),
            dims=dims,
            coords=coords,
        )
    if (
        ETVar.GNDVI.value in spec.inputs
        and ETVar.GNDVI.value not in prepared_data.data_vars
    ):
        prepared_data[ETVar.GNDVI.value] = xr.DataArray(
            utils.compute_gndvi(
                nir=prepared_data[ETVar.NIR.value],
                green=prepared_data[ETVar.GREEN.value],
            ),
            dims=dims,
            coords=coords,
        )
    if (
        ETVar.GLI.value in spec.inputs
        and ETVar.GLI.value not in prepared_data.data_vars
    ):
        prepared_data[ETVar.GLI.value] = xr.DataArray(
            utils.compute_gli(
                blue=prepared_data[ETVar.BLUE.value],
                green=prepared_data[ETVar.GREEN.value],
                red=prepared_data[ETVar.RED.value],
            ),
            dims=dims,
            coords=coords,
        )
    if (
        ETVar.VARI.value in spec.inputs
        and ETVar.VARI.value not in prepared_data.data_vars
    ):
        prepared_data[ETVar.VARI.value] = xr.DataArray(
            utils.compute_vari_green(
                blue=prepared_data[ETVar.BLUE.value],
                green=prepared_data[ETVar.GREEN.value],
                red=prepared_data[ETVar.RED.value],
            ),
            dims=dims,
            coords=coords,
        )
    if (
        ETVar.MSAVI.value in spec.inputs
        and ETVar.MSAVI.value not in prepared_data.data_vars
    ):
        prepared_data[ETVar.MSAVI.value] = xr.DataArray(
            utils.compute_msavi(
                nir=prepared_data[ETVar.NIR.value],
                red=prepared_data[ETVar.RED.value],
            ),
            dims=dims,
            coords=coords,
        )
    return prepared_data


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
    # Default version if needed
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
    le_arr, ef_arr, e_interception_arr, e_soil_arr, t_arr, converged_arr, _ = (
        run_model(
            data=data,
            valid=valid_arr,
            threshold=threshold,
            nb_steps=nb_steps,
            version=version,
        )
    )
    converged_arr = converged_arr.astype(np.int64)
    et_arr = compute_et_from_le(le_arr)
    # Process non-converged pixels
    le_arr[converged_arr == 0] = np.nan
    et_arr[converged_arr == 0] = np.nan
    ef_arr[converged_arr == 0] = np.nan
    e_interception_arr[converged_arr == 0] = np.nan
    e_soil_arr[converged_arr == 0] = np.nan
    t_arr[converged_arr == 0] = np.nan
    flags_arr = np.where(
        converged_arr == 0,
        flags_arr | MSK_PROCESSING_FAILED,
        flags_arr,
    )
    valid_arr = valid_arr & converged_arr
    dims = data[ETVar.LST.value].dims
    # Create dataset and add uncertainty (set to 0 for now)
    return xr.Dataset(
        data_vars={
            ETVar.LE.value: (dims, le_arr),
            ETVar.ET.value: (dims, et_arr),
            ETVar.EF.value: (dims, ef_arr),
            ETVar.EVAPORATION_INTERCEPTION.value: (dims, e_interception_arr),
            ETVar.EVAPORATION_SOIL.value: (dims, e_soil_arr),
            ETVar.TRANSPIRATION.value: (dims, t_arr),
            ETVar.UNCERTAINTY_LE.value: (dims, np.zeros_like(le_arr)),
            ETVar.UNCERTAINTY_ET.value: (dims, np.zeros_like(et_arr)),
            ETVar.UNCERTAINTY_EF.value: (dims, np.zeros_like(ef_arr)),
            ETVar.VALID.value: (dims, valid_arr.astype(FLAGS_TYPE)),
            ETVar.FLAGS.value: (dims, flags_arr.astype(FLAGS_TYPE)),
        },
        coords=data.coords,
        attrs=data.attrs.copy(),
    )
