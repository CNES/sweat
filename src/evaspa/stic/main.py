# Copyright: (c) 2025 LIST / CESBIO / Centre National d'Etudes Spatiales
"""
Module containing the API for STIC
"""

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from evaspa.common.constant import MSK_INPUT_NODATA, PSYCHROMETRIC_CST, ETVar
from evaspa.common.flux import compute_et_from_le, create_net_radiation
from evaspa.logging import LoggerManager
from evaspa.stic.flux import f_g_actualsurface
from evaspa.stic.functions import (
    convert_to_celsius,
    convert_to_local_time,
    convert_to_rh,
    f_psychrometrics,
    f_stateeq,
)
from evaspa.stic.smwetness import (
    f_soilmoisture_initialize,
    f_soilmoisture_iterate,
)

logger = LoggerManager.get_logger(__name__)

# Constants
ALPHA = 1.26


class STICPrepareConfig(BaseModel):
    """
    Configuration for parameters to STIC model
    """

    model_config = ConfigDict(extra="forbid")
    use_topo: bool = Field(default=False)
    selected_radiation: str | None = Field(default=None)


class STICModelConfig(BaseModel):
    """
    Configuration for parameters to STIC model
    """

    model_config = ConfigDict(extra="forbid")
    threshold: float = Field(default=0.01)
    nb_steps: int = Field(default=15)


def run_stic_model_pixel(
    ts: float,
    ta: float,
    td: float,
    rh: float,
    fc: float,
    lai: float,
    rn: float,
    ln: float,
    local_time: float,
    threshold: float = 0.01,
    nb_steps: int = 15,
) -> tuple[float, float, bool]:
    """
    Description
    -----------
    STIC model calulation function for a single pixel

    Parameters
    ----------
    ts: float
        Surface temperature (in Celsius)
    ta: float
        Air temperature (in Celsius)
    td: float
        Dewpoint air temperature (in Celsius)
    rh: float
        Relative humidity (percentage)
    fc: float
        Fraction cover
    lai: float
        Leaf Area Index
    rn: float
        Net radiation
    ln: float
        Longwave net radiation
    local_time: float
        Local time in seconds
    threshold: float
        Threshold value (default=0.01)
    nb_steps: int
        Maximum of ietration number (default = 15)


    Returns
    -------
    res: tuple[float]
        Output arrays
    """
    # 1. Compute psychrometrics
    # -------------------------

    # esstar: saturation vapor pressure at surface temperature (hPa)
    # ea: atmosphere vapour pressure (hPa) at air temperature
    # da: atmosphere vapour pressure deficit (hPa) at the reference height
    # slope slope of saturation vapor pressure versus air temperature at TA (hPa/degC)
    # s1,s2,s3,s4: slopes of saturation vapor pressure versus temperature
    # rho: air density (kg.m-3)
    # cp: specific heat of air at constant pressure (J.kg-1.K-1)
    (
        esstar,
        _,
        ea,
        da,
        slope,
        s1,
        s2,
        s3,
        s4,
        rho,
        cp,
    ) = f_psychrometrics(ts, ta, td, rh)
    # 2. Initialize
    # -------------

    # Compute soi moisture
    # m: surface moisture avalilability (0 - 1)
    # m_soil: surface moisture availability for soil component
    # es: vapor pressure at surface temperature
    # t0d: dewpoint temperature at source/sink height
    # ds: vapor pressure deficit of the air at the surface
    (m, _, m_soil, _, _, es, t0d, ds) = f_soilmoisture_initialize(
        slope,
        ts,
        ta,
        td,
        rn,
        ln,
        fc,
        da,
        ea,
        esstar,
        s1,
        s2,
        s3,
        s4,
    )
    # Saturation vapor pressure at t0
    e0star = esstar
    # Vapor pressure at t0
    e0 = es
    # Priestley taylor parameter
    alpha = ALPHA
    # Save dewpoint temperature at source/sink height
    t0d_old = t0d

    g_flux = f_g_actualsurface(rn, lai, local_time, m_soil)
    available_energy = rn - g_flux

    (g_aero, g_surf, delta_t, ef) = f_stateeq(
        rho, cp, alpha, slope, available_energy, e0, ea, e0star, m
    )
    t0 = delta_t + ta
    slope0 = (e0star - ea) / (t0 - td)

    # Calculate ET and H based on initial results from state eqs.
    # McNaughton and Jarvis (1986)
    omega = ((slope / PSYCHROMETRIC_CST) + 1) / (
        (slope / PSYCHROMETRIC_CST) + 1 + g_aero / g_surf
    )
    le_flux_eq = (available_energy * (slope / PSYCHROMETRIC_CST)) / (
        (slope / PSYCHROMETRIC_CST) + 1
    )
    le_flux_imp = (cp * 0.0289644 / PSYCHROMETRIC_CST) * g_surf * 40 * da
    le_flux = omega * le_flux_eq + (1 - omega) * le_flux_imp

    h_flux = (
        PSYCHROMETRIC_CST * available_energy * (1 + g_aero / g_surf)
        - rho * cp * g_aero * da
    ) / (
        slope + PSYCHROMETRIC_CST * (1 + g_aero / g_surf)
    )  # Deduced from the PM equation

    # 3. Iteration
    # ------------

    # Initialize iteration
    le_flux_old = le_flux
    le_error = 0.05
    steps = 0
    converged = False

    # Iteration step
    while le_error > threshold and steps < nb_steps:
        # Re-estimate saturated vapor pressure at source/sink height
        e0star = ea + (PSYCHROMETRIC_CST * le_flux * (g_aero + g_surf)) / (
            rho * cp * g_aero * g_surf
        )
        e0star = e0star if e0star >= 0 else esstar
        e0star = e0star if e0star < 250 else 250  # noqa PLR2004

        # Re-estimate vapor pressure at source/sink height
        d0 = (
            (g_aero / g_surf)
            * (
                PSYCHROMETRIC_CST
                / (slope + PSYCHROMETRIC_CST * (1 + g_aero / g_surf))
            )
            * (da + ((slope * available_energy) / (rho * cp * g_aero)))
        )
        d0 = d0 if d0 >= 0 else ds

        e0 = e0star - d0
        e0 = es if e0 < 0 else e0
        e0 = es if e0 < ea else e0
        e0 = es if e0 > e0star else e0

        # Re-estimate M (direct LST feedback into M computation)
        t0d = td + (PSYCHROMETRIC_CST * le_flux) / (rho * cp * g_aero * s1)
        t0d = td if t0d < td else t0d
        t0d = t0d_old if t0d > ts else t0d

        (m, _, _, m_soil, _) = f_soilmoisture_iterate(
            slope,
            s1,
            s2,
            s3,
            s4,
            ts,
            ta,
            delta_t,
            td,
            t0d,
            rn,
            ln,
            fc,
            da,
            d0,
            ea,
            e0star,
            esstar,
        )

        # Re-estimate PT coefficient
        alpha = (
            g_surf
            * slope0
            * (t0 - td)
            * (
                2 * slope
                + 2 * PSYCHROMETRIC_CST
                + PSYCHROMETRIC_CST * (g_aero / g_surf) * (1 + m)
            )
        ) / (
            2
            * slope
            * (
                PSYCHROMETRIC_CST * (t0 - ta) * (g_aero + g_surf)
                + g_surf * slope0 * (t0 - td)
            )
        )
        # TODO: Explanation
        alpha = 1.0 if alpha < 0.0 else alpha
        alpha = 2.0 if alpha > 2.0 else alpha  # noqa PRL2004

        # Re-estimate net available energy
        g_flux = f_g_actualsurface(rn, lai, local_time, m)
        available_energy = rn - g_flux

        # Re-estimate conductances and states
        (g_aero, g_surf, delta_t, ef) = f_stateeq(
            rho,
            cp,
            alpha,
            slope,
            available_energy,
            e0,
            ea,
            e0star,
            m,
        )

        t0 = delta_t + ta

        # Re-estimate latent heat flux
        le_flux = (
            (rho * cp / PSYCHROMETRIC_CST)
            * ((g_aero * g_surf) / (g_aero + g_surf))
            * (slope * (t0 - ta) + da)
        )

        if (le_flux < 0) & (le_flux < available_energy):
            le_flux = rho * cp * g_aero * (e0 - ea) / PSYCHROMETRIC_CST

        slope0 = (
            (PSYCHROMETRIC_CST * le_flux) / (rho * cp * g_surf) + (e0 - ea)
        ) / (t0 - td)

        # Error
        le_error = np.abs(le_flux_old - le_flux)
        le_flux_old = le_flux
        steps = steps + 1

        converged = le_error < threshold

    # Final output from the STIC model
    # Compute H flux
    h_flux = (
        PSYCHROMETRIC_CST * available_energy * (1 + g_aero / g_surf)
        - rho * cp * g_aero * da
    ) / (
        slope + PSYCHROMETRIC_CST * (1 + g_aero / g_surf)
    )  # Deduced from the PM equation
    # Compute EF
    ef = le_flux / (le_flux + h_flux)
    ef = np.clip(ef, 0, 1)

    return le_flux, ef, converged


# @njit(types.Tuple([f32[:,:]] * 11)(f32[:,:], f32[:,:], f32[:,:], f32[:,:], f32[:,:], f32[:,:], f32[:,:], f32[:,:], f32[:,:], f32[:,:], f32, f32, f32, f32, f32, f32, f32, f32, i64, boolean), nogil = True, parallel = True, cache = True)
def run_stic_model(
    lst: npt.NDArray,
    ta: npt.NDArray,
    td: npt.NDArray,
    rh: npt.NDArray,
    fc: npt.NDArray,
    lai: npt.NDArray,
    rn: npt.NDArray,
    ln: npt.NDArray,
    local_time: npt.NDArray,
    valid: npt.NDArray | None = None,
    threshold: float = 0.01,
    nb_steps: int = 15,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Description
    -----------
    STIC model calulation function.
    Takes in numpy arrays and constants to calculate spatial outputs.

    Parameters
    ----------
    lst: np.ndarray
        Land surface temperature (in Celsius)
    ta: float
        Air temperature (in Celsius)
    td: float
        Dewpoint temperature (in Celsius)
    rh: float
        Relative humidity (percentage)
    fc: float
        Fraction cover
    lai: float
        Leaf Area Index
    rn: float
        Net radiation
    ln: float
        Longwave net radiation
    local_time: float
        Local time in seconds
    threshold: float
        Threshold value (default=0.01)
    nb_steps: int
        Maximum of ietration number (default = 15)

    Returns
    -------
    res: tuple[float]
        Output arrays (LE, ET, EF, flags)
    """

    # Output array creation
    shape = lst.shape
    le_arr = np.nan * np.ones_like(lst, dtype=float)
    ef_arr = np.nan * np.ones_like(lst, dtype=float)
    converged_arr = np.zeros_like(lst, dtype=float)
    if valid is None:
        valid = np.ones_like(lst)
    # Pixel loop
    for i in range(shape[0]):
        for j in range(shape[1]):
            if valid[i, j] == 1:
                le_arr[i, j], ef_arr[i, j], converged_arr[i, j] = (
                    run_stic_model_pixel(
                        lst[i, j],
                        ta[i, j],
                        td[i, j],
                        rh[i, j],
                        fc[i, j],
                        lai[i, j],
                        rn[i, j],
                        ln[i, j],
                        local_time[i, j],
                        threshold,
                        nb_steps,
                    )
                )
    et_arr = compute_et_from_le(le_arr)
    return le_arr, et_arr, ef_arr, converged_arr


def prepare(
    data: xr.Dataset,
    use_topo: bool = False,
    selected_radiation: str | None = None,
) -> xr.Dataset:
    """
    Description
    -----------
    Prepare data for STIC:
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
        convert_to_celsius, data[ETVar.LST.value]
    )
    new_data[ETVar.TEMPERATURE.value] = xr.apply_ufunc(
        convert_to_celsius, data[ETVar.TEMPERATURE.value]
    )
    new_data[ETVar.DEWPOINT_TEMPERATURE.value] = xr.apply_ufunc(
        convert_to_celsius, data[ETVar.DEWPOINT_TEMPERATURE.value]
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

    # Process nan data
    new_data[ETVar.VALID.value] = (
        (~new_data[ETVar.LST.value].isnull())
        & (~new_data[ETVar.ALBEDO.value].isnull())
        & (~new_data[ETVar.TEMPERATURE.value].isnull())
        & (~new_data[ETVar.DEWPOINT_TEMPERATURE.value].isnull())
        & (~new_data[ETVar.FCOVER.value].isnull())
        & (~new_data[ETVar.LAI.value].isnull())
        & (~new_data[ETVar.EMISSIVITY.value].isnull())
        & (~new_data[ETVar.NET_RADIATION.value].isnull())
        & (~new_data[ETVar.LONGWAVE_NET_RADIATION.value].isnull())
    ).astype(int)
    # TODO: Check type for flags uint8 or uint16
    new_data[ETVar.FLAGS.value] = xr.full_like(
        new_data[ETVar.VALID.value], 0, dtype=np.uint8
    )
    new_data[ETVar.FLAGS.value] = xr.where(
        new_data[ETVar.VALID.value] == 0,
        new_data[ETVar.FLAGS.value] | MSK_INPUT_NODATA,
        new_data[ETVar.FLAGS.value],
    )
    return new_data


def run(
    data: xr.Dataset,
    threshold: float = 0.01,
    nb_steps: int = 15,
) -> xr.Dataset:
    """
    Description
    -----------
    Run STIC

    Parameters
    ----------
    data: xr.Dataset
        Data
    threshold: float
        Threshold value (default=0.01)
    nb_steps: int
        Maximum of iteration number (default = 15)

    Returns
    -------
    res: tuple[float]
        Output arrays
    """
    # Get valid
    valid = None
    if ETVar.VALID.value in data.data_vars:
        valid = data[ETVar.VALID.value].data

    # Run STIC main loop
    le_arr, et_arr, ef_arr, converged_arr = run_stic_model(
        data[ETVar.LST.value].data,
        data[ETVar.TEMPERATURE.value].data,
        data[ETVar.DEWPOINT_TEMPERATURE.value].data,
        data[ETVar.RH.value].data,
        data[ETVar.FCOVER.value].data,
        data[ETVar.LAI.value].data,
        data[ETVar.NET_RADIATION.value].data,
        data[ETVar.LONGWAVE_NET_RADIATION.value].data,
        data[ETVar.LOCAL_TIME.value].data,
        valid=valid,
        threshold=threshold,
        nb_steps=nb_steps,
    )

    # Process non-converged pixels
    le_arr[converged_arr == 0] = np.nan
    et_arr[converged_arr == 0] = np.nan
    ef_arr[converged_arr == 0] = np.nan
    dims = data[ETVar.LST.value].dims
    return xr.Dataset(
        data_vars={
            ETVar.LE.value: (dims, le_arr),
            ETVar.ET.value: (dims, et_arr),
            ETVar.EF.value: (dims, ef_arr),
        },
        coords=data.coords,
        attrs=data.attrs.copy(),
    )
