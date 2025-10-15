# Copyright: (c) 2025 LIST / CESBIO / Centre National d'Etudes Spatiales
"""
Module containing the API for STIC
"""

import numpy as np
import numpy.typing as npt
import xarray as xr
from numba import (
    boolean,
    njit,
    prange,
)
from numba import float32 as f32  # to define f32
from numba import int64 as i64  # to define i64
from numba.types import Array, Tuple
from pydantic import BaseModel, ConfigDict, Field

from sweat.common.constant import (
    FLAGS_TYPE,
    ETVar,
)
from sweat.common.flux import compute_et_from_le, create_net_radiation
from sweat.logging import LoggerManager
from sweat.stic.constant import PSYCHROMETRIC_CST, PT_CST
from sweat.stic.flux import (
    compute_g_flux,
    compute_le_h_fluxes,
    initiate_le_h_fluxes,
)
from sweat.stic.functions import (
    compute_canopy_air_saturation_vapor_pressure,
    compute_canopy_air_vapor_pressure_deficit,
    compute_psychrometrics,
    compute_state_equations,
    convert_kelvin_to_celsius,
    convert_to_local_time,
    convert_to_rh,
)
from sweat.stic.smwetness import (
    initialize_soil_moisture,
    iterate_soil_moisture,
)

logger = LoggerManager.get_logger(__name__)

# if bit 1 activated : The pixel is invalid : STIC not converged
MSK_STIC_NOT_CONVERGED = 1 << 3


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


@njit(
    Tuple((f32, f32, boolean))(
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        i64,
    ),
    nogil=True,
    cache=True,
)
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
    threshold: float,
    nb_steps: int,
) -> tuple[float, float, bool]:
    """
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
        Threshold value
    nb_steps: int
        Maximum of ietration number


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
    ) = compute_psychrometrics(ts, ta, td, rh)

    # 2. Initialize
    # -------------

    # Compute soi moisture
    # m: surface moisture avalilability (0 - 1)
    # m_soil: surface moisture availability for soil component
    # es: vapor pressure at surface temperature
    # t0d: dewpoint temperature at source/sink height
    # ds: vapor pressure deficit of the air at the surface
    (m, _, m_soil, _, _, es, t0d, ds) = initialize_soil_moisture(
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
    # Initialize saturation vapor pressure at t0
    e0star = esstar
    # Initialiaze vapor pressure at t0
    e0 = es
    # Initialize alpha to Priestley taylor parameter
    alpha = f32(PT_CST)
    # Save dewpoint temperature at source/sink height
    t0d_old = t0d
    # Compute G flux
    g_flux = compute_g_flux(rn, lai, local_time, m_soil)
    # Compute available energy
    available_energy = rn - g_flux
    # Compute state equestions
    (g_aero, g_surf, delta_t, ef) = compute_state_equations(
        rho, cp, alpha, slope, available_energy, e0, ea, e0star, m
    )
    # Initialize t0
    t0 = delta_t + ta
    # Compute LE and H fluxes
    le_flux, h_flux = initiate_le_h_fluxes(
        slope, g_aero, g_surf, available_energy, da, rho, cp
    )
    # TODO: To check
    slope0 = (e0star - ea) / (t0 - td)

    # 3. Iteration
    # ------------

    # Initialize iteration loop
    le_flux_old = le_flux
    le_error = f32(0.05)
    steps = 0
    converged = False

    # Iteration step
    while le_error > threshold and steps < nb_steps:
        # Re-estimate saturated vapor pressure at canopy/air height
        e0star = compute_canopy_air_saturation_vapor_pressure(
            le_flux, ea, esstar, g_aero, g_surf, rho, cp, f32(PSYCHROMETRIC_CST)
        )

        # Re-estimate vapor pressure deficit at canopy/air height
        d0 = compute_canopy_air_vapor_pressure_deficit(
            slope, g_aero, g_surf, available_energy, da, ds, rho, cp
        )

        # Re-estimate vapor pressure at canopy/air height (hPa)
        e0 = e0star - d0
        if e0 < f32(0.0):
            e0 = es
        if e0 < ea:
            e0 = es
        if e0 > e0star:
            e0 = es
        # TODO: To check difference with STIC-JPL
        # if e0 < f32(0.0):
        #    e0 = es
        # if e0 > e0star:
        #    e0 = e0star

        # Re-estimate dewpoint temperature at source/sink height
        t0d = td + (f32(PSYCHROMETRIC_CST) * le_flux) / (rho * cp * g_aero * s1)
        if t0d < td:
            t0d = td
        if t0d > ts:
            t0d = t0d_old

        # Re-estimate M (direct LST feedback into M computation)
        (m, _, _, m_soil, _) = iterate_soil_moisture(
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
        # TODO: Check formulation (slope0)
        alpha = (
            g_surf
            * slope0
            * (t0 - td)
            * (
                f32(2) * slope
                + f32(2) * f32(PSYCHROMETRIC_CST)
                + f32(PSYCHROMETRIC_CST) * (g_aero / g_surf) * (f32(1) + m)
            )
        ) / (
            f32(2)
            * slope
            * (
                f32(PSYCHROMETRIC_CST) * (t0 - ta) * (g_aero + g_surf)
                + g_surf * slope0 * (t0 - td)
            )
        )
        # TODO: Explanation
        if alpha < f32(0.0):
            alpha = f32(1.0)
        alpha = min(alpha, f32(2.0))

        # Re-estimate net available energy
        g_flux = compute_g_flux(rn, lai, local_time, m)
        available_energy = rn - g_flux

        # Re-estimate conductances and states
        (g_aero, g_surf, delta_t, ef) = compute_state_equations(
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
        le_flux, h_flux = compute_le_h_fluxes(
            slope, g_aero, g_surf, available_energy, da, ta, t0, ea, e0, rho, cp
        )

        # TODO: To check
        slope0 = (
            (PSYCHROMETRIC_CST * le_flux) / (rho * cp * g_surf) + (e0 - ea)
        ) / (t0 - td)

        # Error
        le_error = np.abs(le_flux_old - le_flux)
        le_flux_old = le_flux
        steps = steps + 1

        converged = le_error < threshold

    # Final output from the STIC model
    # Compute EF
    ef = le_flux / (le_flux + h_flux)
    ef = min(max(ef, f32(0.0)), f32(1.0))

    return le_flux, ef, converged


@njit(
    Tuple((Array(f32, 2, "C"), Array(f32, 2, "C"), Array(f32, 2, "C")))(
        Array(f32, 2, "C"),
        Array(f32, 2, "C"),
        Array(f32, 2, "C"),
        Array(f32, 2, "C"),
        Array(f32, 2, "C"),
        Array(f32, 2, "C"),
        Array(f32, 2, "C"),
        Array(f32, 2, "C"),
        Array(f32, 2, "C"),
        Array(i64, 2, "C"),
        f32,
        i64,
    ),
    nogil=True,
    parallel=True,
    cache=True,
)
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
    valid: npt.NDArray,
    threshold: float,
    nb_steps: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
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
        Threshold value
    nb_steps: int
        Maximum of ietration number

    Returns
    -------
    res: tuple[float]
        Output arrays (LE, EF, flags)
    """

    # Output array creation
    shape = lst.shape
    le_arr = np.empty_like(lst)
    ef_arr = np.empty_like(lst)
    converged_arr = np.empty_like(lst)
    # Pixel loop
    for i in prange(shape[0]):
        for j in prange(shape[1]):
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
            else:
                le_arr[i, j], ef_arr[i, j], converged_arr[i, j] = (
                    f32(np.nan),
                    f32(np.nan),
                    0,
                )

    return le_arr, ef_arr, converged_arr


def prepare(
    data: xr.Dataset,
    use_topo: bool = False,
    selected_radiation: str | None = None,
) -> xr.Dataset:
    """
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

    Returns
    -------
    res: tuple[float]
        Output arrays
    """
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
    le_arr, ef_arr, converged_arr = run_stic_model(
        data[ETVar.LST.value].data.astype(np.float32),
        data[ETVar.TEMPERATURE.value].data.astype(np.float32),
        data[ETVar.DEWPOINT_TEMPERATURE.value].data.astype(np.float32),
        data[ETVar.RH.value].data.astype(np.float32),
        data[ETVar.FCOVER.value].data.astype(np.float32),
        data[ETVar.LAI.value].data.astype(np.float32),
        data[ETVar.NET_RADIATION.value].data.astype(np.float32),
        data[ETVar.LONGWAVE_NET_RADIATION.value].data.astype(np.float32),
        data[ETVar.LOCAL_TIME.value].data.astype(np.float32),
        valid=valid_arr,
        threshold=threshold,
        nb_steps=nb_steps,
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
