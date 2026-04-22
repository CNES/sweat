# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing STIC model v1.4
"""

import numpy as np
import numpy.typing as npt
from numba import (
    boolean,
    njit,
    prange,
)
from numba import float32 as f32  # to define f32
from numba import int64 as i64  # to define i64
from numba.types import Array, Tuple

from sweat.common.constant import ETVar
from sweat.stic.constant import PSYCHROMETRIC_CST, PT_CST
from sweat.stic.models.flux import (
    compute_g_flux,
    compute_le_h_fluxes,
    initiate_le_h_fluxes,
)
from sweat.stic.models.functions import (
    compute_alpha_coefficient,
    compute_canopy_air_saturation_vapor_pressure,
    compute_canopy_air_vapor_pressure_deficit,
    compute_psychrometrics,
    compute_saturated_vapor_pressure,
    compute_state_equations,
    compute_wet_surface_temperature,
)
from sweat.stic.models.v1_4.smwetness import (
    initialize_soil_moisture,
    iterate_soil_moisture,
)

VERSION = "1.4"
IS_DEFAULT = False

# List of variables required by the model
VARIABLES_MAPPING: dict[str, str] = {
    ETVar.LST.value: "ts",
    ETVar.TEMPERATURE.value: "ta",
    ETVar.DEWPOINT_TEMPERATURE.value: "td",
    ETVar.RH.value: "rh",
    ETVar.FCOVER.value: "fc",
    ETVar.LAI.value: "lai",
    ETVar.NET_RADIATION.value: "rn",
    ETVar.LONGWAVE_NET_RADIATION.value: "ln",
    ETVar.NIR.value: "nir",
    ETVar.SWIR.value: "swir",
    ETVar.VARI.value: "vari_green",
    ETVar.GLI.value: "gli",
    ETVar.NDVI.value: "ndvi",
    ETVar.GNDVI.value: "gndvi",
    ETVar.MSAVI.value: "msavi",
    ETVar.LOCAL_TIME.value: "local_time",
}


@njit(
    Tuple((f32,) * 29)(
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
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        boolean,
    ),
    nogil=True,
    cache=True,
)
def init_stic_model_pixel(
    ts: float,
    ta: float,
    td: float,
    rh: float,
    fc: float,
    lai: float,
    rn: float,
    ln: float,
    nir: float,
    swir: float,
    vari_green: float,
    gli: float,
    ndvi: float,
    gndvi: float,
    msavi: float,
    local_time: float,
    debug: boolean,
) -> tuple[float, ...]:
    """
    STIC model initialization function for a single pixel

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
    nir: float
        Near infrared
    swir: float
        Shortwave infrared
    vari_green: float,
        VARI green index
    threshold: float
        Threshold value
    gli: float
        Green Leaf Index
    ndvi: float
        VARI green index
    gndvi: float
        VARI green index
    msavi: float
        VARI green index
    debug: bool
        Mode debug to print intermediate results


    Returns
    -------
    res: tuple[float,...]
        Output arrays
    """
    # 1. Compute psychrometrics
    # -------------------------

    # esstar: saturation vapor pressure at surface temperature (hPa)
    # ea: atmosphere vapour pressure (hPa) at air temperature
    # da: atmosphere vapour pressure deficit (hPa) at the reference height
    # slope slope of saturation vapor pressure versus air temperature
    # at TA (hPa/degC)
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
    if debug:
        print("Psychrometrics = ", esstar, ea, da, slope, rho, cp)  # noqa T201

    # 2. Initialize
    # -------------

    # Compute soi moisture
    # m: surface moisture availability (0 - 1)
    # m_soil: surface moisture availability for soil component
    # es: vapor pressure at surface temperature
    # t0d: dewpoint temperature at source/sink height
    # ds: vapor pressure deficit of the air at the surface
    (m, m_canopy, m_soil, m_surf, m_rz, es, t0d, ds) = initialize_soil_moisture(
        slope,
        ts,
        ta,
        td,
        ln,
        fc,
        ea,
        esstar,
        s1,
        s2,
        s3,
        s4,
        nir,
        swir,
        vari_green,
        gli,
        ndvi,
        gndvi,
        msavi,
    )
    if debug:
        print("Init SM = ", m, m_soil, s1, s2, s3, s4, es, ds)  # noqa T201

    # Initialize saturation vapor pressure at t0
    e0star = esstar
    # Initialize vapor pressure at t0
    e0 = es
    # Compute wet surface temperature
    tw = compute_wet_surface_temperature(ta, ea)
    esstar_w = compute_saturated_vapor_pressure(tw)
    # Initialize alpha to Priestley taylor parameter
    alpha = PT_CST
    # Compute G flux
    g_flux = compute_g_flux(rn, lai, local_time, m_soil)
    if debug:
        print("g_flux = ", g_flux)  # noqa T201

    # Compute available energy
    available_energy = rn - g_flux
    # Compute state equations
    (g_aero, g_surf, delta_t, ef) = compute_state_equations(
        rho, cp, alpha, slope, available_energy, e0, ea, e0star, m
    )
    if debug:
        print("State Eq. = ", g_aero, g_surf, delta_t, ef)  # noqa T201

    # Initialize t0
    t0 = delta_t + ta
    # Compute LE and H fluxes
    le_flux, h_flux = initiate_le_h_fluxes(
        slope, g_aero, g_surf, available_energy, da, rho, cp
    )
    if debug:
        print("LE/H Fluxes = ", le_flux, h_flux)  # noqa T201

    return (
        le_flux,
        h_flux,
        g_flux,
        g_aero,
        g_surf,
        t0,
        t0d,
        m,
        m_canopy,
        m_soil,
        m_surf,
        m_rz,
        da,
        ds,
        es,
        ea,
        e0,
        esstar,
        e0star,
        slope,
        s1,
        s2,
        s3,
        s4,
        rho,
        cp,
        alpha,
        tw,
        esstar_w,
    )


@njit(
    Tuple((f32, f32, f32, f32, f32, f32, f32, f32, boolean))(
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
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        i64,
        boolean,
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
    nir: float,
    swir: float,
    vari_green: float,
    gli: float,
    ndvi: float,
    gndvi: float,
    msavi: float,
    local_time: float,
    threshold: float,
    nb_steps: int,
    debug: boolean,
) -> tuple[float, float, float, float, float, float, float, float, bool]:
    """
    STIC model calculation function for a single pixel

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
    nir: float
        Near infrared
    swir: float
        Shortwave infrared
    vari_green: float
        VARI green index
    gli: float
        Green Leaf Index
    ndvi: float
        VARI green index
    gndvi: float
        VARI green index
    msavi: float
        VARI green index
    local_time: float
        Local time in seconds
    threshold: float
        Threshold value
    nb_steps: int
        Maximum of iteration number
    debug: bool
        Mode debug to print intermediate results


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
    # slope slope of saturation vapor pressure versus air temperature
    # at TA (hPa/degC)
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
    if debug:
        print("Psychrometrics = ", esstar, ea, da, slope, rho, cp)  # noqa T201

    # 2. Initialize
    # -------------

    # Compute soi moisture
    # m: surface moisture availability (0 - 1)
    # m_soil: surface moisture availability for soil component
    # es: vapor pressure at surface temperature
    # t0d: dewpoint temperature at source/sink height
    # ds: vapor pressure deficit of the air at the surface
    (m, _, m_soil, _, _, es, t0d, ds) = initialize_soil_moisture(
        slope,
        ts,
        ta,
        td,
        ln,
        fc,
        ea,
        esstar,
        s1,
        s2,
        s3,
        s4,
        nir,
        swir,
        vari_green,
        gli,
        ndvi,
        gndvi,
        msavi,
    )
    if debug:
        print("Init SM = ", m, m_soil, s1, s2, s3, s4, es, ds)  # noqa T201

    # Initialize saturation vapor pressure at t0
    e0star = esstar
    # Initialize vapor pressure at t0
    e0 = es
    # Compute wet surface temperature
    tw = compute_wet_surface_temperature(ta, ea)
    estar_w = compute_saturated_vapor_pressure(tw)
    # Initialize alpha to Priestley taylor parameter
    alpha = PT_CST
    if debug:
        print("Alpha = ", alpha, tw, estar_w)  # noqa T201
    # Save dewpoint temperature at source/sink height
    t0d_old = t0d
    # Compute G flux
    g_flux = compute_g_flux(rn, lai, local_time, m_soil)
    if debug:
        print("g_flux = ", g_flux)  # noqa T201

    # Compute available energy
    available_energy = rn - g_flux
    # Compute state equations
    (g_aero, g_surf, delta_t, ef) = compute_state_equations(
        rho, cp, alpha, slope, available_energy, e0, ea, e0star, m
    )
    if debug:
        print("State Eq. = ", g_aero, g_surf, delta_t, ef)  # noqa T201

    # Initialize t0
    t0 = delta_t + ta
    # Compute LE and H fluxes
    le_flux, h_flux = initiate_le_h_fluxes(
        slope, g_aero, g_surf, available_energy, da, rho, cp
    )
    if debug:
        print("LE/H Fluxes = ", le_flux, h_flux)  # noqa T201

    # 3. Iteration
    # ------------

    # Initialize iteration loop
    le_flux_old = le_flux
    le_error = f32(0.05)
    steps = 0
    converged = False

    # Iteration step
    while le_error > threshold and steps < nb_steps:
        if debug:
            print("step = ", steps)  # noqa T201
        # Re-estimate saturated vapor pressure at canopy/air height
        e0star = compute_canopy_air_saturation_vapor_pressure(
            le_flux, ea, esstar, g_aero, g_surf, rho, cp, f32(PSYCHROMETRIC_CST)
        )
        if debug:
            print("e0star = ", e0star)  # noqa T201

        # Re-estimate vapor pressure deficit at canopy/air height
        d0 = compute_canopy_air_vapor_pressure_deficit(
            slope, g_aero, g_surf, available_energy, da, ds, rho, cp
        )
        if debug:
            print("d0 = ", d0)  # noqa T201

        # Re-estimate vapor pressure at canopy/air height (hPa)
        e0 = e0star - d0
        if e0 < f32(0.0):
            e0 = es
        if e0 < ea:
            e0 = es
        if e0 > e0star:
            e0 = esstar
        if debug:
            print("e0 = ", e0)  # noqa T201

        # Re-estimate dewpoint temperature at source/sink height
        t0d = td + (f32(PSYCHROMETRIC_CST) * le_flux) / (rho * cp * g_aero * s1)
        if t0d < td:
            t0d = td
        if t0d > ts:
            t0d = t0d_old
        if debug:
            print("t0d = ", t0d)  # noqa T201

        # Re-estimate M (direct LST feedback into M computation)
        (m, _, _, m_soil, _) = iterate_soil_moisture(
            slope,
            s1,
            s2,
            s3,
            s4,
            ts,
            ta,
            td,
            t0d,
            ln,
            fc,
            ea,
            e0star,
            esstar,
            nir,
            swir,
            vari_green,
            gli,
            ndvi,
            gndvi,
            msavi,
        )
        if debug:
            print("SM = ", m, m_soil)  # noqa T201

        # Re-estimate PT coefficient
        alpha = compute_alpha_coefficient(
            slope, g_aero, g_surf, ta, t0, e0star, ea, m
        )
        if debug:
            print("alpha = ", alpha)  # noqa T201

        # Re-estimate net available energy
        g_flux = compute_g_flux(rn, lai, local_time, m)
        available_energy = rn - g_flux
        if debug:
            print("G flux = ", g_flux)  # noqa T201

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
        if debug:
            print("State Eq. = ", g_aero, g_surf, delta_t, ef)  # noqa T201

        t0 = delta_t + ta

        # Re-estimate latent heat flux
        le_flux, h_flux = compute_le_h_fluxes(
            slope, g_aero, g_surf, available_energy, da, ta, t0, ea, e0, rho, cp
        )
        if debug:
            print("LE/H fluxes = ", le_flux, h_flux)  # noqa T201

        # Error
        le_error = np.abs(le_flux_old - le_flux)
        le_flux_old = le_flux
        steps = steps + 1

        converged = le_error < threshold

    if debug:
        print("nb steps = ", steps, ", converged = ", converged)  # noqa T201
    # Final output from the STIC model
    # Compute EF
    ef = le_flux / (le_flux + h_flux)
    ef = min(max(ef, f32(0.0)), f32(1.0))

    return le_flux, h_flux, ef, g_flux, g_aero, g_surf, t0, m, converged


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
    ts: npt.NDArray,
    ta: npt.NDArray,
    td: npt.NDArray,
    rh: npt.NDArray,
    fc: npt.NDArray,
    lai: npt.NDArray,
    rn: npt.NDArray,
    ln: npt.NDArray,
    nir: npt.NDArray,
    swir: npt.NDArray,
    vari_green: npt.NDArray,
    gli: npt.NDArray,
    ndvi: npt.NDArray,
    gndvi: npt.NDArray,
    msavi: npt.NDArray,
    local_time: npt.NDArray,
    valid: npt.NDArray,
    threshold: float,
    nb_steps: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    STIC model calculation function

    Notes
    -----
    The method takes numpy arrays to calculate spatial outputs.

    Parameters
    ----------
    ts: np.ndarray
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
    nir: float
        Near infrared
    swir: float
        Shortwave infrared
    vari_green: float
        VARI green index
    gli: float
        Green Leaf Index
    ndvi: float
        VARI green index
    gndvi: float
        VARI green index
    msavi: float
        VARI green index
    local_time: float
        Local time in seconds
    threshold: float
        Threshold value
    nb_steps: int
        Maximum of iteration number

    Returns
    -------
    res: tuple[float]
        Output arrays (LE, EF, flags)
    """

    # Output array creation
    shape = ts.shape
    le_arr = np.empty_like(ts)
    ef_arr = np.empty_like(ts)
    converged_arr = np.empty_like(ts)
    # Pixel loop
    for i in prange(shape[0]):
        for j in prange(shape[1]):
            if valid[i, j] == 1:
                (
                    le_arr[i, j],
                    _,
                    ef_arr[i, j],
                    _,
                    _,
                    _,
                    _,
                    _,
                    converged_arr[i, j],
                ) = run_stic_model_pixel(
                    ts[i, j],
                    ta[i, j],
                    td[i, j],
                    rh[i, j],
                    fc[i, j],
                    lai[i, j],
                    rn[i, j],
                    ln[i, j],
                    nir[i, j],
                    swir[i, j],
                    vari_green[i, j],
                    gli[i, j],
                    ndvi[i, j],
                    gndvi[i, j],
                    msavi[i, j],
                    local_time[i, j],
                    threshold,
                    nb_steps,
                    False,
                )
            else:
                le_arr[i, j], ef_arr[i, j], converged_arr[i, j] = (
                    f32(np.nan),
                    f32(np.nan),
                    0,
                )

    return le_arr, ef_arr, converged_arr


def run_batch_stic_model(
    data: npt.NDArray,
    threshold: float,
    nb_steps: int,
    debug: bool = False,
) -> npt.NDArray:
    """
    Run

    """
    n = data.shape[0]
    out = np.empty((n, 8))

    for i in range(n):
        (le, h, ef, g, ga, gs, t0, m, _) = run_stic_model_pixel(
            ts=data[i, 0],
            ta=data[i, 1],
            td=data[i, 2],
            rh=data[i, 3],
            fc=data[i, 4],
            lai=data[i, 5],
            rn=data[i, 6],
            ln=data[i, 7],
            nir=data[i, 8],
            swir=data[i, 9],
            vari_green=data[i, 10],
            gli=data[i, 11],
            ndvi=data[i, 12],
            gndvi=data[i, 13],
            msavi=data[i, 14],
            local_time=data[i, 15],
            threshold=threshold,
            nb_steps=nb_steps,
            debug=debug,
        )
        out[i, 0] = le
        out[i, 1] = h
        out[i, 2] = ef
        out[i, 3] = g
        out[i, 4] = ga
        out[i, 5] = gs
        out[i, 6] = t0
        out[i, 7] = m

    return out


def run_batch_init_stic_model(
    data: npt.NDArray,
    debug: bool = False,
) -> npt.NDArray:
    """
    Run

    """
    n = data.shape[0]
    out = np.empty((n, 20))

    for i in range(n):
        (
            le,
            h,
            g,
            ga,
            gs,
            t0,
            t0d,
            m,
            _,
            _,
            m_surf,
            m_rz,
            da,
            ds,
            es,
            ea,
            e0,
            esstar,
            e0star,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            alpha,
            tw,
            esstar_w,
        ) = init_stic_model_pixel(
            ts=data[i, 0],
            ta=data[i, 1],
            td=data[i, 2],
            rh=data[i, 3],
            fc=data[i, 4],
            lai=data[i, 5],
            rn=data[i, 6],
            ln=data[i, 7],
            nir=data[i, 8],
            swir=data[i, 9],
            vari_green=data[i, 10],
            gli=data[i, 11],
            ndvi=data[i, 12],
            gndvi=data[i, 13],
            msavi=data[i, 14],
            local_time=data[i, 15],
            debug=debug,
        )
        out[i, 0] = le
        out[i, 1] = h
        out[i, 2] = g
        out[i, 3] = ga
        out[i, 4] = gs
        out[i, 5] = t0
        out[i, 6] = t0d
        out[i, 7] = m
        out[i, 8] = m_surf
        out[i, 9] = m_rz
        out[i, 10] = da
        out[i, 11] = ds
        out[i, 12] = es
        out[i, 13] = ea
        out[i, 14] = e0
        out[i, 15] = esstar
        out[i, 16] = e0star
        out[i, 17] = alpha
        out[i, 18] = tw
        out[i, 19] = esstar_w

    return out
