# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pytest

from sweat.stic.models import functions


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "ts",
        "ta",
        "td",
        "rh",
        "esstar_expected",
        "eastar_expected",
        "ea_expected",
        "da_expected",
        "slope_expected",
        "s1_expected",
        "s2_expected",
        "s3_expected",
        "s4_expected",
        "rho_expected",
        "cp_expected",
    ),
    [
        pytest.param(
            30,
            25,
            19,
            69.36,
            42.635799,
            31.830929,
            22.077932,
            9.752996,
            1.896020,
            1.369556,
            1.868897,
            2.440350,
            1.625499,
            1.174905,
            1016.492676,
        ),
        pytest.param(
            35,
            28,
            25,
            83.79,
            56.498649,
            37.982050,
            31.825160,
            6.156890,
            2.211536,
            1.887862,
            2.467349,
            3.120363,
            2.052297,
            1.159068,
            1021.627518,
        ),
        pytest.param(
            36,
            28,
            15,
            45.1,
            59.697203,
            37.982050,
            17.129904,
            20.852145,
            2.211536,
            1.098263,
            2.027014,
            3.273146,
            1.604011,
            1.165317,
            1013.900388,
        ),
        pytest.param(
            15,
            15,
            15,
            100.0,
            17.135910,
            17.135910,
            17.135910,
            0.000000,
            1.103222,
            1.098263,
            1.098263,
            1.098263,
            1.098263,
            1.217888,
            1013.903503,
        ),
    ],
)
def test_compute_psychrometrics(
    ts,
    ta,
    td,
    rh,
    esstar_expected,
    eastar_expected,
    ea_expected,
    da_expected,
    slope_expected,
    s1_expected,
    s2_expected,
    s3_expected,
    s4_expected,
    rho_expected,
    cp_expected,
) -> None:
    """
    Test function for computing psychrometrics
    """
    (
        esstar,
        eastar,
        ea,
        da,
        slope,
        s1,
        s2,
        s3,
        s4,
        rho,
        cp,
    ) = functions.compute_psychrometrics(ts, ta, td, rh)
    np.testing.assert_almost_equal(esstar, esstar_expected, decimal=3)
    np.testing.assert_almost_equal(eastar, eastar_expected, decimal=3)
    np.testing.assert_almost_equal(ea, ea_expected, decimal=3)
    np.testing.assert_almost_equal(da, da_expected, decimal=3)
    np.testing.assert_almost_equal(slope, slope_expected, decimal=3)
    np.testing.assert_almost_equal(s1, s1_expected, decimal=3)
    np.testing.assert_almost_equal(s2, s2_expected, decimal=3)
    np.testing.assert_almost_equal(s3, s3_expected, decimal=3)
    np.testing.assert_almost_equal(s4, s4_expected, decimal=3)
    np.testing.assert_almost_equal(rho, rho_expected, decimal=3)
    np.testing.assert_almost_equal(cp, cp_expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "rho",
        "cp",
        "alpha",
        "slope",
        "phi",
        "e0",
        "ea",
        "e0star",
        "m",
        "g_aero_expected",
        "g_surf_expected",
        "delta_t_expected",
        "ef_expected",
    ),
    [
        pytest.param(
            0.05,
            1010,
            1.26,
            1.9,
            100,
            30,
            20,
            30,
            0.3,
            0.1,
            0.1,
            1.0972705395329152,
            0.9315175097276263,
        ),
        pytest.param(
            0.04,
            1026,
            1.3,
            2.0,
            50,
            32.5,
            45.7,
            51.9,
            0.7,
            0.0001,
            0.06,
            5.811882893226179,
            1,
        ),
        pytest.param(
            0.04,
            1026,
            1.3,
            2.0,
            50,
            32.5,
            45.7,
            32.5,
            0.7,
            0.0001,
            0.0001,
            -0.53,
            0.974,
        ),
        pytest.param(
            0.04,
            1026,
            1.3,
            2.0,
            50,
            32.5,
            32.5,
            32.5,
            0.7,
            0.1,
            0.0001,
            0.0,
            0.0001,
        ),
    ],
)
def test_compute_state_equations(
    rho,
    cp,
    alpha,
    slope,
    phi,
    e0,
    ea,
    e0star,
    m,
    g_aero_expected,
    g_surf_expected,
    delta_t_expected,
    ef_expected,
) -> None:
    """
    Test function for computing state equations
    """
    (g_aero, g_surf, delta_t, ef) = functions.compute_state_equations(
        rho,
        cp,
        alpha,
        slope,
        phi,
        e0,
        ea,
        e0star,
        m,
    )
    np.testing.assert_almost_equal(g_aero, g_aero_expected, decimal=3)
    np.testing.assert_almost_equal(g_surf, g_surf_expected, decimal=3)
    np.testing.assert_almost_equal(delta_t, delta_t_expected, decimal=3)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "le_flux",
        "ea",
        "esstar",
        "g_aero",
        "g_surf",
        "rho",
        "cp",
        "gamma",
        "expected",
    ),
    [
        pytest.param(
            209.44928,
            22.07793,
            42.6358,
            0.015451,
            0.009924,
            1.174905,
            1016.4927,
            0.67,
            41.5231,
        ),
    ],
)
def test_compute_canopy_air_saturation_vapor_pressure(
    le_flux, ea, esstar, g_aero, g_surf, rho, cp, gamma, expected
) -> None:
    """
    Test function for computing saturation vapor pressure at canopy/air height
    """
    res = functions.compute_canopy_air_saturation_vapor_pressure(
        le_flux, ea, esstar, g_aero, g_surf, rho, cp, gamma
    )
    np.testing.assert_almost_equal(res, expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "slope",
        "g_aero",
        "g_surf",
        "available_energy",
        "da",
        "ds",
        "rho",
        "cp",
        "expected",
    ),
    [
        pytest.param(
            1.89602,
            0.01545,
            0.009924,
            305.09854,
            9.752996,
            12.51803,
            1.174905,
            1016.492676,
            11.87963,
        ),
    ],
)
def test_compute_canopy_air_vapor_pressure_deficit(
    slope, g_aero, g_surf, available_energy, da, ds, rho, cp, expected
) -> None:
    """
    Test function for computing vapor pressure deficit at canopy/air height
    """
    res = functions.compute_canopy_air_vapor_pressure_deficit(
        slope, g_aero, g_surf, available_energy, da, ds, rho, cp
    )
    np.testing.assert_almost_equal(res, expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "slope",
        "g_aero",
        "g_surf",
        "ta",
        "t0",
        "e0star",
        "ea",
        "m",
        "expected",
    ),
    [
        pytest.param(
            1.89602,
            0.01854,
            0.01039,
            25.0,
            29.23084,
            39.85269,
            22.07793,
            0.359166,
            1.233953,
        ),
    ],
)
def test_compute_alpha_coefficient(
    slope, g_aero, g_surf, ta, t0, e0star, ea, m, expected
) -> None:
    """
    Test function for computing alpha coefficient
    """
    res = functions.compute_alpha_coefficient(
        slope, g_aero, g_surf, ta, t0, e0star, ea, m
    )
    np.testing.assert_almost_equal(res, expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "slope",
        "ta",
        "da",
        "cp",
        "rho",
        "emis",
        "rn",
        "expected",
    ),
    [
        pytest.param(
            1.89602,
            25.0,
            9.75,
            1.2,
            1013,
            0.9,
            400,
            1.07,
        ),
    ],
)
def test_initialize_alpha_coefficient(
    slope, ta, da, cp, rho, emis, rn, expected
) -> None:
    """
    Test function for computing vapor pressure deficit at canopy/air height
    """
    res = functions.initialize_alpha_coefficient(
        slope, ta, da, cp, rho, emis, rn
    )
    np.testing.assert_almost_equal(res, expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "t",
        "expected",
    ),
    [
        pytest.param(
            29.0,
            40.25,
        ),
    ],
)
def test_compute_saturated_vapor_pressure(t, expected) -> None:
    """
    Test function for computing saturated vapor pressure
    """
    res = functions.compute_saturated_vapor_pressure(t)
    np.testing.assert_almost_equal(res, expected, decimal=2)
