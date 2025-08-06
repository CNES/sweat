# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

import datetime as dt

import numpy as np
import pytest

from evaspa.stic import functions


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.timezone.utc),
            -74,
            40,
            None,
            25440.0,
        ),
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.timezone.utc),
            -74,
            40,
            4326,
            25440.0,
        ),
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.timezone.utc),
            585360,
            4428236,
            32618,
            25440.0,
        ),
    ],
)
def test_convert_to_local_time(date, x, y, crs, expected) -> None:
    """
    Test function for converting to local time
    """
    res = functions.convert_to_local_time(date, x, y, crs)
    np.testing.assert_almost_equal(res, expected, decimal=0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        pytest.param(
            0,
            -273.15,
        ),
        pytest.param(
            300,
            26.85,
        ),
    ],
)
def test_convert_to_celsius(temperature, expected) -> None:
    """
    Test function for converting to clesius
    """
    res = functions.convert_to_celsius(temperature)
    np.testing.assert_almost_equal(res, expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t2m", "d2m", "b", "c", "expected"),
    [
        pytest.param(
            20,
            15,
            17.625,
            243.04,
            72.94,
        ),
    ],
)
def test_convert_to_rh(t2m, d2m, b, c, expected) -> None:
    """
    Test function for converting to clesius
    """
    res = functions.convert_to_rh(t2m, d2m, b, c)
    np.testing.assert_almost_equal(res, expected, decimal=2)


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
            42.635787639301256,
            31.830927921627044,
            22.07793160644052,
            9.752996315186525,
            1.8960194745113388,
            1.377511596721743,
            1.8688960029873398,
            2.445493138777766,
            1.6254993858644209,
            1.1747878524431108,
            1016.4926440255699,
        ),
        pytest.param(
            35,
            28,
            25,
            83.79,
            56.4986,
            37.9820,
            31.82516,
            6.15689,
            2.21153,
            1.8960,
            2.46735,
            3.12272,
            2.052297,
            1.15882,
            1021.6275,
        ),
        pytest.param(
            36,
            28,
            15,
            45.1,
            59.6972,
            37.9820,
            17.12990,
            20.85215,
            2.21153,
            1.10322,
            2.02701,
            3.27540,
            1.60401,
            1.16524,
            1013.9004,
        ),
        pytest.param(
            15,
            15,
            15,
            100.0,
            17.136,
            17.136,
            17.136,
            0.0,
            1.1032,
            1.1032,
            1.1032,
            1.1032,
            1.1032,
            1.2178,
            1013.9035,
        ),
    ],
)
def test_f_psychrometrics(
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
    Test function for converting to clesius
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
    ) = functions.f_psychrometrics(ts, ta, td, rh)
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
            0.12358747158762567,
            0.06,
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
            0.0001,
            5.811882893226179,
            1,
        ),
    ],
)
def test_f_stateeq(
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
    TODO check result
    """
    (g_aero, g_surf, delta_t, ef) = functions.f_stateeq(
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
