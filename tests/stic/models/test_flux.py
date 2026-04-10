# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pytest

from sweat.stic.models import flux


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rn", "lai", "local_time", "m", "expected"),
    [
        pytest.param(200, 2, 36000, 0.6, 2.342409),
        pytest.param(-50, 2, 36000, 0.6, 0.585602),
    ],
)
def test_compute_g_flux(rn, lai, local_time, m, expected) -> None:
    """
    Test function for computing g flux
    """
    res = flux.compute_g_flux(rn, lai, local_time, m)
    np.testing.assert_almost_equal(res, expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "slope",
        "g_aero",
        "g_surf",
        "available_energy",
        "da",
        "rho",
        "cp",
        "le_expected",
        "h_expected",
    ),
    [
        pytest.param(
            1.89602,
            0.01545,
            0.00992,
            305.09854,
            9.752996,
            1.1749,
            1016.49268,
            209.427093,
            94.97875,
        ),
    ],
)
def test_initiate_le_h_fluxes(
    slope,
    g_aero,
    g_surf,
    available_energy,
    da,
    rho,
    cp,
    le_expected,
    h_expected,
) -> None:
    """
    Test function for initiating le and h fluxes
    """
    le_flux, h_flux = flux.initiate_le_h_fluxes(
        slope, g_aero, g_surf, available_energy, da, rho, cp
    )
    np.testing.assert_almost_equal(le_flux, le_expected, decimal=3)
    np.testing.assert_almost_equal(h_flux, h_expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "slope",
        "g_aero",
        "g_surf",
        "available_energy",
        "da",
        "ta",
        "t0",
        "ea",
        "e0",
        "rho",
        "cp",
        "le_expected",
        "h_expected",
    ),
    [
        pytest.param(
            1.89602,
            0.01545,
            0.00992,
            305.09854,
            9.752996,
            25.0,
            29.3,
            22.1,
            29.6,
            1.1749,
            1016.49268,
            192.817,
            94.979,
        ),
    ],
)
def test_compute_le_h_fluxes(
    slope,
    g_aero,
    g_surf,
    available_energy,
    da,
    ta,
    t0,
    ea,
    e0,
    rho,
    cp,
    le_expected,
    h_expected,
) -> None:
    """
    Test function for computing le and h fluxes
    """
    le_flux, h_flux = flux.compute_le_h_fluxes(
        slope, g_aero, g_surf, available_energy, da, ta, t0, ea, e0, rho, cp
    )
    np.testing.assert_almost_equal(le_flux, le_expected, decimal=3)
    np.testing.assert_almost_equal(h_flux, h_expected, decimal=3)
