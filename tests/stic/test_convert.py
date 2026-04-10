# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

import datetime as dt

import numpy as np
import pytest

from sweat.stic import convert


@pytest.mark.unit
@pytest.mark.parametrize(
    ("date", "x", "y", "crs", "expected"),
    [
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.UTC),
            -74,
            40,
            None,
            25440.0,
        ),
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.UTC),
            -74,
            40,
            4326,
            25440.0,
        ),
        pytest.param(
            dt.datetime(2025, 6, 16, 12, 0, 0, tzinfo=dt.UTC),
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
    res = convert.convert_to_local_time(date, x, y, crs)
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
def test_convert_kelvin_to_celsius(temperature, expected) -> None:
    """
    Test function for converting Kelvin to Celsius
    """
    res = convert.convert_kelvin_to_celsius(temperature)
    np.testing.assert_almost_equal(res, expected, decimal=3)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        pytest.param(
            -273.15,
            0,
        ),
        pytest.param(
            26.85,
            300,
        ),
    ],
)
def test_convert_celsius_to_kelvin(temperature, expected) -> None:
    """
    Test function for converting Celsius to Kelvin
    """
    res = convert.convert_celsius_to_kelvin(temperature)
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
    Test function for converting dewpoint temperature to relative humidity
    """
    res = convert.convert_to_rh(t2m, d2m, b, c)
    np.testing.assert_almost_equal(res, expected, decimal=2)
