# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import pytest
import xarray as xr

from sweat.timeseries import status_handler as sh


@pytest.mark.unit
def test_is_better() -> None:
    """
    Test the function is_better
    """
    assert sh.is_better(
        xr.DataArray(sh.State.INTERPOLATED.value),
        xr.DataArray(sh.State.FORWARD_EXTRAPOLATED.value),
    )
    assert not sh.is_better(
        xr.DataArray(sh.State.INVALID.value),
        xr.DataArray(sh.State.FORWARD_EXTRAPOLATED.value),
    )
    assert not sh.is_better(
        xr.DataArray(sh.State.BACKWARD_EXTRAPOLATED.value),
        xr.DataArray(sh.State.FORWARD_EXTRAPOLATED.value),
    )


@pytest.mark.unit
def test_is_same() -> None:
    """
    Test the function is_same
    """
    assert not sh.is_same(
        xr.DataArray(sh.State.INTERPOLATED.value),
        xr.DataArray(sh.State.FORWARD_EXTRAPOLATED.value),
    )
    assert sh.is_same(
        xr.DataArray(sh.State.BACKWARD_EXTRAPOLATED.value),
        xr.DataArray(sh.State.FORWARD_EXTRAPOLATED.value),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("bit_position", "expected"),
    [(0, True), (2, False), (3, False)],
)
def test_extract_bit(bit_position, expected) -> None:
    """
    Test the function extract_bit
    """
    status = xr.DataArray(sh.STATUS_TYPE(849))
    xr.testing.assert_equal(
        sh.extract_bit(status, bit_position), xr.DataArray(expected)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("bit_position", "condition", "force_zero", "expected"),
    [
        (0, xr.DataArray(True), False, xr.DataArray(sh.STATUS_TYPE(849))),
        (0, xr.DataArray(False), True, xr.DataArray(sh.STATUS_TYPE(848))),
        (2, xr.DataArray(True), False, xr.DataArray(sh.STATUS_TYPE(853))),
        (2, xr.DataArray(False), False, xr.DataArray(sh.STATUS_TYPE(849))),
    ],
)
def test_set_bit(bit_position, condition, force_zero, expected) -> None:
    """
    Test the function set_bit
    """
    status = xr.DataArray(sh.STATUS_TYPE(849))
    xr.testing.assert_equal(
        sh.set_bit(
            status,
            bit_position=bit_position,
            condition=condition,
            force_zero=force_zero,
        ),
        xr.DataArray(expected),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (0b0000001100010001, sh.State.ACQUISITION),
        (0b0000001100110001, sh.State.INTERPOLATED),
        (0b0000001101010001, sh.State.FORWARD_EXTRAPOLATED),
        (0b0000001101110001, sh.State.BACKWARD_EXTRAPOLATED),
        (0b0000001111010001, sh.State.INVALID),
        (0b0000001111110001, sh.State.NODATA),
    ],
)
def test_extract_state(status, expected) -> None:
    """
    Test the function extract_state
    """
    status = xr.DataArray(sh.STATUS_TYPE(status))
    state = sh.extract_state(status)
    xr.testing.assert_equal(state, xr.DataArray(expected.value))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (0b0000001100110001, 3),
        (0b1111111111110001, 255),
        (0b0000000000010001, 0),
    ],
)
def test_extract_distance(status, expected) -> None:
    """
    Test the function extract_distance
    """
    status = xr.DataArray(sh.STATUS_TYPE(status))
    state = sh.extract_distance(status)
    xr.testing.assert_equal(state, xr.DataArray(expected))


@pytest.mark.unit
def test_decode_status() -> None:
    """
    Test the function decode_status
    """
    status = xr.DataArray(sh.STATUS_TYPE(849))
    updated, state, distance, flags = sh.decode_status(status)
    xr.testing.assert_equal(updated, xr.DataArray(True))
    xr.testing.assert_equal(
        state, xr.DataArray(sh.State.FORWARD_EXTRAPOLATED.value)
    )
    xr.testing.assert_equal(distance, xr.DataArray(3))
    xr.testing.assert_equal(flags, xr.DataArray(0b0001))


@pytest.mark.unit
def test_encode_status() -> None:
    """
    Test the function encode_status
    """
    updated = xr.DataArray(True)
    state = xr.DataArray(sh.State.FORWARD_EXTRAPOLATED.value)
    distance = xr.DataArray(3)
    flags = xr.DataArray(0b0001)
    encoded = sh.encode_status(updated, state, distance, flags)
    xr.testing.assert_equal(encoded, xr.DataArray(sh.STATUS_TYPE(849)))
