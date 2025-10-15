# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import pytest

from sweat.timeseries import status_handler as sh


@pytest.mark.unit
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param(
            (
                sh.State.INTERPOLATED,
                sh.ProcessingMode.NOMINAL,
                True,
                sh.RadiationMode.EXTERNAL,
                sh.AuxDataStatus.MISSING,
                5,
            ),
            0b0000001011010010,
        ),
    ],
)
def test_set_status(params, expected) -> None:
    """
    Test the function set_status
    """
    assert sh.set_status(*params) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "params", "expected"),
    [
        pytest.param(
            0b0000000001000000,
            (
                sh.State.INTERPOLATED,
                sh.ProcessingMode.NOMINAL,
                True,
                sh.RadiationMode.EXTERNAL,
                sh.AuxDataStatus.MISSING,
                5,
            ),
            0b0000001011010010,
        ),
    ],
)
def test_update_status(status, params, expected) -> None:
    """
    Test the function update_status
    """
    assert sh.update_status(status, *params) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(sh.INIT_STATUS, sh.State.NODATA),
        pytest.param(
            0b0000001011010000,
            sh.State.ACQUISITION,
        ),
        pytest.param(0b0000001011010010, sh.State.INTERPOLATED),
        pytest.param(0b0000001011010011, sh.State.EXTRAPOLATED),
        pytest.param(0b0000001011010100, sh.State.BACKWARD_EXTRAPOLATED),
        pytest.param(0b0000001011010101, sh.State.INVALID),
    ],
)
def test_get_state(status, expected) -> None:
    """
    Test the function get_state
    """
    assert sh.get_state(status) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(
            0b0000001011010000,
            sh.ProcessingMode.NOMINAL,
        ),
        pytest.param(
            0b0000001011011000,
            sh.ProcessingMode.DEGRADATED,
        ),
    ],
)
def test_get_processing_mode(status, expected) -> None:
    """
    Test the function get_processing_mode
    """
    assert sh.get_processing_mode(status) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(
            0b0000001011010000,
            sh.RadiationMode.EXTERNAL,
        ),
        pytest.param(
            0b0000001011110000,
            sh.RadiationMode.THEORITICAL,
        ),
    ],
)
def test_get_radiation_mode(status, expected) -> None:
    """
    Test the function get_radiation_mode
    """
    assert sh.get_radiation_mode(status) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(
            0b0000001011000000,
            False,
        ),
        pytest.param(0b0000001011011000, True),
    ],
)
def test_is_updated(status, expected) -> None:
    """
    Test the function is_updated
    """
    assert sh.is_updated(status) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(
            0b0000001011000000,
            5,
        ),
        pytest.param(
            0b0000001111000000,
            7,
        ),
    ],
)
def test_get_distance(status, expected) -> None:
    """
    Test the function get_distance
    """
    assert sh.get_distance(status) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "test", "expected"),
    [
        pytest.param(0b0000001011000010, sh.State.INTERPOLATED, True),
        pytest.param(0b0000001011000010, sh.State.ACQUISITION, False),
    ],
)
def test_check_state(status, test, expected) -> None:
    """
    Test the function check_state
    """
    assert sh.check_state(status, test) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "test", "expected"),
    [
        pytest.param(0b0000001011000010, sh.ProcessingMode.NOMINAL, True),
        pytest.param(0b0000001011000010, sh.ProcessingMode.DEGRADATED, False),
    ],
)
def test_check_processing_mode(status, test, expected) -> None:
    """
    Test the function check_processing_mode
    """
    assert sh.check_processing_mode(status, test) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "test", "expected"),
    [
        pytest.param(0b0000001011000010, sh.RadiationMode.EXTERNAL, True),
        pytest.param(0b0000001011000010, sh.RadiationMode.THEORITICAL, False),
    ],
)
def test_check_radiation_mode(status, test, expected) -> None:
    """
    Test the function check_radiation_mode
    """
    assert sh.check_radiation_mode(status, test) == expected
