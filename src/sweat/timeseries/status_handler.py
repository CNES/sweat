# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for managing pixel status.
A pixel is characterized by
- Bit 0 to 2 "STATE": a state: valid, nodata, interpolated, extrapolated,
- Bit 3 "PROCESSING": a mode for processing: normal or degraded,
backward extrapolated, invalid
- Bit 4 "UPDATED": a flag to indicate is the pixel has been updated
- Bit 5 "RADIATION": a mode for radiation product: external or theoretical
- Bit 6 "AUX_DATA": a status for auxilliary data: complete, missing
- Bit 7 to 10 "DISTANCE": the difference between the two dates used for
interpolation or the difference with the date used for extrapolation
"""

from enum import Enum

import numpy as np

# Status type
STATUS_TYPE = np.uint16
# Blank status
BLANK_STATUS = STATUS_TYPE(0)
INIT_STATUS = STATUS_TYPE(1)


# Bit 0 to 2: Stata
class State(Enum):
    ACQUISITION = 0
    NODATA = 1
    INTERPOLATED = 2
    EXTRAPOLATED = 3
    BACKWARD_EXTRAPOLATED = 4
    INVALID = 5


# Reset state
MSK_STATE = STATUS_TYPE(0b111)


# Bit 3: Processing mode
class ProcessingMode(Enum):
    NOMINAL = 0
    DEGRADED = 1


# Processing mode position
PROCESSING_MODE_POSITION = 3

# Bit 4: Updated flag
UPDATED = 1 << 4
UPDATED_POSITION = 4


# Bit 5: Radiation mode
class RadiationMode(Enum):
    EXTERNAL = 0
    THEORETICAL = 1


# Radiation status position
RADIATION_MODE_POSITION = 5


# Bit 6: Aux data
class AuxDataStatus(Enum):
    COMPLETE = 0
    MISSING = 1


# Radiation status position
AUX_DATA_STATUS_POSITION = 6

# Bit 7 to 10: Distance
MSK_DISTANCE = STATUS_TYPE(0b11110000000)
DISTANCE_POSITION = 7


def _extract_bit(bit_array: STATUS_TYPE, position: int) -> STATUS_TYPE:
    """
    Extract the bit at a specific position from the bit_array.

    Parameters
    ----------
    bit_array: int
        Integer representing the bit array
    position: int
        Position of the bit to extract

    Returns
    -------
    value: int
        The value of the third bit (0 or 1).
    """
    # Right shift by poistion and AND with 1
    return STATUS_TYPE((bit_array >> position) & 0b1)


def _set_bit(bit_array: STATUS_TYPE, position: int, value: int) -> STATUS_TYPE:
    """
    Set the bit at the specified position in the bit_array
    to the given value (0 or 1).

    Parameters
    ----------
    bit_array: int
        Integer representing the bit array.
    position: int
        The position of the bit to set (0-indexed).
    value: int
        Boolean indicating whether to set the bit to 1 (True) or 0 (False).

    Returns
    -------
    return: int
        Updated bit array with the specified bit set to the given value.
    """
    if value:
        # Set the bit to 1
        # OR with 1 shifted to the specified position
        return np.bitwise_or(bit_array, STATUS_TYPE(1 << position))
    # Set the bit to 0
    # AND with NOT of 1 shifted to the specified position
    return np.bitwise_and(bit_array, np.bitwise_not(1 << position))


def set_status(
    state: State,
    processing: ProcessingMode,
    updated: bool,
    radiation: RadiationMode,
    aux_data: AuxDataStatus,
    distance: int,
) -> STATUS_TYPE:
    """
    Set status

    Parameters
    ----------
    state: State
        State of the pixel
    processing: ProcessingMode
        Processing mode of the pixel
    updated: bool
        Boolean to indicate if the pixel has been updated
    radiation: RadiationMode
        Radiation mode used to compute
    aux_data: AuxDataStatus
        Status of auxiliary data used
    distance: int
        Distance used for interpolation/extrapolation

    Returns
    -------
    status: STATUS_TYPE
        Created status
    """
    status = BLANK_STATUS
    # Set state
    status &= np.bitwise_not(MSK_STATE)  # Reset state
    status |= STATUS_TYPE(state.value)  # Set the new state
    # Set processing mode
    status = _set_bit(status, PROCESSING_MODE_POSITION, processing.value)
    status = _set_bit(status, UPDATED_POSITION, int(updated))
    status = _set_bit(status, RADIATION_MODE_POSITION, radiation.value)
    status = _set_bit(status, AUX_DATA_STATUS_POSITION, aux_data.value)
    # Set distance
    status &= np.bitwise_not(MSK_DISTANCE)
    status |= STATUS_TYPE(distance << DISTANCE_POSITION)
    return status


def update_status(
    status: STATUS_TYPE,
    state: State | None = None,
    processing: ProcessingMode | None = None,
    updated: bool | None = None,
    radiation: RadiationMode | None = None,
    aux_data: AuxDataStatus | None = None,
    distance: int | None = None,
) -> STATUS_TYPE:
    """
    Update status

    Parameters
    ----------
    status: STATUS_TYPE
        Actual status
    state: State
        State of the pixel
    processing: ProcessingMode
        Processing mode of the pixel
    updated: bool
        Boolean to indicate if the pixel has been updated
    radiation: RadiationMode
        Radiation mode used to compute
    aux_data: AuxDataStatus
        Status of auxiliary data used
    distance: int
        Distance used for interpolation/extrapolation

    Returns
    -------
    status: STATUS_TYPE
        Updated status
    """
    if state is not None:
        status &= np.bitwise_not(MSK_STATE)  # Reset state
        status |= STATUS_TYPE(state.value)  # Set the new state
    # Set processing mode
    if processing is not None:
        status = _set_bit(status, PROCESSING_MODE_POSITION, processing.value)
    if updated is not None:
        status = _set_bit(status, UPDATED_POSITION, int(updated))
    if radiation is not None:
        status = _set_bit(status, RADIATION_MODE_POSITION, radiation.value)
    if aux_data is not None:
        status = _set_bit(status, AUX_DATA_STATUS_POSITION, aux_data.value)
    if distance is not None:
        status &= np.bitwise_not(MSK_DISTANCE)
        status |= STATUS_TYPE(distance << DISTANCE_POSITION)
    return status


def get_state(bit_array: STATUS_TYPE) -> State:
    """
    Get State

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array

    Returns
    -------
    state: State
        State of the pixel
    """
    return State(bit_array & MSK_STATE)  # Isolate the first two bits


def get_processing_mode(bit_array: STATUS_TYPE) -> ProcessingMode:
    """
    Get processing mode

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array

    Returns
    -------
    mode: ProcessingMode
        Processing mode of the pixel
    """
    return ProcessingMode(_extract_bit(bit_array, PROCESSING_MODE_POSITION))


def get_radiation_mode(bit_array: STATUS_TYPE) -> RadiationMode:
    """
    Get radiation mode

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array

    Returns
    -------
    mode: RadiationMode
        Radiation mode of the pixel
    """
    return RadiationMode(_extract_bit(bit_array, RADIATION_MODE_POSITION))


def get_distance(bit_array: STATUS_TYPE) -> int:
    """
    Get radiation mode

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array

    Returns
    -------
    distance: int
        Distance used for interpolation/extrapolation
    """
    return ((bit_array & MSK_DISTANCE) >> DISTANCE_POSITION).astype(int)


def is_updated(bit_array: STATUS_TYPE) -> bool:
    """
    Check if updated

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array

    Returns
    -------
    updated: bool
        True is if pixel has been updated
    """
    return ((STATUS_TYPE(bit_array) >> UPDATED_POSITION) & 0b1).astype(bool)


def check_state(bit_array: STATUS_TYPE, state: State) -> bool:
    """
    Check state

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array
    state: State
        State to check

    Returns
    -------
    checked: bool
        Return true if states match
    """
    return bit_array & MSK_STATE == state.value


def check_processing_mode(bit_array: STATUS_TYPE, mode: ProcessingMode) -> bool:
    """
    Check processing mode

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array
    mode: ProcessingMode
        Processing mode to check

    Returns
    -------
    checked: bool
        Return true if modes match
    """
    return _extract_bit(bit_array, PROCESSING_MODE_POSITION) == mode.value


def check_radiation_mode(
    bit_array: STATUS_TYPE, radiation: RadiationMode
) -> bool:
    """
    Check radiation status

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array
    radiation: RadiationMode
        Radiation mode to check

    Returns
    -------
    checked: bool
        Return true if modes match
    """
    return _extract_bit(bit_array, RADIATION_MODE_POSITION) == radiation.value


def print_status(bit_array: STATUS_TYPE) -> str:
    """
    Print status

    Parameters
    ----------
    bit_array: STATUS_TYPE
        Bit array

    Returns
    -------
    status: str
        status
    """
    return (
        f"Status : State={get_state(bit_array)}, "
        f"Mode={get_processing_mode(bit_array)}, "
        f"Updated={is_updated(bit_array)}, "
        f"Radiation={get_radiation_mode(bit_array)}, "
        f"Distance={get_distance(bit_array)}"
    )
