# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for managing pixel status.
A pixel is characterized by
- Bit 0 to 2 "STATE": a state: valid, nodata, interpolated, extrapolated,
- Bit 3 "PROCESSING": a mode for processing: normal or degraded,
backward extrapolated, invalid
- Bit 4 "UPDATED": a flag to indicate is the pixel has been updated
- Bit 5 "RADIATION": a mode for radiation product: external or theoritical
- Bit 6 "AUX_DATA": a status for auxilliary data: complete, missing
- Bit 7 to 10 "DISTANCE": the difference between the two dates used for
interpolation or the difference with the date used for extrapolation
"""

from enum import Enum

import numpy as np

# Status type
STATUS_TYPE = np.uint16
# Blank status
BLANK_STATUS = 0
INIT_STATUS = 1


# Bit 0 to 2: Stata
class State(Enum):
    ACQUISITION = 0
    NODATA = 1
    INTERPOLATED = 2
    EXTRAPOLATED = 3
    BACKWARD_EXTRAPOLATED = 4
    INVALID = 5


# Reset state
MSK_STATE = 0b111


# Bit 3: Processing mode
class ProcessingMode(Enum):
    NOMINAL = 0
    DEGRADATED = 1


# Processing mode position
PROCESSING_MODE_POSITION = 3

# Bit 4: Updated flag
UPDATED = 1 << 4
UPDATED_POSITION = 4


# Bit 5: Radiation mode
class RadiationMode(Enum):
    EXTERNAL = 0
    THEORITICAL = 1


# Radiation status position
RADIATION_MODE_POSITION = 5


# Bit 6: Aux data
class AuxDataStatus(Enum):
    COMPLETE = 0
    MISSING = 1


# Radiation status position
AUX_DATA_STATUS_POSITION = 6

# Bit 7 to 10: Distance
MSK_DISTANCE = 0b11110000000
DISTANCE_POSITION = 7


def _extract_bit(bit_array: int, position: int) -> int:
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
    return (bit_array >> position) & 0b1


def _set_bit(bit_array: int, position: int, value: int) -> int:
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
        return bit_array | (
            1 << position
        )  # OR with 1 shifted to the specified position
    # Set the bit to 0
    return bit_array & ~(
        1 << position
    )  # AND with NOT of 1 shifted to the specified position


def set_status(
    state: State,
    processing: ProcessingMode,
    updated: bool,
    radiation: RadiationMode,
    aux_data: AuxDataStatus,
    distance: int,
):
    """
    Set status
    """
    status = BLANK_STATUS
    # Set state
    status &= ~MSK_STATE  # Reset state
    status |= state.value  # Set the new state
    # Set processing mode
    status = _set_bit(status, PROCESSING_MODE_POSITION, processing.value)
    status = _set_bit(status, UPDATED_POSITION, int(updated))
    status = _set_bit(status, RADIATION_MODE_POSITION, radiation.value)
    status = _set_bit(status, AUX_DATA_STATUS_POSITION, aux_data.value)
    # Set distance
    status &= ~MSK_DISTANCE
    status |= distance << DISTANCE_POSITION
    return status


def update_status(
    status: int,
    state: State | None = None,
    processing: ProcessingMode | None = None,
    updated: bool | None = None,
    radiation: RadiationMode | None = None,
    aux_data: AuxDataStatus | None = None,
    distance: int | None = None,
):
    """
    Update status
    """
    if state is not None:
        status &= ~MSK_STATE  # Reset state
        status |= state.value  # Set the new state
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
        status &= ~MSK_DISTANCE
        status |= distance << DISTANCE_POSITION
    return status


def get_state(bit_array: int) -> State:
    """
    Get State
    """
    return State(bit_array & MSK_STATE)  # Isolate the first two bits


def get_processing_mode(bit_array: int) -> ProcessingMode:
    """
    Get processing mode
    """
    return ProcessingMode(_extract_bit(bit_array, PROCESSING_MODE_POSITION))


def get_radiation_mode(bit_array: int) -> RadiationMode:
    """
    Get radiation mode
    """
    return RadiationMode(_extract_bit(bit_array, RADIATION_MODE_POSITION))


def get_distance(bit_array: int) -> int:
    return (bit_array & MSK_DISTANCE) >> DISTANCE_POSITION


def is_updated(bit_array: int) -> bool:
    """
    Test is updated
    """
    return bool((bit_array >> UPDATED_POSITION) & 0b1)


def check_state(bit_array: int, state: State) -> bool:
    """
    Check state
    """
    return State(bit_array & MSK_STATE) == state


def check_processing_mode(bit_array: int, mode: ProcessingMode) -> bool:
    """
    Check processing mode
    """
    return (
        ProcessingMode(_extract_bit(bit_array, PROCESSING_MODE_POSITION))
        == mode
    )


def check_radiation_mode(bit_array: int, radiation: RadiationMode) -> bool:
    """
    Check radiation status
    """
    return (
        RadiationMode(_extract_bit(bit_array, RADIATION_MODE_POSITION))
        == radiation
    )
