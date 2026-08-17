# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for managing pixel status which is encoded in FLAGS

A pixel is characterized by

- Bit 0 "PROCESSING_FAILED": a flag to indicate if the processing failed
- Bit 1 "RADIATION_MISSING": a flag to indicate if radiation data is missing,
this implies that processing failed
- Bit 2 "AUX_DATA_MISSING": a flag to indicate if auxiliary data is missing,
this implies that processing failed
- Bit 3 "FILTERED": a flag to indicate if data is filtered,
this implies that processing failed
- Bit 4 "UPDATED": a flag to indicate if data has been updated
- Bit 5 to 7 "STATE": to indicate the state: valid, interpolated,
forward extrapolated, backward extrapolated, invalid, nodata
- Bit 8 to 15 "DISTANCE": the difference between the two dates used for
interpolation or the difference with the date used for extrapolation

Available states:

- 0: acquisition
- 1: interpolation
- 2: forward extrapolation
- 3: backward extrapolation
- 6: invalid
- 7: nodata
"""

from enum import Enum

import numpy as np
import xarray as xr

# Status type
STATUS_TYPE = np.uint16
# Blank status
BLANK_STATUS = STATUS_TYPE(0)
INIT_STATUS = STATUS_TYPE(0b1111111111100000)  # Distance max and state nodata
MASKED_STATUS = STATUS_TYPE(0xFFFF)
# Bit 0: Processing flag
MSK_PROCESSING_FAILED = 1 << 0
PROCESSING_BIT_POSITION = 0
# Bit 1: Radiation flag
MSK_RADIATION_MISSING = 1 << 1
RADIATION_BIT_POSITION = 1
# Bit 2: Auxiliary data flag
MSK_AUX_DATA_MISSING = 1 << 2
AUX_DATA_BIT_POSITION = 2
# Bit 3: Filtered flag
MSK_FILTERED_DATA = 1 << 3
FILTERED_DATA_BIT_POSITION = 3
# Bit 4: Updated flag
MSK_UPDATED_DATA = 1 << 4
UPDATED_DATA_BIT_POSITION = 4
# Bit 5 to 7: State
MSK_STATE = STATUS_TYPE(0b111)
STATE_BIT_POSITION = 5
STATE_TYPE = np.int8
# Bit 8 to 15: Distance
DISTANCE_BIT_POSITION = 8
MSK_DISTANCE = STATUS_TYPE(0xFF)
DISTANCE_TYPE = np.uint8
# We plan to encode the distance using 8 bits (uint8), which seems sufficient.
DISTANCE_MAX = 255

MSK_VALIDITY_FLAGS = STATUS_TYPE(0b1111)
VALIDITY_FLAGS_TYPE = np.int8


class State(Enum):
    ACQUISITION = 0
    INTERPOLATED = 1
    FORWARD_EXTRAPOLATED = 2
    BACKWARD_EXTRAPOLATED = 3
    INVALID = 6
    NODATA = 7


# Priority between states
#    0: 0,  # acquisition
#    1: 1,  # interpolated
#    2: 2,  # forward extrapolated
#    3: 2,  # backward extrapolated
#    6: 4,  # invalid
#    7: 5,  # nodata
# }
STATE_PRIORITY = np.array([0, 1, 2, 2, np.nan, np.nan, 4, 5])


def is_better(new_state: xr.DataArray, old_state: xr.DataArray) -> xr.DataArray:
    """
    Return if new state is better than old state in terms of priority

    Parameters
    ----------
    new_state : xr.DataArray
        New state data
    old_state : xr.DataArray
        Old state data

    Returns
    -------
    is_better : xr.DataArray
        True if new state has a higher priority
    """
    return STATE_PRIORITY[new_state] < STATE_PRIORITY[old_state]


def is_same(new_state: xr.DataArray, old_state: xr.DataArray) -> xr.DataArray:
    """
    Return if new state is equal than old state in terms of priority

    Parameters
    ----------
    new_state : xr.DataArray
        New state data
    old_state : xr.DataArray
        Old state data

    Returns
    -------
    is_same : xr.DataArray
        True if new state has the same priority
    """
    return STATE_PRIORITY[new_state] == STATE_PRIORITY[old_state]


def extract_bit(data: xr.DataArray, bit_position: int) -> xr.DataArray:
    """
    Extract bit information at bit position specified

    Parameters
    ----------
    data : xr.DataArray
        Data
    bit_position : int
        Position of the bit to extraction information

    Returns
    -------
    bit_array : xr.DataArray
        Bit information returned as boolean array
    """
    # Create mask for the specific bit
    bit_mask = 1 << bit_position
    # True if bit is 1, False if bit is 0
    return (data & bit_mask) != 0


def set_bit(
    data: xr.DataArray,
    bit_position: int,
    condition: xr.DataArray,
    force_zero: bool = False,
) -> xr.DataArray:
    """
    Set or clear a bit in a DataArray based on a condition.

    Parameters
    ----------
    data : xr.DataArray
        Data
    bit_position : int
        Bit position
    condition : xr.DataArray
        Condition stored in boolean
    force_zero : boolean
        If True, set bit to 0 where condition is False;
        If False, only set to 1 where condition is True

    Returns
    -------
    updated : xr.DataArray
        Modified data
    """
    dtype = data.dtype
    if force_zero:
        # Clear bit first, then set based on condition
        # Create a mask
        mask_bit = dtype.type(1 << bit_position)
        data = data & ~mask_bit
        data = data | (condition.astype(dtype) << bit_position)
    else:
        # Only set to 1, leave other states unchanged
        data = data | (condition.astype(dtype) << bit_position)

    return data


def extract_state(status: xr.DataArray) -> xr.DataArray:
    """
    Extract state information

    Parameters
    ----------
    status : xr.DataArray
        Status data stored in STATUS_TYPE

    Returns
    -------
    state : xr.DataArray
        State information
    """
    # Int8 DataArray for bits 5-7
    return ((status >> STATE_BIT_POSITION) & MSK_STATE).astype(np.int8)


def extract_distance(status: xr.DataArray) -> xr.DataArray:
    """
    Extract distance information

    Parameters
    ----------
    status : xr.DataArray
        Status data stored in STATUS_TYPE

    Returns
    -------
    distance : xr.DataArray
        Distance information
    """
    # Unsigned int8 DataArray for bits 8-15 (upper byte)
    return ((status >> DISTANCE_BIT_POSITION) & MSK_DISTANCE).astype(np.uint8)


def decode_status(
    status: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Decode status

    Parameters
    ----------
    status : xr.DataArray
        Status data stored in STATUS_TYPE

    Returns
    -------
    updated : xr.DataArray
        Updated bit
    state : xr.DataArray
        State information
    distance : xr.DataArray
        Distance information
    flags : xr.DataArray
        Flags information
    """
    # Extract flags
    flags = (status & MSK_VALIDITY_FLAGS).astype(np.int8)
    return (
        extract_bit(status, UPDATED_DATA_BIT_POSITION).astype(bool),
        extract_state(status),
        extract_distance(status),
        flags,
    )


def encode_status(
    updated: xr.DataArray,
    state: xr.DataArray,
    distance: xr.DataArray,
    flags: xr.DataArray,
) -> xr.DataArray:
    """
    Encode status

    Parameters
    ----------
    updated : xr.DataArray
        Updated bit
    state : xr.DataArray
        State information
    distance : xr.DataArray
        Distance information
    flags : xr.DataArray
        Flags information

    Returns
    -------
    status : xr.DataArray
        Status data stored in STATUS_TYPE
    """
    # Status is composed:
    # bits 0 to 3: flags
    # bit 4: updated (boolean)
    # bits 5 to 7: state
    # bits 8 to 15: distance
    status = xr.zeros_like(updated, dtype=STATUS_TYPE)
    # Set bits 0-3 from flags (int8)
    status = status | (flags.astype(STATUS_TYPE) & MSK_VALIDITY_FLAGS)
    # Set bit 4 from updated (boolean)
    status = status | (
        (updated.astype(STATUS_TYPE)) << UPDATED_DATA_BIT_POSITION
    )
    # Set bits 5-7 from state (int8)
    status = status | (
        (state.astype(STATUS_TYPE) & MSK_STATE) << STATE_BIT_POSITION
    )
    # Set bits 8-15 from distance (uint8)
    return status | (
        (distance.astype(STATUS_TYPE) & MSK_DISTANCE) << DISTANCE_BIT_POSITION
    )
