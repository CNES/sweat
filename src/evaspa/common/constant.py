# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for constant management
"""

from enum import Enum

import numpy as np

# Constant
PSYCHROMETRIC_CST = 0.67


# Dataset variables
class ETVar(Enum):
    """
    ET variables
    """

    ALBEDO = "albedo"
    ASPECT = "aspect"
    DEWPOINT_TEMPERATURE = "tdp"
    EMISSIVITY = "emis"
    ET = "et"
    EF = "ef"
    FCOVER = "fcover"
    FDIFF = "fdiff"
    FLAGS = "flags"
    HEIGHT = "height"
    LAI = "lai"
    LE = "le"
    LOCAL_TIME = "local_time"
    LONGWAVE_NET_RADIATION = "ln"
    LST = "lst"
    NDVI = "ndvi"
    NET_RADIATION = "rn"
    RH = "rh"
    RLD = "rld"
    RSD = "rsd"
    SLOPE = "slope"
    TEMPERATURE = "ta"
    VALID = "valid"


FLAGS_TYPE = np.uint8
# Constant Flags
# if bit 0 activated : The pixel is invalid : input data contains nodata
MSK_INPUT_NODATA = 1 << 0
# if bit 1 activated : The pixel is invalid : input data are filtered
MSK_INPUT_FILTERED = 1 << 1
# if bit 2 activated : The pixel is invalid : Processing failed
MSK_PROCESSING_FAILED = 1 << 2
