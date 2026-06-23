# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for constant management
"""

from enum import Enum

import numpy as np


# Dataset variables
class ETVar(Enum):
    """
    ET variables
    """

    ALBEDO = "albedo"
    ASPECT = "aspect"
    BLUE = "blue"
    DEWPOINT_TEMPERATURE = "tdp"
    EMISSIVITY = "emis"
    ET = "et"
    EF = "ef"
    FCOVER = "fcover"
    FDIFF = "fdiff"
    FLAGS = "flags"
    GLI = "gli"
    GNDVI = "gndvi"
    GREEN = "green"
    HEIGHT = "height"
    LAI = "lai"
    LE = "le"
    LOCAL_TIME = "local_time"
    LONGWAVE_NET_RADIATION = "ln"
    LST = "lst"
    MSAVI = "msavi"
    NDVI = "ndvi"
    NIR = "nir"
    NET_RADIATION = "rn"
    RH = "rh"
    RED = "red"
    RLD = "rld"
    RSD = "rsd"
    SLOPE = "slope"
    SWIR = "swir"
    TEMPERATURE = "ta"
    UNCERTAINTY_EF = "uncertainty_ef"
    UNCERTAINTY_ET = "uncertainty_et"
    UNCERTAINTY_LE = "uncertainty_le"
    VALID = "valid"
    VARI = "vari_green_index"


FLAGS_TYPE = np.uint8
# Constant Flags
# if bit 0 activated : The pixel is invalid : input data contains nodata
MSK_INPUT_NODATA = 1 << 0
# if bit 1 activated : The pixel is invalid : input data are filtered
MSK_INPUT_FILTERED = 1 << 1
# if bit 2 activated : The pixel is invalid : input data filtered
# during some processing step
MSK_INPUT_FILTERED_DURING_PROCESSING = 1 << 2
# if bit 3 activated : The pixel is invalid : Processing failed
MSK_PROCESSING_FAILED = 1 << 3
