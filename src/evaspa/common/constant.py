from enum import Enum

# Constant
PSYCHROMETRIC_CST = 0.67


# Dataset variables
class ETVar(Enum):
    """
    ET variables
    """

    ALBEDO = "albedo"
    DEWPOINT_TEMPERATURE = "tdp"
    EMISSIVITY = "emis"
    ET = "et"
    EF = "ef"
    FCOVER = "fcover"
    FDIFF = "fdiff"
    FLAGS = "flags"
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
    TEMPERATURE = "ta"
    VALID = "valid"


# Constant Flags
# if bit 0 activated : The pixel is invalid : input data contains nodata
MSK_INPUT_NODATA = 1 << 0
# if bit 1 activated : The pixel is invalid : input data are filtered
MSK_INPUT_FILTERED = 1 << 1
