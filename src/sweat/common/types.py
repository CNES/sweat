# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module for types
"""

from __future__ import annotations

from enum import Enum


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


class PercentileValue:
    """
    Class to manage percentile
    """

    def __init__(self, percentile: float):
        if not (0 <= percentile <= 100):
            msg = "Percentile must be between 0 and 100"
            raise ValueError(msg)
        self.percentile = percentile

    def __repr__(self):
        return f"percentile({self.percentile})"

    def __gt__(self, other: PercentileValue) -> bool:
        return self.percentile > other.percentile

    def __ge__(self, other: PercentileValue) -> bool:
        return self.percentile >= other.percentile
