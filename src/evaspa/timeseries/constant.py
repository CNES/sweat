# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for constant management for time series
"""

from enum import Enum

WINDOW_SIZE = 7


class TimeSeriesVar(Enum):
    ET = "et"
    FLAGS = "flags"
    VALID = "valid"
    RADIATION = "daily_radiation"
    TIME = "time"
    HEIGHT = "height"
    SLOPE = "slope"
    ASPECT = "aspect"
