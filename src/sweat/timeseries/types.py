# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for constant management for time series
"""

from enum import Enum


class TimeSeriesVar(Enum):
    """
    Variables used in time series dataset
    """

    DISTANCE = "distance"
    ET = "et"
    FLAGS = "flags"
    RADIATION = "daily_radiation"
    STATE = "state"
    TIME = "time"
    UPDATED = "updated"
    VALID = "valid"
    VALIDITY_FLAGS = "validity_flags"
    HEIGHT = "height"
    SLOPE = "slope"
    ASPECT = "aspect"
