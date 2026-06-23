# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for types
"""

from __future__ import annotations


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
