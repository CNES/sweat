# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for linear interpolation/extrapolation
"""

import xarray as xr

from evaspa.timeseries import status_handler as sh
from evaspa.timeseries.abstract_updater import Updater
from evaspa.timeseries.constant import TimeSeriesVar as TSVar


class LinearUpdater(Updater):
    """
    Linear upindexr
    """

    def __init__(self, strict_mode: bool, radiation_mode: sh.RadiationMode):
        """
        Init method
        """
        super().__init__(strict_mode, radiation_mode)
        self.variables: list[str] = [TSVar.RADIATION.value]

    def interpolate(
        self, index: int, prev_index: int, next_index: int, data: xr.Dataset
    ) -> tuple[float, sh.ProcessingMode]:
        """
        Interpolate data
        """
        # Retrieve data
        prev_et = (
            data[TSVar.ET.value].isel({TSVar.TIME.value: prev_index}).item()
        )
        next_et = (
            data[TSVar.ET.value].isel({TSVar.TIME.value: next_index}).item()
        )
        radiation = (
            data[TSVar.RADIATION.value].isel({TSVar.TIME.value: index}).item()
        )
        prev_radiation = (
            data[TSVar.RADIATION.value]
            .isel({TSVar.TIME.value: prev_index})
            .item()
        )
        next_radiation = (
            data[TSVar.RADIATION.value]
            .isel({TSVar.TIME.value: next_index})
            .item()
        )
        # Compute ratio
        prev_ratio = prev_et / prev_radiation
        next_ratio = next_et / next_radiation
        ratio = prev_ratio + (next_ratio - prev_ratio) * (
            index - prev_index
        ) / (next_index - prev_index)
        # Processing status
        mode = sh.ProcessingMode.NOMINAL
        if not sh.check_radiation_mode(
            data[TSVar.FLAGS.value].isel({TSVar.TIME.value: index}).item(),
            self.radiation_mode,
        ):
            mode = sh.ProcessingMode.DEGRADATED
        return ratio * radiation, mode

    def extrapolate(
        self, index: int, prev_index: int, data: xr.Dataset
    ) -> tuple[float, sh.ProcessingMode]:
        """
        Extrapolate
        """
        # Retrieve data
        prev_et = (
            data[TSVar.ET.value].isel({TSVar.TIME.value: prev_index}).item()
        )
        radiation = (
            data[TSVar.RADIATION.value].isel({TSVar.TIME.value: index}).item()
        )
        prev_radiation = (
            data[TSVar.RADIATION.value]
            .isel({TSVar.TIME.value: prev_index})
            .item()
        )
        # Compute ratio
        prev_ratio = prev_et / prev_radiation
        # Processing status
        mode = sh.ProcessingMode.NOMINAL
        if not sh.check_radiation_mode(
            data[TSVar.FLAGS.value].isel({TSVar.TIME.value: index}).item(),
            self.radiation_mode,
        ):
            mode = sh.ProcessingMode.DEGRADATED
        return prev_et + prev_ratio * (radiation - prev_radiation), mode

    def backward_extrapolate(
        self, index: int, next_index: int, data: xr.Dataset
    ) -> tuple[float, sh.ProcessingMode]:
        """
        Backward extrapolate
        """
        # Retrieve data
        next_et = (
            data[TSVar.ET.value].isel({TSVar.TIME.value: next_index}).item()
        )
        radiation = (
            data[TSVar.RADIATION.value].isel({TSVar.TIME.value: index}).item()
        )
        next_radiation = (
            data[TSVar.RADIATION.value]
            .isel({TSVar.TIME.value: next_index})
            .item()
        )
        # Compute ratio
        next_ratio = next_et / next_radiation
        # Processing status
        mode = sh.ProcessingMode.NOMINAL
        if not sh.check_radiation_mode(
            data[TSVar.FLAGS.value].isel({TSVar.TIME.value: index}).item(),
            self.radiation_mode,
        ):
            mode = sh.ProcessingMode.DEGRADATED
        return next_et + next_ratio * (radiation - next_radiation), mode
