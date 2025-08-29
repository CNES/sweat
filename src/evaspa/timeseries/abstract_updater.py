# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for time series update
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import xarray as xr

from evaspa.timeseries import status_handler as sh
from evaspa.timeseries.constant import TimeSeriesVar as TSVar


class Updater(ABC):
    """
    Class to update time series
    """

    def __init__(self, strict_mode: bool, radiation_mode: sh.RadiationMode):
        """
        Init method
        """
        self.strict_mode = strict_mode
        self.radiation_mode = radiation_mode
        self.variables: list[str] = []

    def get_requested_variables(self) -> list[str]:
        """
        Get the requested variables for updating time series
        """
        return self.variables

    def find_previous(self, status_arr: npt.NDArray, index: int) -> int:
        """
        Find previous index
        """
        # Iterate through the list in reverse order from the target index
        if index == 0:
            return -1
        prev_index = -1
        for i in range(index - 1, -1, -1):
            state = sh.get_state(status_arr[i])
            mode = sh.get_processing_mode(status_arr[i])
            # Check if the state is 0 and mode is not degraded
            if (
                state == sh.State.ACQUISITION
                and mode == sh.ProcessingMode.NOMINAL
            ):
                return i
            if (
                state == sh.State.INTERPOLATED
                and mode == sh.ProcessingMode.NOMINAL
                and prev_index < i
            ):
                prev_index = i
        if prev_index > -1 and not self.strict_mode:
            return prev_index
        return -1

    def find_next(self, status_arr: npt.NDArray, index: int) -> int:
        """
        Find
        """
        # Iterate through the list in reverse order from the target index
        if index == len(status_arr) - 1:
            return -1
        next_index = len(status_arr)
        for i in range(index + 1, len(status_arr), 1):
            state = sh.get_state(status_arr[i])
            mode = sh.get_processing_mode(status_arr[i])
            # Check if the state is 0 and mode is not degraded
            if (
                state == sh.State.ACQUISITION
                and mode == sh.ProcessingMode.NOMINAL
            ):
                return i
            if (
                state == sh.State.INTERPOLATED
                and mode == sh.ProcessingMode.NOMINAL
                and next_index > i
            ):
                next_index = i
        if next_index < len(status_arr) and not self.strict_mode:
            return next_index
        return -1

    @abstractmethod
    def interpolate(
        self, index: int, prev_index: int, next_index: int, data: xr.Dataset
    ) -> tuple[float, sh.ProcessingMode]:
        """
        Interpolate data
        """

    @abstractmethod
    def extrapolate(
        self, index: int, prev_index: int, data: xr.Dataset
    ) -> tuple[float, sh.ProcessingMode]:
        """
        Extrapolate
        """

    @abstractmethod
    def backward_extrapolate(
        self, index: int, next_index: int, data: xr.Dataset
    ) -> tuple[float, sh.ProcessingMode]:
        """
        Backward extrapolate
        """

    def update_new_acquisition(
        self, data: xr.Dataset, feed: xr.Dataset
    ) -> xr.Dataset:
        """
        Update with new acquisition
        """
        stack = False
        ts = data.copy()
        if "stacked_x_y" in ts.coords:
            stack = True
            ts = ts.drop_vars(
                [
                    "stacked_x_y",
                    "x",
                    "y",
                ],
                errors="ignore",
            )  # Use errors='ignore' to avoid errors if the column doesn't exist
        # Find valid new acquisition data
        acquisition_dates = feed.coords[TSVar.TIME.value].where(
            feed[TSVar.VALID.value], drop=True
        )
        # Select data corresponding to valid acquition dates
        selected_ts = ts.sel({TSVar.TIME.value: acquisition_dates})
        # Select where to update
        selected_condition = xr.apply_ufunc(
            np.vectorize(sh.check_state),
            selected_ts[TSVar.FLAGS.value],
            sh.State.ACQUISITION,
            vectorize=True,
        )
        # Update sub time series with acquisition data
        updated_ts = selected_ts
        # Update ET
        updated_ts[TSVar.ET.value] = updated_ts[TSVar.ET.value].where(
            selected_condition,
            feed[TSVar.ET.value].sel({TSVar.TIME.value: acquisition_dates}),
        )
        # Update Flags
        updated_ts[TSVar.FLAGS.value] = updated_ts[TSVar.FLAGS.value].where(
            selected_condition,
            xr.apply_ufunc(
                np.vectorize(sh.update_status),
                selected_ts[TSVar.FLAGS.value],
                kwargs={
                    "state": sh.State.ACQUISITION,
                    "processing": sh.ProcessingMode.NOMINAL,
                    "updated": True,
                    "distance": 0,
                },
            ),
        )
        # Update the full time series
        ts.loc[{TSVar.TIME.value: acquisition_dates}] = updated_ts
        if stack:
            ts = ts.assign_coords(stacked_x_y=data.coords["stacked_x_y"])
        return ts

    def update_nodata(
        self, index: int, status: sh.STATUS_TYPE, data: xr.Dataset
    ) -> tuple[float, sh.STATUS_TYPE]:
        """
        Update nodata
        """
        prev_index = self.find_previous(data[TSVar.FLAGS.value].data, index)
        next_index = self.find_next(data[TSVar.FLAGS.value].data, index)
        # Case 1: Interpolation
        if prev_index != -1 and next_index != -1:
            value, mode = self.interpolate(index, prev_index, next_index, data)
            status = sh.update_status(
                status,
                state=sh.State.INTERPOLATED,
                processing=mode,
                updated=True,
                distance=next_index - prev_index,
            )
            return value, status

        # Case 2: Extrapolation
        if prev_index != -1 and next_index == -1:
            value, mode = self.extrapolate(index, prev_index, data)
            status = sh.update_status(
                status,
                state=sh.State.EXTRAPOLATED,
                processing=mode,
                updated=True,
                distance=index - prev_index,
            )
            return value, status
        # Case 3: Backward extrapolation
        if prev_index == -1 and next_index != -1:
            value, mode = self.backward_extrapolate(index, next_index, data)
            status = sh.update_status(
                status,
                state=sh.State.BACKWARD_EXTRAPOLATED,
                processing=mode,
                updated=True,
                distance=next_index - index,
            )
            return value, status
        # Case 4: pixel invalid
        status = sh.update_status(
            status,
            state=sh.State.INVALID,
            processing=sh.ProcessingMode.DEGRADATED,
            updated=True,
            distance=0,
        )
        return np.nan, status

    def update_interpolated_data(
        self, index: int, status: sh.STATUS_TYPE, data: xr.Dataset
    ) -> tuple[float, sh.STATUS_TYPE]:
        """
        Update interpolated data
        """
        prev_index = self.find_previous(data[TSVar.FLAGS.value].data, index)
        next_index = self.find_next(data[TSVar.FLAGS.value].data, index)
        # If points used to interpolate exist and
        # If previous value was obtained with a degraded mode or
        # if the previous distance between interpolated points was greater
        if (prev_index != -1 and next_index != -1) and (
            sh.check_processing_mode(status, sh.ProcessingMode.DEGRADATED)
            or (next_index - prev_index) < sh.get_distance(status)
        ):
            # Interpolate
            value, mode = self.interpolate(index, prev_index, next_index, data)
            status = sh.update_status(
                status,
                state=sh.State.INTERPOLATED,
                processing=mode,
                updated=True,
                distance=next_index - prev_index,
            )
            return value, status
        # Do nothing
        return data[TSVar.ET.value].isel(
            {TSVar.TIME.value: index}
        ).item(), status

    def update_extrapolated_data(
        self, index: int, status: sh.STATUS_TYPE, data: xr.Dataset
    ) -> tuple[float, sh.STATUS_TYPE]:
        """
        Update extraplated data
        """
        prev_index = self.find_previous(data[TSVar.FLAGS.value].values, index)
        next_index = self.find_next(data[TSVar.FLAGS.value].values, index)
        # Case 1: If points used to interpolate exist and
        if prev_index != -1 and next_index != -1:
            value, mode = self.interpolate(index, prev_index, next_index, data)
            status = sh.update_status(
                status,
                state=sh.State.INTERPOLATED,
                processing=mode,
                updated=True,
                distance=next_index - prev_index,
            )
            return value, status
        # If previous index exists and
        # If previous value was obtained with a degraded mode or
        # the previous distance between interpolated points was greater
        if (prev_index != -1 and next_index == -1) and (
            sh.check_processing_mode(status, sh.ProcessingMode.DEGRADATED)
            or (index - prev_index) < sh.get_distance(status)
        ):
            # Extrapolate
            value, mode = self.extrapolate(index, prev_index, data)
            status = sh.update_status(
                status,
                state=sh.State.EXTRAPOLATED,
                processing=mode,
                updated=True,
                distance=index - prev_index,
            )
            return value, status
        # If next index exists and
        # If previous value was obtained with a degraded mode or
        # the previous distance between interpolated points was greater
        if (prev_index == -1 and next_index != -1) and (
            sh.check_processing_mode(status, sh.ProcessingMode.DEGRADATED)
            or (next_index - index) < sh.get_distance(status)
        ):
            # Extrapolate
            value, mode = self.backward_extrapolate(index, next_index, data)
            status = sh.update_status(
                status,
                state=sh.State.BACKWARD_EXTRAPOLATED,
                processing=mode,
                updated=True,
                distance=next_index - index,
            )
            return value, status
        # Do nothing
        return data[TSVar.ET.value].isel(
            {TSVar.TIME.value: index}
        ).item(), status

    def update_with_new_acquisitions(
        self, data: xr.Dataset, feed: xr.Dataset
    ) -> xr.Dataset:
        """
        Update time series with new acquisitions
        """
        # Get ET and status time series
        value_ts = data[TSVar.ET.value].values.copy()
        status_ts = data[TSVar.FLAGS.value].values.copy()
        # Find valid new acquisition data
        acquisition_dates = feed.coords[TSVar.TIME.value].where(
            feed[TSVar.VALID.value], drop=True
        )
        for date in acquisition_dates.values:
            value = feed[TSVar.ET.value].sel({TSVar.TIME.value: date}).item()
            status = (
                data[TSVar.FLAGS.value].sel({TSVar.TIME.value: date}).item()
            )
            if not sh.check_state(status, sh.State.ACQUISITION):
                index = data.indexes[TSVar.TIME.value].get_loc(date)
                status = sh.update_status(
                    status,
                    state=sh.State.ACQUISITION,
                    processing=sh.ProcessingMode.NOMINAL,
                    updated=True,
                    distance=0,
                )
                value_ts[index] = value
                status_ts[index] = status
        # Update
        updated_data = data.copy()
        updated_data[TSVar.ET.value].values = value_ts
        updated_data[TSVar.FLAGS.value].values = status_ts
        return updated_data

    def update_time_series(self, data: xr.Dataset) -> xr.Dataset:
        """
        Update time series
        """
        # Get ET and status time series
        value_ts = data[TSVar.ET.value].values.copy()
        status_ts = data[TSVar.FLAGS.value].values.copy()
        # Loop over the values for the time series
        for i, status in enumerate(status_ts):
            state = sh.get_state(status)
            if state.value == sh.State.NODATA.value:
                value_ts[i], status_ts[i] = self.update_nodata(i, status, data)
            if state.value == sh.State.INTERPOLATED.value:
                value_ts[i], status_ts[i] = self.update_interpolated_data(
                    i, status, data
                )
            if state.value == sh.State.EXTRAPOLATED.value:
                value_ts[i], status_ts[i] = self.update_extrapolated_data(
                    i, status, data
                )
            if state.value == sh.State.BACKWARD_EXTRAPOLATED.value:
                value_ts[i], status_ts[i] = self.update_extrapolated_data(
                    i, status, data
                )
        # Update
        updated_data = data.copy()
        updated_data[TSVar.ET.value].values = value_ts
        updated_data[TSVar.FLAGS.value].values = status_ts
        return updated_data

    def update(self, data: xr.Dataset, feed: xr.Dataset | None) -> xr.Dataset:
        """
        Update
        """
        # Get cdimensions with order
        original_dims = list(data.sizes.keys())
        spatial_dims = [x for x in original_dims if x != TSVar.TIME.value]
        x1 = spatial_dims[0]
        x2 = spatial_dims[1]

        def apply_update_time_series(data_ts: xr.Dataset) -> xr.Dataset:
            x1_val = data_ts[x1].values.item()
            x2_val = data_ts[x2].values.item()

            updated_ts = data_ts.copy()
            if feed is not None:
                # Extract corresponding feed
                feed_ts = feed.sel({x1: x1_val, x2: x2_val})

                # Update new acquisition
                updated_ts = self.update_with_new_acquisitions(
                    updated_ts, feed_ts
                )
            # Update time series
            return self.update_time_series(updated_ts)

        # Apply update to each (x, y) time series
        return data.groupby([x1, x2]).map(apply_update_time_series)
