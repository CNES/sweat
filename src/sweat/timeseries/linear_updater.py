# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for linear interpolation/extrapolation
"""

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

import sweat.timeseries.status_handler as sh
from sweat.timeseries.abstract_updater import Updater
from sweat.timeseries.types import TimeSeriesVar as TSVar


class LinearUpdaterParams(BaseModel):
    """
    Parameters to configure linear updater
    """

    model_config = ConfigDict(extra="forbid")
    parallel: bool = Field(default=False)
    num_workers: int = Field(default=4)

    @field_validator("num_workers")
    @classmethod
    def check_num_workers(cls, value: int):
        if value < 1:
            msg = "Number of workers must be greater than 1"
            raise ValueError(msg)
        return value


class LinearUpdater(Updater):
    """
    Linear updater
    """

    variables: ClassVar[Sequence[str]] = [TSVar.RADIATION.value]

    def update_with_acquisitions(
        self,
        et: xr.DataArray,
        radiation: xr.DataArray,
        state: xr.DataArray,
        distance: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Update with linear interpolation of ratio using acquisitions

        Notes
        -----
        Expected behavior is:
            * 2+ acquisitions → linear interpolation of the ratio + propagation
            of the first valid ratio value and the last valid ratio value
            * 1 acquisition → propagate the ratio everywhere
            * 0 acquisition → remain NaN

        Parameters
        ----------
        et : xr.DataArray
            ET data
        radiation : xr.DataArray
            Radiation data
        state : xr.DataArray
            State
        distance : xr.DataArray
            Distance

        Returns
        -------
        updated_data : tuple[xr.DataArray]
            ET data, updated flag, state, distance
        """
        # Condition
        condition = (state == sh.State.ACQUISITION.value) & (
            radiation.notnull()
        )  # Only acquisition
        # Extract reference for interpolation/extrapolation
        et_ref = et.where(condition)
        # Compute next indices
        next_idx = Updater.find_next(state, condition)
        # Compute previous indices
        previous_idx = Updater.find_previous(state, condition)
        # Compute ratio
        ratio_ref = et_ref / radiation
        # Change time coordinates/dimension to numeric
        time_num = (
            ratio_ref.time - ratio_ref.time.isel({TSVar.TIME.value: 0})
        ) / np.timedelta64(1, "D")
        time_size = len(time_num)
        ratio_ref = ratio_ref.assign_coords(
            time_num=(TSVar.TIME.value, time_num.data)
        ).swap_dims({TSVar.TIME.value: "time_num"})
        # Fill NaN values
        ratio_filled = ratio_ref.interpolate_na(
            dim="time_num",
            method="linear",
            # fill_value="extrapolate",
        )
        # Handle cases where only one valid value exists
        # Extrapolate by propagating the ratio
        ratio_filled = ratio_filled.ffill("time_num").bfill("time_num")
        # Restore the original time dimension
        ratio_filled = ratio_filled.swap_dims({"time_num": TSVar.TIME.value})
        ratio_filled = ratio_filled.assign_coords(time=et.time).drop_vars(
            "time_num"
        )
        # Fill ET
        et_filled = ratio_filled * radiation
        # New state
        previous_exists = previous_idx >= 0
        next_exists = next_idx < time_size
        # Interpolation: valid point on both sides
        new_state = xr.full_like(state, sh.State.NODATA.value, dtype=np.int8)
        new_state = xr.where(
            previous_exists & next_exists,
            sh.State.INTERPOLATED.value,
            new_state,
        )
        # Backward extrapolation: no previous point, but a next point exists
        new_state = xr.where(
            ~previous_exists & next_exists,
            sh.State.BACKWARD_EXTRAPOLATED.value,
            new_state,
        )
        # Forward extrapolation: previous point exists, but no next point
        new_state = xr.where(
            previous_exists & ~next_exists,
            sh.State.FORWARD_EXTRAPOLATED.value,
            new_state,
        )
        # Invalid: no previous point, no next point
        new_state = xr.where(
            (~previous_exists & ~next_exists) | (radiation.isnull()),
            sh.State.INVALID.value,
            new_state,
        )
        # Compute distance between previous and next indices
        current_idx = np.arange(time_size)
        # Interpolation: valid point on both sides
        new_distance = xr.full_like(distance, sh.DISTANCE_MAX, dtype=np.uint8)
        new_distance = xr.where(
            previous_exists & next_exists, next_idx - previous_idx, new_distance
        )
        # Backward extrapolation: no previous point, but a next point exists
        new_distance = xr.where(
            ~previous_exists & next_exists, next_idx - current_idx, new_distance
        )
        # Forward extrapolation: previous point exists, but no next point
        new_distance = xr.where(
            previous_exists & ~next_exists,
            current_idx - previous_idx,
            new_distance,
        )
        new_distance = new_distance.where(radiation.notnull(), sh.DISTANCE_MAX)
        # Update
        # To be updated the status must improve or the distance must be reduced
        target = (sh.is_better(new_state, state)) | (  # Better state
            (sh.is_same(new_state, state))
            & (new_distance < distance)  # Same state but better distance
        )
        return (
            xr.where(target, et_filled, et),
            target.astype(bool),
            xr.where(target, new_state, state).astype(np.int8),
            xr.where(target, new_distance, distance).astype(np.uint8),
        )

    def update_with_extrapolation_only(
        self,
        et: xr.DataArray,
        radiation: xr.DataArray,
        state: xr.DataArray,
        distance: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Update using extrapolation data only.

        Parameters
        ----------
        et : xr.DataArray
            ET data
        radiation : xr.DataArray
            Radiation data
        state : xr.DataArray
            State
        distance : xr.DataArray
            Distance

        Returns
        -------
        updated_data : tuple[xr.DataArray]
            ET data, updated flag, state, distance
        """
        # Change time coordinates/dimension to numeric
        time_num = (
            et.time - et.time.isel({TSVar.TIME.value: 0})
        ) / np.timedelta64(1, "D")
        time_size = len(time_num)
        # Forward extrapolation
        condition = (state == sh.State.FORWARD_EXTRAPOLATED.value) & (
            radiation.notnull()
        )  # Extrapolated point
        # Extract reference for interpolation/extrapolation
        et_ref = et.where(condition)
        # Compute previous indices
        previous_idx = Updater.find_previous(state, condition)
        # Compute ratio
        ratio_ref = et_ref / radiation
        ratio_ref = ratio_ref.assign_coords(
            time_num=(TSVar.TIME.value, time_num.data)
        ).swap_dims({TSVar.TIME.value: "time_num"})
        # Fill NaN values
        ratio_ffilled = ratio_ref.ffill("time_num")
        ratio_ffilled = ratio_ffilled.swap_dims({"time_num": TSVar.TIME.value})
        ratio_ffilled = ratio_ffilled.assign_coords(time=et.time).drop_vars(
            "time_num"
        )
        et_ffilled = ratio_ffilled * radiation
        # New state for forward extrapolation
        previous_exists = previous_idx >= 0
        new_fstate = xr.full_like(state, sh.State.NODATA.value, dtype=np.int8)
        new_fstate = xr.where(
            previous_exists, sh.State.FORWARD_EXTRAPOLATED.value, new_fstate
        )
        new_fstate = new_fstate.where(
            radiation.notnull(), sh.State.INVALID.value
        )
        # Compute distance for forward extrapolation
        previous_distance = np.take_along_axis(
            distance.values,
            previous_idx.values.clip(min=0),
            axis=-1,
        )
        current_idx = np.arange(time_size)
        new_fdistance = xr.full_like(distance, sh.DISTANCE_MAX, dtype=np.uint8)
        new_fdistance = xr.where(
            previous_exists,
            previous_distance + current_idx - previous_idx,
            new_fdistance,
        )
        new_fdistance = new_fdistance.where(
            radiation.notnull(), sh.DISTANCE_MAX
        )

        # Backward extrapolation
        condition = (state == sh.State.BACKWARD_EXTRAPOLATED.value) & (
            radiation.notnull()
        )  # Extrapolated point
        # Extract reference for interpolation/extrapolation
        et_ref = et.where(condition)
        # Compute next indices
        next_idx = Updater.find_next(state, condition)
        # Compute ratio
        ratio_ref = et_ref / radiation
        ratio_ref = ratio_ref.assign_coords(
            time_num=(TSVar.TIME.value, time_num.data)
        ).swap_dims({TSVar.TIME.value: "time_num"})
        # Fill NaN values
        ratio_bfilled = ratio_ref.bfill("time_num")
        ratio_bfilled = ratio_bfilled.swap_dims({"time_num": TSVar.TIME.value})
        ratio_bfilled = ratio_bfilled.assign_coords(time=et.time).drop_vars(
            "time_num"
        )
        et_bfilled = ratio_bfilled * radiation
        # New state for forward extrapolation
        next_exists = next_idx.values >= 0
        new_bstate = xr.full_like(state, sh.State.NODATA.value, dtype=np.int8)
        new_bstate = xr.where(
            next_exists, sh.State.BACKWARD_EXTRAPOLATED.value, new_bstate
        )
        new_bstate = new_bstate.where(
            radiation.notnull(), sh.State.INVALID.value
        )
        # Compute distance for backward extrapolation
        next_distance = np.take_along_axis(
            distance.values,
            next_idx.values.clip(max=(time_size - 1)),
            axis=-1,
        )
        new_bdistance = xr.full_like(distance, sh.DISTANCE_MAX, dtype=np.uint8)
        new_bdistance = xr.where(
            next_exists, next_distance + next_idx - current_idx, new_bdistance
        )
        new_bdistance = new_bdistance.where(
            radiation.notnull(), sh.DISTANCE_MAX
        )

        # Update
        forward_condition = new_fdistance <= new_bdistance
        et_filled = xr.where(forward_condition, et_ffilled, et_bfilled)
        new_state = xr.where(forward_condition, new_fstate, new_bstate)
        new_distance = xr.where(forward_condition, new_fdistance, new_bdistance)
        target = (sh.is_better(new_state, state)) | (
            (sh.is_same(new_state, state)) & (new_distance < distance)
        )
        # Update
        return (
            xr.where(target, et_filled, et),
            target.astype(bool),
            xr.where(target, new_state, state).astype(np.int8),
            xr.where(target, new_distance, distance).astype(np.uint8),
        )

    def update_time_series(self, data: xr.Dataset) -> xr.Dataset:
        """
        Update time series

        Parameters
        ----------
        data : xr.Dataset
            Data to update

        Returns
        -------
        updated : xr.Dataset
            Updated data
        """
        # Step 1: Update using acquisitions
        new_et, new_updated, new_state, new_distance = (
            self.update_with_acquisitions(
                et=data[TSVar.ET.value],
                radiation=data[TSVar.RADIATION.value],
                state=data[TSVar.STATE.value],
                distance=data[TSVar.DISTANCE.value],
            )
        )
        updated = (
            data["updated"].astype(bool) | new_updated
        )  # Update with existing one
        # Step 2: Update using extrapolation only
        new_et, new_updated, new_state, new_distance = (
            self.update_with_extrapolation_only(
                et=new_et,
                radiation=data[TSVar.RADIATION.value],
                state=new_state,
                distance=new_distance,
            )
        )
        updated |= new_updated  # Update with existing one

        return xr.Dataset(
            {
                TSVar.ET.value: new_et,
                TSVar.RADIATION.value: data[TSVar.RADIATION.value],
                TSVar.STATE.value: new_state,
                TSVar.DISTANCE.value: new_distance,
                TSVar.UPDATED.value: updated,
                TSVar.VALIDITY_FLAGS.value: data[TSVar.VALIDITY_FLAGS.value],
            },
            attrs=data.attrs.copy(),
        )
