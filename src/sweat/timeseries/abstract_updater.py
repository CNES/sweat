# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for time series update
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import dask
import numpy as np
import xarray as xr

import sweat.timeseries.status_handler as sh
from sweat.timeseries.types import TimeSeriesVar as TSVar


@dataclass(frozen=True)
class Updater(ABC):
    """
    Class to update time series
    """

    # Class attributes
    variables: ClassVar[Sequence[str]] = []

    # Instance attributes
    parallel: bool = False
    num_workers: int = 4

    @staticmethod
    def find_next(state: xr.DataArray, condition: xr.DataArray) -> xr.DataArray:
        """
        Find next index. Return size of time dimensions if not found.

        Parameters
        ----------
        state : xr.DataArray
            State
        condition : xr.DataArray
            Condition to consider next pixel
        Returns
        -------
        next_idx : xr.DataArray
            Next index
        """
        # Time dimension
        time_size = state.sizes[TSVar.TIME.value]

        # Time indices
        idx = np.arange(time_size)

        # Index at each location if it is a target, otherwise time_size
        next_idx = np.where(condition.values, idx, time_size)

        # Propagate the next target index backwards in time
        # and reverse time again
        next_idx = np.minimum.accumulate(next_idx[..., ::-1], axis=-1)[
            ..., ::-1
        ]

        # strictly after the current index
        # next_idx = np.where(
        # condition,
        # np.concatenate(
        #    [
        #        next_idx[..., 1:],
        #        np.full(next_idx[..., :1].shape, time_size)
        #    ],
        #    axis=-1
        # ),
        # next_idx
        # )

        # Create DataArray
        return state.copy(data=next_idx)

    @staticmethod
    def find_previous(
        state: xr.DataArray, condition: xr.DataArray
    ) -> xr.DataArray:
        """
        Find next index. Return size of time dimensions if not found.

        Parameters
        ----------
        state : xr.DataArray
            State
        condition : xr.DataArray
            Condition to consider next pixel

        Returns
        -------
        previous_idx : xr.DataArray
            Next index
        """
        # Time dimension
        time_size = state.sizes[TSVar.TIME.value]

        # Time indices
        idx = np.arange(time_size)

        # Index at each location if it is a target, otherwise -1
        prev_idx = np.where(condition.values, idx, -1)

        # Propagate the next target index backwards in time
        previous_idx = np.maximum.accumulate(
            prev_idx,
            axis=-1,
        )

        # strictly before the current index
        # previous_idx = np.concatenate(
        #    [
        #        np.full(previous_idx[..., :1].shape, -1),
        #        previous_idx[...,:-1],
        #    ],
        #    axis=-1,
        # )

        # Create DataArray
        return state.copy(data=previous_idx)

    def update_with_new_acquisitions(
        self, data: xr.Dataset, feed: xr.Dataset
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Update time series with new acquisitions

        Parameters
        ----------
        data : xr.Dataset
            Data to update
        feed : xr.Dataset
            Data corresponding new acquisitions

        Returns
        -------
        updated : xr.Dataset
            Updated data
        """
        if data.sizes[TSVar.TIME.value] != feed.sizes[TSVar.TIME.value]:
            # Reindex feed to match data's time dimension
            # Fill missing values: NaN for 'et', 0 for 'valid'
            feed = feed.reindex(
                {TSVar.TIME.value: data.time},
                fill_value={
                    TSVar.ET.value: np.nan,
                    TSVar.VALID.value: 0,
                },
            )
        # Condition to update
        condition = feed[TSVar.VALID.value]
        # Fill data with feed where condition is True
        return (
            xr.where(condition, feed[TSVar.ET.value], data[TSVar.ET.value]),
            xr.where(
                condition, sh.State.ACQUISITION.value, data[TSVar.STATE.value]
            ).astype(sh.STATE_TYPE),
            xr.where(condition, True, data[TSVar.UPDATED.value]).astype(bool),
            xr.where(condition, 0, data[TSVar.DISTANCE.value]).astype(
                sh.DISTANCE_TYPE
            ),
        )

    @abstractmethod
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

    def _update_block_with_new_acquisitions(
        self,
        data: xr.Dataset,
        feed: xr.Dataset,
    ) -> xr.Dataset:
        """
        Update with new acquisition

        Parameters
        ----------
        data : xr.Dataset
            Data to update
        feed : xr.Dataset
            Data corresponding new acquisitions

        Returns
        -------
        updated : xr.Dataset
            Updated data
        """
        # Step 1 update with new acquisitions
        new_et, new_state, new_updated, new_distance = (
            self.update_with_new_acquisitions(data, feed)
        )
        # Step 2 update time series
        return self.update_time_series(
            xr.Dataset(
                {
                    TSVar.ET.value: new_et,
                    TSVar.RADIATION.value: data[TSVar.RADIATION.value],
                    TSVar.STATE.value: new_state,
                    TSVar.DISTANCE.value: new_distance,
                    TSVar.UPDATED.value: new_updated,
                    TSVar.VALIDITY_FLAGS.value: data[
                        TSVar.VALIDITY_FLAGS.value
                    ],
                },
                attrs=data.attrs.copy(),
            )
        )

    def _update_block_without_new_acquisitions(
        self,
        data: xr.Dataset,
    ) -> xr.Dataset:
        """
        Update without new acquisition

        Parameters
        ----------
        data : xr.Dataset
            Data to update

        Returns
        -------
        updated : xr.Dataset
            Updated data
        """
        # Update time series
        return self.update_time_series(data)

    def update(
        self, data: xr.Dataset, feed: xr.Dataset | None = None
    ) -> xr.Dataset:
        """
        Update

        Parameters
        ----------
        data : xr.Dataset
            Data to update
        feed : xr.Dataset
            Data corresponding new acquisitions

        Returns
        -------
        updated : xr.Dataset
            Updated data
        """
        if not self.parallel:
            if feed is not None:
                return self._update_block_with_new_acquisitions(
                    data=data, feed=feed
                )
            return self._update_block_without_new_acquisitions(data=data)
        # Chunk spatial dimensions only.
        with dask.config.set(
            scheduler="processes", num_workers=self.num_workers
        ):
            spatial_dims = [x for x in data.dims if x != TSVar.TIME.value]
            data = data.chunk(
                {
                    spatial_dims[0]: 256,
                    spatial_dims[1]: 256,
                    TSVar.TIME.value: -1,
                }
            )

            # Case 1: new acquisition is available
            if feed is not None:
                # Make new_data have the same time coordinate as data.
                # Reindex feed to match data's time dimension
                # Fill missing values: NaN for 'et', 0 for 'valid'
                feed = feed.reindex(
                    {TSVar.TIME.value: data.time},
                    fill_value={
                        TSVar.ET.value: np.nan,
                        TSVar.VALID.value: 0,
                    },
                )
                # Use the same spatial chunking as data.
                feed = feed.chunk(
                    {
                        spatial_dims[0]: 256,
                        spatial_dims[1]: 256,
                        TSVar.TIME.value: -1,
                    }
                )
                return xr.map_blocks(
                    self._update_block_with_new_acquisitions,
                    data,
                    args=(feed,),
                    template=data,
                ).compute()

            # Case 2: no new acquisition
            return xr.map_blocks(
                self._update_block_without_new_acquisitions,
                data,
                template=data,
            ).compute()
