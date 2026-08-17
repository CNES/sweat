# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pytest
import xarray as xr

from sweat.timeseries.abstract_updater import Updater


@pytest.mark.unit
def test_requested_variables() -> None:
    """
    Test requested variables
    """
    assert Updater.variables == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        pytest.param(
            np.array([7, 7, 7, 0, 7, 7, 0]),
            np.array([3, 3, 3, 3, 6, 6, 6]),
            id="case1",
        ),
        pytest.param(
            np.array([0, 7, 7, 7, 7, 7, 0]),
            np.array([0, 6, 6, 6, 6, 6, 6]),
            id="case2",
        ),
        pytest.param(
            np.array([7, 7, 0, 7, 7, 7, 7]),
            np.array([2, 2, 2, 7, 7, 7, 7]),
            id="case2",
        ),
    ],
)
def test_find_next(state, expected) -> None:
    """
    Test function find_next
    """
    # Create DataArray
    da = xr.DataArray(
        state[None, None, :].astype(np.int8),
        dims=["y", "x", "time"],
        coords={"time": np.arange(7), "y": np.arange(1), "x": np.arange(1)},
    )
    ref = xr.DataArray(
        expected[None, None, :].astype(np.int8),
        dims=["y", "x", "time"],
        coords={"time": np.arange(7), "y": np.arange(1), "x": np.arange(1)},
    )
    next_index = Updater.find_next(da, da == 0)
    xr.testing.assert_equal(next_index, ref)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        pytest.param(
            np.array([7, 7, 7, 0, 7, 7, 7]),
            np.array([-1, -1, -1, 3, 3, 3, 3]),
            id="case1",
        ),
        pytest.param(
            np.array([0, 7, 7, 7, 7, 7, 0]),
            np.array([0, 0, 0, 0, 0, 0, 6]),
            id="case2",
        ),
    ],
)
def test_find_previous(state, expected) -> None:
    """
    Test function find_previous
    """
    # Create DataArray
    da = xr.DataArray(
        state[None, None, :].astype(np.int8),
        dims=["y", "x", "time"],
        coords={"time": np.arange(7), "y": np.arange(1), "x": np.arange(1)},
    )
    ref = xr.DataArray(
        expected[None, None, :].astype(np.int8),
        dims=["y", "x", "time"],
        coords={"time": np.arange(7), "y": np.arange(1), "x": np.arange(1)},
    )
    previous_index = Updater.find_previous(da, da == 0)
    xr.testing.assert_equal(previous_index, ref)
