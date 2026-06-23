# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

import numpy as np
import pytest

from sweat.misc.dem import get_dem_from_tile, get_dem_from_tiles


def get_test_data_path() -> str:
    """
    Get test data path for DEM tiles
    """
    return os.path.join(
        os.environ["SWEAT_TEST_DATA_PATH"], "DEM_Copernicus_30m"
    )


@pytest.fixture(autouse=True)
def setup_before_each_test():
    os.environ["MNT_PATH"] = os.environ.get("SWEAT_TEST_DATA_PATH", None)


@pytest.mark.functional
@pytest.mark.require_test_data
def test_get_dem_from_tile() -> None:
    """
    Test get_dem_from_tile() method
    """
    dem = get_dem_from_tile("32TML", base_dir=get_test_data_path())
    assert dem
    assert dem.sizes["y"] == 1830
    assert dem.sizes["x"] == 1830
    assert dem.crs == "EPSG:32632"
    assert dem.resolution == 60
    np.testing.assert_array_equal(
        sorted(dem.data_vars), ["aspect", "height", "slope"]
    )


@pytest.mark.functional
@pytest.mark.require_test_data
def test_get_dem_from_tiles() -> None:
    """
    Test get_dem_from_tiles() method
    """
    dem = get_dem_from_tiles(["32TML"], base_dir=get_test_data_path())
    assert dem
    assert dem.sizes["y"] == 1830
    assert dem.sizes["x"] == 1830
    assert dem.crs == "EPSG:32632"
    assert dem.resolution == 60
    np.testing.assert_array_equal(
        sorted(dem.data_vars),  # type: ignore
        ["aspect", "height", "slope"],
    )

    dem = get_dem_from_tiles(["32TML", "32TNL"], base_dir=get_test_data_path())
    assert dem
    assert dem.sizes["y"] == 1830
    assert dem.sizes["x"] == 3497
    assert dem.crs == "EPSG:32632"
    assert dem.resolution == 60
    np.testing.assert_array_equal(
        sorted(dem.data_vars),  # type: ignore
        ["aspect", "height", "slope"],
    )
