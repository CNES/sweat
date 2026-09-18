# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

import affine
import numpy as np
import pytest
from shapely.geometry import Polygon

from sweat.evaspa import zones
from sweat.misc import trishna
from sweat.misc.dem import get_dem_from_tiles


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
def test_compute_water_mask() -> None:
    """
    Test compute_water_mask() method
    """
    land = trishna.get_land_mask()
    dem = get_dem_from_tiles(["32TLP"], base_dir=get_test_data_path())
    water_mask = zones.compute_water_mask(dem, land)
    assert water_mask.sizes["y"] == 1830
    assert water_mask.sizes["x"] == 1830
    assert water_mask.data[-1, -1]
    assert not water_mask.data[0, 0]


@pytest.mark.functional
@pytest.mark.require_test_data
def test_compute_valid_mask() -> None:
    """
    Test compute_valid_mask() method
    """
    dem = get_dem_from_tiles(["31TCJ"], base_dir=get_test_data_path())
    valid = zones.compute_valid_mask(dem)
    assert not np.isnan(valid.data).all()
    valid = zones.compute_valid_mask(dem, range_threshold=600)
    assert np.sum(valid.data) == 3348900


@pytest.mark.functional
@pytest.mark.require_test_data
def test_define_valid_pixels() -> None:
    """
    Test define_valid_pixels() method
    """
    dem = get_dem_from_tiles(["31TCJ"], base_dir=get_test_data_path())
    pvalid, dem = zones.define_valid_pixels(dem, range_threshold=600)
    np.testing.assert_approx_equal(pvalid, 1.00, significant=2)
    assert np.sum(dem["valid"].data) == 3348900
    land = trishna.get_land_mask()
    dem = get_dem_from_tiles(["32TLP"], base_dir=get_test_data_path())
    pvalid, dem = zones.define_valid_pixels(dem, land)
    np.testing.assert_approx_equal(pvalid, 0.14, significant=2)
    assert not np.isnan(dem["valid"].data).all()


@pytest.mark.unit
def test_polygonize_mask() -> None:
    """
    Test polygonize_mask() method
    """
    mask = np.zeros((100, 100))
    mask[40:60, 40:60] = 1
    identity = affine.Affine(1, 0, 0, 0, 1, 0)
    poly = zones.polygonize_mask(mask, identity)
    assert len(poly) == 1
    assert poly[0] == Polygon(
        [(40, 40), (40, 60), (60, 60), (60, 40), (40, 40)]
    )


@pytest.mark.functional
@pytest.mark.require_test_data
def test_define_valid_zones() -> None:
    """
    Test define_valid_zones() method
    """
    land = trishna.get_land_mask()
    pvalid, nbvalid, polys = zones.define_valid_zones(["32TLP"], land)
    np.testing.assert_approx_equal(pvalid, 0.14, significant=2)
    assert nbvalid == 474657
    assert len(polys.geoms) == 6
