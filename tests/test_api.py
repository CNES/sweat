# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

import geopandas as gpd
import pytest

from evaspa import api, tiling


def get_data_path() -> str:
    """
    Get data path
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_generate_tiles() -> None:
    """
    Test methods for regroup tiles
    """
    data_path = get_data_path()
    tiles, adjs = api.generate_tiles(os.path.join(data_path, "roi.gpkg"))
    assert len(tiles) == 8
    assert len(adjs) == 8
    tiles, adjs = api.generate_tiles(
        os.path.join(data_path, "roi.gpkg"), orbit_id=110
    )
    assert tiles.empty
    assert adjs.empty
    with pytest.raises(ValueError, match="Orbit ID must be between 0 and 114"):
        tiles, adjs = api.generate_tiles(
            os.path.join(data_path, "roi.gpkg"), orbit_id=210
        )


def test_regroup() -> None:
    """
    Test methods for regroup tiles
    """
    data_path = get_data_path()
    tiles = gpd.read_file(os.path.join(data_path, "roi_tiles.gpkg"))
    land = gpd.read_file(os.path.join(data_path, "land_italy.gpkg"))
    adjs = tiling.generate_adjacents(tiles, land)
    group = api.regroup_tiles(tiles, adjs, threshold=100)
    assert len(group) == 9
    group = api.regroup_tiles(tiles, adjs, threshold=1000000)
    assert len(group) == 5


def test_run_evaspa() -> None:
    """
    Test run EVASPA
    """
    entry = {"path": "tests/data/modis_test.tif"}
    params = {"ef": {"check": {"threshold": 0.02}, "models": "default_evaspa"}}
    res = api.run_evaspa(entry, params)
    assert res
