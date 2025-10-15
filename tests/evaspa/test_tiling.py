# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from sweat.evaspa import tiling


def get_data_path() -> str:
    """
    Get data path
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


@pytest.mark.unit
def test_groupby_multipolygon() -> None:
    """
    Test groupby_multipolygon
    """
    df = gpd.GeoDataFrame(
        {
            "a": [0, 0, 1],
            "b": [1, 2, 3],
            "overlap_percentage": [25, 25, 30],
            "overlap_geometry": [
                Polygon([(0, 0), (1, 0), (1, 1)]),
                Polygon([(1, 0), (1, 0), (1, 1)]),
                Polygon([(0, 2), (1, 0), (1, 1)]),
            ],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1)]),
            Polygon([(0, 2), (1, 0), (1, 1)]),
        ],
    )

    grouped = tiling.groupby_multipolygon(df, by="a")
    assert len(grouped) == 2
    assert grouped.loc[grouped.a == 0, "overlap_percentage"].to_numpy()[0] == 50
    assert grouped.loc[grouped.a == 1, "overlap_percentage"].to_numpy()[0] == 30
    assert grouped.loc[grouped.a == 0, "overlap_geometry"].to_numpy()[
        0
    ] == MultiPolygon(
        [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(1, 0), (1, 0), (1, 1)])]
    )
    np.testing.assert_array_equal(
        grouped.geometry.to_numpy(),
        [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(0, 2), (1, 0), (1, 1)])],
    )


@pytest.mark.unit
def test_intersection() -> None:
    """
    Test intersection method
    """
    data_path = get_data_path()
    gdf1 = gpd.read_file(os.path.join(data_path, "bordeaux.gpkg"))
    gdf2 = gpd.read_file(os.path.join(data_path, "europe.gpkg"))
    inter = tiling.intersection(gdf1, gdf2)
    assert len(inter) == 2
    np.testing.assert_almost_equal(
        inter.loc[inter.id == "30TXP", "overlap_percentage"].to_numpy()[0],
        78,
        decimal=0,
    )
    np.testing.assert_almost_equal(
        inter.loc[inter.id == "30TXQ", "overlap_percentage"].to_numpy()[0],
        62,
        decimal=0,
    )

    gdf1 = gpd.read_file("/home/sarrazine/src/evaspa/tests/data/corsica.gpkg")
    gdf2 = gpd.read_file("/home/sarrazine/src/evaspa/tests/data/land.gpkg")
    inter = tiling.intersection(gdf1, gdf2)
    assert len(inter) == 1
    assert len(list(inter.overlap_geometry.to_numpy()[0].geoms)) == 2
    np.testing.assert_almost_equal(
        inter["overlap_percentage"].to_numpy()[0], 17, decimal=0
    )


@pytest.mark.unit
def test_get_adjacent_tiles() -> None:
    """
    Test get_adjacent_tiles() method
    """
    data_path = get_data_path()
    tiles = gpd.read_file(os.path.join(data_path, "italy.gpkg"))
    orbits = gpd.read_file(os.path.join(data_path, "orbit.gpkg"))
    land = gpd.read_file(os.path.join(data_path, "land_italy.gpkg"))

    adjs = tiling.get_adjacent_tiles("32TNL", tiles, land)
    assert adjs
    np.testing.assert_array_equal(adjs, ["32TML", "32TNK"])
    adjs = tiling.get_adjacent_tiles("32TNM", tiles, land)
    assert adjs
    np.testing.assert_array_equal(adjs, ["32TMM", "32TNL", "32TNN"])
    adjs = tiling.get_adjacent_tiles("33SWD", tiles, land)
    assert adjs
    np.testing.assert_array_equal(adjs, ["33SWC", "33SXD", "33TWE"])
    adjs = tiling.get_adjacent_tiles(
        "33SWD", tiles, land, orbit_id=7, orbit=orbits
    )
    assert adjs
    np.testing.assert_array_equal(adjs, ["33SWC", "33TWE"])


@pytest.mark.unit
def test_generate_adjacents() -> None:
    """
    Test check_adjacents() method
    """
    data_path = get_data_path()
    tiles = gpd.read_file(os.path.join(data_path, "tiles2.gpkg"))
    land = gpd.read_file(os.path.join(data_path, "land_italy.gpkg"))
    adjs = tiling.generate_adjacents(tiles, land)
    ref = pd.DataFrame(
        [
            ["31TFH", "31TFJ,31TGH"],
            ["31TFJ", "31TFH"],
            ["31TGH", "31TFH"],
            ["32TMK", "32TML"],
            ["32TML", "32TMK,32TNL"],
            ["32TMM", "32TMN,32TNM"],
            ["32TMN", "32TMM"],
            ["32TNL", "32TML"],
            ["32TNM", "32TMM"],
            ["33SVC", ""],
        ],
        columns=["id", "adjs"],
    ).set_index("id")
    pd.testing.assert_frame_equal(adjs, ref)


@pytest.mark.unit
def test_check_adjacents() -> None:
    """
    Test check_adjacents() method
    """
    df = pd.DataFrame(
        [
            ["31TFH", "31TFJ,31TGH"],
            ["31TFJ", "31TFH,31TFK,31TGJ"],
            ["31TGH", "31TFH,31TGJ"],
            ["32TMK", "32SMJ,32TML,32TNK"],
            ["32TML", "32TMK,32TNL"],
            ["32TMM", "32TML,32TMN,32TNM"],
            ["32TMN", "32TMM,32TNN"],
            ["32TNL", "32TML,32TNK"],
            ["32TNM", "32TMM,32TNL,32TNN"],
            ["33SVC", "33SUC,33SVB,33SWC"],
        ],
        columns=["id", "adjs"],
    ).set_index("id")
    checked_df = tiling.check_adjacents(df)
    ref = pd.DataFrame(
        [
            ["31TFH", "31TFJ,31TGH"],
            ["31TFJ", "31TFH"],
            ["31TGH", "31TFH"],
            ["32TMK", "32TML"],
            ["32TML", "32TMK,32TNL"],
            ["32TMM", "32TMN,32TNM"],
            ["32TMN", "32TMM"],
            ["32TNL", "32TML"],
            ["32TNM", "32TMM"],
            ["33SVC", ""],
        ],
        columns=["id", "adjs"],
    ).set_index("id")
    pd.testing.assert_frame_equal(checked_df, ref)


@pytest.mark.unit
def test_regroup() -> None:
    """
    Test methods for regroup tiles
    """
    data_path = get_data_path()
    roi_tiles = gpd.read_file(os.path.join(data_path, "roi_tiles.gpkg"))
    land = gpd.read_file(os.path.join(data_path, "land_italy.gpkg"))
    adjs = tiling.generate_adjacents(roi_tiles, land)
    regroup_threshold = 100
    tile_df, group_df = tiling.initialize_regroup(roi_tiles, land)
    tile_df, group_df = tiling.regroup(
        tile_df, adjs, group_df, regroup_threshold, land
    )
    assert len(group_df) == 9
    regroup_threshold = 1000000
    tile_df, group_df = tiling.initialize_regroup(roi_tiles, land)
    tile_df, group_df = tiling.regroup(
        tile_df, adjs, group_df, regroup_threshold, land
    )
    assert len(group_df) == 4
