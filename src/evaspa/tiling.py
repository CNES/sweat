#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for tiling management
"""

import os
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sensorsio import mgrs
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from evaspa.zones import define_valid_zones

from .logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def groupby_multipolygon(df: gpd.GeoDataFrame, by: str = "id") -> gpd.GeoDataFrame:
    """
    Regroup multipolygons issued from the
    intersection result

    Parameters
    ----------
    df: GeoDataFrame
    by: str
        Column name used to group elements in df

    Returns
    -------
    aggregated: GeoDataFrame
    """
    data = df.drop(columns=["overlap_geometry"])
    aggfunc = {
        name: ("sum" if name == "overlap_percentage" else "first")
        for name in data.columns
    }
    aggregated_data = data.groupby(by=by).agg(aggfunc)

    # Process spatial component
    def merge_geometries(block):
        polys = []
        for poly in block.values:
            if isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    polys.append(p)
            else:
                polys.append(poly)
        return MultiPolygon(polys)

    g = df.groupby(by=by, group_keys=False)["overlap_geometry"].agg(merge_geometries)

    # Aggregate
    aggregated = gpd.GeoDataFrame(aggregated_data, geometry="geometry", crs=df.crs)
    # Recombine
    aggregated = aggregated.join(g).reset_index(drop=True)
    return aggregated


def intersection(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, by: str = "id"
) -> gpd.GeoDataFrame:
    """
    Intersection between two GeoDataFrames with
    the calculation of the overlap percentage

    Parameters
    ----------
    gdf1: GeoDataFrame
    gdf2: GeoDataFrame
    by: str, default="id"
        Column name used to identify elements in gdf1

    Returns
    -------
    inter: GeoDataFrame
    """
    inter = gpd.GeoDataFrame(
        gpd.overlay(gdf1[[by, "geometry"]], gdf2[["geometry"]], how="intersection")
        .merge(gdf1, how="inner", on=by, suffixes=("_overlap", "_grid"))
        .rename(
            columns={
                "geometry_overlap": "overlap_geometry",
                "geometry_grid": "geometry",
            }
        )
    )
    if len(inter) == 0:
        inter["overlap_percentage"] = pd.Series(dtype="float")
        return inter
    inter["overlap_percentage"] = inter.apply(
        lambda tile: 100 * tile.overlap_geometry.area / tile.geometry.area, axis=1
    )
    return groupby_multipolygon(inter, by=by)


def get_adjacent_tiles(
    tile_id: str,
    tiles: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    orbit_id: int | None = None,
    orbit: gpd.GeoDataFrame | None = None,
    by: str = "id",
    neighbors_overlap: float = 5.0,
    land_overlap: float = 2.0,
    orbit_overlap: float = 25.0,
) -> Optional[List[str]]:
    """
    Compute adjacent tiles

    Parameters
    ----------
    tile_id: str
        Tile ID
    tiles: GeoDataFrame
        List of tiles
    land: GeoDataFrame
        List of land polygons
    orbit_id: int, optional
        Relative orbit number to take into account
    orbit: GeoDataFrame, optional
        List of orbit traces
    neighbors_overlap: float
        Minimum overlap percentage to consider two tiles are adjacent
        (default value is 5 which corresponds to 4-connected neighbors)
    land_overlap: float
        Minimum land overlap percentage between two tiles
        (default value is 2)
    orbir_overlap: float
        Minimum overlap percentage between a tile and an orbit trace
        (default value is 25)

    Returns
    -------
    adjs: List[str] or None
        List of adjacent tile IDs
    """
    grid = tiles.set_index(by)
    # Adjacent tiles must have the same epsg code
    epsg = grid.loc[tile_id].epsg
    subgrid = grid[grid["epsg"] == epsg].drop(index=tile_id).reset_index()

    # Get polygon from tile
    poly = grid.loc[tile_id].geometry

    # Transform polygon from tile to a GeoDataFrame
    roi = gpd.GeoDataFrame(data={"roi": [1], "geometry": [poly]}, crs="EPSG:4326")

    # Find tiles in the same UTM zone that overlap the tile polygon
    tiles = gpd.GeoDataFrame(
        gpd.overlay(subgrid[["id", "geometry"]], roi, how="intersection")
        .merge(subgrid[["id", "geometry"]], how="inner", on="id", suffixes=("_roi", ""))
        .rename(columns={"geometry_roi": "overlap_geometry"})
    )
    if len(tiles) == 0:
        return None
    logger.debug("UTM ", tiles.id.values)

    # Compute overlap and keep only 4 connected neighbors
    tiles["overlap_percentage"] = tiles.apply(
        lambda tile: 100 * tile.overlap_geometry.area / tile.geometry.area, axis=1
    )
    tiles = tiles[tiles["overlap_percentage"] > neighbors_overlap]
    if len(tiles) == 0:
        return None
    logger.debug("4-conn ", tiles.id.values)

    # Take the orbit number into account
    if orbit_id is not None and orbit is not None:
        # Keep tiles whose area is covered by more than 25% of the satellite swath
        orbit_tiles = gpd.GeoDataFrame(
            gpd.overlay(
                tiles[["id", "geometry"]],
                orbit.loc[orbit.orbit_id == orbit_id][["geometry"]],
                how="intersection",
            )
            .merge(
                tiles[["id", "geometry"]], how="inner", on="id", suffixes=("_roi", "")
            )
            .rename(columns={"geometry_roi": "overlap_geometry"})
        )
        if len(orbit_tiles) == 0:
            return None
        orbit_tiles["overlap_percentage"] = orbit_tiles.apply(
            lambda tile: 100 * tile.overlap_geometry.area / tile.geometry.area, axis=1
        )
        orbit_tiles = orbit_tiles[orbit_tiles["overlap_percentage"] > orbit_overlap]
        if len(orbit_tiles) == 0:
            return None
        tiles = tiles[tiles.id.isin(orbit_tiles.id)]
        logger.debug("orbit ", tiles.id.values)

    # Remove tile with overlap in water
    tiles = tiles.reset_index(drop=True)
    overlap = gpd.GeoDataFrame(
        data={"id": tiles.id, "geometry": tiles.overlap_geometry}, crs="EPSG:4326"
    )
    tiles = gpd.GeoDataFrame(
        gpd.overlay(overlap, land[["geometry"]], how="intersection")
        .merge(overlap, how="inner", on="id", suffixes=("_overlap", ""))
        .rename(columns={"geometry_overlap": "overlap_geometry"})
    )
    tiles = groupby_multipolygon(tiles)
    if len(tiles) == 0:
        return None
    tiles["land_percentage"] = tiles.apply(
        lambda tile: 100 * tile.overlap_geometry.area / tile.geometry.area, axis=1
    )
    tiles = tiles[tiles["land_percentage"] > land_overlap]
    if len(tiles) == 0:
        return None
    logger.debug("water ", tiles.id.values)

    # List of land polygon id on the overlap
    poly_ids = gpd.GeoDataFrame(
        gpd.overlay(land[["id", "geometry"]], overlap[["geometry"]], how="intersection")
    ).id.unique()
    poly_overlap = gpd.overlay(land[land["id"].isin(poly_ids)], roi, how="intersection")
    poly_overlap = poly_overlap.to_crs(epsg)
    poly_overlap["area"] = poly_overlap.area
    poly_overlap = poly_overlap.sort_values(by="area", ascending=False)
    poly = land[land["id"] == poly_overlap.iloc[0].id]
    # Keep tiles that intersect the largest polygon
    tiles = groupby_multipolygon(
        gpd.GeoDataFrame(
            gpd.overlay(
                tiles[["id", "geometry"]], poly[["geometry"]], how="intersection"
            )
            .merge(subgrid, how="inner", on="id", suffixes=("_roi", ""))
            .rename(columns={"geometry_roi": "overlap_geometry"})
        )
    )

    return sorted(tiles.id.values)


def generate_adjacents(
    tiles: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    orbit_id: int | None = None,
    orbit: gpd.GeoDataFrame | None = None,
    neighbors_overlap: float = 5.0,
    land_overlap: float = 2.0,
    orbit_overlap: float = 25.0,
) -> pd.DataFrame:
    """
    Generate list of adjacent tiles for a list of tiles

    Parameters
    ----------
    tiles: GeoDataFrame
        List of tiles
    land: GeoDataFrame
        List of land polygons
    orbit_id: int, optional
        Relative orbit number to take into account
    orbit: GeoDataFrame, optional
        List of orbit traces
    neighbors_overlap: float
        Minimum overlap percentage to consider two tiles are adjacent
        (default value is 5 which corresponds to 4-connected neighbors)
    land_overlap: float
        Minimum land overlap percentage between two tiles
        (default value is 2)
    orbir_overlap: float
        Minimum overlap percentage between a tile and an orbit trace
        (default value is 25)

    Returns
    -------
    adjs: DataFrame
    """
    assert "id" in tiles.columns
    assert "epsg" in tiles.columns
    assert "geometry" in tiles.columns
    # Adjacent tiles
    adjs = tiles[["id"]].copy()

    def func(tile_id: str) -> str:
        res = get_adjacent_tiles(tile_id, tiles[["id", "epsg", "geometry"]], land)
        if res is None:
            return ""
        else:
            return ",".join(res)

    adjs["adjs"] = adjs.apply(lambda x: func(x.id), axis=1).astype("str")
    adjs = adjs.set_index("id")
    adjs = check_adjacents(adjs)
    return adjs


def check_adjacents(adjacents: pd.DataFrame, by="adjs") -> pd.DataFrame:
    """
    Check adjacent tiles for each tile and remove adjacent tile
    that has not also the tile among its adjacent tiles

    Parameters
    ----------
    adjacents: GeoDataFrame
        Index column must contains tile ID
        In the column containing adjacent tiles,
        the list of adjacent tiles is given in string form,
        with tile names separated by commas.
    by: str
        Name of the column containing the adjacent tiles list

    Returns
    -------
    adjacents_checked: GeoDataFrame
    """
    pairs: List[List[str]] = []
    for _, row in adjacents.iterrows():
        t1 = row.name
        if row[by] != "":
            for t2 in row[by].split(","):
                pairs.append([t1, t2, "".join(sorted([t1, t2]))])  # type: ignore
    check = pd.DataFrame(data=pairs, columns=["t1", "t2", "pair"])
    check = check.drop_duplicates("pair", keep=False)
    adjacents_checked = adjacents.copy()
    for _, row in check.iterrows():
        adjs = adjacents_checked.loc[row.t1, by].split(",")
        adjs.remove(row.t2)
        adjacents_checked.loc[row.t1, by] = ",".join(adjs)
    return adjacents_checked


def initialize_regroup(
    tiles: pd.DataFrame, land: gpd.GeoDataFrame | None = None, by: str = "id"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Initialize two dataframes :
        - One contains the list of tiles with its ID,
        its group and the possible next group
        - One contains the list of group with the percentage
        of valid pixels, the number of valid pixels and the valid zones

    Parameters
    ----------
    tiles: DataFrame
        List of tiles to process
    land: Optinal(GeoDataFrame)
        Land polygons
    by: str, default="id"
        Name of the column containing the tile ID

    Return
    ------
    tile_df: DataFrame
    group_df: DataFrame
    """
    tile_df = pd.DataFrame(columns=["group", "next"], index=tiles[by])
    tile_df["group"] = tile_df.index
    tile_df["next"] = tile_df.index
    group_df = tile_df[["group"]].reset_index(drop=True)
    group_df[["pvalid", "nbvalid", "zones"]] = group_df.apply(
        lambda x: define_valid_zones(
            x.group.split(","),
            land,
            slope_threshold=30,
            range_threshold=600,
            simplify=True,
            poly_simplify=600,
        ),
        axis=1,
        result_type="expand",
    )
    group_df = group_df.set_index("group")
    return tile_df, group_df


def _get_adjacents(
    tile_ids: List[str], adjacents: pd.DataFrame, by: str = "adjs"
) -> Optional[List[str]]:
    """
    Get adjacent tile list of a list of tile IDs

    Parameters
    ----------
    tile_ids: List[str]
        List of tile IDs
    adjacents: DataFrame
        DataFrame containing adjacent tiles
    by: str
        Name of column containg adjacent tiles

    Return
    ------
    adj_list: List[str]
    """
    adj_list: List[str] = []
    for tile in tile_ids:
        adj_list += str(adjacents.loc[tile, by]).split(",")
    if adj_list == [""]:
        return None
    return list(set(adj_list))


def _join_group(
    tiles: pd.DataFrame, adjs: pd.DataFrame, groups: pd.DataFrame, threshold: int
) -> None:
    """
    Join tile to a group if the number of valid pixel is below a threshold

    Parameter
    ---------
    tiles: DataFrame
        List of tiles with its ID, its group and the possible next group
    adjs: DataFrame
        List of tiles with its adjacent tiles
    groups: DataFrame
        List of groups with the percentage of valid pixels, the number of valid pixels and the valid zones
    threshold: int
        Threshold on minimum number of valid pixels
    """
    for _, row in groups.iterrows():
        if row.nbvalid < threshold:
            # List of adjacent tiles
            adj_list = _get_adjacents(str(row.name).split(","), adjs)
            if adj_list is None:
                continue
            # List of groups that contain the adjacent tiles
            group_list = list(
                np.unique(tiles[tiles.index.isin(adj_list)].group.to_numpy())
            )
            if len(group_list) == 0:
                continue
            # Select the best group among candidates.
            # The best group is the group which has the most valid pixels
            best_candidate = groups.loc[groups.index.isin(group_list)][
                "nbvalid"
            ].idxmax()
            # The constraint is that a tile can only belong to one group.
            # To avoid a tile ending up in two groups, the next column is used to update associations between groups.
            best_grp = str(tiles.loc[tiles.group == best_candidate, "next"].values[0])
            next_grp = str(tiles.loc[tiles.group == row.name, "next"].values[0])
            # Merge the actual group with the best group
            new_group = sorted(
                list(
                    set(
                        best_grp.split(",")
                        + str(row.name).split(",")
                        + next_grp.split(",")
                    )
                )
            )
            # print(row.name, new_group)
            # Update
            tiles.loc[tiles["group"] == best_candidate, "next"] = ",".join(new_group)
            tiles.loc[tiles["group"] == row.name, "next"] = ",".join(new_group)
            tiles.loc[tiles["next"] == best_grp, "next"] = ",".join(new_group)
            tiles.loc[tiles["next"] == next_grp, "next"] = ",".join(new_group)


def regroup(
    tiles: pd.DataFrame,
    adjs: pd.DataFrame,
    groups: pd.DataFrame,
    threshold: int,
    land: gpd.GeoDataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Regroup tiles

    Parameter
    ---------
    tiles: DataFrame
        List of tiles with its ID, its group and the possible next group
    adjs: DataFrame
        List of tiles with its adjacent tiles
    groups: DataFrame
        List of groups with the percentage of valid pixels, the number of valid pixels and the valid zones
    threshold: int
        Threshold on minimum number of valid pixels
    land: Optinal(GeoDataFrame)
        Land polygons

    Return
    ------
    tile_df: DataFrame
    group_df: DataFrame
    """
    tile_df = tiles.copy()
    group_df = groups.copy()
    # Join
    _join_group(tile_df, adjs, group_df, threshold=threshold)
    # Update the group
    tile_df["group"] = tile_df["next"]
    # Drop duplicates
    group_df = tile_df[["group"]].reset_index(drop=True).drop_duplicates("group")
    # Compute valid pixels with new groups
    group_df[["pvalid", "nbvalid", "zones"]] = group_df.apply(
        lambda x: define_valid_zones(
            x.group.split(","),
            land,
            slope_threshold=30,
            range_threshold=600,
            simplify=True,
            poly_simplify=600,
        ),
        axis=1,
        result_type="expand",
    )
    group_df = group_df.set_index("group")
    return tile_df, group_df


def write_regroup(
    df: pd.DataFrame,
    filename: str = "groups.shp",
    columns: List[str] = ["group", "nbvalid"],
) -> None:
    """
    Write groups of tiles in a shapefile

    Parameter
    ---------
    df: DataFrame
        List of groups
    filename: str
        Path to the shapefile
    columns: List[str]
        List of columns to write
    """

    def get_polygon_from_tiles(tile_ids: List[str]) -> None:
        mgrs_grid = gpd.read_file(
            "/vsizip/"
            + os.path.join(
                os.path.dirname(os.path.abspath(mgrs.__file__)),
                "data/sentinel2/mgrs_tiles.gpkg.zip",
                "mgrs_tiles.gpkg",
            )
        )
        return unary_union(mgrs_grid[mgrs_grid.Name.isin(tile_ids)].geometry.values)

    geometry: pd.Series = df.apply(
        lambda x: get_polygon_from_tiles(str(x.group).split(",")), axis=1
    )
    gdf = gpd.GeoDataFrame(df[columns], crs="EPSG:4326", geometry=geometry)
    gdf.to_file(filename, driver="ESRI Shapefile")
