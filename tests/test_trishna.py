# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


from evaspa import trishna


def test_get_trishna_tiles() -> None:
    """
    Test get_trishna_tiles method
    """
    tiles = trishna.get_trishna_tiles()
    assert len(tiles) == 24509
    assert tiles.crs.to_string() == "EPSG:4326"


def test_get_land_mask() -> None:
    """
    Test get_land_mask() method
    """
    land = trishna.get_land_mask()
    assert len(land) == 32830
    assert land.crs.to_string() == "EPSG:4326"


def test_get_trishna_orbits() -> None:
    """
    Test get_trishna_orbits() method
    """
    orbits = trishna.get_trishna_orbits()
    assert len(orbits) == 115
    assert orbits.crs.to_string() == "EPSG:4326"
