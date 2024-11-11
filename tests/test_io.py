#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
import pytest

import evaspa.io as io


def get_test_data_dir() -> str:
    """
    Get directory path for test data
    """
    if os.environ["EVASPA_TEST_DATA_PATH"] is None:
        raise Exception("Variable EVASPA_TEST_DATA_PATH must be set")
    return os.environ["EVASPA_TEST_DATA_PATH"]


@pytest.mark.requires_test_data
def test_read_data_from_file() -> None:
    """
    Test read data from a file
    """
    input_path = os.path.join(get_test_data_dir(), "Landsat_20230327_28PCA.tif")
    xarr = io.read_data_from_file(input_path)
    assert xarr.sizes["x"] == 1832
    assert xarr.sizes["y"] == 1832
    assert set([i for i in xarr.data_vars]) == set(
        [
            "qa",
            "ndvi",
            "rld",
            "emis",
            "red",
            "blue",
            "cloud",
            "lai",
            "albedo",
            "green",
            "lst",
            "water",
            "rsd",
            "nir",
        ]
    )


@pytest.mark.requires_test_data
def test_read_data() -> None:
    """
    Test read data
    """
    input_path = os.path.join(get_test_data_dir(), "Landsat_20230327_28PCA")
    xarr = io.read_data(input_path)
    assert xarr.sizes["x"] == 1832
    assert xarr.sizes["y"] == 1832
    assert set([i for i in xarr.data_vars]) == set(
        [
            "qa",
            "ndvi",
            "rld",
            "emis",
            "red",
            "blue",
            "cloud",
            "lai",
            "albedo",
            "green",
            "lst",
            "water",
            "rsd",
            "nir",
        ]
    )
