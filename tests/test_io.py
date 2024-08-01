#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

import evaspa.io as io


def test_read_data() -> None:
    """
    Test read data
    """
    input_path = os.path.join(os.path.dirname(__file__), "data", "modis_20180222.tif")
    xarr = io.read_data(input_path)
    assert xarr.sizes["x"] == 145
    assert xarr.sizes["y"] == 145
    assert set([i for i in xarr.data_vars]) == set(
        ["lst", "lai", "albedo", "emis", "ndvi", "rg", "ra"]
    )
