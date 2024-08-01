#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

import scipy

if __name__ == "__main__":
    from .api import compute_le

    input = os.path.join(os.environ["DATA_PATH"], "test_evaspa/2018136/input/")
    output = os.getcwd()
    lst = scipy.io.loadmat(os.path.join(input, "LST_2018136_center.mat"))["LST_tmp"]
    lai = scipy.io.loadmat(os.path.join(input, "LAI_2018136_center.mat"))["LAI"]
    albedo = scipy.io.loadmat(os.path.join(input, "Alb_2018136_center.mat"))["Alb"]
    emis = scipy.io.loadmat(os.path.join(input, "Emis__doy2018136_center.mat"))["Emis"]
    ndvi = scipy.io.loadmat(os.path.join(input, "NDVI_2018136_center.mat"))["NDVI"]
    atm_flux = scipy.io.loadmat(os.path.join(input, "Ra_2018136_center.mat"))["Ra"]
    sun_flux = scipy.io.loadmat(os.path.join(input, "Rg_2018136_center.mat"))["Rg"]
    compute_le(lst, emis, albedo, lai, ndvi, sun_flux, atm_flux)
