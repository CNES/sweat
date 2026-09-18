# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing various functions for notebooks
"""

import os


def locate_test_data() -> str:
    """
    Locate test data directory
    """
    current = os.getcwd()
    if os.path.isdir(os.path.join(current, "tests")):
        return os.path.join(current, "tests", "data")
    parent = os.path.dirname(current)
    if os.path.isdir(os.path.join(parent, "tests")):
        return os.path.join(parent, "tests", "data")
    parent = os.path.dirname(parent)
    if os.path.isdir(os.path.join(parent, "tests")):
        return os.path.join(parent, "tests", "data")
    msg = "Test data directory not found"
    raise FileNotFoundError(msg)
