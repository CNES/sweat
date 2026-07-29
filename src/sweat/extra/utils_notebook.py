# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing various functions for notebooks
"""

import os


def locate_test_data() -> str:
    """
    Locate test data directory
    """
    current = os.getcwd()
    parent = os.path.dirname(current)
    if os.path.isdir(os.path.join(parent, "tests")):
        return os.path.join(parent, "tests", "data")
    parent = os.path.dirname(parent)
    if os.path.isdir(os.path.join(parent, "tests")):
        return os.path.join(parent, "tests", "data")
    msg = "Test data directory not found"
    raise FileNotFoundError(msg)
