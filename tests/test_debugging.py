# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import time

import pytest

from sweat.debugging import (
    configure_debugging,
    debugging,
    get_registered_functions,
    register_debugging,
)


@debugging(profile=True, verbose=True)
def foo():
    time.sleep(1)


@register_debugging
def foo1():
    time.sleep(1)


@register_debugging
def foo2():
    time.sleep(1)


@pytest.mark.unit
def test_register() -> None:
    """Test register functions for debugging"""
    res = get_registered_functions()
    assert "tests.test_debugging.foo1" in sorted(res.keys())
    assert "tests.test_debugging.foo2" in sorted(res.keys())


@pytest.mark.unit
def test_configure() -> None:
    """Test configure debugging"""
    func = foo1
    assert not func.profile
    assert not func.verbose
    configure_debugging(profile=True, verbose=True, path="out")
    assert func.profile
    assert func.verbose
    assert func.out == "out"


@pytest.mark.unit
def test_debugging() -> None:
    func = foo
    assert func.profile
    assert func.verbose
