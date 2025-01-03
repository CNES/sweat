# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import time

from evaspa.debugging import (
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


def test_register() -> None:
    """Test register functions for debugging"""
    res = get_registered_functions()
    assert "foo1" in sorted(res.keys())
    assert "foo2" in sorted(res.keys())


def test_configure() -> None:
    """Test configure debugging"""
    func = foo1
    assert not func.profile
    assert not func.verbose
    configure_debugging(profile=True, verbose=True, path="out")
    assert func.profile
    assert func.verbose
    assert func.out == "out"


def test_debugging() -> None:
    func = foo
    assert func.profile
    assert func.verbose
