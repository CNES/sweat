# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing register decorators for STIC models
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ModelSpec:
    func: Callable
    inputs: dict[str, str]


MODEL_REGISTRY: dict[str, ModelSpec] = {}
MODEL_PIXEL_REGISTRY: dict[str, ModelSpec] = {}


def register_model(version: str, inputs: dict[str, str]):
    """
    Decorator to register function to MODEL_REGISTRY
    """

    def decorator(func):
        MODEL_REGISTRY[version] = ModelSpec(func=func, inputs=inputs)
        return func

    return decorator


def register_model_pixel(version: str, inputs: dict[str, str]):
    """
    Decorator to register function to MODEL_PIXEL_REGISTRY
    """

    def decorator(func):
        MODEL_PIXEL_REGISTRY[version] = ModelSpec(func=func, inputs=inputs)
        return func

    return decorator
