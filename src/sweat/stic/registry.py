# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing register decorators for STIC models
"""

import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from sweat.stic import models


@dataclass
class ModelSpec:
    raster_func: Callable
    batch_func: Callable
    init_func: Callable
    inputs: dict[str, str]
    required_inputs: list[str]


MODEL_REGISTRY = {}
DEFAULT_VERSION = "none"

for module in pkgutil.iter_modules(models.__path__):
    if module.ispkg:
        imported_module = import_module(
            f"sweat.stic.models.{module.name}.model"
        )
        if not callable(imported_module.run_stic_model):
            msg = f"no function run_stic_model in {module.name}.model"
            raise ValueError(msg)
        if not callable(imported_module.run_batch_stic_model):
            msg = f"no function run_batch_stic_model in {module.name}.model"
            raise ValueError(msg)
        for v in ["VERSION", "VARIABLES_MAPPING"]:
            if not hasattr(imported_module, "VERSION"):
                msg = f"no {v} in {module.name}.model"
                raise ValueError(msg)

        version = imported_module.VERSION
        mapping = imported_module.VARIABLES_MAPPING
        required = imported_module.REQUIRED_INPUTS
        batch_func = imported_module.run_batch_stic_model
        raster_func = imported_module.run_stic_model
        init_func = imported_module.run_batch_init_stic_model
        MODEL_REGISTRY[version] = ModelSpec(
            batch_func=batch_func,
            raster_func=raster_func,
            init_func=init_func,
            inputs=mapping,
            required_inputs=required,
        )

        if getattr(imported_module, "IS_DEFAULT", False):
            DEFAULT_VERSION = version


if DEFAULT_VERSION == "none":
    msg = "No default version defined"
    raise ValueError(msg)
