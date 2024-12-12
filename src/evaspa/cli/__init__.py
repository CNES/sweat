#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import json
import logging
from pathlib import Path

import click

from evaspa.__about__ import __version__
from evaspa.api import run_evaspa, generate_tiles, regroup_tiles
from evaspa.configuration import InputFile
from evaspa.logging import LoggerManager
from evaspa.tiling import write_regroup
from evaspa.ef import write_ef

logger = LoggerManager.get_logger(__name__)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Debug mode",
    show_default=True,
)
@click.option("--roi", required=False, type=str, help="ROI file path")
@click.option("--orbit", required=False, type=int, help="Orbit relative number")
@click.option(
    "--land_percentage",
    required=False,
    type=float,
    default=10,
    show_default=True,
    help="Minimum percentage of land coverage to keep a tile",
)
@click.option(
    "--orbit_percentage",
    required=False,
    type=float,
    default=25,
    show_default=True,
    help="Minimum percentage of orbit coverage to keep a tile",
)
@click.option(
    "--threshold",
    required=False,
    type=int,
    default=600000,
    show_default=True,
    help="Threshold on number of valid pixels",
)
@click.option(
    "--output",
    required=False,
    type=str,
    default="group.shp",
    show_default=True,
    help="Path to the output file",
)
@click.version_option(version=__version__, prog_name="evaspa-tiling")
def evaspa_tiling(
    debug, roi, orbit, land_percentage, orbit_percentage, threshold, output
):
    # Configure logging
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
    LoggerManager.set_level(log_level)
    # Run
    logger.info("Run evaspa-tiling...")
    roi_tiles, adjs = generate_tiles(
        roi=roi,
        orbit_id=orbit,
        land_percentage=land_percentage,
        orbit_percentage=orbit_percentage,
    )
    logger.info("Tile generation: OK")
    group = regroup_tiles(roi_tiles, adjs, threshold=threshold)
    logger.info("Title association: OK")
    # Write results
    write_regroup(group.reset_index(), filename=output)
    logger.info("Writing results: OK")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--debug/--no-debug", default=False, help="Debug mode")
@click.argument(
    "input_file", type=click.Path(exists=True, file_okay=True, readable=True)
)
@click.version_option(version=__version__, prog_name="evaspa")
def evaspa(debug, input_file):
    # Configure logging
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
    LoggerManager.set_level(log_level)
    # Set configuration
    logger.debug(f"Configuration file: {input_file}")
    with open(input_file) as json_file:
        json_data = json.load(json_file)
    # Verify config and manage default parameters
    config = InputFile.model_validate(json_data)
    # Run
    logger.info("Run evaspa...")
    ef = run_evaspa(
        config.input,
        config.params,
    )
    # Write results
    output_dir = Path(config.output.path)
    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=4, default=lambda x: x.value)
    if ef is not None:
        # Write Evaporative farction
        filename = output_dir / "ef.tif"
        write_ef(ef, filename=filename)
    logger.info("Writing results: OK")
