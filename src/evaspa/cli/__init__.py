#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging

import click

from evaspa.__about__ import __version__
from evaspa.api import run_evaspa, generate_tiles, regroup_tiles
from evaspa.logging import LoggerManager
from evaspa.tiling import write_regroup

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
@click.version_option(version=__version__, prog_name="evaspa-tiling")
def evaspa_tiling(debug, roi, orbit, land_percentage, orbit_percentage, threshold):
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
    write_regroup(group.reset_index())
    logger.info("Writing results: OK")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--debug/--no-debug", default=False, help="Debug mode")
@click.option("-i", "--input", required=True, type=str, help="Input file path")
@click.option("-o", "--output", required=True, type=str, help="Output directory path")
@click.option("-c", "--conf", required=False, type=str, help="Configuration file path")
@click.version_option(version=__version__, prog_name="evaspa")
def evaspa(debug, input, output, conf):
    logger.info("Run evaspa...")
    run_evaspa(input, output, conf)
