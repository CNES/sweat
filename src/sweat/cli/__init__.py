# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
from pathlib import Path

import click

from sweat.__about__ import __version__
from sweat.api import (
    generate_tiles,
    read_input_data,
    read_ts_input_data,
    regroup_tiles,
    run_evaspa,
    run_stic,
    run_timeseries,
    run_window_time_series,
)
from sweat.common.config import read_config, write_config
from sweat.common.io import write_dataset
from sweat.evaspa.config import EVASPAInputFile
from sweat.evaspa.tiling import write_regroup
from sweat.logging import LoggerManager
from sweat.stic.config import STICInputFile
from sweat.timeseries.config import TimeSeriesInputFile
from sweat.timeseries.io_handler import write_timeseries

logger = LoggerManager.get_logger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Debug logging mode",
)
@click.argument(
    "input_file", type=click.Path(exists=True, file_okay=True, readable=True)
)
@click.version_option(version=__version__, prog_name="evaspa")
def evaspa(verbose, input_file):
    # Configure logging
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    LoggerManager.set_level(log_level)
    # Set configuration
    msg = f"Configuration file: {input_file}"
    logger.debug(msg)
    dict_config = read_config(input_file)
    # Verify config and manage default parameters
    config = EVASPAInputFile.model_validate(dict_config)
    output_dir = Path(config.output.path)
    # Read input data
    logger.debug("Read input data...")
    input_data = read_input_data(config.input)
    logger.info("Read input data: OK")
    # Run
    logger.debug("Run evaspa...")
    res = run_evaspa(input_data, config.params, config.debug)
    logger.info("Run evaspa: OK")
    # Save configuration
    logger.debug("Write results...")
    write_config(config.model_dump(), output_dir, fmt="json")
    logger.info("Write configuration: OK")
    # Write results
    if res is not None:
        # Write instantaneous results
        filename = "evaspa_inst.tif"
        write_dataset(
            res[0], filename=filename, directory=output_dir, separate=True
        )
        filename = "evaspa_daily.tif"
        write_dataset(
            res[1], filename=filename, directory=output_dir, separate=True
        )
        logger.info("Writing results: OK")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Debug logging mode",
)
@click.argument(
    "input_file", type=click.Path(exists=True, file_okay=True, readable=True)
)
@click.version_option(version=__version__, prog_name="stic")
def stic(verbose, input_file):
    # Configure logging
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    LoggerManager.set_level(log_level)
    # Set configuration
    msg = f"Configuration file: {input_file}"
    logger.debug(msg)
    dict_config = read_config(input_file)
    # Verify config and manage default parameters
    config = STICInputFile.model_validate(dict_config)
    output_dir = Path(config.output.path)
    # Read input data
    logger.debug("Read input data...")
    input_data = read_input_data(config.input)
    logger.info("Read input data: OK")
    # Run
    logger.debug("Run stic...")
    res = run_stic(input_data, config.params, config.debug)
    logger.info("Run stic: OK")
    # Save configuration
    logger.debug("Write results...")
    write_config(config.model_dump(), output_dir, fmt="json")
    logger.info("Write configuration: OK")
    # Write results
    if res is not None:
        # Write instantaneous results
        filename = "stic_inst.tif"
        write_dataset(
            res[0], filename=filename, directory=output_dir, separate=True
        )
        filename = "stic_daily.tif"
        write_dataset(
            res[1], filename=filename, directory=output_dir, separate=True
        )
        logger.info("Writing results: OK")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Debug logging mode",
)
@click.argument(
    "input_file", type=click.Path(exists=True, file_okay=True, readable=True)
)
@click.version_option(version=__version__, prog_name="timeseries")
def timeseries(verbose, input_file):
    # Configure logging
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    LoggerManager.set_level(log_level)
    # Set configuration
    msg = f"Configuration file: {input_file}"
    logger.debug(msg)
    dict_config = read_config(input_file)
    # Verify config and manage default parameters
    config = TimeSeriesInputFile.model_validate(dict_config)
    output_dir = Path(config.output.path)
    # Read input data
    logger.debug("Read input data...")
    et_ts, radiation_ts, et_sd, dem = read_ts_input_data(config.input)
    logger.info("Read input data: OK")
    # Run
    logger.debug("Run stic...")
    res = run_timeseries(
        et_ts, radiation_ts, et_sd, dem, config.params, config.debug
    )
    logger.info("Run timeseries: OK")
    # Save configuration
    logger.debug("Write results...")
    write_config(config.model_dump(), output_dir, fmt="json")
    logger.info("Write configuration: OK")
    # Write results
    if res is not None:
        write_timeseries(res, root_name="et_timeseries", directory=output_dir)
        logger.info("Writing results: OK")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Debug logging mode",
)
@click.argument("period_start", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.argument("period_end", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.argument(
    "et_single_date_dir",
    type=click.Path(exists=True, dir_okay=True, readable=True),
)
@click.argument(
    "radiation_dir",
    type=click.Path(exists=True, dir_okay=True, readable=True),
)
@click.argument(
    "et_time_series_dir",
    type=click.Path(dir_okay=True, readable=True, exists=False),
)
@click.argument(
    "window",
    type=click.IntRange(1, 30),
)
@click.argument(
    "shift",
    type=click.IntRange(1, 30),
)
@click.option(
    "--json_dir",
    required=False,
    type=click.Path(dir_okay=True, readable=True, exists=False),
    default=None,
    help="Directory to store the json file if json_verbose option selected",
)
@click.option(
    "--json_verbose/--no-json_verbose",
    default=False,
    help="If selected, configuration will be stored as a .json file",
)
@click.version_option(version=__version__, prog_name="timeseries")
def window_timeseries(
    verbose,
    period_start,
    period_end,
    et_single_date_dir,
    radiation_dir,
    et_time_series_dir,
    window,
    shift,
    json_dir,
    json_verbose,
):
    # Configure logging
    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    LoggerManager.set_level(log_level)
    # Check
    if period_start > period_end:
        msg_dates = "End date must be more recent than start date"
        raise click.BadParameter(msg_dates)
    if window < shift:
        msg_window = "Window size can not be inferior to shift."
        raise click.BadParameter(msg_window)
    # Converting dates from datetime to str
    period_start_str = period_start.strftime("%Y-%m-%d")
    period_end_str = period_end.strftime("%Y-%m-%d")
    # Run
    run_window_time_series(
        period_start_str,
        period_end_str,
        et_single_date_dir,
        radiation_dir,
        et_time_series_dir,
        window,
        shift,
        json_dir,
        json_verbose,
    )
