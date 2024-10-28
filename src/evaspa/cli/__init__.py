#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import click

from evaspa.__about__ import __version__


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--debug/--no-debug", default=False, help="Debug mode")
@click.option("--roi", required=False, type=str, help="ROI file path")
@click.version_option(version=__version__, prog_name="evaspa-tiling")
def evaspa_tiling(debug, roi):
    click.echo("Run evaspa-tiling")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--debug/--no-debug", default=False, help="Debug mode")
@click.option("-i", "--input", required=True, type=str, help="Input file path")
@click.option("-o", "--output", required=True, type=str, help="Output directory path")
@click.option("-c", "--conf", required=False, type=str, help="Configuration file path")
@click.version_option(version=__version__, prog_name="evaspa")
def evaspa(debug, input, output, conf):
    click.echo("Run evaspa")
