#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os

from click.testing import CliRunner
from evaspa import cli


def test_command_evaspa(tmp_path):
    """
    Test evaspa CLI
    """
    output_dir = tmp_path / "tmp_out"
    output_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(cli.evaspa, [os.path.join("tests", "data", "input.json")])
    assert result.exit_code == 0


def test_command_evaspa_tiling(tmp_path):
    """
    Test evaspa-tiling CLI
    """
    output_dir = tmp_path / "tmp_out"
    output_dir.mkdir()
    filename = output_dir / "group.shp"
    runner = CliRunner()
    result = runner.invoke(
        cli.evaspa_tiling,
        args=f'--roi {os.path.join("tests", "data", "corsica.gpkg")} '
        f"--output {filename}",
    )
    assert result.exit_code == 0
