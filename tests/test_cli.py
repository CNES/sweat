#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
import json

from click.testing import CliRunner
from evaspa import cli


def test_command_evaspa(tmp_path):
    """
    Test evaspa CLI
    """
    output_dir = tmp_path / "tmp_out"
    output_dir.mkdir()
    input_file = os.path.join("tests", "data", "input.json")
    tmp_input_file = output_dir / "input.json"
    with open(input_file) as fi:
        input_data = json.load(fi)
        input_data["output"]["path"] = str(output_dir)
        with open(tmp_input_file, mode="w") as fo:
            json.dump(input_data, fo)
    runner = CliRunner()
    result = runner.invoke(cli.evaspa, [str(tmp_input_file)])
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
