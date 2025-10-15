# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import json
import os

import pytest
from click.testing import CliRunner

from sweat import cli


@pytest.mark.end_to_end
@pytest.mark.parametrize(
    "filename",
    ["evaspa_input.json", "evaspa_input_default.json"],
)
def test_command_evaspa(filename, tmp_path):
    """
    Test evaspa CLI
    """
    output_dir = tmp_path / "tmp_out"
    output_dir.mkdir()
    input_file = os.path.join("tests", "data", filename)
    tmp_input_file = output_dir / "input.json"
    with open(input_file) as fi:
        input_data = json.load(fi)
        input_data["output"]["path"] = str(output_dir)
        with open(tmp_input_file, mode="w") as fo:
            json.dump(input_data, fo)
    runner = CliRunner()
    result = runner.invoke(cli.evaspa, [str(tmp_input_file)])
    assert result.exit_code == 0


@pytest.mark.end_to_end
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
        args=f"--roi {os.path.join('tests', 'data', 'corsica.gpkg')} "
        f"--output {filename}",
    )
    assert result.exit_code == 0


@pytest.mark.end_to_end
@pytest.mark.parametrize(
    "filename",
    ["stic_input.json", "stic_input_default.json"],
)
def test_command_stic(filename, tmp_path):
    """
    Test evaspa CLI
    """
    output_dir = tmp_path / "tmp_out"
    output_dir.mkdir()
    input_file = os.path.join("tests", "data", filename)
    tmp_input_file = output_dir / "input.json"
    with open(input_file) as fi:
        input_data = json.load(fi)
        input_data["output"]["path"] = str(output_dir)
        with open(tmp_input_file, mode="w") as fo:
            json.dump(input_data, fo)
    runner = CliRunner()
    result = runner.invoke(cli.stic, [str(tmp_input_file)])
    assert result.exit_code == 0


@pytest.mark.end_to_end
@pytest.mark.parametrize(
    "filename",
    ["timeseries_input.json", "timeseries_input_default.json"],
)
def test_command_timeseries(filename, tmp_path):
    """
    Test evaspa CLI
    """
    output_dir = tmp_path / "tmp_out"
    output_dir.mkdir()
    input_file = os.path.join("tests", "data", filename)
    tmp_input_file = output_dir / "input.json"
    with open(input_file) as fi:
        input_data = json.load(fi)
        input_data["output"]["path"] = str(output_dir)
        with open(tmp_input_file, mode="w") as fo:
            json.dump(input_data, fo)
    runner = CliRunner()
    result = runner.invoke(cli.timeseries, [str(tmp_input_file)])
    assert result.exit_code == 0
