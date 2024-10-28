#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from click.testing import CliRunner
from evaspa import cli


def test_command_evaspa():
    runner = CliRunner()
    result = runner.invoke(cli.evaspa, ["-i Input", "-o Output", "-c Conf"])
    assert result.exit_code == 0


def test_command_evaspa_tiling():
    runner = CliRunner()
    result = runner.invoke(cli.evaspa_tiling)
    assert result.exit_code == 0
