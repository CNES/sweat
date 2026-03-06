# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
from pathlib import Path

import pytest

import sweat.evaspa.config as cfg
from sweat.__about__ import __version__
from sweat.evaspa import merging


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"ef": {"models": "default_evaspa"}},
        {"ef": {"check": {"threshold": 10}, "models": "default_evaspa"}},
        {
            "filtering": {"cloud": {"op": "!=", "value": 1}},
            "ef": {"models": "default_evaspa"},
        },
        {"ef": {"models": "default_evaspa"}, "seb": {"use_topo": True}},
        {
            "filtering": {"water": {"op": "!=", "value": 1}},
            "ef": {"models": "default_evaspa"},
            "daily": {"use_topo": True, "method": "toa"},
        },
        {
            "filtering": {
                "cloud": {"op": "==", "value": 0},  # keep clear pixels
                "water": {"op": "==", "value": 0},  # keep non water pixels
                "qa": {"op": "==", "value": 1},  # keep pixels with qa = 1
                "ndvi": {
                    "and": [{"op": ">=", "value": 0}, {"op": "<=", "value": 1}]
                },  # Keep NDVI between [0,1]
                "fcover": {
                    "and": [{"op": ">=", "value": 0}, {"op": "<=", "value": 1}]
                },  # Keep Fcover between [0,1]
                "albedo": {
                    "and": [{"op": ">=", "value": 0}, {"op": "<=", "value": 1}]
                },  # Keep Fcover between [0,1]
                "height": {
                    "and": [
                        {"op": ">=", "value": 0},
                        {"op": "<=", "value": 300},
                    ]
                },  # Keep height between [0,300]
            },
            "ef": {"models": "default_evaspa"},
            "seb": {"use_topo": True},
            "daily": {"use_topo": True, "method": "toa"},
        },
        {
            "filtering": {"cloud": {"op": "==", "value": 0}},
            "ef": {
                "check": {"threshold": 0.02},
                "models": "default_evaspa",
                "options": {
                    "filtering": {
                        "albedo": {
                            "and": [
                                {"op": ">=", "value": 0.1},
                                {"op": "<=", "value": 0.3},
                            ]
                        }
                    },
                    "selection": True,
                    "merging": {"merging_method": "mean"},
                },
            },
            "seb": {"use_topo": True},
            "daily": {"use_topo": True, "method": "toa"},
        },
    ],
)
def test_paramsconfig(config) -> None:
    """
    Test EVASPA ParamsConfig
    """
    assert cfg.EVASPAParamsConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "merging_expected", "uncertainty_expected"),
    [
        pytest.param(
            {
                "ef": {
                    "models": "default_evaspa",
                    "options": {
                        "merging": {
                            "merging_method": "median",
                            "uncertainty_method": "interquartile",
                        }
                    },
                }
            },
            merging.MergingMethod.MEDIAN,
            merging.UncertaintyMethod.INTERQUARTILE,
        ),
        pytest.param(
            {
                "ef": {"models": "default_evaspa"},
                "seb": {
                    "merging": {
                        "merging_method": "mean",
                        "uncertainty_method": "std",
                    }
                },
            },
            merging.MergingMethod.MEDIAN,
            merging.UncertaintyMethod.INTERQUARTILE,
        ),
        pytest.param(
            {
                "ef": {
                    "models": "default_evaspa",
                    "options": {
                        "merging": {
                            "merging_method": "mean",
                            "uncertainty_method": "interquartile",
                        }
                    },
                },
                "seb": {
                    "merging": {
                        "merging_method": "median",
                        "uncertainty_method": "nmad",
                    }
                },
            },
            merging.MergingMethod.MEAN,
            merging.UncertaintyMethod.INTERQUARTILE,
        ),
    ],
)
def test_paramsconfig_check_merging(
    config, merging_expected, uncertainty_expected, caplog
) -> None:
    """
    Test check merging for EVASPA ParamsConfig
    """
    caplog.set_level(logging.WARNING)
    res = cfg.EVASPAParamsConfig.model_validate(config)
    assert "Merging methods differ, use the one from EF step" in caplog.text
    assert "Uncertainty methods differ, use the one from EF step" in caplog.text
    assert res.ef.options.merging.merging_method.value == merging_expected.value
    assert res.ef.options.merging.uncertainty_method
    assert (
        res.ef.options.merging.uncertainty_method.value
        == uncertainty_expected.value
    )
    assert res.seb.merging.merging_method.value == merging_expected.value
    assert res.seb.merging.uncertainty_method
    assert (
        res.seb.merging.uncertainty_method.value == uncertainty_expected.value
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "version",
    [None, "0"],
)
def test_version(version, tmp_path) -> None:
    """
    Test version in InputFile
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {"path": "tests/data/modis_test.tif"},
        "output": {"path": str(d)},
        "params": {
            "ef": {
                "check": {"threshold": 10},
                "models": "default_evaspa",
            }
        },
    }
    if version is not None:
        config["version"] = version
    entry = cfg.EVASPAInputFile.model_validate(config)
    assert entry.version == __version__


@pytest.mark.unit
def test_check() -> None:
    """
    Test check method
    """
    config = {
        "input": {
            "path": "tests/data/modis_test.tif",
            "date": "2018-05-16T10:00:00Z",
        },
        "output": {"path": "out"},
        "params": {
            "filtering": {
                "cloud": {"op": "==", "value": 0},
                "water": {"op": "!=", "value": 1},
                "ndvi": {
                    "and": [{"op": ">=", "value": 0}, {"op": "<=", "value": 1}],
                },
                "qa": {
                    "or": [{"op": "==", "value": 2}, {"op": "==", "value": 10}],
                },
            },
            "ef": {"models": "default_evaspa"},
            "seb": {"use_topo": True},
            "daily": {"use_topo": True, "method": "toa"},
        },
    }
    ref_config = {
        "input": {
            "path": "tests/data/modis_test.tif",
            "date": "2018-05-16T10:00:00Z",
        },
        "output": {"path": "out"},
        "params": {
            "filtering": {
                "cloud": {"op": "==", "value": 0},
                "water": {"op": "!=", "value": 1},
                "ndvi": {
                    "and": [{"op": ">=", "value": 0}, {"op": "<=", "value": 1}],
                    "or": None,
                },
                "qa": {
                    "or": [{"op": "==", "value": 2}, {"op": "==", "value": 10}],
                    "and": None,
                },
            },
            "ef": {
                "models": [
                    {
                        "name": "EF1",
                        "dry_edge": {
                            "type": "LinearEdge",
                            "config": {
                                "interval_type": "density",
                                "interval_nb": 20,
                                "percentile": 2,
                                "selection": "median",
                            },
                        },
                        "wet_edge": {
                            "type": "LinearEdge",
                            "config": {
                                "interval_type": "density",
                                "interval_nb": 20,
                                "percentile": 2,
                                "selection": "median",
                            },
                        },
                        "var": "albedo",
                    },
                    {
                        "name": "EF8",
                        "dry_edge": {
                            "type": "LinearEdge",
                            "config": {
                                "interval_type": "density",
                                "interval_nb": 20,
                                "percentile": 5,
                                "selection": "median",
                            },
                        },
                        "wet_edge": {"type": "FlatEdge", "config": {}},
                        "var": "albedo",
                    },
                ],
                "options": {
                    "filtering": {},
                    "selection": False,
                    "merging": {
                        "merging_method": "median",
                        "uncertainty_method": "interquartile",
                    },
                },
                "check": {"threshold": 0.02},
            },
            "seb": {
                "use_topo": True,
                "models": ["kustas"],
                "merging": {
                    "merging_method": "median",
                    "uncertainty_method": "interquartile",
                },
            },
            "daily": {"use_topo": True, "method": "toa"},
        },
        "debug": {"profile": False, "verbose": False, "path": "out/debug"},
    }
    checked_config = cfg.check_config_evaspa(config)
    assert checked_config["input"] == ref_config["input"]
    assert checked_config["output"] == ref_config["output"]
    assert checked_config["params"] == ref_config["params"]
    assert checked_config["debug"] == ref_config["debug"]


@pytest.mark.unit
def test_check_debug() -> None:
    """
    Test check method
    """
    config = {
        "input": {"path": "tests/data/modis_test.tif", "date": None},
        "output": {"path": "out"},
        "params": {
            "ef": {
                "check": {"threshold": 10},
                "models": "default_evaspa",
                "options": {"merging": {"merging_method": "mean"}},
            },
        },
        "debug": {"profile": True, "verbose": True},
    }
    ref_debug = {"profile": True, "verbose": True, "path": "out/debug"}
    checked_config = cfg.check_config_evaspa(config)
    assert checked_config["debug"] == ref_debug
