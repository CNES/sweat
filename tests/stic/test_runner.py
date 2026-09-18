# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sweat.common.types import ETVar
from sweat.stic.runner import (
    run_batch_init_model,
    run_batch_model,
    run_model,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "ts",
        "ta",
        "td",
        "rh",
        "fc",
        "lai",
        "rn",
        "ln",
        "local_time",
        "valid",
        "threshold",
        "nb_steps",
        "version",
        "le_expected",
        "ef_expected",
        "e_interception_expected",
        "e_soil_expected",
        "t_expected",
        "converged_expected",
        "stressed_expected",
    ),
    [
        pytest.param(
            np.full((2, 2), 34.35, dtype=np.float32),
            np.full((2, 2), 31.575, dtype=np.float32),
            np.full((2, 2), 20.51, dtype=np.float32),
            np.full((2, 2), 52.0, dtype=np.float32),
            np.full((2, 2), 0.54, dtype=np.float32),
            np.full((2, 2), 1.61, dtype=np.float32),
            np.full((2, 2), 529.72, dtype=np.float32),
            np.full((2, 2), -89.6, dtype=np.float32),
            np.full((2, 2), 38440, dtype=np.float32),
            np.ones((2, 2), dtype=np.int64),
            0.01,
            15,
            "1.3",
            np.full((2, 2), 241.79, dtype=np.float32),
            np.full((2, 2), 0.49, dtype=np.float32),
            np.full((2, 2), 14.88, dtype=np.float32),
            np.full((2, 2), 9.02, dtype=np.float32),
            np.full((2, 2), 217.88, dtype=np.float32),
            np.full((2, 2), True, dtype=bool),
            np.full((2, 2), False, dtype=bool),
        ),
    ],
)
def test_run_stic_model(
    ts,
    ta,
    td,
    rh,
    fc,
    lai,
    rn,
    ln,
    local_time,
    valid,
    threshold,
    nb_steps,
    version,
    le_expected,
    ef_expected,
    e_interception_expected,
    e_soil_expected,
    t_expected,
    converged_expected,
    stressed_expected,
) -> None:
    """
    Test runner for STIC model
    """
    data = xr.Dataset(
        data_vars={
            ETVar.LST.value: (["y", "x"], ts),
            ETVar.TEMPERATURE.value: (["y", "x"], ta),
            ETVar.DEWPOINT_TEMPERATURE.value: (["y", "x"], td),
            ETVar.RH.value: (["y", "x"], rh),
            ETVar.FCOVER.value: (["y", "x"], fc),
            ETVar.LAI.value: (["y", "x"], lai),
            ETVar.NET_RADIATION.value: (["y", "x"], rn),
            ETVar.LONGWAVE_NET_RADIATION.value: (["y", "x"], ln),
            ETVar.LOCAL_TIME.value: (["y", "x"], local_time),
        },
        coords={
            "y": ("y", np.array([0, 1])),
            "x": ("x", np.array([0, 1])),
        },
        attrs={"description": "Test data"},
    )
    le, ef, e_interception, e_soil, t, cv, st = run_model(
        data=data,
        valid=valid,
        threshold=threshold,
        nb_steps=nb_steps,
        version=version,
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
    np.testing.assert_almost_equal(
        e_interception, e_interception_expected, decimal=2
    )
    np.testing.assert_almost_equal(e_soil, e_soil_expected, decimal=2)
    np.testing.assert_almost_equal(t, t_expected, decimal=2)
    np.testing.assert_equal(cv, converged_expected)
    np.testing.assert_equal(st, stressed_expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "variables",
        "data",
        "threshold",
        "nb_steps",
        "version",
        "le_expected",
        "ef_expected",
    ),
    [
        pytest.param(
            [
                "ts",
                "ta",
                "td",
                "rh",
                "fc",
                "lai",
                "rn",
                "ln",
                "local_time",
            ],
            np.array(
                [
                    [
                        34.35,
                        31.575,
                        20.51,
                        52.0,
                        0.54,
                        1.61,
                        529.72,
                        -89.6,
                        38440,
                    ]
                ],
                dtype=np.float32,
            ),
            0.01,
            15,
            "1.3",
            241.79,
            0.49,
        ),
    ],
)
def test_run_batch_model(
    variables,
    data,
    threshold,
    nb_steps,
    version,
    le_expected,
    ef_expected,
) -> None:
    """
    Test runner for STIC model (batch mode)
    """
    df = pd.DataFrame(data, columns=variables)
    res = run_batch_model(
        data=df,
        threshold=threshold,
        nb_steps=nb_steps,
        debug=False,
        version=version,
        mapping=True,
    )
    assert res.shape == (1, 13)
    np.testing.assert_almost_equal(res[0, 0], le_expected, decimal=2)
    np.testing.assert_almost_equal(res[0, 2], ef_expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "variables",
        "data",
        "version",
        "le_expected",
    ),
    [
        pytest.param(
            [
                "ts",
                "ta",
                "td",
                "rh",
                "fc",
                "lai",
                "rn",
                "ln",
                "local_time",
            ],
            np.array(
                [
                    [
                        34.35,
                        31.575,
                        20.51,
                        52.0,
                        0.54,
                        1.61,
                        529.72,
                        -89.6,
                        38440,
                    ]
                ],
                dtype=np.float32,
            ),
            "1.3",
            241.776,
        ),
    ],
)
def test_run_batch_init_model(
    variables,
    data,
    version,
    le_expected,
) -> None:
    """
    Test runner for STIC model (init only)
    """
    df = pd.DataFrame(data, columns=variables)
    res = run_batch_init_model(
        data=df,
        version=version,
        debug=False,
        mapping=True,
    )
    assert res.shape == (1, 25)
    np.testing.assert_almost_equal(res[0, 0], le_expected, decimal=2)
