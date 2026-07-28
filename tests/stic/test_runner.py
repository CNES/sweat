# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales


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
        "converged_expected",
        "stressed_expected",
    ),
    [
        pytest.param(
            np.array([[30.0, 30.0], [30.0, 30.0]], dtype=np.float32),
            np.array([[25, 25], [25, 25]], dtype=np.float32),
            np.array([[19, 19], [19, 19]], dtype=np.float32),
            np.array([[69.36, 69.36], [69.36, 69.36]], dtype=np.float32),
            np.array([[0.86, 0.86], [0.86, 0.86]], dtype=np.float32),
            np.array([[4, 4], [4, 4]], dtype=np.float32),
            np.array([[300, 300], [300, 300]], dtype=np.float32),
            np.array([[100, 100], [100, 100]], dtype=np.float32),
            np.array([[20000, 20000], [20000, 20000]], dtype=np.float32),
            np.array([[1, 1], [1, 1]], dtype=np.int64),
            0.01,
            15,
            "1.3",
            np.array([[210.53, 210.53], [210.53, 210.53]]),
            np.array([[0.6925, 0.6925], [0.6925, 0.6925]]),
            np.array([[True, True], [True, True]]),
            np.array([[False, False], [False, False]]),
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
    le, ef, cv, st = run_model(
        data=data,
        valid=valid,
        threshold=threshold,
        nb_steps=nb_steps,
        version=version,
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
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
                [[30.0, 25, 19, 69.36, 0.86, 4, 300, 100, 20000]],
                dtype=np.float32,
            ),
            0.01,
            15,
            "1.3",
            210.53,
            0.6925,
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
    assert res.shape == (1, 10)
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
                [[30.0, 25, 19, 69.36, 0.86, 4, 300, 100, 20000]],
                dtype=np.float32,
            ),
            "1.3",
            210.1425,
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
    assert res.shape == (1, 21)
    np.testing.assert_almost_equal(res[0, 0], le_expected, decimal=2)
