# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pytest

from sweat.stic.models.v1_3.model import (
    run_batch_init_stic_model,
    run_batch_stic_model,
    run_stic_model,
    run_stic_model_pixel,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "lst",
        "ta",
        "td",
        "rh",
        "fc",
        "lai",
        "rn",
        "ln",
        "local_time",
        "threshold",
        "nb_steps",
        "le_expected",
        "ef_expected",
        "converged_expected",
        "stressed_expected",
    ),
    [
        pytest.param(
            30,
            25,
            19,
            69.36,
            0.86,
            4,
            300,
            100,
            20000,
            0.01,
            15,
            210.528,
            0.6925,
            True,
            False,
        ),
    ],
)
def test_run_stic_model_pixel(
    lst,
    ta,
    td,
    rh,
    fc,
    lai,
    rn,
    ln,
    local_time,
    threshold,
    nb_steps,
    le_expected,
    ef_expected,
    converged_expected,
    stressed_expected,
) -> None:
    """
    Test function for STIC model calculation function for a single pixel
    """
    le, _, ef, _, _, _, _, _, converged, stressed = run_stic_model_pixel(
        lst,
        ta,
        td,
        rh,
        fc,
        lai,
        rn,
        ln,
        local_time,
        threshold,
        nb_steps,
        debug=False,
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
    np.testing.assert_equal(converged, converged_expected)
    np.testing.assert_equal(stressed, stressed_expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "lst",
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
            np.array([[210.528, 210.528], [210.528, 210.528]]),
            np.array([[0.6925, 0.6925], [0.6925, 0.6925]]),
            np.array([[True, True], [True, True]]),
            np.array([[False, False], [False, False]]),
        ),
    ],
)
def test_run_stic_model(
    lst,
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
    le_expected,
    ef_expected,
    converged_expected,
    stressed_expected,
) -> None:
    """
    Test function for STIC model calculation function (raster mode)
    """
    le, ef, cv, st = run_stic_model(
        lst, ta, td, rh, fc, lai, rn, ln, local_time, valid, threshold, nb_steps
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
    np.testing.assert_equal(cv, converged_expected)
    np.testing.assert_equal(st, stressed_expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "data",
        "threshold",
        "nb_steps",
        "le_expected",
        "ef_expected",
    ),
    [
        pytest.param(
            np.array(
                [[30.0, 25, 19, 69.36, 0.86, 4, 300, 100, 20000]],
                dtype=np.float32,
            ),
            0.01,
            15,
            210.528,
            0.6925,
        ),
    ],
)
def test_run_batch_stic_model(
    data,
    threshold,
    nb_steps,
    le_expected,
    ef_expected,
) -> None:
    """
    Test function for STIC model calculation function (batch mode)
    """
    res = run_batch_stic_model(data, threshold, nb_steps, debug=False)
    assert res.shape == (1, 10)
    np.testing.assert_almost_equal(res[0, 0], le_expected, decimal=2)
    np.testing.assert_almost_equal(res[0, 2], ef_expected, decimal=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "data",
        "le_expected",
    ),
    [
        pytest.param(
            np.array(
                [
                    [
                        30.0,
                        25,
                        19,
                        69.36,
                        0.86,
                        4,
                        300,
                        100,
                        20000,
                    ]
                ],
                dtype=np.float32,
            ),
            210.142,
        ),
    ],
)
def test_run_batch_init_stic_model(
    data,
    le_expected,
) -> None:
    """
    Test function for STIC model calculation function (init only)
    """
    res = run_batch_init_stic_model(data, debug=False)
    assert res.shape == (1, 21)
    np.testing.assert_almost_equal(res[0, 0], le_expected, decimal=2)
