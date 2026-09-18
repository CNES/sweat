# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025 CESBIO / Centre National d'Etudes Spatiales


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
        "e_interception_expected",
        "e_soil_expected",
        "t_expected",
        "converged_expected",
        "stressed_expected",
    ),
    [
        pytest.param(
            34.35,
            31.575,
            20.51,
            52.0,
            0.54,
            1.61,
            529.72,
            -89.6,
            38440,
            0.01,
            15,
            241.79,
            0.49,
            14.88,
            9.02,
            217.88,
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
    e_interception_expected,
    e_soil_expected,
    t_expected,
    converged_expected,
    stressed_expected,
) -> None:
    """
    Test function for STIC model calculation function for a single pixel
    """
    le, _, ef, e_interception, e_soil, t, _, _, _, _, _, converged, stressed = (
        run_stic_model_pixel(
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
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
    np.testing.assert_almost_equal(
        e_interception, e_interception_expected, decimal=2
    )
    np.testing.assert_almost_equal(e_soil, e_soil_expected, decimal=2)
    np.testing.assert_almost_equal(t, t_expected, decimal=2)
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
    e_interception_expected,
    e_soil_expected,
    t_expected,
    converged_expected,
    stressed_expected,
) -> None:
    """
    Test function for STIC model calculation function (raster mode)
    """
    le, ef, e_interception, e_soil, t, cv, st = run_stic_model(
        lst, ta, td, rh, fc, lai, rn, ln, local_time, valid, threshold, nb_steps
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
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
        "data",
        "threshold",
        "nb_steps",
        "le_expected",
        "ef_expected",
    ),
    [
        pytest.param(
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
            241.79,
            0.49,
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
    assert res.shape == (1, 13)
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
            241.776,
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
    assert res.shape == (1, 25)
    np.testing.assert_almost_equal(res[0, 0], le_expected, decimal=2)
