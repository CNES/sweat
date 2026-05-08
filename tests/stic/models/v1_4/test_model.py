# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pytest

from sweat.stic.models.v1_4.model import (
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
        "nir",
        "swir",
        "vari_green",
        "gli",
        "ndvi",
        "gndvi",
        "msavi",
        "emis",
        "local_time",
        "threshold",
        "nb_steps",
        "le_expected",
        "ef_expected",
        "converged_expected",
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
            0.5,
            0.3,
            0.8,
            0.8,
            0.8,
            0.8,
            0.8,
            0.9,
            20000,
            0.01,
            15,
            203.02,
            0.67,
            True,
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
    nir,
    swir,
    vari_green,
    gli,
    ndvi,
    gndvi,
    msavi,
    emis,
    local_time,
    threshold,
    nb_steps,
    le_expected,
    ef_expected,
    converged_expected,
) -> None:
    """
    Test function for STIC model calculation function for a single pixel
    """
    le, _, ef, _, _, _, _, _, converged = run_stic_model_pixel(
        lst,
        ta,
        td,
        rh,
        fc,
        lai,
        rn,
        ln,
        nir,
        swir,
        vari_green,
        gli,
        ndvi,
        gndvi,
        msavi,
        emis,
        local_time,
        threshold,
        nb_steps,
        debug=False,
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
    np.testing.assert_equal(converged, converged_expected)


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
        "nir",
        "swir",
        "vari_green",
        "gli",
        "ndvi",
        "gndvi",
        "msavi",
        "emis",
        "local_time",
        "valid",
        "threshold",
        "nb_steps",
        "le_expected",
        "ef_expected",
        "converged_expected",
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
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
            np.array([[0.3, 0.3], [0.3, 0.3]], dtype=np.float32),
            np.array([[0.8, 0.8], [0.8, 0.8]], dtype=np.float32),
            np.array([[0.8, 0.8], [0.8, 0.8]], dtype=np.float32),
            np.array([[0.8, 0.8], [0.8, 0.8]], dtype=np.float32),
            np.array([[0.8, 0.8], [0.8, 0.8]], dtype=np.float32),
            np.array([[0.8, 0.8], [0.8, 0.8]], dtype=np.float32),
            np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
            np.array([[20000, 20000], [20000, 20000]], dtype=np.float32),
            np.array([[1, 1], [1, 1]], dtype=np.int64),
            0.01,
            15,
            np.array([[203.02, 203.02], [203.02, 203.02]]),
            np.array([[0.67, 0.67], [0.67, 0.67]]),
            np.array([[True, True], [True, True]]),
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
    nir,
    swir,
    vari_green,
    gli,
    ndvi,
    gndvi,
    msavi,
    emis,
    local_time,
    valid,
    threshold,
    nb_steps,
    le_expected,
    ef_expected,
    converged_expected,
) -> None:
    """
    Test function for STIC model calculation function
    """
    le, ef, cv = run_stic_model(
        lst,
        ta,
        td,
        rh,
        fc,
        lai,
        rn,
        ln,
        nir,
        swir,
        vari_green,
        gli,
        ndvi,
        gndvi,
        msavi,
        emis,
        local_time,
        valid,
        threshold,
        nb_steps,
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
    np.testing.assert_equal(cv, converged_expected)


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
                        30.0,
                        25,
                        19,
                        69.36,
                        0.86,
                        4,
                        300,
                        100,
                        0.5,
                        0.3,
                        0.8,
                        0.8,
                        0.8,
                        0.8,
                        0.8,
                        0.9,
                        20000,
                    ]
                ],
                dtype=np.float32,
            ),
            0.01,
            15,
            203.02,
            0.67,
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
    Test function for STIC model calculation function
    """
    res = run_batch_stic_model(data, threshold, nb_steps, debug=False)
    assert res.shape == (1, 8)
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
                        0.5,
                        0.3,
                        0.8,
                        0.8,
                        0.8,
                        0.8,
                        0.8,
                        0.9,
                        20000,
                    ]
                ],
                dtype=np.float32,
            ),
            202.93,
        ),
    ],
)
def test_run_batch_init_model(
    data,
    le_expected,
) -> None:
    """
    Test function for STIC model calculation function
    """
    res = run_batch_init_stic_model(data, debug=False)
    assert res.shape == (1, 20)
    np.testing.assert_almost_equal(res[0, 0], le_expected, decimal=2)
