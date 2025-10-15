# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales

import datetime as dt
import warnings

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from evaspa.common.constant import ETVar
from evaspa.stic import main


def setup_data(
    lst: tuple[float, float],
    albedo: tuple[float, float],
    temperature: tuple[float, float],
    dewpoint_temperature: tuple[float, float],
    rh: tuple[float, float],
    fcover: tuple[float, float],
    lai: tuple[float, float],
    emissivity: tuple[float, float],
    rn: tuple[float, float],
    ln: tuple[float, float],
    local_time: tuple[float, float],
    valid: tuple[float, float],
    size: int = 4,
    seed: int = 0,
) -> xr.Dataset:
    """
    Generate data for tests
    """
    np.random.seed(seed)
    x = np.linspace(start=1.0, stop=2.0, num=size, dtype=np.float32)
    y = np.linspace(start=43.0, stop=44.0, num=size, dtype=np.float32)
    valid_arr = np.random.choice(
        [0, 1], size=(size, size), p=[valid[0], valid[1]]
    )
    lst_arr = np.random.uniform(low=lst[0], high=lst[1], size=(size, size))
    albedo_arr = np.random.uniform(
        low=albedo[0], high=albedo[1], size=(size, size)
    )
    temperature_arr = np.random.uniform(
        low=temperature[0], high=temperature[1], size=(size, size)
    )
    dewpoint_temperature_arr = np.random.uniform(
        low=dewpoint_temperature[0],
        high=dewpoint_temperature[1],
        size=(size, size),
    )
    rh_arr = np.random.uniform(low=rh[0], high=rh[1], size=(size, size))
    fcover_arr = np.random.uniform(
        low=fcover[0], high=fcover[1], size=(size, size)
    )
    lai_arr = np.random.uniform(low=lai[0], high=lai[1], size=(size, size))
    emissivity_arr = np.random.uniform(
        low=emissivity[0], high=emissivity[1], size=(size, size)
    )
    rn_arr = np.random.uniform(low=rn[0], high=rn[1], size=(size, size))
    ln_arr = np.random.uniform(low=ln[0], high=ln[1], size=(size, size))
    localtime_arr = np.random.uniform(
        low=local_time[0], high=local_time[1], size=(size, size)
    )
    data_vars = {
        ETVar.LST.value: (["y", "x"], lst_arr),
        ETVar.ALBEDO.value: (["y", "x"], albedo_arr),
        ETVar.TEMPERATURE.value: (["y", "x"], temperature_arr),
        ETVar.DEWPOINT_TEMPERATURE.value: (
            ["y", "x"],
            dewpoint_temperature_arr,
        ),
        ETVar.RH.value: (["y", "x"], rh_arr),
        ETVar.FCOVER.value: (["y", "x"], fcover_arr),
        ETVar.LAI.value: (["y", "x"], lai_arr),
        ETVar.EMISSIVITY.value: (["y", "x"], emissivity_arr),
        ETVar.NET_RADIATION.value: (["y", "x"], rn_arr),
        ETVar.LONGWAVE_NET_RADIATION.value: (["y", "x"], ln_arr),
        ETVar.LOCAL_TIME.value: (["y", "x"], localtime_arr),
        "valid": (["y", "x"], valid_arr),
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "y": ("y", y),
            "x": ("x", x),
        },
        attrs={
            "description": "Test data",
            "date": dt.datetime(2025, 7, 24, 12, 30, tzinfo=dt.timezone.utc),
            "crs": 4326,
        },
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {},
            {
                "use_topo": False,
                "selected_radiation": None,
            },
        ),
        pytest.param(
            {
                "use_topo": True,
                "selected_radiation": "msg",
            },
            {
                "use_topo": True,
                "selected_radiation": "msg",
            },
        ),
        pytest.param(
            {
                "use_topo": True,
            },
            {
                "use_topo": True,
                "selected_radiation": None,
            },
        ),
    ],
)
def test_sticprepareconfig(config, expected) -> None:
    """
    Test STICModelConfig
    """
    res = main.STICPrepareConfig.model_validate(config)
    assert res.model_dump() == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {
            "foo": False,
            "use_topo": True,
        },
        {
            "use_topo": True,
            "selected_radiation": 10,
        },
    ],
)
def test_sticprepareconfig_error(config) -> None:
    """
    Test STICModelConfig
    """
    with pytest.raises(ValidationError):
        main.STICPrepareConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {},
            {
                "threshold": 0.01,
                "nb_steps": 15,
            },
        ),
        pytest.param(
            {
                "threshold": 0.01,
                "nb_steps": 10,
            },
            {
                "threshold": 0.01,
                "nb_steps": 10,
            },
        ),
        pytest.param(
            {
                "threshold": 0.05,
            },
            {
                "threshold": 0.05,
                "nb_steps": 15,
            },
        ),
    ],
)
def test_sticmodelconfig(config, expected) -> None:
    """
    Test STICModelConfig
    """
    res = main.STICModelConfig.model_validate(config)
    assert res.model_dump() == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {
            "foo": False,
            "use_topo": True,
            "threshold": 0.01,
            "nb_steps": 15,
        },
    ],
)
def test_sticmodelconfig_error(config) -> None:
    """
    Test STICModelConfig
    """
    with pytest.raises(ValidationError):
        main.STICModelConfig.model_validate(config)


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
            20000,
            0.01,
            15,
            210.980,
            0.6925,
            True,
        ),
    ],
)
def test_run_stic_model_pixel(
    ts,
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
) -> None:
    """
    Test function for STIC model calulation function for a single pixel
    """
    le, ef, converged = main.run_stic_model_pixel(
        ts, ta, td, rh, fc, lai, rn, ln, local_time, threshold, nb_steps
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
    np.testing.assert_equal(converged, converged_expected)


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
            np.array([[20000, 20000], [20000, 20000]], dtype=np.float32),
            np.array([[1, 1], [1, 1]], dtype=np.int64),
            0.01,
            15,
            np.array([[210.980, 210.980], [210.980, 210.980]]),
            np.array([[0.6925, 0.6925], [0.6925, 0.6925]]),
            np.array([[True, True], [True, True]]),
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
    le_expected,
    ef_expected,
    converged_expected,
) -> None:
    """
    Test function for STIC model calulation funtion
    """
    le, ef, cv = main.run_stic_model(
        ts, ta, td, rh, fc, lai, rn, ln, local_time, valid, threshold, nb_steps
    )
    np.testing.assert_almost_equal(le, le_expected, decimal=2)
    np.testing.assert_almost_equal(ef, ef_expected, decimal=2)
    np.testing.assert_equal(cv, converged_expected)


@pytest.mark.functional
def test_prepare():
    """
    Test function for STIC prepare function
    """
    data = xr.Dataset(
        data_vars={
            "lst": (["y", "x"], np.array([[293, np.nan], [293, 293]])),
            "albedo": (["y", "x"], np.array([[0.5, 0.5], [np.nan, 0.5]])),
            "ta": (["y", "x"], np.full((2, 2), 290)),
            "tdp": (["y", "x"], np.full((2, 2), 285)),
            "fcover": (["y", "x"], np.full((2, 2), 0.6)),
            "lai": (["y", "x"], np.full((2, 2), 2)),
            "emis": (["y", "x"], np.full((2, 2), 0.8)),
            "rsd": (["y", "x"], np.full((2, 2), 400)),
            "rld": (["y", "x"], np.full((2, 2), 200)),
        },
        coords={
            "y": ("y", np.array([40.0, 40.1])),
            "x": ("x", np.array([0, 0.1])),
        },
        attrs={
            "description": "Test data",
            "date": dt.datetime.now(tz=dt.timezone.utc),
            "crs": 4326,
        },
    )
    res = main.prepare(data)
    assert res


@pytest.mark.unit
def test_prepare_exc():
    """
    Test function for STIC prepare function (with exception)
    """
    data = xr.Dataset(
        data_vars={
            "albedo": (["y", "x"], np.array([[0.5, 0.5], [np.nan, 0.5]])),
            "ta": (["y", "x"], np.full((2, 2), 290)),
            "tdp": (["y", "x"], np.full((2, 2), 285)),
            "fcover": (["y", "x"], np.full((2, 2), 0.6)),
            "lai": (["y", "x"], np.full((2, 2), 2)),
            "emis": (["y", "x"], np.full((2, 2), 0.8)),
        },
        coords={
            "y": ("y", np.array([0, 1])),
            "x": ("x", np.array([0, 1])),
        },
        attrs={"description": "Test data"},
    )
    with pytest.raises(
        KeyError, match="Variable lst is missing in the dataset"
    ):
        main.prepare(data)


@pytest.mark.functional
def test_run():
    """
    Test function for STIC run function
    """
    warnings.filterwarnings("error")
    data = setup_data(
        lst=(20, 30),
        albedo=(0.2, 0.4),
        temperature=(15, 25),
        dewpoint_temperature=(12, 19),
        rh=(0.4, 0.6),
        fcover=(0.4, 0.7),
        lai=(1, 4),
        emissivity=(0.4, 0.7),
        rn=(300, 300),
        ln=(50, 50),
        local_time=(30000, 40000),
        valid=(0, 1),
    )

    res = main.run(
        data,
        threshold=0.01,
        nb_steps=15,
    )
    assert res
