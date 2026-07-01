# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales


import datetime as dt

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from pyproj import CRS

from sweat.common import flux


def setup_data(
    nb: int = 1,
    size: int = 125,
    seed: int = 0,
    valid: float = 0.9,
    nan: bool = False,
) -> xr.Dataset:
    """
    Generate data for tests
    """
    assert nb > 0
    np.random.seed(seed)
    lon = np.linspace(start=1.0, stop=2.0, num=size, dtype=np.float32)
    lat = np.linspace(start=43.0, stop=44.0, num=size, dtype=np.float32)
    albedo_arr = np.random.uniform(low=0, high=1.0, size=(size, size))
    lst_arr = np.random.uniform(low=290, high=310, size=(size, size))
    emis_arr = np.random.uniform(low=0, high=1.0, size=(size, size))
    ndvi_arr = np.random.uniform(low=-1.0, high=1.0, size=(size, size))
    lai_arr = np.random.uniform(low=0, high=10.0, size=(size, size))
    fcover_arr = np.random.uniform(low=0, high=1.0, size=(size, size))
    valid_arr = np.random.choice(
        [0, 1], size=(size, size), p=[1 - valid, valid]
    )
    if nan:
        lst_arr = np.where(valid_arr == 1, lst_arr, np.nan)
        emis_arr = np.where(valid_arr == 1, emis_arr, np.nan)
        lai_arr = np.where(valid_arr == 1, lai_arr, np.nan)
        ndvi_arr = np.where(valid_arr == 1, ndvi_arr, np.nan)
        fcover_arr = np.where(valid_arr == 1, fcover_arr, np.nan)
        albedo_arr = np.where(valid_arr == 1, albedo_arr, np.nan)
    rsd_arr = []
    rld_arr = []
    for i in np.arange(nb):
        rsd_tmp = (400 + i) * np.ones((size, size))
        rld_tmp = (200 + i) * np.ones((size, size))
        if nan:
            rsd_tmp = np.where(valid_arr == 1, rsd_tmp, np.nan)
            rld_tmp = np.where(valid_arr == 1, rld_tmp, np.nan)
        rsd_arr.append(rsd_tmp)
        rld_arr.append(rld_tmp)
    data_vars = {
        "albedo": (["lat", "lon"], albedo_arr),
        "lst": (["lat", "lon"], lst_arr),
        "emis": (["lat", "lon"], emis_arr),
        "ndvi": (["lat", "lon"], ndvi_arr),
        "lai": (["lat", "lon"], lai_arr),
        "fcover": (["lat", "lon"], fcover_arr),
        "valid": (["lat", "lon"], valid_arr),
    }
    data_vars["rsd"] = (["lat", "lon"], rsd_arr[0])
    data_vars["rld"] = (["lat", "lon"], rld_arr[0])
    for i in np.arange(1, nb):
        data_vars[f"rsd_{i}"] = (["lat", "lon"], rsd_arr[i])
        data_vars[f"rld_{i}"] = (["lat", "lon"], rld_arr[i])
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
        },
        attrs={"description": "Test data", "crs": CRS(4326)},
    )


def setup_datasets(
    lst: npt.NDArray,
    albedo: npt.NDArray,
    emis: npt.NDArray,
    rsd: npt.NDArray,
    fdiff: npt.NDArray,
    rld: npt.NDArray,
    height: npt.NDArray,
    aspect: npt.NDArray,
    slope: npt.NDArray,
    rn: npt.NDArray,
    ln: npt.NDArray,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Create a dataset
    """
    lon = np.linspace(start=1.0, stop=2.0, num=rsd.shape[0], dtype=np.float32)
    lat = np.linspace(start=43.0, stop=44.0, num=rsd.shape[1], dtype=np.float32)

    return (
        xr.Dataset(
            data_vars={
                "lst": (["lat", "lon"], lst),
                "albedo": (["lat", "lon"], albedo),
                "emis": (["lat", "lon"], emis),
                "rsd": (["lat", "lon"], rsd),
                "fdiff": (["lat", "lon"], fdiff),
                "rld": (["lat", "lon"], rld),
                "height": (["lat", "lon"], height),
                "apsect": (["lat", "lon"], aspect),
                "slope": (["lat", "lon"], slope),
            },
            coords={
                "lon": ("lon", lon),
                "lat": ("lat", lat),
            },
            attrs={"description": "Test data"},
        ),
        xr.Dataset(
            data_vars={
                "rn": (["lat", "lon"], rn),
            },
            coords={
                "lon": ("lon", lon),
                "lat": ("lat", lat),
            },
            attrs={"description": "Test data"},
        ),
        xr.Dataset(
            data_vars={
                "ln": (["lat", "lon"], ln),
            },
            coords={
                "lon": ("lon", lon),
                "lat": ("lat", lat),
            },
            attrs={"description": "Test data"},
        ),
    )


def setup_array(data: npt.NDArray) -> xr.DataArray:
    """
    Create dataarray
    """
    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={
            "x": (["x"], np.linspace(-1.0, 1.0, num=data.shape[0])),
            "y": (["y"], np.linspace(42.0, 44.0, num=data.shape[1])),
        },
    )


@pytest.mark.unit
def test_stefan_boltzmann_constant() -> None:
    """
    Stefan Boltzmann constant
    """
    np.testing.assert_approx_equal(flux.CST_SB, 5.670374419e-8, significant=7)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lst", "emis", "albedo", "rsd", "rld", "rn_expected", "ln_expected"),
    [
        pytest.param(0.0, 0.0, 0.0, 100.0, 0.0, 100.0, 0.0),
        pytest.param(
            300.0,
            1.0,
            0.0,
            0.0,
            0.0,
            -flux.CST_SB * 300.0**4,
            -flux.CST_SB * 300.0**4,
        ),
        pytest.param(0.0, 1.0, 1.0, 0.0, 50.0, 50.0, 50.0),
        pytest.param(0.0, 1.0, 1.0, 100.0, 50.0, 50.0, 50.0),
        pytest.param(0.0, 1.0, 0.5, 100.0, 50.0, 100.0, 50.0),
    ],
)
def test_compute_rn(
    lst, emis, albedo, rsd, rld, rn_expected, ln_expected
) -> None:
    """
    Test compute net radiation Rn
    """
    # Case 1: Black body
    rsd_arr = rsd * np.ones((2, 2))
    rld_arr = rld * np.ones((2, 2))
    albedo_arr = albedo * np.ones((2, 2))
    emis_arr = emis * np.ones((2, 2))
    lst_arr = lst * np.ones((2, 2))
    rn, ln = flux.compute_rn(lst_arr, emis_arr, albedo_arr, rsd_arr, rld_arr)
    rn_ref = rn_expected * np.ones((2, 2))
    ln_ref = ln_expected * np.ones((2, 2))
    np.testing.assert_allclose(rn, rn_ref)
    np.testing.assert_allclose(ln, ln_ref)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rsd", "sza", "saa", "slope", "aspect", "expected"),
    [
        pytest.param(1000, 20, 0, 30, 0, 1048.01),
        pytest.param(1000, 20, 0, 0, 0, 1000),
    ],
)
def test_correct_direct_radiation(rsd, sza, saa, slope, aspect, expected):
    """
    Test function for correcting direct shortwave radiation
    """
    res = flux.correct_direct_radiation(rsd, sza, saa, slope, aspect)
    np.testing.assert_approx_equal(res, expected, significant=2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rsd", "slope", "aspect", "fdiff", "sza", "saa", "expected"),
    [
        pytest.param(
            setup_array(1000 * np.ones((2, 2))),
            setup_array(30 * np.ones((2, 2))),
            setup_array(80 * np.ones((2, 2))),
            None,
            None,
            None,
            setup_array(
                np.array(
                    [
                        [1082, 1066],
                        [1082, 1066],
                    ]
                )
            ),
        ),
        pytest.param(
            setup_array(1000 * np.ones((2, 2))),
            setup_array(30 * np.ones((2, 2))),
            setup_array(80 * np.ones((2, 2))),
            setup_array(np.zeros((2, 2))),
            None,
            None,
            setup_array(
                np.array(
                    [
                        [1082, 1066],
                        [1082, 1066],
                    ]
                )
            ),
        ),
        pytest.param(
            setup_array(1000 * np.ones((2, 2))),
            setup_array(30 * np.ones((2, 2))),
            setup_array(80 * np.ones((2, 2))),
            setup_array(np.ones((2, 2))),
            None,
            None,
            setup_array(
                np.array(
                    [
                        [1000, 1000],
                        [1000, 1000],
                    ]
                )
            ),
        ),
        pytest.param(
            setup_array(1000 * np.ones((2, 2))),
            setup_array(30 * np.ones((2, 2))),
            setup_array(80 * np.ones((2, 2))),
            setup_array(np.zeros((2, 2))),
            setup_array(30 * np.ones((2, 2))),
            setup_array(80 * np.ones((2, 2))),
            setup_array(
                np.array(
                    [
                        [1154, 1154],
                        [1154, 1154],
                    ]
                )
            ),
        ),
    ],
)
def test_correct_shortwave_radiation(
    rsd, slope, aspect, fdiff, sza, saa, expected
):
    """
    Test function for correcting direct shortwave radiation
    """
    date = dt.datetime(2025, 6, 10, 10, 0, 0, tzinfo=dt.UTC)
    res = flux.correct_shortwave_radiation(
        rsd=rsd,
        slope=slope,
        aspect=aspect,
        fdiff=fdiff,
        date=date,
        sza=sza,
        saa=saa,
        crs=CRS(4326),
    )
    np.testing.assert_allclose(res, expected, atol=5.0, rtol=0.1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rsd", "slope", "aspect", "fdiff", "sza", "saa", "msg"),
    [
        pytest.param(
            setup_array(1000 * np.ones((2, 2))),
            setup_array(30 * np.ones((2, 2))),
            setup_array(80 * np.ones((2, 2))),
            None,
            setup_array(30 * np.ones((2, 2))),
            setup_array(80 * np.ones((2, 2))),
            "No diffuse fraction data",
        ),
        pytest.param(
            setup_array(1000 * np.ones((2, 2))),
            setup_array(30 * np.ones((2, 2))),
            setup_array(80 * np.ones((2, 2))),
            setup_array(np.zeros((2, 2))),
            None,
            None,
            "Sun angle not present, theoretical calculation done",
        ),
    ],
)
def test_correct_shortwave_radiation_with_warnings(
    rsd, slope, aspect, fdiff, sza, saa, msg, caplog
):
    """
    Test function for correcting direct shortwave radiation
    with warnings
    """
    caplog.clear()
    date = dt.datetime(2025, 6, 10, 10, 0, 0, tzinfo=dt.UTC)
    flux.correct_shortwave_radiation(
        rsd=rsd,
        slope=slope,
        aspect=aspect,
        fdiff=fdiff,
        date=date,
        sza=sza,
        saa=saa,
        crs=CRS(4326),
    )
    assert msg in caplog.text


@pytest.mark.functional
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param({"nb": 1}, 1),
        pytest.param({"nb": 3}, 3),
    ],
)
def test_get_radiation_variables(params, expected) -> None:
    """
    Test the functions to get radiation variables
    """
    data = setup_data(**params)
    rsd_data, rld_data = flux.get_radiation_variables(data)
    rsd_data_expected = [f"rsd_{i}" for i in np.arange(1, params["nb"])]
    rsd_data_expected.append("rsd")
    rld_data_expected = [f"rld_{i}" for i in np.arange(1, params["nb"])]
    rld_data_expected.append("rld")
    assert len(rsd_data) == expected
    assert len(rld_data) == expected
    assert sorted(rsd_data) == sorted(rsd_data_expected)
    assert sorted(rld_data) == sorted(rld_data_expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "lst",
        "albedo",
        "emis",
        "rsd",
        "fdiff",
        "rld",
        "height",
        "aspect",
        "slope",
        "use_topo",
        "rn_expected",
        "ln_expected",
    ),
    [
        pytest.param(
            np.full((2, 2), 0.0),
            np.full((2, 2), 1.0),
            np.full((2, 2), 1.0),
            np.full((2, 2), 100.0),
            np.full((2, 2), 0.0),
            np.full((2, 2), 50.0),
            np.full((2, 2), 0.0),
            np.full((2, 2), 0.0),
            np.full((2, 2), 0.0),
            False,
            np.full((2, 2), 50.0),
            np.full((2, 2), 50.0),
        ),
        pytest.param(
            np.full((2, 2), 0.0),
            np.full((2, 2), 0.5),
            np.full((2, 2), 1.0),
            np.full((2, 2), 100.0),
            np.full((2, 2), 0.0),
            np.full((2, 2), 50.0),
            np.full((2, 2), 0.0),
            np.full((2, 2), 0.0),
            np.full((2, 2), 0.0),
            False,
            np.full((2, 2), 100.0),
            np.full((2, 2), 50.0),
        ),
    ],
)
def test_create_net_radiation(
    lst,
    albedo,
    emis,
    rsd,
    fdiff,
    rld,
    height,
    aspect,
    slope,
    use_topo,
    rn_expected,
    ln_expected,
) -> None:
    """
    Test create net radiation dataset
    """
    data, rn_ref, ln_ref = setup_datasets(
        lst,
        albedo,
        emis,
        rsd,
        fdiff,
        rld,
        height,
        aspect,
        slope,
        rn_expected,
        ln_expected,
    )
    rn, ln = flux.create_net_radiation(data, use_topo)
    xr.testing.assert_allclose(rn, rn_ref)
    xr.testing.assert_allclose(ln, ln_ref)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param("rsd", r"No RSD data available"),
        pytest.param(
            "rld", r"No RLD data \(rld\) associated to RSD data \(rsd\)"
        ),
    ],
)
def test_create_net_radiation_exc(params, expected) -> None:
    """
    Test create net radiation dataset (with exception)
    """
    data = setup_data()
    data = data.drop_vars(params)
    with pytest.raises(ValueError, match=expected):
        flux.create_net_radiation(data)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param({"nb": 1}, 1),
        pytest.param({"nb": 3}, 3),
    ],
)
def test_create_net_radiation_multiple(params, expected) -> None:
    """
    Test create net radiation dataset
    with multiple input radiation data
    """
    data = setup_data(**params)
    rn, ln = flux.create_net_radiation(data)
    assert len(rn.data_vars) == expected
    assert len(ln.data_vars) == expected


@pytest.mark.unit
def test_compute_et_from_le():
    """
    Test method for computing ET from LE
    """
    le = 100 * np.ones((2, 2))
    ref = np.ones((2, 2)) * 100 / flux.LATENT_HEAT_VAPORIZATION
    et = flux.compute_et_from_le(le)
    np.testing.assert_allclose(et, ref)
