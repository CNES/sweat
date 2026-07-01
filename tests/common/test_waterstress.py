# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError
from pyproj import CRS

from sweat.common import waterstress as ws


def setup_et_data(
    variables: list[str] | None = None,
    size: int = 125,
    seed: int = 0,
    valid: float = 0.9,
) -> xr.Dataset:
    """
    Generate data for tests
    """
    np.random.seed(seed)
    lon = np.linspace(start=1.0, stop=2.0, num=size, dtype=np.float32)
    lat = np.linspace(start=43.0, stop=44.0, num=size, dtype=np.float32)
    ef_arr = np.random.uniform(low=0, high=1, size=(size, size))
    valid_arr = np.random.choice(
        [0, 1], size=(size, size), p=[1 - valid, valid]
    )
    ef_arr = np.where(valid_arr == 1, ef_arr, np.nan)
    data_vars = {
        "ef": (["lat", "lon"], ef_arr),
        "valid": (["lat", "lon"], valid_arr),
    }
    data = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
        },
        attrs={"description": "Test data", "crs": CRS(4326)},
    )
    if variables is None:
        return data
    to_remove = list(set(data_vars) - set(variables))
    return data.drop_vars(to_remove)


@pytest.mark.unit
def test_compute_waterstress_from_ef() -> None:
    """
    Test function for computing water stress from
    evaporative fraction
    """
    ef = np.random.rand(2, 2)
    water_stress = ws.compute_waterstress_from_ef(ef)
    ref = 1 - ef
    np.testing.assert_allclose(water_stress, ref)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"method": ws.DEFAULT_METHOD},
        {"method": "ef"},
    ],
)
def test_waterstressconfig(config) -> None:
    """
    Test WaterStressConfig
    """
    assert ws.WaterStressConfig.model_validate(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"foo": "foo"},
        {"method": "foo"},
    ],
)
def test_waterstressconfig_with_exc(config) -> None:
    """
    Test WaterStressConfig with exception
    """
    with pytest.raises(ValidationError):
        ws.WaterStressConfig.model_validate(config)


@pytest.mark.functional
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"method": ws.DEFAULT_METHOD},
        {"method": "ef"},
    ],
)
def test_run(config):
    """
    Test run method
    """
    et_data = setup_et_data()
    ws_xr = ws.run(et_data, **config)
    assert len(ws_xr.data_vars) > 0
    assert all(ws_xr[var].count() > 0 for var in ws_xr.data_vars)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("et_data", "config", "exception", "message"),
    [
        (
            setup_et_data(),
            {"method": "foo"},
            ValueError,
            "Unknown method for water stress: foo",
        ),
        (
            setup_et_data(variables=["valid"]),
            {"method": "ef"},
            ValueError,
            "Data missing for EF method",
        ),
    ],
)
def test_run_with_exc(et_data, config, exception, message):
    """
    Test run method with exception
    """
    with pytest.raises(exception, match=message):
        ws.run(et_data, **config)
