import numpy as np
from models_tools import (
    et_sd_list,
    find_directories,
    rad_list,
    ratio_list,
    var_list,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def simulated_api(
    dir_rad: str,
    dir_sd: str,
    dir_var: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
    k: float,
    freq: int,
    lamb: float,
) -> list[float]:
    """
    Compute the simulated daily acccumulated precipitation
    for a given pixel over a specified period

    Parameters
    ----------
    dir_rad: str
        Directory where the `radiation_YYYYMMDD.tif` files are stored
    dir_sd: str
        Directory where the `et_single_date_YYYYMMDD.tif` are stored
    dir_var: str
        Directory where the `explanatory_YYYYMMDD.tif` are stored
    start_date: dt.datetime
        Start date
    end_date: dt.datetime
        End date
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    k: float
        Scaling parameter
    freq: int
        Acquisition frequency
    lamb: float
        Latent heat of vaporization

    Returns
    -------
    list[float]
        Simulated daily acccumulated precipitation
    """
    ratio = ratio_list(dir_rad, dir_sd, start_date, end_date, x, y)
    tp = var_list(dir_var, "tp", start_date, end_date, x, y)
    api: list[float] = []
    for i in range(len(ratio)):
        if i % freq == 0:
            api_i = lamb * ratio[i] * 0.2
        else:
            api_i = k * api[i - 1] + tp[i]
            api_i = min(api_i, 100)
        api.append(api_i)
    return api


def et_ts_list_ap(
    dir_rad: str,
    dir_sd: str,
    dir_var: str,
    start_date,
    end_date,
    x: int,
    y: int,
    k: float,
    freq: int,
    lamb: float,
    alpha_dry: float,
    alpha_wet: float,
) -> list[float]:
    """
    Compute the simulated daily evapotranspiration
    according to the accumulated precipitaton-based model
    for a given pixel over a specified period

    Parameters
    ----------
    dir_rad: str
        Directory where the `radiation_YYYYMMDD.tif` files are stored
    dir_sd: str
        Directory where the `et_single_date_YYYYMMDD.tif` are stored
    dir_var: str
        Directory where the `explanatory_YYYYMMDD.tif` are stored
    start_date: dt.datetime
        Start date
    end_date: dt.datetime
        End date
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    k: float
        Scaling parameter
    freq: int
        Acquisition frequency
    lamb: float
        Latent heat of vaporization
    alpha_dry: float
        Smoothing parameter used when precipitation occurred the previous day
    alpha_wet: float
        smoothing parameter used when no precipitation occurred the previous day

    Returns
    -------
    list[float]
        Simulated daily evapotranspiration
    """
    tp = var_list(dir_var, "tp", start_date, end_date, x, y)
    api_ts = simulated_api(
        dir_rad, dir_sd, dir_var, start_date, end_date, x, y, k, freq, lamb
    )
    ratio_filt = 0.0
    rad = rad_list(dir_rad, start_date, end_date, x, y)
    et_ts: list[float] = []
    for i in range(len(api_ts)):
        ratio_i = (1 / lamb) * api_ts[i] / 0.2
        # Smoothing ratio values
        if i > 0:
            alpha = alpha_wet if tp[i - 1] > 0 else alpha_dry
            ratio_filt = alpha * ratio_filt + (1 - alpha) * ratio_i
        else:
            ratio_filt = ratio_i
        et_ts_i = rad[i] * ratio_i if i % freq == 0 else rad[i] * ratio_filt
        et_ts.append(et_ts_i)
    return et_ts


def et_ts_list_regr(
    input_dir: str,
    start_date: str,
    end_date: str,
    model_for: Pipeline,
    model_dry_for: LinearRegression,
    model_back: Pipeline,
    model_dry_back: LinearRegression,
    switch_threshold: float,
    freq: int,
    x: int,
    y: int,
) -> list[float]:
    """
    Compute the simulated daily evapotranspiration
    according to the mulitple linear regression-based model
    for a given pixel over a specified period

    Parameters
    ----------
    input_dir: str
        Path to the directory where all the variable products are stored
    start_date: dt.datetime
        Start date
    end_date: dt.datetime
        End date
    model_for: Pipeline
        General polynomial regression model for forward extrapolation
    model_dry_for:
        Dry-period-specific linear regression model for forward extrapolation
    model_for: Pipeline
        General polynomial regression model for backward extrapolation
    model_dry_for:
        Dry-period-specific linear regression model for backward extrapolation
    switch_threshold: float
        Threshold for switching from the general model
        the to dry-period_specific model
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)

    Returns
    -------
    list[float]
        Simulated daily evapotranspiration
    """
    dir_sd, dir_rad, dir_var = find_directories(input_dir)
    rad = rad_list(dir_rad, start_date, end_date, x, y)
    et_sd = et_sd_list(dir_sd, start_date, end_date, x, y)
    sw = var_list(dir_var, "sw", start_date, end_date, x, y)
    et_ts: list[float] = []
    h = (freq - 1) // 2
    for i in range(len(et_sd)):
        if i % freq == 0:
            # Keeping observed values
            et_ts_i = et_sd[i]
            # Backward correction
            if i > 0:
                for j in range(1, h + 1):
                    x_input = np.array(
                        [
                            rad[i - j],
                            rad[i - j + 1],
                            et_sd[i - j + 1],
                            sw[i - j],
                        ]
                    ).reshape(1, -1)
                    model_back = (
                        model_back
                        if sw[i] > switch_threshold
                        else model_dry_back
                    )
                    et_ts[i - j] = model_back.predict(x_input)[0]
        else:
            # Forward extrapolation
            x_input = np.array(
                [rad[i], rad[i - 1], et_sd[i - 1], sw[i]]
            ).reshape(1, -1)
            model_for = model_for if sw[i] > switch_threshold else model_dry_for
            et_ts_i = model_for.predict(x_input)[0]
        et_ts.append(et_ts_i)
    return et_ts
