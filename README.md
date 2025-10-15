# Evapotranspiration for TRISHNA

This contains contains several algorithms
used in TRISHNA mission to compute evapotranspiration:  
  
* [EVASPA][1]
* [STIC][2] 
* ET time series

## Installation

### Clone the repository

```console
git clone https://src.koda.cnrs.fr/trishna/sweat.git
```

### Install prerequisites

The prerequisites are installed by [pixi](https://pixi.sh/).
```console
cd sweat/
pixi install
``` 

To activate the environment
```console
pixi shell
```

### Install notebook
```console
pip install .[notebook]
```

## Usage

### Command-line interface

#### EVASPA

To run EVASPA 
```console
evaspa INPUT_FILE
```

An example of configuration file:
```json
{
    "input":{
        "path":"tests/data/modis_test_geo.tif",
        "date":"2018-05-16T10:00:00-00:00"
    },
    "output":{
        "path":"out"
    },
    "params":{
        "ef":{
            "models": "default_evaspa"
        }
    }
}
```

#### STIC

To run STIC
```console
stic INPUT_FILE
```

An example of configuration file:
```json
{
    "input":{
        "path":"tests/data/modis_test_full.tif",
        "date":"2018-05-16T10:00:00-00:00"
    },
    "output":{
        "path":"out"
    },
}
```

#### ET time series

To run timeseries
```console
timeseries INPUT_FILE
```

An example of configuration file:
```json
{
    "input": {
        "dates": [
            "2025-08-23",
            "2025-08-24",
            "2025-08-25",
            "2025-08-26",
        ],
        "et_time_series": [
            "tests/data/timeseries/et_time_series_20250823.tif",
            "tests/data/timeseries/et_time_series_20250824.tif",
            "tests/data/timeseries/et_time_series_20250825.tif",
        ],
        "radiation": [
            "tests/data/timeseries/radiation_20250823.tif",
            "tests/data/timeseries/radiation_20250824.tif",
            "tests/data/timeseries/radiation_20250825.tif",
            "tests/data/timeseries/radiation_20250826.tif",
        ],
        "et_single_date": [
            "tests/data/timeseries/et_single_date_20250826.tif",
        ]
    },
    "output": {
        "path": "out"
    },
    "params": {}
}
```

### Notebooks

To launch the notebooks
```console
jupyter-lab
```

[1]: https://doi.org/10.1016/j.proenv.2013.06.035 "B. Gallego-Elvira et al., EVASPA (EVapotranspiration Assessment from SPAce) Tool: An overview, Procedia Environmental Sciences, Volume 19, 2013."

[2]: https://doi.org/10.1016/j.rse.2013.10.022 "K. Mallick *et al.*, A Surface Temperature Initiated Closure (STIC) for surface energy balance fluxes, Remote Sensing of Environment, Volume 141, 2014."