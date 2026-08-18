# Usage

## Run timeseries

### Command-line interface

To run ET time series
```console
timeseries INPUT_FILE
```

### Input file description

The input file is composed of 4 sections:

  * **input**: provide information for input data
  * **output**: provide information to write results
  * **params**: provide parameters to processing steps
  * **debug** (optional): provide information for debug mode

An example of input file:
```json
{
    "input": {
        "dates": [
            "2025-08-23",
            "2025-08-24",
            "2025-08-25",
            "2025-08-26"
        ],
        "et_time_series": [
            "tests/data/timeseries/et_time_series_20250823.tif",
            "tests/data/timeseries/et_time_series_20250824.tif",
            "tests/data/timeseries/et_time_series_20250825.tif"
        ],
        "radiation": [
            "tests/data/timeseries/radiation_20250823.tif",
            "tests/data/timeseries/radiation_20250824.tif",
            "tests/data/timeseries/radiation_20250825.tif",
            "tests/data/timeseries/radiation_20250826.tif"
        ],
        "et_single_date": [
            "tests/data/timeseries/et_single_date_20250826.tif"
        ]
    },
    "output": {
        "path": "out"
    },
    "params": {}
}
```

#### Input section

The **input** section is composed of:

| Name | Description | Type | Mandatory | Default value |
|------|-------------|------|-----------|--------------|
| `et_time_series` | List of file paths of previous ET time series | list[str] | yes | - |
| `dates` | List of dates to compute the new ET time series. If not provided, the dates correspond to ET time series | list[str] | non | - |
| `radiation` | List of file paths of daily radiation. If not provided, theoretical radiation is computed. | list[str] | no | - |
| `et_single_date` | List of file paths of new ET products used to update the tET time series. | list[str] | no | - |
| `dem` | Path to the DEM | str | no | - |

The path of input data must be a GeoTIF file.
The code assumes that each band of the GeoTIF file
has a band description in order to retrieve the name of the band to use.

Example:
```json
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
    ],
    "dem": "tests/data/timeseries/dem.tif",
},
```

##### Format of input data

Input data must be supplied in the form of a single file.
Data must be supplied in GeoTIFF format.
For each band, a *band description* option/tag must be used when writing the GeoTIFF file.
The code uses this information to find out which band it is.
The table below lists the expected nomenclature for band names.

| Band | Tag name |
|------|----------|
| Evapotranspiration | `et` |
| ET Flags | `flags` |
| Daily radiation | `daily_radiation` |
| DEM elevation | `height` |
| DEM slope | `slope` |
| DEM aspect | `aspect` |

#### Output section

The **output** section is composed of:

| Name | Description | Type | Mandatory | Default value |
|------|-------------|------|-----------|---------------|
| `path` | Path to write results | str | yes | - |

Example
:
```json
"output":{
    "path":"out"
}
```
#### Params section

The **params** section describe the parameters used for the processing. See [detailed parameters configuration](configuration.md)

#### Debug section

The **debug** section is composed of:

| Name | Description | Type | Mandatory | Default value |
|------|-------------|------|-----------|---------------|
| `profile` | Activate profiling | bool | no | false |
| `verbose` | Activate intermediate result writing | bool | no | false |
| `path` | Path to write intermediate results | str | no | debug |

Example:

```json
"debug": {
    "profile": true,
    "verbose": true,
    "path": "out/debug"
}
```

### Output directory description

**timeseries** produces the following results:

 - files corresponding to ET time series **et_time_series_YYYYMMDD.tif**
 - a file **config.json** which contains the detailed configuration used
 - a directory **debug** if debug mode is activated

Here is the directory tree for the results:

```bash
output_dir/
out/
├── config.json
├── debug
└── et_timeseries_YYYYMMDD.tif
```

## Run timeseries over period

### Command-line interface

To run ET time series
```console
window-timeseries INPUT_FILE
```

### Input file description

The input file is composed of 4 sections:

  * **input**: provide information for input data
  * **output**: provide information to write results
  * **params** (optional): provide parameters to processing steps
  * **debug** (optional): provide information for debug mode

An example of input file:

```json
{
    "input": {
        "period_start": "2025-08-23",
        "period_end": "2025-08-29",
        "et_single_date_dir": "tests/data/timeseries/",
        "radiation_dir": "tests/data/timeseries/"
    },
    "output": {
        "path": "out",
        "fmt": "json "
    }
}
```

#### Input section

The **input** section is composed of:


| Name | Description | Type | Mandatory | Default value |
|------|-------------|------|-----------|--------------|
| `period_start` | Date for period start (YYYY-MM-DD) | str | yes | - |
| `period_end` | Date for period end (YYYY-MM-DD) | str | yes | - |
| `et_single_date` | Directory with ET products. | str | yes | - |
| `radiation_dir` | Directory with daily radiation. If not provided, theoretical radiation is computed. | str | no | - |
| `window` | Size of the window to compute the time series at each step | int | no | 7 |
| `shift` | Shift used for the window at each step | int | no | 1 |
| `dem` | Path to the DEM | str | no | - |

Example:

```json
"input": {
    "period_end": "2025-08-29",
    "period_start": "2025-08-23",
    "et_single_date_dir": "tests/data/timeseries",
    "radiation_dir": "tests/data/timeseries",
    "shift": 1,
    "window": 7,
    "dem": "tests/data/timeseries/dem.tif"
}
```

#### Output section

The **output** section is composed of:

| Name | Description                      | Type | Mandatory | Default value | Possible value       |
|------|----------------------------------|------|-----------|---------------|----------------------|
| `path` | Path to write results            | str | yes | -             | -                    |
 |`fmt` | Format of the output config file | str | no | "json"        | "json", "yaml, "yml" |

Example:

```json
"output":{
    "path":"out",
    "fmt": "json"
}
```
#### Params section

The **params** section describe the parameters used for the processing. See [detailed parameters configuration](configuration.md)

#### Debug section

The **debug** section is composed of:

| Name | Description | Type | Mandatory | Default value |
|------|-------------|------|-----------|---------------|
| `profile` | Activate profiling | bool | no | false |
| `verbose` | Activate intermediate result writing | bool | no | false |
| `path` | Path to write intermediate results | str | no | debug |
| `config_verbose` | Activate time series configuration writing at each step | bool | no | false |
| `config_dir` | Path to write time series configuration files | str | no | config_dir |

Example:

```json
"debug": {
    "config_dir": "out/timeseries/config_dir",
    "config_verbose": true,
    "path": "out/timeseries/debug",
    "profile": false,
    "verbose": false
}
```

### Output directory description

**window-timeseries** produces the following results:

 - files corresponding to ET time series **et_time_series_YYYYMMDD.tif**
 - a file **config.(json/JSON/yaml/yml)** which contains the detailed configuration used
 - a directory **debug** containing intermediary results if debug mode is activated

Here is the directory tree for the results:

```bash
output_dir/
├── config_dir
│   └── config_YYYYMMDD_YYYYMMDD.json
├── config.json
├── debug
└── et_timeseries_YYYYMMDD.tif
```

### Flag description

| Bit | Description |
|-----|-------------|
| 0   | `PROCESSING_FAILED`: a flag to indicate if the processing failed |
| 1   | `RADIATION_MISSING`: a flag to indicate if radiation data is missing, this implies that processing failed |
| 2   | `AUX_DATA_MISSING`: a flag to indicate if auxiliary data is missing, this implies that processing failed |
| 3   | `FILTERED`: a flag to indicate if data is filtered, this implies that processing failed |
| 4   | `UPDATED`: a flag to indicate if data has been updated |
| 5 to 7 | `STATE`: to indicate the state: valid, interpolated, forward extrapolated, backward extrapolated, invalid, no data |
| 8 to 15 | `DISTANCE`: the difference between the two dates used for interpolation or the difference with the date used for extrapolation |

Available states:

- `0`: acquisition
- `1`: interpolation
- `2`: forward extrapolation
- `3`: backward extrapolation
- `6`: invalid
- `7`: n odata
