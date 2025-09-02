# Usage

## Command-line interface

To run ET time series 
```console
timeseries INPUT_FILE
```

## Input file description

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

STIC produces the following results:

 - files corresponding to ET time series **et_time_series_YYYYMMDD.tif** 
 - a file **config.json** which contains the detailed configuration used
 - a directory **debug** if debug mode is activated

### Input section

The **input** section is composed of:

| Name | Description | Type | Mandatory | Default value |
|------|-------------|------|-----------|--------------|
| et_time_series | List of file paths of previous ET time series | list[str] | yes | - |
| dates | List of dates to compute the new ET time series. If not provided, the dates correspond to ET time series | list[str] | non | - |
| radiation | List of file paths of daily radiation. If not provided, theoritical radiation is computed. | list[str] | no | - |
| et_single_date | List of file paths of new ET products used to update the tET time series. | list[str] | no | - |
| dem | Path to the DEM | str | no | - |

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

#### Format of input data

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

### Output section

The **output** section is composed of:

| Name | Description | Type | Mandatory | Default value |
|------|-------------|------|-----------|---------------|
| path | Path to write results | str | yes | - |

Example:
```json
"output":{
    "path":"out"
}
```
### Params section

The **params** section describe the parameters used for the processing. See [detailed parameters configuration](configuration.md)

### Debug section

The **debug** section is composed of:

| Name | Description | Type | Mandatory | Default value |
|------|-------------|------|-----------|---------------|
| profile | Activate profiling | bool | no | false |
| verbose | Activate intermediate result wrinting | bool | no | false |
| path | Path to write intermediate results | str | no | debug |

Example:
```json
"debug": {
    "profile": true,
    "verbose": true,
    "path": "out/debug"
}
```

## Output directory description

ET time series produces the following files in the output directory: 

```bash
output_dir/
out/
├── config.json
├── debug
└── et_timeseries_YYYYMMDD.tif
```

* `config.json` corresponds to the exact configurtaion used to run STIC.
* `et_time_series_YYYYMMDD.tif` corresponding to all files of the new ET time series
* `debug` (optional) contains intermediary results if verbose mode is active in debug section