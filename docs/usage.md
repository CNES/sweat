# Overview

## Command-line interface

To run EVASPA 
```console
evaspa INPUT_FILE
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

EVASPA produces the following results:

 - a directory **evaspa_daily** which contains daily products: LE and ET 
 - a directory **evaspa_inst** which contains instantaneous products: EF, LE, ET 
 - a file **config.json** which contains the detailed configuration used
 - a directory **debug** if debug mode is activated

### Input section

The **input** section is composed of:

| Name | Description | Type | Mandatory | Defult value |
|------|-------------|------|-----------|--------------|
| path | Path of input data (file or directory) | str | yes | - |
| date | Date of the acquisition | str | non | - |

The path of input data can be either a GeoTIF file or a directory 
containing GeoTIF files. The code assumes that each band of the GeoTIF file 
has a band description in order to retrieve the name of the band to use.

Example:
```json
"input":{
    "path":"tests/data/modis_test_geo.tif",
    "date":"2018-05-16T10:00:00-00:00"
}
```

### Output section

The **input** section is composed of:

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

EVASPA produces the following files in the output directory: 

```bash
output_dir/
out/
├── config.json
├── debug
│   ├── evaspa.ef_run_0.nc
│   ├── evaspa.ef_run_1.nc
│   ├── evaspa.filter_determine_valid_pixels.nc
│   ├── evaspa.seb_create_net_radiation.nc
│   ├── evaspa.seb_create_ratio.nc
│   ├── evaspa.seb_run_0.nc
│   └── evaspa.seb_run_1.nc
├── evaspa_daily
│   ├── evaspa_daily_et.tif
│   └── evaspa_daily_le.tif
└── evaspa_inst
    ├── evaspa_inst_ef.tif
    ├── evaspa_inst_et.tif
    ├── evaspa_inst_le.tif
    ├── evaspa_inst_uncertainty_ef.tif
    └── evaspa_inst_uncertainty_le.tif
```

* `config.json` corresponds to the exact configurtaion used to run EVASPA.
* `evasa_inst` contains instantaneous products
* `evasa_daily` contains daily products
* `debug` (optional) contains intermediary results if verbose mode is active in debug section