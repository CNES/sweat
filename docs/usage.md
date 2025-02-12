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
