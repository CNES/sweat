# Detailed parameters configuration for processing steps

## Prepare step

This configuration describes how to prepare data for the STIC model.

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| use_topo | Use DEM to correct solar direct radiation with slope and aspect. | bool | no | false | true orfalse |
| selected_radiation | Select which radiation data is selected | str | no | null | - |

Example
```json
{
    "prepare": {
            "use_topo": true,
            "selected_radiation": null
    }
}
```

## Filtering step

The configuration describes how to find valid pixels. It is a dictionary.
The key corresponds to the variable to look at. 
The value is either a single condition or a combined conditionThe key correspond to the data variables.

### Simple condition

A simple condition is composed of an operator and a value.
The list of available operators are `"==", "!=", ">", ">=", "<", "<="`.
The value can be either a float or a string in percentile format like 'percentile(90)'.

```json
{"op": "OPERATOR", "value": "VALUE"}
```

Example
```json
{"op": "==", "value": 0}
```
or
```json
{"op": "<=", "value": "percentile(95)"}
```

### Combined condition

A combined condition offers the possibility to use **and** or **or** operator to combine a list of several simple conditions. 

Example
```json
"and": [{"op": "!=", "value": 40}, {"op": "!=", "value": 50}]
```

A default configuration is applied:
```json
{
    "filtering": {
            "tdp": {
                "op": ">=",
                "value": -30.0
            }
    }
}
```

### Example

```json
{
    "filtering": {
        "lst": {"and": [{"op": ">", "value": 291}, {"op": "<", "value": 298}]},
        "cloud": {"op": "!=", "value": 1},
        "water": {"op": "!=", "value": 1},
        "lulc": {
            "and": [{"op": "!=", "value": 40}, {"op": "!=", "value": 50}],
        },
    }
}
```


## STIC processing step

This configuration describes the parameters for STIC main processing step

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| threshold | Iteration threshold value used for check convergence | float | no | 0.01 | - |
| nb_steps | Number of iteration steps | int | no | 15 | - |



```json
{
    "stic": {
        "threshold": 0.01,
        "nb_steps": 15
    }
}
```

## Daily extrapolation step

This ocnfiguration describes the parameters for daily extrapolation processing step

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| method | Method used daily extrapolation | str | no | "toa" | "toa" |
| use_topo | Use topographic corrections for daily extrapolation | bool | no | false | true or false |

```json
{
    "daily": {
        "use_topo": false,
        "method": "toa"
    }
}
```
