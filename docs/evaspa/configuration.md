# Detailed parameters configuration for processing steps


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

## Evaporative fraction step

This configuration describes the parameters for EF processing step.

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| models | EF model description. See [EF models](ef_model.md#ef-models-configuration) | str or list[dict] | yes | - | - |
| options.selection | Activate model selection (Not implemented yet) | bool | no | false | true or false |
| options.merging | Method used to merge EF data | str | no | "mean" | "median","mean" |
| check.threshold | Threshold used to check land surface temperature varaibility | float | no | 0.02 | - |

```json
{
    "ef": {
        "models": [
            {
                "name": "model1",
                "dry_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "size",
                        "interval_nb": 20,
                        "percentile": [98, 100],
                        "selection": "median"
                    }
                },
                "wet_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "size",
                        "interval_nb": 20,
                        "percentile": [0, 2],
                        "selection": "median"
                    }
                },
                "var": "albedo"
            },
            {
                "name": "model2",
                "dry_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "density",
                        "interval_nb": 20,
                        "percentile": [95, 100],
                        "selection": "median"
                    }
                },
                "wet_edge": {
                    "type": "FlatEdge"
                },
                "var": "albedo"
            }

        ],
        "options": {
            "selection": false,
            "merging": "mean"
        },
        "check": {
            "threshold": 0.02
        }
    }
}
```

```json
{
    "ef":{
        "models": "default_evaspa",
        "options": {
            "selection": false,
            "merging": "mean"
        },
        "check": {
            "threshold": 0.02
        }
    }
}
```


## Latent heat flux step

This configuration describes the parameters for SEB processing step.

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| models | Models used to compute $G/R_n$ | list[str] | no | ["kustas"] | "kustas","su","choudhury" |
| use_topo | Use topographic corrections for net radiation computation | bool | no | false | true or false |
| merging | Method used to merge LE data | str | no | "mean" | "median","mean" |

```json
{
    "seb": {
        "use_topo": false,
        "models": [
            "kustas"
        ],
        "merging": "mean"
    }
}
```

## Daily extrapolation step

This configuration describes the parameters for daily extrapolation processing step

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
