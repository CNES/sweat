# Detailed parameters configuration for processing steps


## Filtering step

The configuration describes how to find valid pixels, i.e.
where ET is computed.
The configuration is stored in the form of a dictionary.
The key corresponds to the variable to look at.
The value is either a single condition or a combined condition.

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
| `models` | EF model description. See [EF models](ef_model.md#ef-models-configuration) | str or list[dict] | yes | - | - |
| `options.filtering` | Use to define points to be considered for computing the regression points for dry/wet edges | dict | no | - | - |
| `options.selection` | Activate model selection (Not implemented yet) | bool | no | false | true or false |
| `options.merging` | Methods used to merge EF data and compute uncertainty | str | no | "mean" | "median","mean" |
| `check.threshold` | Threshold used to check land surface temperature variability | float | no | 0.02 | - |

### Options: filtering

This section of the configuration corresponds to a second level of filtering.
It is used to describe which pixels must be kept for selecting the regression points based for the dry/wet edges computation.
This configuration is similar to the one used for define valid pixels, see [Filtering](#filtering-step).

### Options: Merging

This section of the configuration describes the methods used to :

- to merge EF data
- to compute uncertainty

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| `merging_method` | Methods used to merge EF data and compute uncertainty | str | no | "mean"  | "mean", "median" |
| `uncertainty_method` | Methods used to merge EF data and compute uncertainty | str | no | "interquartile" | "std", "nmad", "interquartile" |

```json
"merging": {
    "merging_method": "median",
    "uncertainty_method": "interquartile"
}
```

If `merging_method` is provided but not `uncertainty_method`, then `uncertainty_method` is set to
- **std** if `merging_method` is equal to **mean**
- **nmad** if `merging_method` is equal to **median**

### Example

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
            "filtering": {
                    "albedo": {
                        "and": [
                            {
                                "op": ">=",
                                "value": 0.1
                            },
                            {
                                "op": "<=",
                                "value": 0.3
                            }
                        ]
                    }
                },
            "selection": false,
            "merging": {
                "merging_method": "median",
                "uncertainty_method": "interquartile"
            }
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
            "merging": {
                "merging_method": "median"
            }
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
| `models` | Models used to compute $G/R_n$ | list[str] | no | ["kustas"] | "kustas","su","choudhury" |
| `use_topo` | Use topographic corrections for net radiation computation | bool | no | false | true or false |
| `merging` | Methods used to merge LE data and to compute uncertainty | dict | no | - | - |

### Merging

This section of the configuration describes the methods used to :

- to merge LE data
- to compute uncertainty

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| `merging_method` | Methods used to merge LE data and compute uncertainty | str | no | "mean"  | "mean", "median" |
| `uncertainty_method` | Methods used to merge LE data and compute uncertainty | str | no | "interquartile" | "std", "nmad", "interquartile" |

```json
"merging": {
    "merging_method": "median",
    "uncertainty_method": "interquartile"
}
```

The *merging* configuration should be the same that the one used for [EF step](#options-merging).
If not, a warning is raised and the configuration is changed to correspond to the EF step.


### Example

```json
{
    "seb": {
        "use_topo": false,
        "models": [
            "kustas"
        ],
        "merging": {
            "merging_method": "median",
            "uncertainty_method": "interquartile"
        }
    }
}
```

## Daily extrapolation step

This configuration describes the parameters for daily extrapolation processing step

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| `method` | Method used daily extrapolation | str | no | "toa" | "toa" |
| `use_topo` | Use topographic corrections for daily extrapolation | bool | no | false | true or false |

```json
{
    "daily": {
        "use_topo": false,
        "method": "toa"
    }
}
```
