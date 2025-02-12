# Detailed parameters configuration for processing steps

## Filtering step

Describe how to identify valid pixels

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| cloud | Name of variables that contains cloud mask in input data. If not provided, cloud mask is not applied | str | no | null | - |
| water | Name of variables that contains water mask in input data. If not provided, cloud mask is not applied | str | no | null | - |
| qa | Name of variables that contains quality mask in input data. If not provided, cloud mask is not applied | str | no | null | - |
| zones | Name of variables that contains valid zone mask in input data. If not provided, cloud mask is not applied | str | no | null | - |
| cover | Name of variables that contains land cover / land use mask in input data. If not provided, cloud mask is not applied | str | no | null | - |
| config | Parameters for the filter. The parameters correspond to the values used for masking | dict | no | - | - |

```json
{
    "filtering": {
        "cloud": null,
        "water": null,
        "qa": null,
        "zones": null,
        "cover": null,
        "config": {
            "cloud": 0,
            "water": 0,
            "qa": 0,
            "zones": 0,
            "cover": [
                10,
                60,
                80
            ]
        }
    }
}
```

## Evaporative fraction step

Describe the parameters for EF processing step

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

Describe the parameters for EF processing step

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

Describe the parameters for EF processing step

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
