# Detailed parameters configuration for processing steps

## Stack step

The stack step is composed of a filtering step to identify valid data and
a stack stack to create the stack for processing the time series.

### Filtering step

The configuration describes how to find valid pixels, i.e.
where ET is computed.
The configuration is stored in the form of a dictionary.
The key corresponds to the variable to look at.
The value is either a single condition or a combined condition.

#### Simple condition

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

#### Combined condition

A combined condition offers the possibility to use **and** or **or** operator to combine a list of several simple conditions.

Example
```json
"and": [{"op": "!=", "value": 40}, {"op": "!=", "value": 50}]
```

A default configuration is applied:
```json
{
    "filtering": {
            "flags": {
                "op": "==",
                "value": 0
            }
    }
}
```

### Example

```json
{
    "stack": {
        "et_single_date_filtering": {
            "flags": {
                "op": "==",
                "value": 0
            }
        }
    }
}
```


## Update step

This configuration describes the parameters for update processing step

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| `method` | Type of update method used | str | no | "linear" | "linear" |
| `params` | Parameters used to configure the update method | dict | no | - | - |


### Linear updater

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| `parallel` | Use Dask parallelization |  bool | no | false | true or false |
| `num_workers` | Number of workers used for parallelization | int | no | 4 | - |


```json
{
    "params": {
        "parallel": true,
        "num_workers": 8
    }
}
```

### Example

```json
{
    "update": {
        "method": "linear",
        "params": {
            "parallel": true,
            "num_workers": 8
        }
    }
}
```
