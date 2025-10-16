# Detailed parameters configuration for processing steps

## Stack step

The stack step is composed of a filtering step to identify valid data and
a stack stack to create the stack for processing the time series.

### Filtering step

The configuration describes how to find valid pixels. It is a dictionary.
The key corresponds to the variable to look at.
The value is either a single condition or a combined conditionThe key correspond to the data variables.

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
| method | Type of update method used | str | no | "linear" | "linear" |
| params | Parameters used to configure the update method | dict | no | - | - |


### Linear updater

| Name | Description | Type | Mandatory | Default value | Possible value |
|------|-------------|------|-----------|---------------|----------------|
| strict_mode | Only use acquisition to interpolate/extrapolate | bool | no | true | true or false |
| radiation_mode | Radiation mode to use | str | no | "EXTERNAL" | "THEORITICAL", "EXTERNAL" |


```json
{
    "params": {
        "strict_mode": true,
        "radiation_mode": "EXTERNAL"
    }
}
```

### Example

```json
{
    "update": {
        "method": "linear",
        "params": {
            "strict_mode": true,
            "radiation_mode": "EXTERNAL"
        }
    }
}
```
