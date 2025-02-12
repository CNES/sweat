# EF model

Evaporative fraction $EF$, that represents the ratio of latent heat flux to available energy is computed 
from the position of the surface temperature value respectively to the dry edge and the wet edges. 
A EF model is defined by:

* the space to consider: for instance, albedo vs. temperature or fcover vs. temperature
* the method to compute the dry edge 
* the method to compute the wet edge

By varying the characteristics of the EF model, we can create a wide range of models.

## Generic method to estimate edges

### Definition
The methods for estimating dry and wet edges in the literature share common mathematical basis. May it be wet or dry, an edge's shape is picked amongst six possible shapes, each with a unique mathematical function:

* Flat edge: *FlatEdge*
* Linear edge: *LinearEdge*
* Linear edge with threshold: *ThresholdLinearEdge*
* Linear edge with inflexion point : *FlatLinearEdge*
* Double regression edge: *DoubleLinearEdge*
* Parabolic edge: *ParabolicEdge*

![edge](images/edge.png){ width="300" }
/// caption
Different dry edge types: flat edge (orange line), linear edge (red line), linear edge with threshold (dashed red line), linear edge with inflexion point (dotted red line), double regression edge (dashed red line), parabolic edge (yellow line)
///


For flat edges, the principle is to take the maximum for dry edge (or minimum for wet edge), excluding outliers or not. For other types of edge, 
the principle is to define points and then estimate the chosen regression. In these cases, each method divides the space (in relation to albedo or Fcover) into intervals. 
In our case, we could choose between intervals with same size or intervals with the same density of points.  
The number of intervals can be modified according to the method being used or tile being processed. 
For each interval, the objective is to define the point used to estimate the edge. Its x-axis coordinate is defined by taking the median of the points in the interval. 
Its LST-axis coordinate is defined by considering a percentile of points in the interval and taking the maximum (or the minimum) or the median. 
The Figure gives an example for a dry edge estimation.

![edge](images/dry_edge.png){ width="300" }
/// caption
Example of dry edge
///

### Edge configuration

Therefore, an edge is characterized by:

* **type**: edge class,
* for regression edge, a configuration **config**: 
    - **interval_type**: an interval type either fixed size *size* or fixed density *density*
    - **interval_nb**: a number of intervals (for density interval type) or **interval_size**: a size of intervals (for fixed size type)
    - **percentile**: a percentile interval to considered for point selection 
    - **selection**: a regression point selection criteria (*median*,*mean*,*max*,*min*)

### Examples

* Flat edge: 

```json
{
    "type": "FlatEdge"
}
```

* Linear edge: 

```json
{
    "type": "LinearEdge",
    "config": {
        "interval_type": "density",
        "interval_nb": 20,
        "percentile": [95, 100],
        "selection": "median"
    }
}
```

or 

```json
{
    "type": "LinearEdge",
    "config": {
        "interval_type": "size",
        "interval_size": 0.05,
        "percentile": [95, 100],
        "selection": "median"
    }
}
```

* Linear edge with threshold: 

```json
{
    "type": "ThresholdLinearEdge",
    "config": {
        "interval_type": "size",
        "interval_size": 0.01,
        "percentile": [95, 100],
        "selection": "median"
    }
}
```

 * Linear edge with inflexion point : 

```json
{
    "type": "FlatLinearEdge",
    "config": {
        "interval_type": "size",
        "interval_size": 0.01,
        "percentile": [95, 100],
        "selection": "median"
    }
}
```

* Double regression edge: 

```json
{
    "type": "DoubleLinearEdge",
    "config": {
        "interval_type": "size",
        "interval_size": 0.05,
        "percentile": [95, 100],
        "selection": "median"
    }
}
```

* Parabolic edge: 

```json
{
    "type": "ParabolicEdge",
    "config": {
        "interval_type": "size",
        "interval_size": 0.05,
        "percentile": [95, 100],
        "selection": "median"
    }
}
```

## Generic EF model

### Configuration

A EF model is defined by:

* **name**: a name used to identified the model
* **dry_edge**: a configuration for the dry edge
* **wet_edge**: a configuration for the wet edge
* **var**: the variable to considered

### Examples
```json
{
    "name": "EF1",
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
        "type": "LinearEdge",
        "config": {
            "interval_type": "density",
            "interval_nb": 20,
            "percentile": [0, 5],
            "selection": "median"
        }
    },
    "var": "albedo"
}
```


```json
{
    "name": "EF8",
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
```

### EF models configuration

In the input file, the description of EF models can be provided as a list of EF models. For instance:

```json
{
    "models": [
            {
                "name": "EF1",
                "dry_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "density",
                        "interval_nb": 20,
                        "percentile": [98, 100],
                        "selection": "median"
                    }
                },
                "wet_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "density",
                        "interval_nb": 20,
                        "percentile": [0, 2],
                        "selection": "median"
                    }
                },
                "var": "albedo"
            },
            {
                "name": "EF8",
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
    ]
}
```

See [EF model](#generic-ef-model) for detail to describe an EF model.  

But, it is also possible to use pre-defined configuration:

- "default_evaspa"
- "hsm_evaspa"
- "avignon_evaspa"

For instance 

```json
{
    "models": "default_evaspa"
}
```

