# EVASPA

-----


## Installation

### Clone the repository

```console
git clone https://src.koda.cnrs.fr/trishna/evaspa.git
```

### Install prerequisites

The prerequisites are installed by [pixi](https://pixi.sh/).
```console
cd evaspa/
pixi install
``` 

To activate the environment
```console
pixi shell
```

### Install evaspa
```console
pip install .
```

To use notebook
```console
pip install .[notebook]
```

## Usage

### Command-line interface

To run EVASPA 
```console
evaspa INPUT_FILE
```

An example of configuration file:
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

### Notebooks

To launch the notebooks
```console
jupyter-lab
```
