# EVASPA

-----

**Table of Contents**

- [Installation](#installation)
- [Development](#Development)


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

## Development

For development
```console
pip install -e .[dev,notebook]
```

```console
hatch test # for pytest
hatch test --cover # for coverage
hatch fmt # for ruff
hatch run types:check # for mypy
```