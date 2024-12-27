# EVASPA

-----

**Table of Contents**

- [Installation](#installation)

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

For development
```console
pip install -e .[dev,notebook]
```
