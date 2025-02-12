# Development guide

## Useful developper commands

Installation for development
```console
pip install -e .[dev,notebook]
```
Run static analysis
```console
hatch fmt # for ruff
hatch run types:check # for mypy
```

Run tests
```console
hatch test # for pytest
hatch test --cover # for coverage
```

Install pre-commit scripts
```console
pre-commit install
```

Run pre-commit
```console
pre-commit run --all-files
```

## Documentation generation

Installation for documentation
```console
pip install -e .[docs]
```

Build the documenttaion
```console
mkdocs build
``` 

Clean the documenttaion
```console
mkdocs build --clean
``` 

Start the live-reloading docs server
```console
mkdocs serve
``` 
