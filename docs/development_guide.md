# Development guide

## Useful developper commands

Installation for development
```console
pip install -e .[dev,docs,notebook]
```
Run static analysis
```console
make ruff
make mypy
```

Run tests
```console
make test # for pytest
make test-cov # for coverage
```

Install pre-commit scripts
```console
pre-commit install
```

Run pre-commit
```console
make precommit # For staged files
make precommit-all # For all files
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
