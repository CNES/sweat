# Development guide

## Useful developer commands

Installation for development
```console
pixi install -e dev
```
Run static analysis
```console
make ruff
make mypy
```

Run tests

/!\ Make sure to get the sweat_test_data repository and set the environment variable SWEAT_TEST_DATA_PATH
to the path of the repository before running tests.

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

Installation for documentation (not necessary for development environment)
```console
pip install -e .[docs]
```

Build the documentation
```console
mkdocs build
```

Clean the documentation
```console
mkdocs build --clean
```

Start the live-reloading docs server
```console
mkdocs serve
```
