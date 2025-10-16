RUN = pixi run --locked -q --no-progress -e dev

ruff:
	$(RUN) ruff format .
	$(RUN) ruff check . --fix

mypy:
	$(RUN) mypy src/ tests/

lint: ruff mypy

.PHONY: test
TESTARGS?=tests/  # default argument for the make test target
test:
	# you can use
	# make test TESTARGS="-k mytest"
	# make test TESTARGS="-m \"unit\""
	$(RUN) pytest $(TESTARGS)

test-unit:
	$(RUN) pytest --no-header -m unit $(TESTARGS)

test-cov:
	$(RUN) pytest --cov=sweat $(TESTARGS)

check: ruff mypy test

precommit:
	$(RUN) pre-commit run

precommit-all:
	$(RUN) pre-commit run --all-files

docs:
	$(RUN) mkdocs build
