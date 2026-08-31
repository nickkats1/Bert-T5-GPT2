.PHONY: style quality test test-fast coverage

# Override when the venv is not activated, e.g. `make quality PYTHON=venv/bin/python`
PYTHON ?= python

check_dirs := headlines tests examples

style:
	$(PYTHON) -m ruff check --fix $(check_dirs)
	$(PYTHON) -m ruff format $(check_dirs)

quality:
	$(PYTHON) -m ruff check $(check_dirs)
	$(PYTHON) -m ruff format --check $(check_dirs)

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not integration"

coverage:
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=html
