.PHONY: test test-unit test-int cov lint e2e clean help

help:
	@echo "Available targets:"
	@echo "  test       - Run unit and integration tests (default)"
	@echo "  test-unit  - Run unit tests only"
	@echo "  test-int   - Run integration tests only"
	@echo "  cov        - Run unit and integration tests with coverage report on core"
	@echo "  lint       - Run ruff linter and mypy type checks"
	@echo "  e2e        - Run Playwright end-to-end tests"
	@echo "  clean      - Remove cache and temporary test artifacts"

test:
	pytest -m "unit or integration"

test-unit:
	pytest -m unit

test-int:
	pytest -m integration

cov:
	pytest --cov=callshield.core --cov-report=term-missing --cov-report=xml tests/

lint:
	ruff check .
	mypy src tests

e2e:
	pytest -m e2e

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist *.egg-info
