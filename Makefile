PYTHON ?= python3.12
VENV ?= .venv
SRC := src
TESTS := tests
PYTHONPATH := $(SRC)
export PYTHONPATH

.PHONY: help
help: ## Show this help message
	@echo "pjepa development targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

.PHONY: doctor
doctor: ## Run the capability probe report
	$(VENV)/bin/pjepa doctor

.PHONY: hardware
hardware: ## Print the active compute backend
	$(VENV)/bin/pjepa hardware

.PHONY: install
install: ## Install the project in editable mode with dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev,ogb]"

.PHONY: lint
lint: ## Run ruff
	$(VENV)/bin/ruff check $(SRC) $(TESTS)

.PHONY: format
format: ## Auto-format with ruff
	$(VENV)/bin/ruff format $(SRC) $(TESTS)

.PHONY: typecheck
typecheck: ## Run pytype in strict mode
	$(VENV)/bin/pytype $(SRC)/pjepa

.PHONY: test
test: ## Run the test suite (parallel)
	$(VENV)/bin/pytest $(TESTS) -n auto

.PHONY: test-fast
test-fast: ## Run only fast tests
	$(VENV)/bin/pytest $(TESTS) -n auto -m "not slow"

.PHONY: coverage
coverage: ## Run tests with coverage
	$(VENV)/bin/pytest $(TESTS) -n auto --cov=pjepa --cov-report=term-missing --cov-fail-under=80

.PHONY: audit
audit: ## Run dependency vulnerability audit
	$(VENV)/bin/pip-audit --strict

.PHONY: mutation
mutation: ## Run mutation testing on objectives and dynamics
	$(VENV)/bin/cosmic-ray run --baseline=src/pjepa/objectives src/pjepa/objectives
	$(VENV)/bin/cosmic-ray run --baseline=src/pjepa/dynamics src/pjepa/dynamics

.PHONY: bench-retrieval
bench-retrieval: ## Run the (1 - 1/e) retrieval benchmark
	$(VENV)/bin/python experiments/run_exp_a_retrieval.py

.PHONY: bench-distortion
bench-distortion: ## Run the hyperbolic distortion benchmark
	$(VENV)/bin/python experiments/run_exp_b_distortion.py

.PHONY: docs
docs: ## Build the documentation site
	$(VENV)/bin/mkdocs build --strict

.PHONY: docs-serve
docs-serve: ## Serve the documentation site locally
	$(VENV)/bin/mkdocs serve

.PHONY: clean
clean: ## Remove build and cache artefacts
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/ results/checkpoints/ results/logs/ results/optuna/ results/metrics/

.PHONY: all
all: lint typecheck test coverage ## Run lint, typecheck, test, and coverage

.PHONY: ci
ci: lint typecheck test audit ## Run the CI checklist