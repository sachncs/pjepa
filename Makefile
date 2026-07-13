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

.PHONY: install
install: ## Install the project in editable mode with dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev,ogb]"

.PHONY: doctor
doctor: ## Run the capability probe report
	$(VENV)/bin/pjepa doctor

.PHONY: hardware
hardware: ## Print the active compute backend
	$(VENV)/bin/pjepa hardware

.PHONY: lint
lint: ## Run ruff
	$(VENV)/bin/ruff check $(SRC) $(TESTS)

.PHONY: format
format: ## Auto-format with ruff
	$(VENV)/bin/ruff format $(SRC) $(TESTS)

.PHONY: format-check
format-check: ## Verify ruff formatting (no changes)
	$(VENV)/bin/ruff format --check $(SRC) $(TESTS)

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
	$(VENV)/bin/mutmut run --paths-to-mutate=src/pjepa/objectives
	$(VENV)/bin/mutmut run --paths-to-mutate=src/pjepa/dynamics

.PHONY: bench-retrieval
bench-retrieval: ## Run the (1 - 1/e) retrieval benchmark
	$(VENV)/bin/pjepa benchmark retrieval

.PHONY: bench-distortion
bench-distortion: ## Run the hyperbolic distortion benchmark
	$(VENV)/bin/pjepa benchmark distortion

.PHONY: bench-encoder
bench-encoder: ## Run the encoder ablation benchmark (Proposition 3)
	$(VENV)/bin/pjepa benchmark encoder-ablation

.PHONY: tune-tu
tune-tu: ## Optuna hyperparameter search for TU datasets
	$(VENV)/bin/pjepa tune tu configs/tu.yaml

.PHONY: train-tu
train-tu: ## Train on TU datasets using the supplied config
	$(VENV)/bin/pjepa train tu configs/tu.yaml

.PHONY: train-cl
train-cl: ## Train on CL datasets using the supplied config
	$(VENV)/bin/pjepa train cl configs/cl.yaml

.PHONY: train-ogb
train-ogb: ## Train on OGB-arxiv using the supplied config
	$(VENV)/bin/pjepa train ogb configs/ogb.yaml

.PHONY: reproduce-tu
reproduce-tu: ## Reproduce Phase 8 TU SOTA experiment
	$(VENV)/bin/pjepa train tu configs/tu.yaml

.PHONY: reproduce-cl
reproduce-cl: ## Reproduce Phase 9 CL SOTA experiment
	$(VENV)/bin/pjepa train cl configs/cl.yaml

.PHONY: reproduce-ogb
reproduce-ogb: ## Reproduce Phase 10 OGB-arxiv experiment
	$(VENV)/bin/pjepa train ogb configs/ogb.yaml

.PHONY: reproduce-all
reproduce-all: reproduce-tu reproduce-cl reproduce-ogb ## Reproduce every experiment

.PHONY: verify-claims
verify-claims: ## Cross-check reproduced numbers against paper claims
	$(VENV)/bin/python -m pytest tests/test_tu_sota_aggregation.py tests/test_phase11_experiments.py -q

.PHONY: ablation
ablation: ## Run the Phase 11 ablation study
	$(VENV)/bin/python experiments/run_exp_h_ablations.py

.PHONY: sensitivity
sensitivity: ## Run the Phase 11 sensitivity sweep
	$(VENV)/bin/python experiments/run_sensitivity.py

.PHONY: decoupling
decoupling: ## Run the inference-storage decoupling measurement
	$(VENV)/bin/python experiments/run_exp_g_decoupling.py

.PHONY: aggregate
aggregate: ## Aggregate every experiment's results under results/
	$(VENV)/bin/pjepa aggregate results

.PHONY: profile
profile: ## Run a CPU/memory profile of the pretraining loop
	$(VENV)/bin/python -c "import cProfile, pstats, io, torch; \
from pjepa.training.pretrain import pretrain_loop, PretrainConfig; \
from pjepa.encoders import DualGeometricEncoder, JEPAPredictor, TargetEncoder; \
enc = DualGeometricEncoder(vertex_dim=8, hidden_dim=16, num_layers=2); \
pred = JEPAPredictor(hidden_dim=16); \
tgt = TargetEncoder(enc); \
x = torch.randn(8, 8); \
pr = cProfile.Profile(); pr.enable(); \
pretrain_loop(enc, x, JEPAPredictor=pred, target=tgt, config=PretrainConfig(epochs=1, batch_size=2)); \
pr.disable(); \
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(20); \
print(s.getvalue())"

.PHONY: package
package: ## Build sdist + wheel locally (dry-run, no upload)
	$(VENV)/bin/python -m build --sdist --wheel

.PHONY: release
release: ## Dry-run the release packaging flow (no tag, no upload)
	$(VENV)/bin/python -m build --sdist --wheel
	@echo "release: artefacts built under dist/ (no tag, no upload performed)"

.PHONY: docs
docs: ## Build the documentation site (strict)
	$(VENV)/bin/mkdocs build --strict

.PHONY: docs-serve
docs-serve: ## Serve the documentation site locally
	$(VENV)/bin/mkdocs serve

.PHONY: clean
clean: ## Remove build and cache artefacts
	rm -rf build/ dist/ site/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .pytype/ \
	       .coverage coverage.xml coverage-*.xml htmlcov/ \
	       src/pjepa.egg-info/ src/*.egg-info/ \
	       $(VENV) \
	       results/checkpoints/ results/logs/ results/optuna/ results/metrics/ \
	       results/tu_smoke/ results/exp_a_smoke/ results/exp_b_smoke/ \
	       results/exp_c_smoke/ results/ogb_smoke/ results/decoupling_smoke/ \
	       results/ablation_smoke/ results/sensitivity_smoke/ \
	       results/cl/ results/ogb/ results/ablation/ results/decoupling/ \
	       results/sensitivity_B/ results/tables/ results/plots/ results/all_runs.jsonl
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '*.py[co]' -delete

.PHONY: setup
setup: ## Run the project setup script (setup.sh)
	bash setup.sh

.PHONY: cleanup
cleanup: ## Run the project cleanup script (cleanup.sh)
	bash cleanup.sh

.PHONY: all
all: lint format-check typecheck test coverage docs ## Run lint, format, typecheck, test, coverage, and docs

.PHONY: ci
ci: lint format-check typecheck test coverage docs ## Run the full blocking CI checklist
