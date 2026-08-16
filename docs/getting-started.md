# Getting Started

## Installation

Persistent-JEPA requires Python 3.10, 3.11, or 3.12. We recommend 3.12.

### From source — Makefile workflow (recommended)

```bash
git clone https://github.com/sachncs/pjepa.git
cd pjepa
make install
```

`make install` creates a Python 3.12 virtual environment at `.venv`,
installs the package in editable mode, installs the dev and OGB
extras, and verifies every CI tool (`pjepa`, `pytest`, `ruff`,
`pytype`, `pip-audit`, `mkdocs`, `python -m build`, `optuna`) is
importable.

### From source — manual

If you prefer not to use the Makefile:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ogb]"
```

### From source — `setup.sh` script

For an opinionated single-shot install (creates the venv,
installs everything, verifies every CI tool, runs `pjepa doctor`,
executes the test suite):

```bash
git clone https://github.com/sachncs/pjepa.git
cd pjepa
bash setup.sh
```

### With Docker

```bash
docker build -t pjepa .
docker run --rm pjepa pjepa doctor
```

## Verify your environment

`pjepa` ships six capability probes that exercise the active
compute backend. Run them all at once:

```bash
source .venv/bin/activate   # if not already active
pjepa doctor
```

This prints a report similar to:

```
Backend:    mps
Device:     Apple Silicon (MPS)
Python:     3.12.7
PyTorch:    2.13.0
Platform:   macOS-26.6.2-arm64-arm-64bit
CPU count:  12

Capability probes:
  [GREEN ] matmul
  [GREEN ] scatter_add
  [GREEN ] torch.compile
  [GREEN ] hyperbolic
  [GREEN ] pyg_scatter
  [GREEN ] cpu_fallback
```

`exit code 0` when every probe is GREEN; `exit code 2` when at
least one probe is RED.

## Run the cheap validation benchmarks

The paper makes three central claims that have fast,
deterministic validations:

```bash
# Theorem 3: greedy retrieval achieves (1 - 1/e) ≈ 0.632 of optimum
pjepa benchmark retrieval

# Proposition 7: hyperbolic per-edge distortion is �(log D / (D log b))
pjepa benchmark distortion

# Proposition 3: dual-geometric beats Euclidean-only
pjepa benchmark encoder-ablation
```

Each prints a structured JSON summary to stdout.

## Run a headline experiment

The headline reproduction is a k-fold CV head-to-head of `gin`
vs `dual_geometric` on PROTEINS:

```bash
# 30-minute smoke
python experiments/train_real.py \
    --epochs 100 --seeds 3 --folds 5 \
    --methods gin dual_geometric \
    --output-dir results/proteins

# Multi-hour full reproduction
python experiments/train_real.py \
    --epochs 200 --seeds 3 --folds 10 \
    --methods gin dual_geometric \
    --output-dir results/proteins_full
```

Or via the CLI dispatch:

```bash
pjepa train tu configs/tu.yaml         # TU SOTA (6 datasets × 7 methods)
pjepa train cl configs/cl.yaml         # CL SOTA (3 datasets × 5 methods)
pjepa train ogb configs/ogb.yaml       # OGB-arxiv
pjepa tune  tu configs/tu.yaml         # Optuna search
pjepa aggregate results                # collate every result under results/
```

## Next steps

- Researchers: [Persistent Graph World Model](researcher/01_persistent_graph_world_model.md)
- Developers: [Quickstart for Developers](developer/01_quickstart.md)
- API reference: [Reference](reference/api.md) (auto-generated from docstrings)
