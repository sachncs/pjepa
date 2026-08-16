<p align="center">
  <h1 align="center">pjepa</h1>
  <p align="center">Persistent-JEPA — production-grade persistent graph world model for continual developmental learning.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
    <a href="https://github.com/sachncs/pjepa/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/pjepa/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachncs/pjepa/stargazers"><img src="https://img.shields.io/github/stars/sachncs/pjepa" alt="Stars"></a>
  </p>
</p>

**Persistent-JEPA** (`pjepa`) is a production-grade, open-source
implementation of a persistent graph world model for continual
developmental learning. It separates long-term knowledge (a persistent
graph), transient reasoning (a working graph), and learning dynamics (a
fast-weight kernel), all governed by a single information-theoretic
variational objective.

This repository contains the library, the training infrastructure, the
experiments, and the reproducibility package.

---

## Features

- **Persistent + Working Graphs** — `Graph` (the immutable substrate), `State` (long-term knowledge wrapper with commit/reject audit trail), and `Working` (budget-bounded retrieval view) — see `pjepa.graphs`.
- **Dual-Geometric Encoder** — `Euclidean` (GIN-style MPNN) + `Hyperbolic` (Poincaré projection) composed via `DualGeometric`, all rooted in a polymorphic `Encoder` ABC. A separate `Predictor` / `Target` head pair (`Head` ABC) drives JEPA training.
- **Greedy Retrieval with (1 − 1/e) Guarantee** — `Retrieval` realises the Theorem 3 matroid-greedy bound; the utility hierarchy (`Utility` ABC) provides `Facility` (provably submodular) and `InfoGain` (information-gain with per-vertex cost).
- **Hyperbolic Distortion Bound** — encoders and retrieval are dimensioned for the Proposition 7 hyperbolic vs Euclidean guarantee.
- **HRG / Bisimulation Rewriting** — `Criterion` ABC with the headline `FourConditions` verifier, plus HRG, bisimulation, and DPO drivers in `pjepa.rewriting`.
- **Sleep-Cadence Scheduler** — PPO trainer, `Buffer` replay storage (`Storage` ABC), and `Sleep` cadence (`Cadence` ABC) in `pjepa.scheduler`.
- **Variational Objective** — `𝒥` free-energy functional with information-bottleneck (IB) and minimum-description-length (MDL) terms — `pjepa.objectives`.
- **Performance Infra** — `safe_compile`, autocast, EMA, fused scatter, sync helpers — `pjepa.perf`.
- **Augmentation Suite** — DropEdge, DropNode, DropFeature, FeatureMask, RandomWalk, plus `Transform` / `Pipeline` (the polymorphic augmentation ABCs) and `TensorDropFeature`.
- **Training Stack** — pretrain / train / eval loops, SWA, TTA, Ensemble, Distillation, plus TU / CL / OGB runners — `pjepa.training`.
- **Baselines** — GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM, BGRL, GraphSAGE, PackNet, Naive.
- **8-Class Test Taxonomy** — 495 tests covering happy / bad / ugly / leaky / round-trip / cross-backend / distributional / property.
- **mkdocs --strict** — researcher, developer, and reference doc trees.
- **Real multi-hour training** — `experiments/train_real.py` is the canonical k-fold CV training script. The full PROTEINS run (3 seeds × 10 folds × 200 epochs × 2 methods, 60 fits, ~2 hours on a single CPU) ships in `results/proteins_full/`.

---

## Installation

### From source

```bash
git clone https://github.com/sachncs/pjepa.git
cd jepa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ogb]"
```

### With Docker

```bash
docker build -t pjepa .
docker run --rm pjepa pjepa doctor
```

**Requirements**: Python 3.10–3.12 (3.12 recommended), pip ≥ 21.

---

## Quick Start

### CLI

```bash
# Verify your environment
pjepa doctor

# Validate paper claims cheaply
pjepa benchmark retrieval        # Theorem 3 — (1 - 1/e) retrieval approximation
pjepa benchmark distortion       # Proposition 7 — hyperbolic vs Euclidean distortion
pjepa benchmark encoder-ablation # Proposition 3 — dual-geometric advantage

# Run headline experiments
pjepa tune tu configs/tu.yaml    # Optuna search for Persistent-JEPA
pjepa train tu configs/tu.yaml   # TU SOTA (6 datasets × 7 methods)
pjepa train cl configs/cl.yaml   # CL SOTA (3 datasets × 5 methods)
pjepa train ogb configs/ogb.yaml # OGB-arxiv

# Aggregate results across phases
pjepa aggregate results          # writes results/all_runs.jsonl + tables
```

### Python API

```python
import torch
from pjepa.graphs import Graph, State
from pjepa.encoders import Euclidean, DualGeometric, Hyperbolic, Predictor
from pjepa.retrieval import Retrieval, Facility, InfoGain

# Build a typed attributed graph and wrap it in a persistent state
v = torch.randn((6, 8))
ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)
g = Graph(v, ei, torch.zeros((ei.shape[1], 2)))
state = State(graph=g)

# Encode with the dual-geometric stack (returns a concat of Euclidean + hyperbolic)
encoder = DualGeometric(input_dim=8, euclidean_dim=16, hyperbolic_dim=4, num_layers=2)
embedding = encoder.encode(g)           # shape (6, 20)

# Encode with the Euclidean-only stack (returns a single tensor)
eu = Euclidean(input_dim=8, hidden_dim=16, num_layers=2, output_dim=16)
eu_embedding = eu(g)                    # shape (6, 16)

# The predictor takes a context vector and produces a target.
predictor = Predictor(input_dim=20, hidden_dim=32, output_dim=20)
prediction = predictor(embedding)

# Submodular working-graph retrieval
retriever = Retrieval(budget=4)
result = retriever.select(g, torch.randn(4, 8), utility=Facility(g.vertex_features))
print(result.utility, result.iterations)
```

---

## Configuration

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| Python version | `PYTHON_VERSION` | `3.12` | Recommended; 3.10/3.11 also supported |
| Extras | — | — | Install with `.[dev]`, `.[ogb]`, or `.[dev,ogb]` |
| Compile mode | `PJEPA_SAFE_COMPILE` | `0` | Set to `1` to enable `safe_compile` |
| Autocast | `PJEPA_AUTOCAST` | `1` | Mixed-precision autocast toggle |
| Result root | `PJEPA_RESULTS_DIR` | `results` | Where `pjepa aggregate` writes tables |
| Preserve | `PRESERVE` | `0` | Set to `1` to keep `results/` during `cleanup.sh` |

See `configs/*.yaml` for the canonical TU / CL / OGB experiment configs.

---

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `pjepa.graphs.Graph` | class | Typed attributed graph primitive |
| `pjepa.graphs.State` | class | Long-term knowledge container |
| `pjepa.graphs.Working` | class | Transient reasoning container |
| `pjepa.encoders.Euclidean` | class | Euclidean message-passing encoder |
| `pjepa.encoders.Hyperbolic` | class | Hyperbolic projection encoder |
| `pjepa.encoders.DualGeometric` | class | Composition of Euclidean + Hyperbolic |
| `pjepa.encoders.Predictor` | class | JEPA predictor head |
| `pjepa.encoders.Target` | class | EMA target encoder |
| `pjepa.retrieval.Retrieval` | class | (1 − 1/e) matroid-greedy retrieval |
| `pjepa.retrieval.Utility` | class (ABC) | Retrieval-utility base class |
| `pjepa.retrieval.Facility` | class | Provably-submodular coverage utility |
| `pjepa.retrieval.InfoGain` | class | Information-gain utility with per-vertex cost |
| `pjepa.rewriting.{HRG,Bisimulation,Criterion}` | class | Verified rewriting drivers |
| `pjepa.rewriting.FourConditions` | class | The four-conditions acceptance criterion |
| `pjepa.scheduler` | package | PPO trainer, replay buffer, sleep cadence |
| `pjepa.scheduler.PPOTrainer` | class | Clipped-surrogate PPO trainer |
| `pjepa.scheduler.Buffer` | class | FIFO replay buffer (concrete `Storage`) |
| `pjepa.scheduler.Sleep` | class | Rolling-statistic sleep cadence (concrete `Cadence`) |
| `pjepa.objectives` | package | `𝒥` free-energy functional, IB, MDL |
| `pjepa.objectives.FreeEnergy` | class | The four-term `𝒥` functional |
| `pjepa.dynamics` | package | Evolution operator `F`, contraction analysis |
| `pjepa.dynamics.EvolutionOperator` | class | Configuration for the contraction analysis |
| `pjepa.augmentations` | package | DropEdge, DropNode, DropFeature, … |
| `pjepa.augmentations.Transform` | class (ABC) | Augmentation base class |
| `pjepa.augmentations.Pipeline` | class | Composition of multiple `Transform`s |
| `pjepa.training` | package | pretrain/train/eval, SWA, TTA, Ensemble, Distillation |
| `pjepa.eval` | package | metrics, bootstrap CI, statistical tests |
| `pjepa.eval.aggregate` | module | Canonical result aggregator |
| `pjepa.perf` | package | safe_compile, autocast, EMA, fused scatter, sync |
| `pjepa.perf.EMATarget` | class | Cosine-schedule EMA wrapper |
| `pjepa.baselines` | package | GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM, BGRL, GraphSAGE, PackNet, Naive |
| `pjepa.compat` | module | Backward-compatible aliases for renamed symbols |

---

## Examples

```bash
# Validate Theorem 3 cheaply (single GPU)
pjepa benchmark retrieval

# Run Phase 8 TU SOTA on a single dataset
pjepa train tu configs/tu.yaml --dataset MUTAG --methods gin,dual_geometric

# Run the headline k-fold-CV reproduction on PROTEINS
python experiments/train_real.py --epochs 200 --seeds 3 --folds 10 \
    --methods gin dual_geometric --output-dir results/proteins_full
```

```python
# Retrieve against a persistent state and inspect the guarantee
from pjepa.retrieval import Retrieval

retriever = Retrieval(budget=4)
result = retriever.select(state.graph, torch.randn(4, 8))
print(result.utility, result.iterations)  # utility, iterations actually used
```

---

## Project Structure

```
pjepa/
├── docs/                       # User-facing documentation
│   ├── researcher/             # Deep-dive explanations
│   ├── developer/              # API guides, extension tutorials
│   └── reference/              # Auto-generated API docs
├── experiments/                # Runnable experiment scripts
│   ├── run_exp_a_retrieval.py  # (1 - 1/e) validation
│   ├── run_exp_b_distortion.py # Hyperbolic distortion bound
│   ├── run_exp_c_encoder_ablation.py
│   ├── run_exp_d_tu_sota.py
│   ├── run_exp_e_continual.py
│   ├── run_exp_f_ogb_arxiv.py
│   ├── run_exp_g_decoupling.py
│   ├── run_exp_h_ablations.py
│   └── train_real.py           # k-fold CV training script (PROTEINS head-to-head)
├── src/pjepa/                  # The library
│   ├── graphs/                 # Graph, State, Working
│   ├── encoders/               # Encoder (ABC), Euclidean, Hyperbolic, DualGeometric, Head/Predictor/Target
│   ├── retrieval/              # Retrieval, Utility (ABC), Facility, InfoGain
│   ├── rewriting/              # HRG, Bisimulation, Criterion (ABC), FourConditions
│   ├── scheduler/              # PPOTrainer, Buffer, Storage (ABC), Cadence/Sleep
│   ├── objectives/             # FreeEnergy, ib_lagrangian, description_length
│   ├── dynamics/               # EvolutionOperator, contractivity_bound, fixed_point_iteration
│   ├── augmentations/          # Transform (ABC), Pipeline, DropEdge, DropNode, …
│   ├── training/               # pretrain/train/eval loops, SWA, TTA, Ensemble, Distillation
│   ├── eval/                   # metrics, bootstrap CI, statistical tests
│   ├── perf/                   # safe_compile, autocast, EMATarget, fused scatter, sync
│   ├── data/                   # TUDataset, OGB-arxiv, class-incremental splits
│   ├── baselines/              # GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM, BGRL, GraphSAGE, PackNet, Naive
│   └── cli/                    # Typer-based CLI
├── tests/                      # 495 tests (8-class taxonomy)
├── configs/                    # TU, CL, OGB experiment configs
├── pyproject.toml              # PEP 621 metadata
└── Dockerfile                  # Reproducible container image
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev,ogb]"

# Tests
pytest                              # 495 tests
pytest -m "not slow"                # skip slow tests
pytest --cov=pjepa tests/           # with coverage

# Lint / format / type
ruff check src/ tests/
ruff format src/ tests/
pytype src/pjepa

# Docs
mkdocs build --strict
mkdocs serve

# Audits
pip-audit
vulture src/pjepa
```

The `setup.sh` / `cleanup.sh` scripts at the repository root are the
canonical environment contract: `setup.sh` creates the venv, installs
the project with `.[dev,ogb]`, verifies every CI tool (`pjepa`,
`pytest`, `ruff`, `pytype`, `pip-audit`, `mkdocs`, `python -m build`,
`optuna`), runs `pjepa doctor`, and executes the test suite;
`cleanup.sh` removes the venv, build artefacts, type-checker caches,
the mkdocs site, Python bytecode caches, and `results/` (preserved
when `PRESERVE=1`).

---

## Testing

```bash
pytest                              # run the 8-class taxonomy tests
pytest -m "not slow"                # skip slow tests
```

---

## Build

```bash
python -m build
```

Distribution artefacts include the sdist, the wheel, and the
`Dockerfile` image.

---

## Release

Versions follow [Semantic Versioning](https://semver.org/). Releases are
tracked in [CHANGELOG.md](CHANGELOG.md) and the citation metadata in
[CITATION.cff](CITATION.cff).

**Local 1.0.0 scope**: the library, the CLI dispatcher, the experiment
runners, the aggregator, the docs site (mkdocs strict), the package
artefacts (`make package`), and the changelog are all in the initial
release.

**External 1.0.0 scope (intentionally not executed here)**:

- Docker image push to a registry — requires credentials.
- GitHub Release `v1.0.0` with attached sdist + wheel.
- PyPI upload — requires credentials and a maintainer decision.
- Read the Docs build trigger — requires RTD credentials.
- Full 70-hour reproduction (`make reproduce-all`) — saturates the CI
  runner; the reproduction matrix in `experiments/REPRODUCE.md` remains
  the source of truth for one-command re-runs.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10–3.12 |
| Compute | PyTorch (CUDA / ROCm / MPS) |
| Search | Optuna |
| Docs | mkdocs (strict) |
| Lint | ruff |
| Type Check | pytype |
| Audit | pip-audit |
| Dead Code | vulture |
| Testing | pytest |
| Container | Docker |
| Citation | CITATION.cff |

---

## Roadmap

- **v1.0.0** — Current: library, CLI dispatcher, experiment runners, aggregator, mkdocs strict docs, package artefacts, full Google-style docstrings on every public symbol, polymorphic ABC roots for every major hierarchy.
- **v1.1.0** — Distributed (multi-GPU) training; persistent-state compression.
- **v2.0.0** — External release (Docker push, GH Release, PyPI upload, RTD trigger).

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold (venv, pyproject, CI, Docker, mkdocs, doctor) | ✅ Complete |
| 1 | Core library + 8-class tests | ✅ Complete |
| 2 | Performance infra (safe_compile, autocast, EMA, fused scatter, sync) | ✅ Complete |
| 3 | Augmentation suite | ✅ Complete |
| 4 | Validation experiments (Exp A retrieval, B distortion, C encoder-ablation) | ✅ Complete |
| 5 | Training infra (SWA, TTA, Ensemble, Distillation) | ✅ Complete |
| 6 | Optuna hyperparameter search | ✅ Complete |
| 7 | Baselines | ✅ Complete |
| 8 | TU SOTA experiment runner | ✅ Complete |
| 9 | CL SOTA experiment runner | ✅ Complete |
| 10 | OGB-arxiv experiment runner | ✅ Complete |
| 11 | Decoupling measurement + ablations | ✅ Complete |
| 12 | Reporting + 1.0.0 release (local) | ✅ Complete |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/developer/01_quickstart.md](docs/developer/01_quickstart.md) for the
workflow. All public symbols need Google-style docstrings; tests must
cover the eight-class taxonomy.

## Code of Conduct

Contributors are expected to follow the
[Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

## Citation

If you use `pjepa` in academic work, please cite the paper. The BibTeX
entry is in [CITATION.cff](CITATION.cff).

## License

[Apache-2.0](LICENSE) © 2026 Sachin
