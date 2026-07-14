<p align="center">
  <h1 align="center">pjepa</h1>
  <p align="center">Persistent-JEPA — production-grade persistent graph world model for continual developmental learning.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
    <a href="https://github.com/sachncs/jepa/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/jepa/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachncs/jepa/stargazers"><img src="https://img.shields.io/github/stars/sachncs/jepa" alt="Stars"></a>
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

- **Persistent + Working Graphs** — `PersistentState` (long-term knowledge) and `WorkingGraph` (transient reasoning) with typed-attributed graph primitives.
- **Dual-Geometric Encoder** — `EuclideanMPNN` + `HyperbolicProjection` composed via `DualGeometricEncoder` with a `JEPAPredictor` head.
- **Greedy Retrieval with (1 − 1/e) Guarantee** — `retrieval.GreedyRetrieval` realises the Theorem 3 matroid-greedy bound.
- **Hyperbolic Distortion Bound** — encoders and retrieval are dimensioned for the Proposition 7 hyperbolic vs Euclidean guarantee.
- **HRG / Bisimulation Rewriting** — four-conditions rewriting with HRG, bisimulation, and DPO drivers in `pjepa.rewriting`.
- **Sleep-Cadence Scheduler** — PPO trainer, replay buffer, sleep cadence in `pjepa.scheduler`.
- **Variational Objective** — `𝒥` free-energy functional with information-bottleneck (IB) and minimum-description-length (MDL) terms.
- **Performance Infra** — `safe_compile`, autocast, EMA, fused scatter, sync helpers in `pjepa.perf`.
- **Augmentation Suite** — DropEdge, DropNode, DropFeature, FeatureMask, RandomWalk, and `TensorDropFeature`.
- **Training Stack** — SWA, TTA, Ensemble, Distillation, plus TU / CL / OGB runners.
- **Baselines** — GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM.
- **8-Class Test Taxonomy** — 436 tests covering happy / bad / ugly / leaky / round-trip / cross-backend / distributional / property.
- **mkdocs --strict** — researcher, developer, and reference doc trees.

---

## Installation

### From source

```bash
git clone https://github.com/sachncs/jepa.git
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
import pjepa
from pjepa.graphs import TypedAttributedGraph, PersistentState, WorkingGraph
from pjepa.encoders import DualGeometricEncoder, JEPAPredictor

# Build a typed attributed graph and wrap it in a persistent state
graph = TypedAttributedGraph.from_dataset("MUTAG")
state = PersistentState(initial=graph)

# Encode with the dual-geometric stack
encoder = DualGeometricEncoder(in_dim=state.feature_dim)
predictor = JEPAPredictor(latent_dim=encoder.latent_dim)
embedding = encoder.encode(state.working_view())
prediction = predictor(embedding)
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
| `pjepa.graphs.TypedAttributedGraph` | class | Typed attributed graph primitive |
| `pjepa.graphs.PersistentState` | class | Long-term knowledge container |
| `pjepa.graphs.WorkingGraph` | class | Transient reasoning container |
| `pjepa.encoders.EuclideanMPNN` | class | Euclidean message-passing encoder |
| `pjepa.encoders.HyperbolicProjection` | class | Hyperbolic projection encoder |
| `pjepa.encoders.DualGeometricEncoder` | class | Composition of Euclidean + Hyperbolic |
| `pjepa.encoders.JEPAPredictor` | class | JEPA predictor head |
| `pjepa.retrieval.GreedyRetrieval` | class | (1 − 1/e) matroid-greedy retrieval |
| `pjepa.rewriting.{HRG,Bisimulation,DPO}` | class | Four-conditions rewriting drivers |
| `pjepa.scheduler` | package | PPO trainer, replay buffer, sleep cadence |
| `pjepa.objectives` | package | `𝒥` free-energy functional, IB, MDL |
| `pjepa.dynamics` | package | Evolution operator `F`, contraction analysis |
| `pjepa.augmentations` | package | DropEdge, DropNode, DropFeature, … |
| `pjepa.training` | package | pretrain/train/eval, SWA, TTA, Ensemble, Distillation |
| `pjepa.eval` | package | metrics, bootstrap CI, statistical tests |
| `pjepa.perf` | package | safe_compile, autocast, EMA, fused scatter, sync |
| `pjepa.baselines` | package | GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM |

---

## Examples

```bash
# Validate Theorem 3 cheaply (single GPU)
pjepa benchmark retrieval --budget small

# Run Phase 8 TU SOTA on a single dataset
pjepa train tu configs/tu.yaml --dataset MUTAG --methods pjepa,gin,graphmae
```

```python
# Retrieve against a persistent state and inspect the guarantee
from pjepa.retrieval import GreedyRetrieval

retriever = GreedyRetrieval(persistent=state)
hits = retriever.retrieve(query=embedding, k=10)
guarantee = retriever.last_approximation_ratio  # >= (1 - 1/e)
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
│   └── run_exp_h_ablations.py
├── src/pjepa/                  # The library
│   ├── graphs/                 # TypedAttributedGraph, PersistentState, WorkingGraph
│   ├── encoders/               # EuclideanMPNN, HyperbolicProjection, DualGeometricEncoder, JEPAPredictor
│   ├── retrieval/              # GreedyRetrieval with (1 - 1/e) guarantee
│   ├── rewriting/              # HRG, bisimulation, four-conditions, DPO
│   ├── scheduler/              # PPO trainer, replay buffer, sleep cadence
│   ├── objectives/             # 𝒥 free-energy functional, IB, MDL
│   ├── dynamics/               # Evolution operator F, contraction analysis
│   ├── augmentations/          # DropEdge, DropNode, DropFeature, FeatureMask, RandomWalk, Tensor
│   ├── training/               # pretrain/train/eval loops, SWA, TTA, Ensemble, Distillation
│   ├── eval/                   # metrics, bootstrap CI, statistical tests
│   ├── perf/                   # safe_compile, autocast, EMA, fused scatter, sync
│   ├── data/                   # TUDataset, OGB-arxiv, class-incremental splits
│   ├── baselines/              # GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM
│   └── cli/                    # Typer-based CLI
├── tests/                      # 436 tests (8-class taxonomy)
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
pytest                              # 436 tests
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

- **v1.0.0** — Current: library, CLI dispatcher, experiment runners, aggregator, mkdocs strict docs, package artefacts.
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
