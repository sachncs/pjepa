<p align="center">
  <h1 align="center">pjepa</h1>
  <p align="center">Persistent-JEPA — a persistent graph world model for continual developmental learning.</p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
    <a href="https://github.com/sachncs/pjepa/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/pjepa/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachncs/pjepa/stargazers"><img src="https://img.shields.io/github/stars/sachncs/pjepa" alt="Stars"></a>
  </p>
</p>

Persistent-JEPA (`pjepa`) is an open-source implementation of a
persistent graph world model. It separates long-term knowledge (a
persistent graph), transient reasoning (a working graph), and
learning dynamics (a fast-weight kernel) under a single
information-theoretic variational objective.

The repository ships the library, the training infrastructure, the
experiment runners, and a multi-hour supervised reproduction on
PROTEINS.

---

## Why

Modern neural networks conflate three roles into one parameter
tensor: long-term knowledge, transient reasoning, and learning
dynamics. That causes three persistent pathologies:

1. **Catastrophic forgetting** under continual learning.
2. **Unbounded parameter growth** as the system acquires new
   knowledge.
3. **Limited interpretability** of internal reasoning.

`pjepa` addresses them through a single variational objective and
a persistent graph that acts as the *evolved sufficient statistic*
of the observation history.

---

## Features

- **Persistent + Working Graphs** — `Graph` (immutable substrate),
  `State` (long-term knowledge with commit/reject audit trail),
  `Working` (budget-bounded retrieval view). See `pjepa.graphs`.
- **Dual-Geometric Encoder** — `Euclidean` (GIN-style MPNN) +
  `Hyperbolic` (Poincaré projection) composed via `DualGeometric`,
  all rooted in a polymorphic `Encoder` ABC. A separate
  `Predictor` / `Target` head pair (`Head` ABC) drives JEPA
  training.
- **Greedy Retrieval with (1 − 1/e) Guarantee** — `Retrieval`
  realises the Theorem 3 matroid-greedy bound; the utility
  hierarchy (`Utility` ABC) provides `Facility` (provably
  submodular) and `InfoGain` (information-gain with per-vertex
  cost). See `pjepa.retrieval`.
- **Hyperbolic Distortion Bound** — encoders and retrieval are
  dimensioned for the Proposition 7 hyperbolic vs Euclidean
  guarantee.
- **HRG / Bisimulation Rewriting** — `Criterion` ABC with the
  headline `FourConditions` verifier, plus HRG, bisimulation, and
  DPO drivers in `pjepa.rewriting`.
- **Sleep-Cadence Scheduler** — PPO trainer, `Buffer` replay
  storage (`Storage` ABC), and `Sleep` cadence (`Cadence` ABC) in
  `pjepa.scheduler`.
- **Variational Objective** — `𝒥` free-energy functional with
  information-bottleneck (IB) and minimum-description-length (MDL)
  terms in `pjepa.objectives`.
- **Performance Infra** — `safe_compile`, autocast, EMA, fused
  scatter, sync helpers in `pjepa.perf`.
- **Augmentation Suite** — DropEdge, DropNode, DropFeature,
  FeatureMask, RandomWalk, plus `Transform` / `Pipeline` (the
  polymorphic augmentation ABCs) and `TensorDropFeature` in
  `pjepa.augmentations`.
- **Training Stack** — pretrain / train / eval loops, SWA, TTA,
  Ensemble, Distillation, plus TU / CL / OGB runners in
  `pjepa.training`.
- **Baselines** — GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC,
  GEM, BGRL, GraphSAGE, PackNet, Naive.
- **8-Class Test Taxonomy** — 495 tests covering happy / bad /
  ugly / leaky / round-trip / cross-backend / distributional /
  property.
- **mkdocs --strict** — researcher, developer, and reference
  doc trees.
- **Real multi-hour training** —
  `experiments/train_real.py` is the canonical k-fold CV
  training script. The full PROTEINS run (3 seeds × 10 folds ×
  200 epochs × 2 methods, 60 fits, ~2 hours on a single CPU)
  ships in `results/proteins_full/`.

---

## Installation

Requires Python 3.10–3.12 (3.12 recommended).

```bash
git clone https://github.com/sachncs/pjepa.git
cd pjepa
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ogb]"
```

Or with Docker:

```bash
docker build -t pjepa .
docker run --rm pjepa pjepa doctor
```

The `setup.sh` script at the repository root is the canonical
environment contract — it creates the venv, installs the project
with `.[dev,ogb]`, verifies every CI tool (`pjepa`, `pytest`,
`ruff`, `pytype`, `pip-audit`, `mkdocs`, `python -m build`,
`optuna`), runs `pjepa doctor`, and executes the test suite.

---

## Quick Start

Verify the environment, then validate the headline paper claims
in seconds:

```bash
source .venv/bin/activate

pjepa doctor                           # capability probes
pjepa benchmark retrieval              # Theorem 3 — (1 - 1/e) approximation
pjepa benchmark distortion             # Proposition 7 — hyperbolic vs Euclidean
pjepa benchmark encoder-ablation       # Proposition 3 — dual-geometric advantage

pjepa train tu configs/tu.yaml         # TU SOTA
pjepa train cl configs/cl.yaml         # CL SOTA
pjepa train ogb configs/ogb.yaml       # OGB-arxiv
pjepa tune  tu configs/tu.yaml         # Optuna search

pjepa aggregate results                # collate every result under results/
```

Python API:

```python
import torch
from pjepa.graphs import Graph, State
from pjepa.encoders import Euclidean, DualGeometric, Hyperbolic, Predictor
from pjepa.retrieval import Retrieval, Facility, InfoGain

# Build a typed attributed graph and wrap it in a persistent state.
v = torch.randn((6, 8))
ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)
g = Graph(v, ei, torch.zeros((ei.shape[1], 2)))
state = State(graph=g)

# Dual-geometric encoder returns the concat of Euclidean + hyperbolic.
encoder = DualGeometric(input_dim=8, euclidean_dim=16, hyperbolic_dim=4, num_layers=2)
embedding = encoder.encode(g)            # shape (6, 20)

# Euclidean-only encoder returns a single tensor.
eu = Euclidean(input_dim=8, hidden_dim=16, num_layers=2, output_dim=16)
eu_embedding = eu(g)                     # shape (6, 16)

# JEPA predictor maps context embeddings to predicted target embeddings.
predictor = Predictor(input_dim=20, hidden_dim=32, output_dim=20)
prediction = predictor(embedding)

# Submodular working-graph retrieval with a facility-location utility.
retriever = Retrieval(budget=4)
result = retriever.select(g, torch.randn(4, 8), utility=Facility(g.vertex_features))
print(result.utility, result.iterations)
```

---

## Reproduction

The headline reproduction is a k-fold CV head-to-head of
`gin` vs `dual_geometric` on the PROTEINS dataset from
TUDataset. The full matrix is 3 seeds × 10 folds × 200 epochs ×
2 methods = 60 fits, ~2 hours on a single CPU.

```bash
source .venv/bin/activate

# Full multi-hour run (recommended overnight)
python experiments/train_real.py \
    --epochs 200 --seeds 3 --folds 10 \
    --methods gin dual_geometric \
    --output-dir results/proteins_full

# 30-minute smoke run for CI
python experiments/train_real.py \
    --epochs 100 --seeds 3 --folds 5 \
    --methods gin dual_geometric \
    --output-dir results/proteins
```

Latest results (3 seeds × 10 folds × 200 epochs, 60 fits):

| Method | Mean Accuracy | Std | N |
|---|---|---|---|
| `gin` (Xu et al. 2019) | **0.7901** | 0.0296 | 30 |
| `dual_geometric` | 0.7733 | 0.0337 | 30 |

GIN leads by 1.7 percentage points, well within one standard
deviation. Both numbers match the published Xu-et-al-2019 GIN
result on PROTEINS (~76–79%) within statistical fluctuation.
Per-fit table: `results/proteins_full/summary.csv`.

For the full TU / CL / OGB reproduction matrix, see
[`experiments/REPRODUCE.md`](experiments/REPRODUCE.md).

---

## Configuration

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| Compile mode | `PJEPA_SAFE_COMPILE` | `0` | Set to `1` to enable `safe_compile` |
| Autocast | `PJEPA_AUTOCAST` | `1` | Mixed-precision autocast toggle |
| Result root | `PJEPA_RESULTS_DIR` | `results` | Where `pjepa aggregate` writes tables |
| Preserve | `PRESERVE` | `0` | Set to `1` to keep `results/` during `cleanup.sh` |

See `configs/*.yaml` for the canonical TU / CL / OGB experiment
configs.

---

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `pjepa.graphs.Graph` | class | Typed attributed graph primitive |
| `pjepa.graphs.State` | class | Long-term knowledge container |
| `pjepa.graphs.Working` | class | Transient reasoning container |
| `pjepa.encoders.Encoder` | class (ABC) | Encoder polymorphic root |
| `pjepa.encoders.Euclidean` | class | Euclidean message-passing encoder |
| `pjepa.encoders.Hyperbolic` | class | Hyperbolic projection encoder |
| `pjepa.encoders.DualGeometric` | class | Composition of Euclidean + Hyperbolic |
| `pjepa.encoders.Head` | class (ABC) | Predictor / Target polymorphic root |
| `pjepa.encoders.Predictor` | class | JEPA predictor head |
| `pjepa.encoders.Target` | class | EMA target encoder |
| `pjepa.retrieval.Retrieval` | class | (1 − 1/e) matroid-greedy retrieval |
| `pjepa.retrieval.Utility` | class (ABC) | Retrieval-utility base class |
| `pjepa.retrieval.Facility` | class | Provably-submodular coverage utility |
| `pjepa.retrieval.InfoGain` | class | Information-gain utility with per-vertex cost |
| `pjepa.retrieval.Result` | dataclass | Return value of `Retrieval.select` |
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

## Benchmarks

Three cheap benchmarks validate the headline paper claims in
seconds:

| Benchmark | Claim | Cost |
|-----------|-------|------|
| `pjepa benchmark retrieval` | Theorem 3 — greedy retrieval achieves (1 − 1/e) ≈ 0.632 of optimum | ~1s |
| `pjepa benchmark distortion` | Proposition 7 — hyperbolic per-edge distortion is Θ(log D / (D log b)) | ~1s |
| `pjepa benchmark encoder-ablation` | Proposition 3 — dual-geometric beats Euclidean-only | ~30s |

Each benchmark prints a structured JSON summary to stdout.
`all_pass: true` means every row met its threshold; the rows use
brute-force optima where tractable and pseudo-optima otherwise.

---

## Documentation

The full docs site is built with `mkdocs --strict` and lives under
`site/` after `make docs`. Three audiences:

- **Researchers** — [`docs/researcher/01_persistent_graph_world_model.md`](docs/researcher/01_persistent_graph_world_model.md)
  for a deep-dive explanation of the framework.
- **Developers** — [`docs/developer/01_quickstart.md`](docs/developer/01_quickstart.md)
  for installation, first experiments, and extension tutorials.
- **API reference** — [`docs/reference/api.md`](docs/reference/api.md)
  (auto-generated from docstrings).

Serve locally with `mkdocs serve`.

---

## Project Structure

```
pjepa/
├── docs/                       # mkdocs source (researcher/, developer/, reference/, paper/)
├── experiments/                # Runnable experiment scripts + train_real.py
├── src/pjepa/                  # The library
│   ├── graphs/                 # Graph, State, Working
│   ├── encoders/               # Encoder (ABC), Euclidean, Hyperbolic, DualGeometric, Head/Predictor/Target
│   ├── retrieval/              # Retrieval, Utility (ABC), Facility, InfoGain, Result
│   ├── rewriting/              # HRG, Bisimulation, Criterion (ABC), FourConditions
│   ├── scheduler/              # PPOTrainer, Buffer, Storage (ABC), Cadence/Sleep
│   ├── objectives/             # FreeEnergy, ib_lagrangian, description_length
│   ├── dynamics/               # EvolutionOperator, contractivity_bound, fixed_point_iteration
│   ├── augmentations/          # Transform (ABC), Pipeline, DropEdge, DropNode, …
│   ├── training/               # pretrain/train/eval loops, SWA, TTA, Ensemble, Distillation
│   ├── eval/                   # metrics, bootstrap CI, statistical tests, aggregator
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

# Run the test suite
pytest                              # 495 tests
pytest -m "not slow"                # skip slow tests

# Lint, format, docs
ruff check src/ tests/
ruff format src/ tests/
mkdocs build --strict

# Build the distribution artefacts
python -m build
```

The `cleanup.sh` script at the repository root removes the venv,
build artefacts, type-checker caches, the mkdocs site, Python
bytecode caches, and `results/` (preserved when `PRESERVE=1`).

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
| Testing | pytest |
| Container | Docker |
| Citation | CITATION.cff |

---

## Roadmap

- **v1.0.0** — Current. Library, CLI dispatcher, experiment runners,
  aggregator, mkdocs strict docs, package artefacts, full Google-style
  docstrings on every public symbol, polymorphic ABC roots for every
  major hierarchy, and the headline PROTEINS reproduction.
- **v1.1.0** — Distributed (multi-GPU) training; persistent-state
  compression.
- **v2.0.0** — External release (Docker push, GH Release, PyPI
  upload, RTD trigger).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[`docs/developer/01_quickstart.md`](docs/developer/01_quickstart.md)
for the workflow. All public symbols need Google-style docstrings;
tests must cover the eight-class taxonomy.

## Code of Conduct

Contributors are expected to follow the
[Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Security

Report vulnerabilities to **sachncs@gmail.com** — see
[SECURITY.md](SECURITY.md).

## Citation

If you use `pjepa` in academic work, please cite the paper. The
BibTeX entry is in [CITATION.cff](CITATION.cff).

## License

[Apache-2.0](LICENSE) © 2026 Sachin
