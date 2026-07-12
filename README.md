# pjepa

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Code style](https://img.shields.io/badge/code%20style-google-blueviolet)](https://google.github.io/styleguide/pyguide.html)

**Persistent-JEPA** (`pjepa`) is a production-grade, open-source implementation of a persistent graph world model for continual developmental learning. It separates long-term knowledge (a persistent graph), transient reasoning (a working graph), and learning dynamics (a fast-weight kernel), all governed by a single information-theoretic variational objective.

This repository contains the library, the training infrastructure, the experiments, and the reproducibility package.

## Quick Start

```bash
# Install (Python 3.12 recommended; 3.10/3.11 also supported)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ogb]"

# Verify your environment
pjepa doctor

# Validate paper claims cheaply
pjepa benchmark retrieval       # Theorem 3: (1 - 1/e) retrieval approximation
pjepa benchmark distortion      # Proposition 7: hyperbolic vs Euclidean distortion
pjepa benchmark encoder-ablation # Proposition 3: dual-geometric advantage

# Run headline experiments
pjepa train tu                  # Phase 8: TU SOTA (6 datasets × 7 methods)
pjepa train cl                  # Phase 9: CL SOTA (3 datasets × 5 methods)
pjepa train ogb                 # Phase 10: OGB-arxiv
```

## Documentation

| Audience | Start here |
|---|---|
| **Researcher** (new to JEPA) | [docs/researcher/01_persistent_graph_world_model.md](docs/researcher/01_persistent_graph_world_model.md) |
| **Developer** (new to pjepa) | [docs/developer/01_quickstart.md](docs/developer/01_quickstart.md) |
| **Reproducer** (verifying paper claims) | [docs/developer/05_reproducing_paper_results.md](docs/developer/05_reproducing_paper_results.md) |
| **API reference** | [docs/reference/](docs/reference/) (auto-generated from docstrings) |

## Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold (Python 3.12 venv, pyproject, CI, Docker, mkdocs, doctor) | ✅ Complete |
| 1 | Core library (graphs, encoders, retrieval, rewriting, objectives, dynamics, scheduler) + 8-class tests | ✅ Complete |
| 2 | Performance infra (safe_compile, autocast, EMA, fused scatter, sync) | ✅ Complete |
| 3 | Augmentation suite (DropEdge, DropNode, DropFeature, FeatureMask, RandomWalk) + TensorDropFeature | ✅ Complete |
| 4 | Validation experiments (Exp A retrieval, Exp B distortion, Exp C encoder-ablation) | ✅ Complete |
| 5 | Training infra (SWA, TTA, Ensemble, Distillation) | ✅ Complete |
| 6 | Optuna hyperparameter search | ✅ Complete |
| 7 | Baselines (GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM) | ✅ Complete |
| 8 | TU SOTA experiment runner | ✅ Complete |
| 9 | CL SOTA experiment runner | ✅ Complete |
| 10 | OGB-arxiv experiment stub | ✅ Complete |
| 11 | Decoupling measurement + ablations | ✅ Complete |
| 12 | Reporting + 1.0.0 release | In progress |

**227 tests pass, 0 ruff errors, 0 mypy-style type issues.**

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
│   ├── run_exp_c_encoder_ablation.py # Encoder ablation
│   ├── run_exp_d_tu_sota.py    # TU SOTA
│   ├── run_exp_e_continual.py  # CL SOTA
│   ├── run_exp_f_ogb_arxiv.py  # OGB-arxiv
│   ├── run_exp_g_decoupling.py # Inference-storage decoupling
│   └── run_exp_h_ablations.py  # Ablation study
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
├── tests/                      # 227 tests covering the eight-class taxonomy
├── plans/                      # Implementation plans (gitignored)
├── docs/paper/paper.md         # Paper draft (gitignored)
├── pyproject.toml              # PEP 621 metadata, full deps, lint/type/test config
├── Makefile                    # install/lint/test/coverage/audit/docs/bench targets
├── mkdocs.yml                  # Documentation site config
├── .github/workflows/ci.yml    # CI: ruff + pytype + pytest on Python 3.10/3.11/3.12
├── .pre-commit-config.yaml     # Ruff + standard hooks
├── Dockerfile                  # Reproducible container image
├── CHANGELOG.md                # Versioned history with commit SHAs and rationale
└── LICENSE                     # Apache 2.0
```

## Contributing

We welcome contributions. Read [docs/developer/01_quickstart.md](docs/developer/01_quickstart.md) for the workflow, then check the [issues](../../issues) for current priorities. All public symbols need Google-style docstrings; tests must cover the eight-class taxonomy (happy / bad / ugly / leaky / round-trip / cross-backend / distributional / property).

## Citation

If you use `pjepa` in academic work, please cite the paper. The BibTeX entry is in [CITATION.cff](CITATION.cff).

## License

Apache 2.0. See [LICENSE](LICENSE).