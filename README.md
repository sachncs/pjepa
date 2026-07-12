# pjepa

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Code style](https://img.shields.io/badge/code%20style-google-blueviolet)](https://google.github.io/styleguide/pyguide.html)

**Persistent-JEPA** (`pjepa`) is a production-grade, open-source implementation of the Persistent Graph World Model for continual developmental learning described in our paper. The framework unifies predictive representation learning, continual learning via persistent sufficient-statistic memory, structural abstraction through verified graph rewriting, and a developmental scheduler, all under a single information-theoretic variational objective.

This repository contains the library, the training infrastructure, the benchmarks, and the reproducibility package.

## Quick Start

```bash
# Install (Python 3.12 recommended)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify your environment
pjepa doctor

# Pretrain on a TU dataset
pjepa pretrain configs/tu.yaml --dataset PROTEINS --seeds 5

# Train (supervised) and evaluate
pjepa train tu configs/tu.yaml --dataset MUTAG
pjepa eval  tu results/runs/<run-id>

# Continual learning
pjepa train cl configs/cl.yaml --dataset PROTEINS-CL5

# Validate paper claims
pjepa benchmark retrieval       # Theorem 3: (1 - 1/e) retrieval approximation
pjepa benchmark distortion      # Proposition 7: hyperbolic vs Euclidean distortion
pjepa benchmark encoder-ablation # Proposition 3: dual-geometric encoder
```

## Documentation

| Audience | Start here |
|---|---|
| **Researcher** (new to JEPA) | [docs/researcher/01_why_jepa.md](docs/researcher/01_why_jepa.md) → [docs/researcher/02_persistent_graph.md](docs/researcher/02_persistent_graph.md) |
| **Researcher** (familiar with JEPA) | [docs/researcher/03_free_energy.md](docs/researcher/03_free_energy.md) → [docs/researcher/04_verified_rewriting.md](docs/researcher/04_verified_rewriting.md) |
| **Developer** (new to pjepa) | [docs/developer/01_quickstart.md](docs/developer/01_quickstart.md) → [docs/developer/02_architecture.md](docs/developer/02_architecture.md) |
| **Developer** (extending pjepa) | [docs/developer/03_adding_an_encoder.md](docs/developer/03_adding_an_encoder.md) → [docs/developer/04_adding_a_baseline.md](docs/developer/04_adding_a_baseline.md) |
| **Reproducer** (verifying paper claims) | [docs/developer/05_reproducing_paper_results.md](docs/developer/05_reproducing_paper_results.md) |

The full paper (under `docs/paper/paper.md`) is the academic source.

## Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold | In progress |
| 1 | Core library + tests | Pending |
| 2 | Performance infra | Pending |
| 3 | Augmentation suite | Pending |
| 4 | Validation experiments | Pending |
| 5 | Training infra (SWA, TTA, ensemble, distillation) | Pending |
| 6 | Optuna hyperparameter search | Pending |
| 7 | Baselines + target verification | Pending |
| 8 | TU SOTA | Pending |
| 9 | CL SOTA | Pending |
| 10 | OGB-arxiv | Pending |
| 11 | Decoupling measurement + ablations | Pending |
| 12 | Reporting + 1.0.0 release | Pending |

See [CHANGELOG.md](CHANGELOG.md) for what shipped when.

## Contributing

We welcome contributions. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, then check the [open issues](../../issues) for current priorities. Code style is Google Python Style Guide; all public symbols need docstrings; tests must cover the eight-class taxonomy (happy / bad / ugly / leaky / round-trip / cross-backend / distributional / property).

## Citation

If you use `pjepa` in academic work, please cite the paper. The BibTeX entry is in [CITATION.cff](CITATION.cff).

## License

Apache 2.0. See [LICENSE](LICENSE).