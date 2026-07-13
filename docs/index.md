# Persistent-JEPA

**Persistent-JEPA** (`pjepa`) is a production-grade, open-source implementation of a persistent graph world model for continual developmental learning. It separates long-term knowledge (a persistent graph), transient reasoning (a working graph), and learning dynamics (a fast-weight kernel), all governed by a single information-theoretic variational objective.

## What problem does this solve?

Modern neural networks conflate three roles into one parameter tensor: long-term knowledge, transient reasoning, and learning dynamics. This causes three persistent pathologies:

1. **Catastrophic forgetting** under continual learning.
2. **Unconstrained parameter growth** as the system acquires new knowledge.
3. **Limited interpretability** of internal reasoning.

Existing remedies — replay buffers, retrieval-augmented networks, parameter isolation, neuro-symbolic modules — address these in isolation. **Persistent-JEPA unifies them** through a single variational objective and a persistent graph that acts as the *evolved sufficient statistic* of the observation history.

## Quick start

```bash
git clone https://github.com/sachncs/persistent-jepa.git
cd persistent-jepa
make install
make doctor
make bench-retrieval    # validates the (1 - 1/e) retrieval guarantee
make bench-distortion   # validates the hyperbolic distortion bound
make bench-encoder      # validates the encoder ablation (Proposition 3)
make aggregate          # collates results under results/
```

## Documentation

The documentation is organised by audience:

* **For researchers** — start at [Persistent Graph World Model](researcher/01_persistent_graph_world_model.md) for a deep-dive explanation of the framework.
* **For developers** — start at [Quickstart](developer/01_quickstart.md) for installation and first experiments.
* **API reference** — see [Reference](reference/api.md) (auto-generated from docstrings).

## Project status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold | ✅ Complete |
| 1 | Core library + 8-class tests | ✅ Complete |
| 2 | Performance infrastructure | ✅ Complete |
| 3 | Augmentation suite | ✅ Complete |
| 4 | Validation experiments (retrieval + distortion + encoder ablation) | ✅ Complete |
| 5 | Training infrastructure (SWA, TTA, ensemble, distillation) | ✅ Complete |
| 6 | Optuna hyperparameter search | ✅ Complete |
| 7 | Baselines + target verification (8 baselines) | ✅ Complete |
| 8 | TU SOTA runner (6 datasets × 7 methods × 5 seeds) | ✅ Complete |
| 9 | CL SOTA runner (3 datasets × 5 methods × 5 seeds) | ✅ Complete |
| 10 | OGB-arxiv runner | ✅ Complete |
| 11 | Decoupling measurement + ablations | ✅ Complete |
| 12 | Reporting + 1.0.0 release (local) | ✅ Complete |
| 12 | Reporting + 1.0.0 release (external: Docker push, GH Release, PyPI, RTD) | ⏭ Out of local scope |

See the [changelog](changelog.md) for what shipped when and the
honest split between local- and external-release actions.

## Contributing

We welcome contributions. Please read
[CONTRIBUTING.md on GitHub](https://github.com/sachncs/persistent-jepa/blob/master/CONTRIBUTING.md)
for the workflow, then check the issues. All public symbols need
Google-style docstrings; tests follow the eight-class taxonomy.

## Citation

If you use `pjepa` in academic work, please cite the paper. The
BibTeX entry is in [CITATION.cff on GitHub](https://github.com/sachncs/persistent-jepa/blob/master/CITATION.cff).

## License

Apache 2.0. See [LICENSE on GitHub](https://github.com/sachncs/persistent-jepa/blob/master/LICENSE).