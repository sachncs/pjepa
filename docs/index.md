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
| 1 | Core library + tests | ✅ Complete (182 tests passing) |
| 2 | Performance infrastructure | Pending |
| 3 | Augmentation suite | ✅ Complete |
| 4 | Validation experiments | ✅ Complete (retrieval + distortion + encoder ablation) |
| 5 | Training infrastructure (SWA, TTA, ensemble, distillation) | Pending |
| 6 | Optuna hyperparameter search | Pending |
| 7 | Baselines + target verification | ✅ Complete (8 baselines) |
| 8 | TU SOTA (6 datasets × 7 methods × 5 seeds) | Pending |
| 9 | CL SOTA (3 datasets × 5 methods × 5 seeds) | Pending |
| 10 | OGB-arxiv | Pending |
| 11 | Decoupling measurement + ablations | Pending |
| 12 | Reporting + 1.0.0 release | Pending |

See [CHANGELOG](changelog.md) for what shipped when.

## Contributing

We welcome contributions. Please read [CONTRIBUTING](CONTRIBUTING.md) (to be written) for the workflow, then check the issues. All public symbols need Google-style docstrings; tests follow the eight-class taxonomy.

## Citation

If you use `pjepa` in academic work, please cite the paper (citation details to be added at 1.0.0 release).

## License

Apache 2.0. See [LICENSE](LICENSE).