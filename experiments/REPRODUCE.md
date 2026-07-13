# Reproduction Guide

> One-command reproduction of every experiment in the paper.

## Quick Reference

| Command | What it runs | Wall-clock (M3 Pro) |
|---|---|---|
| `make bench-retrieval` | Validates Theorem 3 (1-1/e retrieval) | < 1 second |
| `make bench-distortion` | Validates Proposition 7 (hyperbolic distortion) | < 1 second |
| `make bench-encoder` | Validates Proposition 3 (encoder ablation) | ~30 seconds |
| `make reproduce-tu` | TU SOTA (6 datasets × 7 methods × 5 seeds × 10 folds) | ~30 hours |
| `make reproduce-cl` | CL SOTA (3 datasets × 5 methods × 5 seeds × 5 tasks) | ~15 hours |
| `make reproduce-ogb` | OGB-arxiv (5 methods × 3 seeds) | ~25 hours |
| `make reproduce-all` | Everything (sequential) | ~70 hours |

## Smoke Tests

Before running the full reproduction, validate your environment:

```bash
# Verify all 6 capability probes are GREEN
pjepa doctor

# Run the cheap validations
pjepa benchmark retrieval       # validates (1 - 1/e)
pjepa benchmark distortion      # validates Sarkar/LLR bounds
pjepa benchmark encoder-ablation # validates Hierarchical Consistency
```

Each benchmark prints a structured JSON summary. All should report `all_pass: true` (or `accuracy > 0` for the encoder ablation).

## TU SOTA Reproduction

```bash
# Default: 5 seeds, 10 folds, 6 datasets
pjepa train tu configs/tu.yaml

# Faster smoke version (1 seed, 3 folds, 1 dataset)
python experiments/run_exp_d_tu_sota.py --datasets MUTAG --seeds 1 --folds 3 --epochs 100
```

Outputs:
- `results/tu/tu_results.csv` — per-fold results
- `results/tu/tu_summary.csv` — mean ± std aggregated per (dataset, method)

## CL SOTA Reproduction

```bash
pjepa train cl configs/cl.yaml

# Or run directly:
python experiments/run_exp_e_continual.py --datasets PROTEINS --n-tasks 5 --seeds 3
```

Outputs:
- `results/cl/cl_results.csv` — per-task accuracies

## OGB-arxiv Reproduction

```bash
pjepa train ogb configs/ogb.yaml

# Or run directly:
python experiments/run_exp_f_ogb_arxiv.py --seeds 3 --epochs 100
```

Outputs:
- `results/ogb/ogb_results.csv` — per-seed test accuracies

## Ablation Study

```bash
python experiments/run_exp_h_ablations.py --dataset MUTAG --seeds 3 --folds 5
```

Outputs:
- `results/ablation.csv` — accuracy per ablation variant

## Optuna Hyperparameter Search

```bash
python experiments/run_optuna_search.py --datasets PROTEINS MUTAG --n-trials 20
```

Outputs:
- `results/optuna/<dataset>/best_config.yaml` — best hyperparameters per dataset

## Inference-Storage Decoupling

```bash
python experiments/run_exp_g_decoupling.py
```

Outputs:
- `results/decoupling.csv` — wall-clock vs persistent graph size

## Resuming an Interrupted Run

The runners are not checkpointed; re-running them produces fresh outputs. If you need to resume, modify the experiment runner's loop variables or use the `--seeds` and `--folds` arguments to skip completed portions.

## Verifying Reproduced Numbers

After `make reproduce-tu`, check the headline table:

```bash
cat results/tu/tu_summary.csv
```

Compare against the published targets in [`docs/paper/paper.md`](../paper/paper.md). The framework's SOTA claim is honest reporting; if our numbers are below the published baseline, we publish the gap with a full ablation table to diagnose.

## Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 18 GB+ (M3 Pro) |
| Accelerators | CPU only | MPS / CUDA |
| Disk | 5 GB | 20 GB+ (for datasets + checkpoints) |
| Wall-clock | 70 hours | N/A (compute is the bottleneck) |

On MPS (Apple Silicon), expect 2-3× slower than a modern NVIDIA GPU for the GCN/GIN baselines and 5-10× slower for the message-passing-heavy methods.

## Troubleshooting

### "Out of memory" on OGB-arxiv

Reduce `num_neighbors` in `configs/ogb.yaml` (e.g., `[10, 5, 3]` instead of `[15, 10, 5]`).

### "NaN" during hyperbolic encoder training

Reduce the curvature $c$ from `1.0` to `0.5` in `configs/default.yaml`.

### Submodular utility property test fails

This indicates a bug in your utility implementation. Check that
`f(S ∪ {v}) - f(S) >= f(T ∪ {v}) - f(T)` for $S \subseteq T$.

### Bisimulation metric too expensive

Set `configs/<dataset>.yaml::pjepa.bisimulation.proxy = "wl"` to use the
Weisfeiler-Leman proxy (much faster, slightly weaker).

## Where to Look Next

* [Quickstart for Developers](../developer/01_quickstart.md)
* [Eight-Class Test Taxonomy](../developer/06_test_taxonomy.md)
* [Persistent Graph World Model](../researcher/01_persistent_graph_world_model.md)