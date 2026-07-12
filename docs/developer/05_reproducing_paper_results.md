# Reproducing Paper Results

> One-command reproduction of every experiment in the paper.

## Quick Start

The reproduction suite is exposed via the Makefile. Every target
re-runs an experiment with the *frozen* config from the corresponding
phase boundary.

```bash
# Cheap validations (< 1 minute each)
make bench-retrieval          # Validates Theorem 3
make bench-distortion         # Validates Proposition 7

# Full reproductions (longer)
make reproduce-tu             # 6 datasets × 7 methods × 5 seeds (~30 hours)
make reproduce-cl             # 3 datasets × 5 methods × 5 seeds (~15 hours)
make reproduce-ogb            # 5 methods × 3 seeds (~25 hours)
make reproduce-all            # Everything (~70 hours)
```

All results land in `results/`:

```
results/
├── tables/        # CSV + Markdown
├── plots/         # PDF + PNG (300 dpi)
├── metrics/       # Per-run CSV
├── checkpoints/   # Saved model weights
└── logs/          # Structured logs per run
```

## What Each Reproduce Target Does

### `make reproduce-tu`

Runs the **TU SOTA** experiment:

* **Datasets:** PROTEINS, MUTAG, NCI1, IMDB-B, REDDIT-B, DD
* **Methods:** GCN, GIN, GraphMAE, GraphCL, InfoGraph, naive
  fine-tune, Persistent-JEPA (ours)
* **Protocol:** 10-fold cross-validation, 5 seeds, linear-probe
  evaluation for self-supervised methods
* **Output:**
  * `results/tables/tu_summary.csv` — mean ± std per method per dataset
  * `results/plots/tu_radar.png` — radar chart
  * `results/plots/tu_heatmap.png` — accuracy heatmap

### `make reproduce-cl`

Runs the **CL SOTA** experiment:

* **Datasets:** PROTEINS-CL5, MUTAG-CL5, NCI1-CL5 (5-task
  class-incremental splits)
* **Methods:** Naive fine-tune, EWC, GEM, PackNet-style, Persistent-JEPA
* **Protocol:** Sequential training across 5 tasks; measure backward
  transfer (forgetting) and forward transfer
* **Output:**
  * `results/tables/cl_summary.csv` — per-method metrics
  * `results/plots/cl_forgetting_curves.png` — forgetting rate per task

### `make reproduce-ogb`

Runs the **OGB-arxiv** experiment:

* **Dataset:** ogbn-arxiv (169K nodes, 70 classes)
* **Methods:** GCN, GraphSAGE, BGRL, GraphMAE, Persistent-JEPA
* **Protocol:** Standard OGB evaluation; neighbour sampling for
  memory efficiency
* **Output:**
  * `results/tables/ogb_summary.csv` — test accuracy per method
  * Optional OGB submission via `pjepa eval ogb --submit`

### `make bench-retrieval` and `make bench-distortion`

Run the *cheap validation benchmarks*:

* **Retrieval:** generates 100 random monotone submodular functions,
  computes the greedy retrieval's ratio to the brute-force optimum,
  and verifies the ratio is ≥ (1 - 1/e) ≈ 0.632 in expectation.
* **Distortion:** generates b-ary trees of varying depths, computes
  hyperbolic and Euclidean embeddings, and verifies the ratio
  matches the Θ(log D / (D log b)) bound from Proposition 7.

These run in seconds and are the recommended **first sanity check**
after installing `pjepa`.

## Verifying Reproduction

After `make reproduce-tu`, check the headline table:

```bash
cat results/tables/tu_summary.csv
```

Expected: a CSV with rows for each (dataset, method) pair and columns
for mean accuracy, std, and confidence interval.

For SOTA claims, run the verification hook:

```bash
make verify-claims
```

This compares the reproduced numbers against the published targets in
`docs/paper/sota_targets.md` and exits non-zero if any number is more
than 2× bootstrap CI away from the claim.

## Troubleshooting

### Slow reproduction

If `make reproduce-tu` is running slower than expected, profile the
bottleneck:

```bash
pjepa profile --run-dir results/runs/<run_id>
```

The most common bottleneck on M3 Pro is **scatter_add_ on MPS**, which
falls back to CPU. Fix by upgrading to PyG 2.9+ or pinning PyG to a
version with native MPS support.

### OOM on OGB-arxiv

OGB-arxiv with neighbour sampling uses ~6 GB RAM. If you hit OOM:

* Reduce `num_neighbors` in `configs/ogb.yaml` (e.g. `[10, 5, 3]`).
* Reduce `batch_size` for the linear-probe stage.

### Numerical instability in hyperbolic encoder

If the hyperbolic encoder produces NaN during training, the curvature
`c` is too high. Reduce to `c = 0.5` in `configs/default.yaml`.

### Bisimulation metric too expensive

The bisimulation metric is O(B²). For `B > 256`, switch to the WL-test
proxy via `config.bisimulation.proxy = "wl"`.

## Resuming an Interrupted Run

Every run is checkpointed. To resume:

```bash
pjepa train tu configs/tu.yaml --resume results/runs/<run_id>
```

The trainer picks up from the last saved epoch and continues training.

## Customising the Reproduction

The configs in `configs/` are YAML and fully editable. Common
customisations:

* **Different number of seeds:** edit `n_seeds: 5` in
  `configs/tu.yaml`.
* **Different number of folds:** edit `n_folds: 10` in
  `configs/tu.yaml`.
* **Different optimizer:** edit `training.optimizer` and the
  hyperparameters below it.
* **Different encoder:** edit `model.encoder` to one of
  `euclidean_mpnn`, `hyperbolic`, `dual_geometric`,
  `gcn`, `gin`.

After editing, re-run the reproduction:

```bash
make reproduce-tu
```

## Where to Look Next

* [Quickstart](01_quickstart.md)
* [Architecture overview](02_architecture.md)
* [Adding a custom encoder](03_adding_an_encoder.md)
* [Adding a custom baseline](04_adding_a_baseline.md)