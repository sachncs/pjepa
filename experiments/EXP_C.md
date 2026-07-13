# Experiment C — Encoder Ablation

> **Validates**: Proposition 3 (Hierarchical Consistency) — the Dual-Geometric encoder (Euclidean + Hyperbolic) is at least as accurate as either single-geometry encoder on hierarchical structure.
> **Scope**: Synthetic AST-like graphs of varying depth.
> **Wall-clock (smoke)**: ~2 s on M3 Pro.

## What

We train three encoder variants on a collection of synthetic b-ary
trees with one-hot depth features and report their held-out accuracy
on a vertex-level depth-prediction task:

1. **`EuclideanMPNN`** — a three-layer GIN-style MPNN operating in `R^d`.
2. **`HyperbolicMPNN`** — the same MPNN followed by a `HyperbolicProjection`
   into the Poincaré ball.
3. **`DualGeometricEncoder`** — the canonical pjepa encoder that emits
   both Euclidean and hyperbolic components.

The "structural-prediction" task is to predict each vertex's depth from
its graph embedding. We split the AST graphs into disjoint train and
test sets so that the encoder must generalise across structurally
distinct trees rather than memorising a single tree.

## How

```bash
# Smoke defaults: D ∈ {5, 10}, 6 graphs, 2 seeds, 30 epochs.
PYTHONPATH=src python experiments/run_exp_c_encoder_ablation.py \
    --output-dir results/exp_c_smoke

# Plan-compliant sweep: D ∈ {5, 10, 15, 20}, 8 graphs, 3 seeds, 50 epochs.
PYTHONPATH=src python experiments/run_exp_c_encoder_ablation.py \
    --depths 5 10 15 20 --n-graphs 8 --seeds 3 --epochs 50 \
    --output-dir results/exp_c_full
```

## Outputs

* `encoder_ablation.csv` — per-trial `(depth, seed, encoder,
  n_train_graphs, n_test_graphs, n_vertices_per_graph, accuracy)`.
* `encoder_ablation.png` — mean ± std held-out accuracy vs AST depth,
  one line per encoder, with the per-depth random-chance baseline.

## Held-out Evaluation

For each depth `D` we generate `n_graphs` ASTs (default 6) with the
same branching factor but distinct seeds; we hold out 25% of them for
evaluation and train on the remainder. The encoder must learn
topology-aware depth prediction rather than memorising a single
graph's features.

## Result Interpretation

A run is considered "pass" when the Dual-Geometric encoder achieves a
mean held-out accuracy at least as high as either single-geometry
encoder over the depth sweep. Because the Dual encoder emits both
Euclidean and hyperbolic representations, it is expected to dominate
each component alone on hierarchical tasks; the experiment provides
an empirical lower bound on that advantage.

## Practical Defaults

The default invocation runs 12 trials (2 depths × 2 seeds × 3 encoders)
and completes in a couple of seconds. The number of vertices per
graph grows as `2^{D+1} − 1`; for `D = 20` this is over one million
vertices, so the plan-compliant sweep uses small graphs and shallow
depths unless `--epochs` is increased proportionally.
