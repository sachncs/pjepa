# Experiment B — Hyperbolic vs Euclidean Distortion

> **Validates**: Proposition 7 (Sarkar's construction embeds trees in `H²` with bounded distortion; Bourgain-style multi-scale projection into `R^d` has distortion that grows like `O(log D / log b)`).
> **Scope**: Synthetic b-ary trees of varying depth and branching.
> **Wall-clock (smoke)**: < 1 s on M3 Pro.

## What

We embed every b-ary tree of depth `D` and branching `b ∈ {2, 4}` in
both:

1. **Hyperbolic `H²`** via Sarkar's deterministic Delaunay construction.
   The root is placed at the origin; each child is placed at hyperbolic
   distance `α = 2 · asinh(1)` from its parent along the angular slot
   that the parent assigns it. The construction guarantees bounded
   distortion on trees.
2. **Euclidean `R^d`** via Bourgain-style multi-scale projection. Each
   level `k` contributes a random unit vector per "slot" (vertex at that
   level); each vertex's embedding is the weighted sum of its ancestor
   slots' unit vectors with weight `1 / (k + 1)`.

For each edge `(u, v)` we compute the per-edge Euclidean distance
`d_E(u, v)` and per-edge hyperbolic distance `d_H(u, v)`, then the
per-edge ratio `r_e = d_E(u, v) / d_H(u, v)`. The ratio is bounded
above by 1 for our parameterisation — Bourgain-style *under*-stretches
deep edges, so Euclidean distances are tighter than hyperbolic ones on
deep edges. The trend of the ratio as a function of `D` is what
characterises the embedding: the mean ratio decays like `log D / (D log b)`
as `D → ∞`, while the max ratio is dominated by the constant
root-to-child edge length.

## How

```bash
# Smoke defaults: D ∈ {3, 5, 7}, b ∈ {2, 4}, d ∈ {4, 8}, 2 seeds.
PYTHONPATH=src python experiments/run_exp_b_distortion.py \
    --output-dir results/exp_b_smoke

# Plan-compliant sweep: D ∈ {5, 10, 20, 50, 100}, b ∈ {2, 4}, d ∈ {4, 8, 16, 32}, 5 seeds.
PYTHONPATH=src python experiments/run_exp_b_distortion.py \
    --depths 5 10 20 50 100 --branchings 2 4 --dims 4 8 16 32 --seeds 5 \
    --output-dir results/exp_b_full
```

## Outputs

* `distortion.csv` — per-trial `(depth, branching, d, seed, hyp_*,
  euc_*, euc_over_hyp_mean, euc_over_hyp_max)`.
* `distortion.png` — two-panel plot of mean and max per-edge
  `d_E / d_H` ratios vs tree depth, one line per embedding dimension,
  grouped by branching factor.

## Distortion Statistics Reported

For every `(b, d, D, seed)` cell we record, per edge `(u, v)`:

* `d_H(u, v)` — hyperbolic geodesic distance on the Poincaré disk,
  computed in closed form via `arcosh`.
* `d_E(u, v)` — Euclidean distance in `R^d`.
* `r_e = d_E / d_H` — the per-edge ratio.

For the (b, d, D) cell as a whole we report the mean, standard
deviation, min, and max of each distribution. The CSV also records the
per-edge ratio's mean and max, which are the entries that carry
distortion information across depth.

## Practical Defaults

The default invocation runs 24 trials (2 branchings × 3 depths × 2
dimensions × 2 seeds) on trees of up to 2¹⁵ = 32 768 vertices (depth 7,
branching 4) and completes in a few seconds. Larger depths (50, 100)
can be explored with the CLI; the hyperbolic embedding remains
numerically stable up to the Poincaré-disk curvature limit.
