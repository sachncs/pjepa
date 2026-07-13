# Encoders Module

> Graph-to-embedding maps, dual-geometric for hierarchy + locality.

The framework uses a *dual-geometric* representation: each vertex carries both an Euclidean embedding (for locality-preserving message passing) and a hyperbolic embedding (for hierarchical structure).

## Why dual-geometric?

Trees and tree-like structures (ASTs, scope hierarchies, inheritance chains) are *exponentially* easier to embed in hyperbolic space than in Euclidean space. Sarkar (2011) proves every tree has a Delaunay embedding into $\mathbb{H}^2$ with $O(\log D)$ per-edge distortion, while any fixed-dim Euclidean embedding requires $\Omega(D \log b)$ per-edge distortion (Linial-London-Rabinovich 1995).

The combined ratio is $\Theta(\log D / (D \log b))$ — polynomial, *widening* with depth.

## Architecture

The dual-geometric encoder concatenates:

1. An **Euclidean MPNN** (`EuclideanMPNN`) producing per-vertex Euclidean embeddings $e_v \in \mathbb{R}^{d_\text{euc}}$.
2. A **hyperbolic projection** (`HyperbolicProjection`) mapping $e_v$ into the Poincaré ball of curvature $-c$, producing $h_v \in \mathbb{H}^{d_\text{hyp}}$.

For tasks that don't benefit from hierarchy, the hyperbolic component can be ablated (see `docs/developer/04_adding_a_baseline.md` for the ablation runner).

## Implementations

| Class | Purpose |
|---|---|
| `Encoder` (Protocol) | The interface every encoder must satisfy: `forward(graph)`, `to(device)`. |
| `EuclideanMPNN` | GIN-style message passing with sum aggregation. |
| `HyperbolicProjection` | Tanh-map into the Poincaré ball with norm clamping. |
| `DualGeometricEncoder` | Euclidean + hyperbolic concatenated representation. |
| `JEPAPredictor` | Predictor head for the JEPA objective. |
| `TargetEncoder` | BYOL-style EMA target encoder. |

## Numerical Stability

`HyperbolicProjection` enforces:

* Clamped norms strictly below `1 - 1e-5` (Poincaré ball boundary).
* `log1p`-based precision in the tanh map.
* Float64 on MPS (where supported) for the bisimulation metric.

If you observe NaN/Inf during training, the curvature $c$ is likely too high; reduce it in `configs/default.yaml`.

## Adding a New Encoder

See [`docs/developer/03_adding_an_encoder.md`](../developer/03_adding_an_encoder.md) for a worked example (Spectral GCN).

## Where to Look Next

* [Retrieval](retrieval.md) — uses encoder embeddings to score vertices.
* [Verified Rewriting](verified-rewriting.md) — uses bisimulation metric on encodings.
* [Free Energy 𝒥](free-energy.md) — the unified objective that governs every component.