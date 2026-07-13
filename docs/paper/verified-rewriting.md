# Verified Rewriting Module

> The engine that grows the persistent graph — with proof that every commit is safe.

## Why Verified?

An unverified rewrite engine can silently corrupt the persistent graph. The framework's commitment is that **every committed change passes a four-conditions acceptance criterion**, and the verification is enforced *at runtime* via [`pjepa.exceptions.GraphError`](../reference/api.md#pjepaexceptions) and *at compile time* via the type system.

## The Four Conditions

A candidate $\widehat{G}_{t+1}$ is accepted iff all of:

1. **Variational descent**: $\Delta \mathcal{J} < 0$ (strict).
2. **Grammar conformance**: the rewrite is produced by a rule in the HRG.
3. **Behavioural bisimilarity**: $d_\sim(\widehat{G}_{t+1}, G_t) \le \varepsilon$.
4. **Bounded cost**: $\mathrm{DL}(\widehat{G}_{t+1}) - \mathrm{DL}(G_t) \le \eta$.

Conditions 1 and 4 are *information-theoretic* (they ensure strict descent of 𝒥 and bounded graph size). Conditions 2 and 3 are *structural* (they ensure the rewrite is well-formed and behaviour-preserving).

## Components

| Class | Description |
|---|---|
| `HRG(nonterminals, terminals, productions, start)` | Hyperedge-replacement grammar. |
| `HRGProduction(lhs, rhs_edge_index, rhs_edge_features)` | A single production rule. |
| `BisimulationMetric(epsilon, max_iters)` | Bisimulation metric configuration. |
| `bisimulation_distance(graph_a, graph_b, metric)` | Compute the pseudometric. |
| `FourConditions(beta_ib, lambda_mdl, gamma_forward, bisimulation_eps, max_cost)` | Acceptance thresholds. |
| `accept_candidate(candidate, current, observation, grammar, thresholds)` | Evaluate the four conditions. |
| `DPOConfig` / `dpo_loss` | Knowledge-distillation-style loss for the predictor. |

## Why HRG?

The hyperedge-replacement grammar is the *smallest* graph grammar class that supports:

1. **Edge substitution** (e.g., DFG replacement): a single production can replace a hyperedge with a subgraph.
2. **Node merging** via hyperedge fusion.
3. **Clean Double-Pushout (DPO) semantics** under the dangling condition.

These properties make the rewriting engine compositional, verifiable, and tractable.

## Why Behavioural Bisimulation?

Two graphs are *behaviourally bisimilar* if every observation (under the SSCG relation set $R$) produces the same response. Bisimulation is a *coarsening* of graph isomorphism that preserves semantic equivalence while being computable in $O(B^2)$ time.

## Implementation Notes

* `bisimulation_distance` uses the value-iteration Bellman operator with a per-vertex signature as the base distance.
* The runtime uses `float64` on CUDA and falls back to `float32` on MPS (which lacks `float64`).
* `accept_candidate` returns `(accepted, info)` where `info` contains per-condition values for diagnostics.

## Example

```python
from pjepa.rewriting import (
    HRG,
    HRGProduction,
    FourConditions,
    accept_candidate,
)

hrg = HRG(nonterminals=("S",), terminals=("t",), productions=(), start="S")
thresholds = FourConditions(bisimulation_eps=0.1)
accepted, info = accept_candidate(candidate, current, observation, hrg, thresholds)
assert info["reason"] in (
    "all four conditions satisfied",
    "cost exceeds max_cost",
    "delta_j is non-negative",
    "bisimilarity violated",
    "no grammar supplied",
)
```

## Where to Look Next

* [Graphs](graphs.md) — the substrate being rewritten.
* [Free Energy 𝒥](free-energy.md) — the optimisation target.
* [Dynamics](dynamics.md) — analyses the evolution of $G_t$.