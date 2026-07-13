# Retrieval Module

> Working-graph selection via submodular maximisation.

The retriever selects a *fixed-budget* vertex-induced subgraph $W_t \subseteq G_t$ with $|V(W_t)| \le B$ that maximises a monotone submodular utility.

## Why Submodular?

The framework's retrieval utility (typically facility-location) is **monotone submodular**: adding a vertex to a smaller subset yields more marginal gain than adding it to a larger one. Submodular functions admit the Nemhauser-Wolsey-Fisher 1978 guarantee:

$$\text{Greedy}(W) \;\ge\; \left(1 - \frac{1}{e}\right) \text{OPT}(W) \;\approx\; 0.632 \cdot \text{OPT}(W).$$

This is the best possible polynomial-time approximation for general submodular maximisation.

## Implementations

| Class | Description |
|---|---|
| `RetrievalUtility` (Protocol) | `__call__(vertex_subset, observation) -> float`. |
| `FacilityLocationUtility` | Provably submodular: $f(W) = \sum_i \max_{v \in W} \text{sim}(v, i)$. |
| `InformationGainUtility` | Information-gain proxy with per-vertex cost. |
| `GreedyRetrieval(budget)` | Algorithm 1 of the paper. |

## Algorithm 1

For each iteration $k = 1, \ldots, B$:

1. For each candidate vertex $v \notin W^{(k-1)}$, compute the marginal gain $f(W^{(k-1)} \cup \{v\}) - f(W^{(k-1)})$.
2. Add the vertex with the largest marginal gain.
3. Stop when the marginal gain is non-positive (or the budget is exhausted).

Complexity: $O(B \cdot n \cdot \text{utility\_eval})$.

## Validation

The `tests/test_retrieval.py::test_one_minus_one_over_e_on_synthetic` test verifies the $(1 - 1/e)$ guarantee against brute-force optimum on 8-vertex facility-location problems.

```python
from pjepa.retrieval import GreedyRetrieval, FacilityLocationUtility

util = FacilityLocationUtility(vertex_features=features)
result = GreedyRetrieval(budget=8).select(graph, observation, utility=util)
assert result.utility >= (1 - 1 / e) * opt - 1e-5
```

## Decoupling from Storage

The retrieval utility is computed only over the *persistent graph's* vertex features. The working graph $W_t$ is then sampled. Because retrieval runs in $O(Bn)$ time and inference runs in $O(B^2 d)$, the cost is dominated by $B$, not $n$. This is the operational form of *inference–storage decoupling* (paper §3).

See `experiments/run_exp_g_decoupling.py` for the measurement (source-tree file outside the docs site).

## Where to Look Next

* [Verified Rewriting](verified-rewriting.md) — operates on the persistent graph after retrieval.
* [Encoders](encoders.md) — produces the embeddings retrieval uses.
* [Free Energy 𝒥](free-energy.md) — provides the optimisation target.