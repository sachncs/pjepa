# Graphs Module

> The substrate on which every other component operates.

## Architectural Rationale

The framework's central commitment is that **the persistent graph $G_t$ is the only object on which learning is deposited**. Every other component operates on a *bounded working graph* $W_t \subseteq G_t$ with $|V(W_t)| \le B$.

This document explains the design of the three primitives:

* [`TypedAttributedGraph`](#typedattributedgraph)
* [`PersistentState`](#persistentstate)
* [`WorkingGraph`](#workinggraph)

## `TypedAttributedGraph`

Immutable dataclass storing vertex features, edge index, edge features, optional labels, and optional global features. Frozen (`frozen=True`) to eliminate an entire class of bugs around in-place mutation.

### Invariants enforced at construction

1. Vertex features are 2-D `[N, d_v]`.
2. Edge index is `[2, E]` and `dtype == torch.long`.
3. Every edge endpoint lies within `[0, N)`.
4. Edge features are 2-D `[E, d_e]`.
5. Edge features row count matches edge index column count.
6. Vertex label count matches vertex count.

Any violation raises [`GraphError`](../reference/api.md#pjepaexceptions).

### Functional updates

* `with_features(**kwargs)` returns a new instance with the specified fields replaced and `version + 1`. Used by augmentations and rewrites.
* `subgraph(vertex_mask)` returns the vertex-induced subgraph.
* `to(device)` moves every tensor to the given device.

## `PersistentState`

Wrapper around the persistent graph with an immutable commit/reject audit trail.

### Invariants

1. The graph is the only state that survives across observations.
2. Every commit is logged with a timestamp and a cost.
3. Every reject is logged with a reason and a cost.
4. The wrapper never mutates in place; every operation returns a new instance.

### API

* `commit(candidate, cost, timestamp, delta_j=None)` — accept a candidate and append a `CommitRecord`. When `delta_j` is provided, it must be strictly negative.
* `reject(reason, cost)` — record a rejected candidate.
* `num_commits()` / `num_rejections()` — introspection.

## `WorkingGraph`

Bounded subgraph used by the kernel for inference. Enforces $|V(W_t)| \le B$ at construction.

### Why a wrapper?

The `WorkingGraph` wrapper guarantees the budget invariant at the type level: every operation that would violate the budget raises `GraphError` at the Python boundary, not silently at the GPU kernel.

## Example

```python
from pjepa.graphs import (
    TypedAttributedGraph,
    PersistentState,
    WorkingGraph,
)
import torch

# Build a persistent graph.
g = TypedAttributedGraph(
    vertex_features=torch.zeros((3, 2)),
    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
)

# Wrap it.
state = PersistentState(graph=g)

# Commit a candidate.
candidate = g.with_features(global_features=torch.zeros((4,)))
state2 = state.commit(candidate, cost=0.1, timestamp=0.0, delta_j=-0.5)
assert state2.num_commits() == 1
assert state2.graph.version == 1

# Build a working graph within the budget.
working = WorkingGraph(graph=g, budget=10)
assert working.utilisation() == 0.3
```

## Where to Look Next

* [Encoders](encoders.md) — how features are extracted from a `TypedAttributedGraph`.
* [Retrieval](retrieval.md) — how the working graph is selected.
* [Rewriting](verified-rewriting.md) — how the persistent graph is updated.