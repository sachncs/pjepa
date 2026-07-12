# Adding a Custom Baseline

> A worked example: implementing PNA (Principal Neighbourhood Aggregation).

## What is a Baseline?

A *baseline* is a published method that we re-implement for SOTA
comparison. The existing baselines (GCN, GIN, GraphMAE, GraphCL,
InfoGraph, EWC, GEM) live in `src/pjepa/baselines/`. They are
deliberately minimal: enough to reproduce the published accuracy on
the TU benchmarks, no more.

## Worked Example: PNA

PNA (Corso et al., NeurIPS 2020) is a multi-aggregator GNN that
combines mean, max, std, and sum aggregators with degree-scaled
amplifiers. It is competitive on OGB and a strong TU baseline.

```python
"""PNA baseline (Corso et al., 2020)."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import degree
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from pjepa.graphs import TypedAttributedGraph

__all__ = ["PNA"]


def _compute_log_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute the log-scaled degree for each vertex (the PNA input feature)."""
    deg = degree(edge_index[1], num_nodes=num_nodes).clamp(min=1).float()
    return torch.log(deg / deg.mean() + 1e-6)


class _PNALayer(nn.Module):
    """A single PNA layer."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 1, towers: int = 1) -> None:
        super().__init__()
        self.towers = towers
        # Four aggregators: mean, max, std, sum
        self.pre_mlp = nn.Linear(in_dim * 4 + edge_dim * 2 + 1, out_dim * towers)
        self.post_mlp = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg_log: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = x[src]
        # Aggregators
        mean_agg = torch.zeros_like(x)
        mean_agg.index_add_(0, dst, msg)
        cnt = torch.zeros(x.shape[0], device=x.device).index_add_(
            0, dst, torch.ones_like(msg[:, 0])
        ).clamp(min=1).unsqueeze(-1)
        mean_agg = mean_agg / cnt
        max_agg = torch.full_like(x, float("-inf"))
        max_agg = torch.scatter_reduce(
            torch.zeros_like(x), 0, dst.unsqueeze(-1).expand_as(msg), msg, reduce="amax", include_self=True
        )
        std_agg = torch.zeros_like(x)
        std_agg.index_add_(0, dst, (msg - mean_agg[dst]) ** 2)
        std_agg = (std_agg / cnt).sqrt()
        sum_agg = torch.zeros_like(x)
        sum_agg.index_add_(0, dst, msg)
        # Concatenate and apply MLP
        h = torch.cat([mean_agg, max_agg, std_agg, sum_agg, deg_log], dim=-1)
        return self.post_mlp(torch.relu(self.pre_mlp(h)))


class PNA(nn.Module):
    """Principal Neighbourhood Aggregation baseline.

    Attributes:
        num_layers: Number of PNA layers.
        hidden_dim: Width of each PNA layer.
        num_classes: Output dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0 or num_classes <= 0:
            raise ValueError("PNA: dims must be positive")
        self.input_proj = nn.Linear(input_dim + 1, hidden_dim)
        self.layers = nn.ModuleList([_PNALayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        if graph.num_vertices() == 0:
            return torch.zeros((1, self.num_classes))
        deg_log = _compute_log_degrees(graph.edge_index, graph.num_vertices()).unsqueeze(-1)
        h = self.input_proj(torch.cat([graph.vertex_features, deg_log], dim=-1))
        for layer in self.layers:
            h = torch.relu(layer(h, graph.edge_index, deg_log))
        device = h.device
        batch = torch.zeros(h.shape[0], dtype=torch.long, device=device)
        pooled = global_add_pool(h, batch)
        return self.classifier(pooled)

    def embed(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return the pooled graph embedding without the classifier."""
        deg_log = _compute_log_degrees(graph.edge_index, graph.num_vertices()).unsqueeze(-1)
        h = self.input_proj(torch.cat([graph.vertex_features, deg_log], dim=-1))
        for layer in self.layers:
            h = torch.relu(layer(h, graph.edge_index, deg_log))
        device = h.device
        batch = torch.zeros(h.shape[0], dtype=torch.long, device=device)
        return global_add_pool(h, batch)
```

## Tests

```python
"""Tests for the PNA baseline."""

from __future__ import annotations

import pytest
import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph
from my_module import PNA


def test_happy_pna_forward() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    )
    model = PNA(input_dim=4, hidden_dim=8, num_layers=2, num_classes=2)
    out = model(g)
    assert out.shape == (1, 2)


def test_bad_pna_zero_dim() -> None:
    with pytest.raises(ValueError):
        PNA(input_dim=0)


def test_ugly_pna_empty_graph() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    model = PNA(input_dim=4, hidden_dim=8, num_layers=2, num_classes=2)
    out = model(g)
    assert out.shape == (1, 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_pna_forward() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    ).to("mps")
    model = PNA(input_dim=4, hidden_dim=8, num_layers=2, num_classes=2).to("mps")
    out = model(g)
    assert out.device.type == "mps"


def test_property_pna_output_shape() -> None:
    for num_vertices in [1, 10, 100]:
        g = TypedAttributedGraph(
            vertex_features=torch.randn((num_vertices, 4)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        model = PNA(input_dim=4, hidden_dim=8, num_layers=2, num_classes=2)
        out = model(g)
        assert out.shape == (1, 2)
```

## Register the Baseline

For SOTA comparison, add the baseline to the experiment runner in
`experiments/run_exp_d_tu_sota.py` (to be created in Phase 8). For now,
just make sure the baseline imports cleanly and passes its tests:

```python
from pjepa.baselines import PNA  # noqa: F401
```

## Common Pitfalls

* **The degree feature:** PNA expects a per-vertex log-degree feature.
  Compute it once, cache it (PNA's effectiveness depends on this).

* **The four aggregators:** PNA uses mean, max, std, sum in *every*
  layer. Don't skip one. Std is the most expensive (requires
  mean first) but contributes meaningfully.

* **Towers and scalers:** The original PNA paper uses degree-scaled
  amplifiers; the implementation above is simplified. For a faithful
  reproduction, use the [official PNA implementation](https://github.com/lcsigilli/PNA).

* **Empty graphs:** PNA's degree computation can fail on a graph with
  no edges. The check `graph.num_vertices() == 0` is essential.

## Where to Look Next

* [Adding a custom encoder](03_adding_an_encoder.md)
* [Reproducing paper results](05_reproducing_paper_results.md)
* [Eight-class test taxonomy](06_test_taxonomy.md)