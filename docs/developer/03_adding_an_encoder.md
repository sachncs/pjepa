# Adding a Custom Encoder

> A worked example: a spectral GCN encoder.

## What is an Encoder?

An encoder maps a :class:`pjepa.graphs.TypedAttributedGraph` to an
embedding tensor. There are three flavours in the framework:

* **Per-vertex encoders** (EuclideanMPNN, HyperbolicProjection) emit a
  tensor of shape ``[N, output_dim]``.
* **Graph-level encoders** (GCN, GIN) emit a tensor of shape
  ``[1, output_dim]`` after pooling.
* **Dual encoders** (DualGeometricEncoder) emit a tuple
  ``(per_vertex_euclidean, per_vertex_hyperbolic)``.

The framework treats encoders as duck-typed via the
:class:`pjepa.encoders.base.Encoder` protocol. Any class with
``forward(graph)`` and ``to(device)`` is a valid encoder.

## Worked Example: Spectral GCN

A spectral GCN uses the eigendecomposition of the graph Laplacian to
define a convolution. Here is a minimal implementation:

```python
"""Spectral GCN encoder for pjepa."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian

from pjepa.graphs import TypedAttributedGraph

__all__ = ["SpectralGCN"]


def _laplacian_eigendecomp(edge_index: torch.Tensor, num_vertices: int, k: int) -> torch.Tensor:
    """Compute the k smallest non-trivial eigenvectors of the Laplacian."""
    import numpy as np
    from scipy.sparse.linalg import eigsh

    row, col = edge_index
    edge_weight = torch.ones(row.shape[0])
    L = to_scipy_sparse_matrix(torch.stack([col, row]), edge_weight, num_vertices)
    eigvals, eigvecs = eigsh(L.astype(float), k=k + 1, which="SM")
    return torch.tensor(eigvecs[:, 1:k + 1], dtype=torch.float32)  # skip first trivial


class SpectralGCN(nn.Module):
    """Spectral GCN using Laplacian eigenmaps as positional encodings.

    Attributes:
        output_dim: Output feature dimension.
        k: Number of eigenvectors to use as positional encodings.
    """

    output_dim: int = 64

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64, k: int = 16) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0 or k <= 0:
            raise ValueError("SpectralGCN: dims and k must be positive")
        self.input_proj = nn.Linear(input_dim + k, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.k = k
        self.cached_eigvecs: dict[int, torch.Tensor] = {}

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        if graph.num_vertices() == 0:
            return torch.zeros((0, self.output_dim))
        if graph.num_vertices() not in self.cached_eigvecs:
            self.cached_eigvecs[graph.num_vertices()] = _laplacian_eigendecomp(
                graph.edge_index, graph.num_vertices(), self.k
            )
        eigvecs = self.cached_eigvecs[graph.num_vertices()]
        h = torch.cat([graph.vertex_features, eigvecs], dim=-1)
        return self.mlp(self.input_proj(h))

    def to(self, device: torch.device) -> "SpectralGCN":
        """Move parameters to device."""
        return super().to(device)
```

## Tests

The encoder should pass the eight-class taxonomy:

```python
"""Tests for the SpectralGCN encoder."""

from __future__ import annotations

import pytest
import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph
from my_module import SpectralGCN


def test_happy_spectral_gcn_forward() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    )
    encoder = SpectralGCN(input_dim=4, hidden_dim=8, output_dim=4, k=4)
    out = encoder(g)
    assert out.shape == (5, 4)


def test_bad_spectral_gcn_zero_dim() -> None:
    with pytest.raises(ValueError):
        SpectralGCN(input_dim=0, hidden_dim=8)


def test_ugly_spectral_gcn_single_vertex() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((1, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    encoder = SpectralGCN(input_dim=4, hidden_dim=8, output_dim=4, k=1)
    out = encoder(g)
    assert out.shape == (1, 4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_spectral_gcn() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    ).to("mps")
    encoder = SpectralGCN(input_dim=4, hidden_dim=8, output_dim=4, k=2).to("mps")
    out = encoder(g)
    assert out.device.type == "mps"
```

## Register the Encoder

For Phase 2, the encoder registry will allow users to reference the
encoder by name in config files. For now, instantiate directly:

```python
from my_module import SpectralGCN
encoder = SpectralGCN(input_dim=10, hidden_dim=64, output_dim=128)
```

## Common Pitfalls

* **Forgetting `to(device)`:** the framework's hardware module requires
  every encoder to support `.to(device)`. Implement it even if you
  think users won't move the encoder.

* **Not handling empty graphs:** always check `graph.num_vertices() == 0`
  and return a tensor of the right shape with zero rows.

* **Forgetting dtype consistency:** when concatenating features with
  positional encodings, make sure dtypes match.

* **Caching the eigendecomposition:** the Laplacian is expensive; cache
  it keyed by vertex count. Be aware that graphs with the same vertex
  count but different edges will produce different Laplacians; key by
  edge signature if you need correctness.

## Where to Look Next

* [Adding a custom baseline](04_adding_a_baseline.md) — for SOTA comparison.
* [Architecture overview](02_architecture.md) — module dependency graph.
* [Eight-class test taxonomy](06_test_taxonomy.md) — what good tests look like.