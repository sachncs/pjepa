"""GraphCL baseline (You et al., 2020).

Contrastive learning between two augmented views of the same graph.
Uses NT-Xent loss (the normalised temperature-scaled cross-entropy
estimator from Chen et al., 2020).

## Algorithm

Let ``z_a, z_b ∈ ℝ^{N × D}`` be the projections of two augmented
views through the encoder + projector; we concatenate them along the
batch axis (``2N`` rows) and compute the pairwise cosine-similarity
matrix scaled by ``1/τ``. The positive pair is the ``i``-th row of
``z_a`` matched against the ``i``-th row of ``z_b`` (and vice
versa); every other entry is treated as a negative. The
cross-entropy loss with this label assignment is the NT-Xent loss.

## Complexity

* :meth:`embed` — ``O(N * D)`` forward pass.
* :meth:`loss` — ``O(N² * D)`` for the pairwise similarity matrix
  plus ``O(N²)`` for the cross-entropy.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["GraphCL"]


class GraphCL(nn.Module):
    """Graph contrastive learning with NT-Xent loss.

    Attributes:
        hidden_dim: Width of the projection head.
        temperature: NT-Xent temperature parameter.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("GraphCL: dims must be positive")
        if temperature <= 0:
            raise ValueError(f"GraphCL: temperature must be positive; got {temperature}")
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temperature = temperature
        self.hidden_dim = hidden_dim

    def embed(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return the graph-level embedding (mean of vertex embeddings).

        Args:
            graph: The input graph.

        Returns:
            ``[1, hidden_dim]`` graph-level embedding.

        Raises:
            GraphError: When the graph is empty.
        """
        if graph.num_vertices() == 0:
            raise GraphError("GraphCL.embed: cannot embed an empty graph")
        h = self.encoder(graph.vertex_features)
        return h.mean(dim=0, keepdim=True)

    def loss(self, view_a: TypedAttributedGraph, view_b: TypedAttributedGraph) -> torch.Tensor:
        """Compute the NT-Xent loss between two augmented views.

        Args:
            view_a: The first augmented view.
            view_b: The second augmented view.

        Returns:
            Scalar NT-Xent loss across the ``2N`` projected
            embeddings.
        """
        z_a = self.projector(self.embed(view_a))
        z_b = self.projector(self.embed(view_b))
        z = torch.cat([z_a, z_b], dim=0)
        n = z.shape[0] // 2
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temperature
        labels = torch.cat([torch.arange(n) + n, torch.arange(n)])
        mask = torch.eye(2 * n, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        return F.cross_entropy(sim, labels)
