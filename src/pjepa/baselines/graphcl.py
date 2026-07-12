"""GraphCL baseline (You et al., 2020).

Contrastive learning between two augmented views of the same graph.
Uses NT-Xent loss.
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
        """Return the graph-level embedding (mean of vertex embeddings)."""
        if graph.num_vertices() == 0:
            raise GraphError("GraphCL.embed: cannot embed an empty graph")
        h = self.encoder(graph.vertex_features)
        return h.mean(dim=0, keepdim=True)

    def loss(self, view_a: TypedAttributedGraph, view_b: TypedAttributedGraph) -> torch.Tensor:
        """Compute the NT-Xent loss between two augmented views."""
        z_a = self.projector(self.embed(view_a))
        z_b = self.projector(self.embed(view_b))
        z = torch.cat([z_a, z_b], dim=0)
        n = z.shape[0] // 2
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temperature
        labels = torch.cat([torch.arange(n) + n, torch.arange(n)])
        mask = torch.eye(2 * n, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        return F.cross_entropy(sim, labels)
