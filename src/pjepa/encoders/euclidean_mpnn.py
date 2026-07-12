"""Euclidean message-passing neural network encoder.

A simple GIN-style MPNN (Xu et al. 2019) operating on the Euclidean
component of the persistent graph. Returns per-vertex embeddings.
"""

from __future__ import annotations

import torch

from torch import nn

from pjepa.graphs import TypedAttributedGraph

__all__ = ["EuclideanMPNN"]


class _MLP(nn.Module):
    """A two-layer MLP used inside the MPNN."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.relu(self.lin1(x)))


class EuclideanMPNN(nn.Module):
    """GIN-style message-passing encoder.

    Attributes:
        hidden_dim: Width of the message-passing layers.
        num_layers: Number of message-passing layers.
        output_dim: Dimensionality of the per-vertex embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0 or output_dim <= 0:
            raise ValueError(
                f"EuclideanMPNN: all dimensions must be positive; "
                f"got input_dim={input_dim}, hidden_dim={hidden_dim}, "
                f"num_layers={num_layers}, output_dim={output_dim}"
            )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Update MLP takes concatenated [h || agg] so its input is 2 * hidden_dim.
        self.update = _MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Encode the graph and return a per-vertex embedding.

        Args:
            graph: The input graph.

        Returns:
            A ``[N, output_dim]`` tensor of per-vertex embeddings.
        """
        x = graph.vertex_features
        h = self.input_proj(x)
        edge_index = graph.edge_index
        for _ in range(self.num_layers):
            if edge_index.numel() == 0:
                agg = torch.zeros_like(h)
            else:
                src = edge_index[0]
                dst = edge_index[1]
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, h[src])
            h = self.update(torch.cat([h, agg], dim=-1))
        return self.out_proj(h)

    def to(self, device: torch.device) -> "EuclideanMPNN":
        """Move parameters to ``device`` and return self."""
        return super().to(device)