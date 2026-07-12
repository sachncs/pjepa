"""GraphMAE baseline (Hou et al., 2022).

Masked autoencoder for graphs. The encoder is GIN; the decoder
reconstructs masked node features. Trained with MSE on masked
positions only.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["GraphMAE"]


class GraphMAE(nn.Module):
    """GraphMAE encoder + decoder.

    Attributes:
        hidden_dim: Width of the GIN layers.
        mask_ratio: Fraction of vertices to mask during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("GraphMAE: dims must be positive")
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"GraphMAE: mask_ratio must be in [0, 1); got {mask_ratio}")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.mask_ratio = mask_ratio
        self.hidden_dim = hidden_dim

    def encode(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Encode the (masked) graph and return per-vertex embeddings."""
        h = self.input_proj(graph.vertex_features)
        for layer in self.encoder:
            h = torch.relu(layer(h, graph.edge_index))
        return h

    def forward(self, graph: TypedAttributedGraph) -> dict[str, torch.Tensor]:
        """Run the masked autoencoder and return embeddings + reconstruction.

        Args:
            graph: The input graph.

        Returns:
            A dict with ``embedding`` (pooled graph embedding),
            ``mask`` (boolean mask of masked vertices), and
            ``reconstruction`` (per-vertex reconstruction).
        """
        n_vertices = graph.num_vertices()
        if n_vertices == 0:
            raise GraphError("GraphMAE.forward: cannot encode an empty graph")
        n_mask = max(1, int(self.mask_ratio * n_vertices))
        perm = torch.randperm(n_vertices)[:n_mask]
        mask = torch.zeros(n_vertices, dtype=torch.bool)
        mask[perm] = True
        masked_features = graph.vertex_features.clone()
        masked_features[mask] = 0.0
        masked_graph = graph.with_features(vertex_features=masked_features)
        h = self.encode(masked_graph)
        reconstruction = self.decoder(h)
        batch = torch.zeros(h.shape[0], dtype=torch.long)
        embedding = global_add_pool(h, batch)
        return {
            "embedding": embedding,
            "mask": mask,
            "reconstruction": reconstruction,
        }
