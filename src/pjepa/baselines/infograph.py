"""InfoGraph baseline (Sun et al., 2020).

Maximises mutual information between graph-level and node-level
representations using a bilinear discriminator.
"""

from __future__ import annotations

import torch
from torch import nn

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["InfoGraph"]


class InfoGraph(nn.Module):
    """InfoGraph encoder + bilinear discriminator.

    Attributes:
        hidden_dim: Width of the encoder.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("InfoGraph: dims must be positive")
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def encode(self, graph: TypedAttributedGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(node_embeddings, graph_embedding)``."""
        if graph.num_vertices() == 0:
            raise GraphError("InfoGraph.encode: cannot encode an empty graph")
        node = self.encoder(graph.vertex_features)
        graph_emb = node.mean(dim=0, keepdim=True)
        return node, graph_emb

    def loss(
        self,
        node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        shuffled_node_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the InfoGraph mutual-information loss."""
        n = node_emb.shape[0]
        pos = self.discriminator(node_emb, graph_emb.expand_as(node_emb))
        neg = self.discriminator(shuffled_node_emb, graph_emb.expand_as(node_emb))
        logits = torch.cat([pos, neg], dim=0)
        labels = torch.cat([torch.ones(n), torch.zeros(n)])
        return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)
