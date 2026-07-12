"""Graph Convolutional Network baseline (Kipf & Welling, 2017).

A simple two-layer GCN with mean pooling. Used as the supervised
baseline in the TU SOTA comparison.
"""

from __future__ import annotations

import torch

from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["GCN"]


class GCN(nn.Module):
    """Two-layer GCN encoder with a linear classifier.

    Attributes:
        hidden_dim: Width of the convolutional layers.
        num_classes: Output dimension of the classifier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_classes <= 0:
            raise ValueError("GCN: dims must be positive")
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Encode the graph and return per-graph logits."""
        h = self.conv1(graph.vertex_features, graph.edge_index)
        h = torch.relu(h)
        h = self.conv2(h, graph.edge_index)
        device = h.device
        batch = torch.zeros(h.shape[0], dtype=torch.long, device=device)
        pooled = global_mean_pool(h, batch)
        return self.classifier(pooled)

    def embed(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return the pooled graph embedding without the classifier."""
        h = self.conv1(graph.vertex_features, graph.edge_index)
        h = torch.relu(h)
        h = self.conv2(h, graph.edge_index)
        device = h.device
        batch = torch.zeros(h.shape[0], dtype=torch.long, device=device)
        return global_mean_pool(h, batch)