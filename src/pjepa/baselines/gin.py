"""Graph Isomorphism Network baseline (Xu et al., 2019).

GIN with sum pooling. Optional virtual-node (VN) trick for stronger
graph-level representation (Xu et al. 2019 §5.3). The VN flag is
controlled at construction time.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

from pjepa.graphs import TypedAttributedGraph

__all__ = ["GIN"]


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Create a 2-layer MLP used inside a GINConv."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class GIN(nn.Module):
    """Graph Isomorphism Network with optional virtual node.

    Attributes:
        hidden_dim: Width of the GIN layers.
        num_layers: Number of GIN layers.
        num_classes: Output dimension of the classifier.
        use_virtual_node: Whether to use the virtual-node trick.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        use_virtual_node: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0 or num_classes <= 0:
            raise ValueError("GIN: dims must be positive")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GINConv(_make_mlp(hidden_dim, hidden_dim, hidden_dim)) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.use_virtual_node = use_virtual_node
        if use_virtual_node:
            self.virtual_node = nn.Parameter(torch.zeros(hidden_dim))
        self.num_classes = num_classes

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Encode the graph and return per-graph logits."""
        h = self.input_proj(graph.vertex_features)
        if self.use_virtual_node:
            h = h + self.virtual_node
        for layer in self.layers:
            h = torch.relu(layer(h, graph.edge_index))
            if self.use_virtual_node:
                h = h + self.virtual_node
        batch = torch.zeros(h.shape[0], dtype=torch.long)
        pooled = global_add_pool(h, batch)
        return self.classifier(pooled)

    def embed(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return the pooled graph embedding without the classifier."""
        h = self.input_proj(graph.vertex_features)
        if self.use_virtual_node:
            h = h + self.virtual_node
        for layer in self.layers:
            h = torch.relu(layer(h, graph.edge_index))
            if self.use_virtual_node:
                h = h + self.virtual_node
        batch = torch.zeros(h.shape[0], dtype=torch.long)
        return global_add_pool(h, batch)
