"""Graph Convolutional Network baseline (Kipf & Welling, 2017).

A two-layer GCN with mean pooling. Used as the supervised baseline in
the TU SOTA comparison and as the node-classification backbone for
the OGB-arxiv experiment.

The class exposes both a graph-level :meth:`forward` (mean pool then
classify, used for TU experiments) and a per-vertex :meth:`node_logits`
path used by the OGB-arxiv trainers.

## Architecture

```
   h1 = relu(GCNConv(x, edge_index))
   h2 = relu(GCNConv(h1, edge_index))
   y  = classifier(global_mean_pool(h2))
```

## Complexity

* :meth:`forward` — ``O(|E| * H)`` per GCNConv message-passing call;
  two layers and a final linear projection.
* :meth:`node_logits` — same cost as :meth:`forward` minus the
  pooling and classifier.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool  # type: ignore[import-not-found]

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
        self.hidden_dim = hidden_dim

    def encode(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return per-vertex embeddings of shape ``[N, hidden_dim]``.

        Args:
            graph: The input graph.

        Returns:
            The per-vertex embeddings after two GCN convolutions.
        """
        h = self.conv1(graph.vertex_features, graph.edge_index)
        h = torch.relu(h)
        return self.conv2(h, graph.edge_index)

    def node_logits(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return per-vertex logits of shape ``[N, num_classes]``.

        Args:
            graph: The input graph.

        Returns:
            The per-vertex logits produced by the linear classifier.
        """
        return self.classifier(self.encode(graph))

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Encode the graph and return per-graph logits.

        Args:
            graph: The input graph.

        Returns:
            ``[1, num_classes]`` per-graph logits obtained by mean
            pooling the per-vertex embeddings and feeding the
            pooled vector to the linear classifier.
        """
        h = self.encode(graph)
        device = h.device
        batch = torch.zeros(h.shape[0], dtype=torch.long, device=device)
        pooled = global_mean_pool(h, batch)
        return self.classifier(pooled)

    def embed(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return the pooled graph embedding without the classifier.

        Args:
            graph: The input graph.

        Returns:
            ``[1, hidden_dim]`` mean-pooled graph embedding.
        """
        h = self.encode(graph)
        device = h.device
        batch = torch.zeros(h.shape[0], dtype=torch.long, device=device)
        return global_mean_pool(h, batch)
