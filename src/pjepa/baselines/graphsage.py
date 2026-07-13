"""GraphSAGE baseline (Hamilton et al., 2017).

A configurable-width, configurable-depth GraphSAGE encoder with
mean pooling. Designed to be friendly to neighbour-sampled
subgraphs (the :meth:`forward` and :meth:`embed` methods accept a
:class:`TypedAttributedGraph` argument so the caller can
pre-extract the induced subgraph for the current mini-batch).

For node-classification tasks the public :meth:`node_logits`
helper returns the per-node predictions without any pooling. The
:meth:`embed` method returns pooled graph-level features,
matching the shape expected by the TU aggregator.

## Algorithm

The :class:`SAGEConv` layers perform mean-aggregation of
neighbour features followed by a ReLU + linear projection. The
encoder is paired with an optional linear classifier; passing
``num_classes=0`` disables the classifier (embed-only mode).

## Complexity

* Per-layer cost — ``O(|E| * H)``.
* :meth:`embed` — ``O(|V| * H)`` for the mean pool at the end.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv  # type: ignore[import-not-found]

from pjepa.graphs import TypedAttributedGraph

__all__ = ["GraphSAGE"]


class GraphSAGE(nn.Module):
    """GraphSAGE encoder with optional linear classifier.

    Attributes:
        hidden_dim: Width of the SAGE layers.
        num_layers: Number of SAGE convolutions.
        num_classes: Output dimension of the classifier. ``0`` skips
          the classifier (embed-only mode).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("GraphSAGE: dims must be positive")
        if num_classes < 0:
            raise ValueError(f"GraphSAGE: num_classes must be >= 0; got {num_classes}")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, aggr="mean") for _ in range(num_layers)]
        )
        self.classifier: nn.Module | None
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifier = None
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

    def encode(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return per-vertex embeddings of shape ``[N, hidden_dim]``.

        Args:
            graph: The input graph (any :class:`TypedAttributedGraph`
              whose ``edge_index`` describes the message-passing
              edges; for neighbour-sampled training this is the
              induced subgraph).

        Returns:
            ``[N, hidden_dim]`` per-vertex embeddings.
        """
        h = self.input_proj(graph.vertex_features)
        for layer in self.layers:
            h = torch.relu(layer(h, graph.edge_index))
        return h

    def embed(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return mean-pooled graph embedding of shape ``[1, hidden_dim]``.

        Args:
            graph: The input graph.

        Returns:
            ``[1, hidden_dim]`` graph embedding. When the graph has
            zero vertices the function returns a zeros vector so
            downstream consumers (mean-per-class accuracy, etc.) can
            process the result uniformly.
        """
        h = self.encode(graph)
        if h.shape[0] == 0:
            return torch.zeros((1, self.hidden_dim), dtype=h.dtype, device=h.device)
        return h.mean(dim=0, keepdim=True)

    def node_logits(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return per-vertex logits of shape ``[N, num_classes]``.

        Args:
            graph: The input graph.

        Returns:
            ``[N, num_classes]`` per-vertex logits.

        Raises:
            RuntimeError: When the classifier is disabled
              (``num_classes=0``).
        """
        if self.classifier is None:
            raise RuntimeError("GraphSAGE: classifier is disabled (num_classes=0)")
        return self.classifier(self.encode(graph))

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return per-graph logits of shape ``[1, num_classes]``.

        Args:
            graph: The input graph.

        Returns:
            For ``num_classes > 0``: ``[1, num_classes]`` per-graph
            logits. For ``num_classes == 0``: the mean-pooled graph
            embedding ``[1, hidden_dim]``.
        """
        if self.classifier is None:
            return self.embed(graph)
        return self.classifier(self.embed(graph))
