"""Naive baseline.

A trivial mean-pooled linear classifier over the raw vertex
features. Used as a sanity baseline: any non-trivial model must
beat this on the TU benchmarks to be considered a real comparison.

## Algorithm

```
   pooled  = global_mean_pool(vertex_features)
   hidden  = projection(pooled)         # skipped when hidden_dim == 0
   logits  = classifier(hidden)
```

The hidden projection is disabled (identity) when ``hidden_dim=0``,
which lets the baseline compare a raw-feature classifier against an
``hidden_dim=64`` projection variant. The latter is the
configuration used in the TU SOTA comparison.

## Complexity

``O(|V| * D)`` for the mean-pool followed by ``O(D * H)`` and
``O(H * C)`` for the linear layers. Memory is ``O(|V| * D)``.
"""

from __future__ import annotations

import torch
from torch import nn

from pjepa.graphs import TypedAttributedGraph

__all__ = ["Naive"]


class Naive(nn.Module):
    """Mean-pool the vertex features and apply a linear classifier.

    Attributes:
        hidden_dim: Width of the optional projection; ``0`` skips
          the projection and feeds the raw features directly to the
          classifier.
        num_classes: Output dimension of the classifier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 0:
            raise ValueError("Naive: input_dim and num_classes must be positive")
        if hidden_dim < 0:
            raise ValueError(f"Naive: hidden_dim must be >= 0; got {hidden_dim}")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        if hidden_dim == 0:
            self.projection: nn.Module = nn.Identity()
            classifier_in = input_dim
        else:
            self.projection = nn.Linear(input_dim, hidden_dim)
            classifier_in = hidden_dim
        self.classifier = nn.Linear(classifier_in, num_classes)
        self.embedding_dim = classifier_in

    def embed(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return the pooled graph embedding before the classifier.

        Args:
            graph: The input graph.

        Returns:
            ``[1, input_dim | hidden_dim]`` pooled projection.
        """
        features = graph.vertex_features
        if features.shape[0] == 0:
            pooled = torch.zeros(
                (1, self.embedding_dim),
                dtype=features.dtype,
                device=features.device,
            )
        else:
            pooled = features.mean(dim=0, keepdim=True)
        return self.projection(pooled)

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return per-graph logits computed from mean-pooled vertex features.

        Args:
            graph: The input graph.

        Returns:
            ``[1, num_classes]`` per-graph logits.
        """
        return self.classifier(self.embed(graph))
