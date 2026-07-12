"""Feature-space graph augmentations.

* :class:`DropFeature` zeros a random fraction of feature dimensions.
* :class:`FeatureMask` replaces a random fraction of feature values
  with a learnable mask token (initialised to zero at construction).
"""

from __future__ import annotations

import torch
from torch import nn

from pjepa.augmentations.base import Augmentation
from pjepa.graphs import TypedAttributedGraph

__all__ = ["DropFeature", "FeatureMask"]


class DropFeature(Augmentation):
    """Zero out a fraction ``strength`` of feature dimensions.

    The same set of dimensions is dropped across all vertices, as in
    the GraphMAE feature-masking strategy.
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation."""
        if graph.num_vertices() == 0:
            return graph
        n_dim = graph.vertex_features.shape[1]
        n_drop = int(self.strength * n_dim)
        if n_drop == 0:
            return graph
        perm = torch.randperm(n_dim, generator=self.generator)[:n_drop]
        new_features = graph.vertex_features.clone()
        new_features[:, perm] = 0.0
        return graph.with_features(vertex_features=new_features, version=graph.version + 1)


class FeatureMask(Augmentation):
    """Replace a fraction ``strength`` of feature values with a learnable mask token.

    The mask token is registered as a buffer and initialised to zeros.
    """

    def __init__(
        self,
        feature_dim: int,
        strength: float = 0.2,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(strength=strength, generator=generator)
        self.feature_dim = feature_dim
        self.mask_token = nn.Parameter(torch.zeros(feature_dim))

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation."""
        if graph.num_vertices() == 0:
            return graph
        n_dim = graph.vertex_features.shape[1]
        n_mask = int(self.strength * n_dim)
        if n_mask == 0:
            return graph
        perm = torch.randperm(n_dim, generator=self.generator)[:n_mask]
        new_features = graph.vertex_features.clone()
        mask_value = self.mask_token.detach().to(new_features.device)
        new_features[:, perm] = mask_value[perm]
        return graph.with_features(vertex_features=new_features, version=graph.version + 1)
