"""Tensor-compatible augmentation wrapper.

Useful for tests and small models that work on raw tensors rather
than :class:`TypedAttributedGraph`. Operates on a 2-D tensor by
zeroing a random fraction of feature columns.
"""

from __future__ import annotations

import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["TensorDropFeature", "tensor_drop_feature"]


class TensorDropFeature:
    """Drop a fraction ``strength`` of feature columns from a 2-D tensor.

    The augmentation accepts either a 2-D tensor directly or a
    :class:`TypedAttributedGraph`; the graph branch updates only the
    vertex-feature tensor and returns a fresh graph.
    """

    def __init__(
        self,
        strength: float = 0.2,
        generator: torch.Generator | None = None,
    ) -> None:
        if not 0.0 <= strength <= 1.0:
            raise GraphError(f"TensorDropFeature: strength must be in [0, 1]; got {strength}")
        self.strength = strength
        self.generator = generator

    def __call__(
        self, tensor: torch.Tensor | TypedAttributedGraph
    ) -> torch.Tensor | TypedAttributedGraph:
        """Drop a random fraction of feature columns.

        Args:
            tensor: Either a 2-D feature tensor or a
                :class:`TypedAttributedGraph`. When a graph is passed,
                its vertex features are augmented and a new graph is
                returned.

        Returns:
            Either the augmented tensor (same shape) or the augmented
            :class:`TypedAttributedGraph`. The concrete type mirrors
            the input type.

        Raises:
            GraphError: If the input tensor's leading dimension is not
                ``2`` or the input is a graph with a non-2-D feature
                tensor.
        """
        if isinstance(tensor, TypedAttributedGraph):
            feats = tensor.vertex_features
        else:
            feats = tensor
        if feats.ndim != 2:
            raise GraphError(
                f"TensorDropFeature: tensor must be 2-D; got shape {tuple(feats.shape)}"
            )
        n_dim = feats.shape[1]
        n_drop = int(self.strength * n_dim)
        if n_drop == 0:
            result = feats.clone()
        else:
            perm = torch.randperm(n_dim, generator=self.generator)[:n_drop]
            result = feats.clone()
            result[:, perm] = 0.0
        if isinstance(tensor, TypedAttributedGraph):
            return tensor.with_features(vertex_features=result)
        return result


def tensor_drop_feature(
    tensor: torch.Tensor,
    strength: float = 0.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Convenience function: drop features from a tensor in one call.

    Equivalent to ``TensorDropFeature(strength, generator)(tensor)``;
    provided so call sites that always work with raw tensors do not
    have to construct an instance.
    """
    return TensorDropFeature(strength=strength, generator=generator)(tensor)
