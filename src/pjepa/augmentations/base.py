"""Augmentation base classes.

An augmentation is a callable that transforms a
:class:`TypedAttributedGraph` into a new graph. The pipeline class
supports composing augmentations in three modes: sequential, random
sample one, and random sample k.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["Augmentation", "AugmentationPipeline", "PipelineMode"]


class PipelineMode:
    """Composition modes for :class:`AugmentationPipeline`."""

    SEQUENTIAL = "sequential"
    RANDOM_SAMPLE_ONE = "random_sample_one"
    RANDOM_SAMPLE_K = "random_sample_k"


class Augmentation(ABC):
    """Base class for all augmentations."""

    def __init__(self, strength: float = 0.2, generator: torch.Generator | None = None) -> None:
        if not 0.0 <= strength <= 1.0:
            raise GraphError(f"Augmentation: strength must be in [0, 1]; got {strength}")
        self.strength = strength
        self.generator = generator

    @abstractmethod
    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation to ``graph`` and return the result."""


class AugmentationPipeline:
    """Compose multiple augmentations into a single transform.

    Args:
        augmentations: The list of augmentations to compose.
        mode: One of :class:`PipelineMode`. ``SEQUENTIAL`` applies all
          augmentations in order; ``RANDOM_SAMPLE_ONE`` samples one
          augmentation uniformly and applies it; ``RANDOM_SAMPLE_K``
          samples ``k`` augmentations without replacement and applies
          them in the sampled order.
        k: Number of augmentations to sample under ``RANDOM_SAMPLE_K``.
        generator: Optional ``torch.Generator`` for reproducibility.
    """

    def __init__(
        self,
        augmentations: Sequence[Augmentation],
        mode: str = PipelineMode.RANDOM_SAMPLE_ONE,
        k: int = 2,
        generator: torch.Generator | None = None,
    ) -> None:
        if not augmentations:
            raise GraphError("AugmentationPipeline: at least one augmentation is required")
        if mode not in (
            PipelineMode.SEQUENTIAL,
            PipelineMode.RANDOM_SAMPLE_ONE,
            PipelineMode.RANDOM_SAMPLE_K,
        ):
            raise GraphError(f"AugmentationPipeline: unknown mode {mode!r}")
        if k <= 0 or k > len(augmentations):
            raise GraphError(
                f"AugmentationPipeline: k must be in [1, {len(augmentations)}]; got {k}"
            )
        self.augmentations = list(augmentations)
        self.mode = mode
        self.k = k
        self.generator = generator

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the pipeline to ``graph`` and return the result."""
        if self.mode == PipelineMode.SEQUENTIAL:
            current = graph
            for aug in self.augmentations:
                current = aug(current)
            return current
        if self.mode == PipelineMode.RANDOM_SAMPLE_ONE:
            idx = torch.randint(0, len(self.augmentations), (1,), generator=self.generator).item()
            return self.augmentations[idx](graph)
        # RANDOM_SAMPLE_K
        n = len(self.augmentations)
        perm = torch.randperm(n, generator=self.generator)[: self.k].tolist()
        current = graph
        for i in perm:
            current = self.augmentations[i](current)
        return current
