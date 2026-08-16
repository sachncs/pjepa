"""Transform base classes.

An augmentation is a callable that transforms a
:class:`Graph` into a new graph. The pipeline class
supports composing augmentations in three modes: sequential, random
sample one, and random sample k.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import Graph

__all__ = ["Pipeline", "PipelineMode", "Transform"]


class PipelineMode:
    """Composition modes for :class:`Pipeline`.

    Attributes:
        SEQUENTIAL: Apply every augmentation in declaration order.
        RANDOM_SAMPLE_ONE: Sample one augmentation uniformly and apply it.
        RANDOM_SAMPLE_K: Sample ``k`` augmentations without
            replacement and apply them in the sampled order.
    """

    SEQUENTIAL = "sequential"
    RANDOM_SAMPLE_ONE = "random_sample_one"
    RANDOM_SAMPLE_K = "random_sample_k"


class Transform(ABC):
    """Base class for all augmentations.

    Subclasses must implement :meth:`__call__`; they are free to read
    :attr:`strength` and :attr:`generator` to make stochastic
    decisions.

    Attributes:
        strength: A scalar in ``[0, 1]`` controlling the strength of
            the augmentation; each subclass interprets the magnitude
            differently (fraction of vertices, edges, etc.).
        generator: Optional :class:`torch.Generator` for reproducible
            randomness. When ``None``, augmentations read from the
            global PyTorch generator.

    Raises:
        GraphError: At construction if ``strength`` is outside
            ``[0, 1]``.
    """

    def __init__(self, strength: float = 0.2, generator: torch.Generator | None = None) -> None:
        """Initialise the augmentation base class.

        Args:
            strength: A scalar in ``[0, 1]`` controlling the strength
                of the augmentation; each subclass interprets the
                magnitude differently (fraction of vertices,
                edges, etc.).
            generator: Optional :class:`torch.Generator` for
                reproducible randomness. When ``None``,
                augmentations read from the global PyTorch
                generator.

        Raises:
            GraphError: If ``strength`` is outside ``[0, 1]``.
        """
        if not 0.0 <= strength <= 1.0:
            raise GraphError(f"Transform: strength must be in [0, 1]; got {strength}")
        self.strength = strength
        self.generator = generator

    @abstractmethod
    def __call__(self, graph: Graph) -> Graph:
        """Apply the augmentation to ``graph`` and return the result.

        The returned graph may equal ``graph`` (an explicit no-op) but
        is always a new object unless the subclass deliberately
        returns the input.
        """


class Pipeline:
    """Compose multiple augmentations into a single transform.

    Args:
        augmentations: The list of augmentations to compose; must be
            non-empty.
        mode: One of :class:`PipelineMode`. ``SEQUENTIAL`` applies
            every augmentation in order; ``RANDOM_SAMPLE_ONE``
            samples exactly one uniformly; ``RANDOM_SAMPLE_K``
            samples ``k`` without replacement and applies them in
            the sampled order.
        k: Number of augmentations to sample under
            ``RANDOM_SAMPLE_K``. Must be in ``[1, len(augmentations)]``.
        generator: Optional ``torch.Generator`` for reproducibility.

    Raises:
        GraphError: If ``augmentations`` is empty, ``mode`` is
            unknown, or ``k`` is out of range.
    """

    def __init__(
        self,
        augmentations: Sequence[Transform],
        mode: str = PipelineMode.RANDOM_SAMPLE_ONE,
        k: int = 2,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initialise the augmentation pipeline.

        Args:
            augmentations: The list of augmentations to compose.
                Must be non-empty.
            mode: One of :class:`PipelineMode`. ``SEQUENTIAL``
                applies every augmentation in order;
                ``RANDOM_SAMPLE_ONE`` samples exactly one
                uniformly; ``RANDOM_SAMPLE_K`` samples ``k``
                without replacement and applies them in the sampled
                order.
            k: Number of augmentations to sample under
                ``RANDOM_SAMPLE_K``. Must be in
                ``[1, len(augmentations)]``.
            generator: Optional :class:`torch.Generator` for
                reproducibility.

        Raises:
            GraphError: If ``augmentations`` is empty, ``mode`` is
                unknown, or ``k`` is out of range.
        """
        if not augmentations:
            raise GraphError("Pipeline: at least one augmentation is required")
        if mode not in (
            PipelineMode.SEQUENTIAL,
            PipelineMode.RANDOM_SAMPLE_ONE,
            PipelineMode.RANDOM_SAMPLE_K,
        ):
            raise GraphError(f"Pipeline: unknown mode {mode!r}")
        if k <= 0 or k > len(augmentations):
            raise GraphError(f"Pipeline: k must be in [1, {len(augmentations)}]; got {k}")
        self.augmentations = list(augmentations)
        self.mode = mode
        self.k = k
        self.generator = generator

    def __call__(self, graph: Graph) -> Graph:
        """Apply the pipeline to ``graph``.

        Args:
            graph: The input graph.

        Returns:
            A new :class:`Graph` after the configured sequence of
            augmentations has been applied.
        """
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
