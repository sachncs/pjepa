"""Identity augmentation.

Returns the input graph unchanged. Useful as the identity element in
:class:`AugmentationPipeline` compositions and as a placeholder when
the chosen strength would otherwise result in a no-op.
"""

from __future__ import annotations

from pjepa.augmentations.base import Augmentation
from pjepa.graphs import TypedAttributedGraph

__all__ = ["Identity"]


class Identity(Augmentation):
    """Return the input graph unchanged.

    The ``strength`` argument is accepted for API symmetry with other
    augmentations but is ignored.
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Return ``graph`` unchanged."""
        return graph
