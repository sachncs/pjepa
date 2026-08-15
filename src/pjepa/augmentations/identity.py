"""Identity augmentation.

Returns the input graph unchanged. Useful as the identity element in
:class:`Pipeline` compositions and as a placeholder when
the chosen strength would otherwise result in a no-op.
"""

from __future__ import annotations

from pjepa.augmentations.base import Transform
from pjepa.graphs import Graph

__all__ = ["Identity"]


class Identity(Transform):
    """Return the input graph unchanged.

    The ``strength`` argument is accepted for API symmetry with other
    augmentations but is ignored.
    """

    def __call__(self, graph: Graph) -> Graph:
        """Return ``graph`` unchanged."""
        return graph
