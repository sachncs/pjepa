"""Documented cross-version compatibility aliases.

These aliases exist so external code that imports older or PyG-flavoured
names keeps working. They are stable exports; new aliases are added in
minor versions and never removed without a deprecation cycle.

Each alias is the canonical class itself (not a subclass), so
``isinstance`` checks and ``type()`` comparisons against the original
types continue to hold:

* :data:`Graph` -> :class:`pjepa.graphs.TypedAttributedGraph`
* :data:`PersistentGraph` -> :class:`pjepa.graphs.PersistentState`
* :data:`GraphState` -> :class:`pjepa.graphs.WorkingGraph`
* :data:`PJEPAEncoder` -> :class:`pjepa.encoders.base.Encoder`
* :data:`PJEPAAugmentation` -> :class:`pjepa.augmentations.base.Augmentation`

The helper :func:`make_typed_graph` is also exported here so callers
that prefer a single entry point regardless of subpackage layout can
import it without reaching into :mod:`pjepa.graphs`.
"""

from __future__ import annotations

from pjepa.augmentations.base import Augmentation as AugmentationBase
from pjepa.encoders.base import Encoder as EncoderBase
from pjepa.graphs import (
    PersistentState,
    TypedAttributedGraph,
    WorkingGraph,
)

Graph = TypedAttributedGraph
"""Alias matching the convention used in some downstream packages."""

PersistentGraph = PersistentState
"""Alias: ``PersistentGraph`` is the framework's persistent-state container."""

GraphState = WorkingGraph
"""Alias: ``GraphState`` is the framework's working-graph container."""

PJEPAEncoder = EncoderBase
"""Alias of :class:`pjepa.encoders.base.Encoder` for type annotations."""

PJEPAAugmentation = AugmentationBase
"""Alias of :class:`pjepa.augmentations.base.Augmentation` for type annotations."""

__all__ = [
    "Graph",
    "GraphState",
    "PJEPAAugmentation",
    "PJEPAEncoder",
    "PersistentGraph",
    "TypedAttributedGraph",
    "WorkingGraph",
    "make_typed_graph",
]


def make_typed_graph(
    vertex_features: object,
    edge_index: object,
    edge_features: object | None = None,
    **kwargs: object,
) -> TypedAttributedGraph:
    """Construct a :class:`TypedAttributedGraph` from positional tensors.

    Convenience helper for callers that prefer keyword-style imports or
    want a single entry point regardless of package layout.

    Args:
        vertex_features: A ``[N, d_v]`` tensor of vertex features.
        edge_index: A ``[2, E]`` ``long`` tensor in COO format.
        edge_features: Optional ``[E, d_e]`` tensor of edge features.
        **kwargs: Forwarded verbatim to :class:`TypedAttributedGraph`.

    Returns:
        A new :class:`TypedAttributedGraph`.

    Raises:
        pjepa.exceptions.GraphError: If the supplied tensors violate
            the structural invariants documented on
            :class:`TypedAttributedGraph`.

    Example:
        >>> vf = torch.randn((3, 4))
        >>> ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        >>> g = make_typed_graph(vf, ei)
    """
    if edge_features is None:
        return TypedAttributedGraph(
            vertex_features=vertex_features, edge_index=edge_index, **kwargs
        )
    return TypedAttributedGraph(
        vertex_features=vertex_features,
        edge_index=edge_index,
        edge_features=edge_features,
        **kwargs,
    )
