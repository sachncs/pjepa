"""Minimum Description Length (MDL) on graphs.

Given a :class:`TypedAttributedGraph`, :func:`description_length`
returns a non-negative scalar estimate of the number of nats required
to encode the graph under a fixed hyperedge-replacement grammar. The
estimate uses three additive terms:

* ``log(|V| + 1)`` — vertex-count contribution,
* ``log(|E| + 1)`` — edge-count contribution,
* ``log(nnz(F) + 1)`` — non-zero feature contribution.

The estimate is intentionally simple; the grammar-aware description
length is reserved for a future revision.
"""

from __future__ import annotations

import math

from pjepa.graphs import TypedAttributedGraph

__all__ = ["description_length"]


def description_length(graph: TypedAttributedGraph) -> float:
    """Estimate the description length of a graph in nats.

    Args:
        graph: The graph whose description length to estimate.

    Returns:
        A strictly positive ``float`` in nats for any non-empty
        graph. The empty graph yields ``0`` because all three counts
        are zero and ``log1p(0) == 0``.

    Example:
        >>> import torch
        >>> g = TypedAttributedGraph(
        ...     vertex_features=torch.zeros((3, 4)),
        ...     edge_index=torch.zeros((2, 0), dtype=torch.long),
        ... )
        >>> description_length(g) > 0.0
        True
    """
    n_vertices = graph.num_vertices()
    n_edges = graph.num_edges()
    nnz_features = int((graph.vertex_features != 0).sum().item())
    return math.log1p(n_vertices) + math.log1p(n_edges) + math.log1p(nnz_features)
