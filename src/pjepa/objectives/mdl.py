"""Minimum Description Length (MDL) on graphs.

Given a :class:`TypedAttributedGraph`, ``description_length`` returns
a non-negative scalar estimate of the number of bits required to
encode the graph under a fixed hyperedge-replacement grammar. The
estimate is intentionally simple — it counts vertices, edges, and
non-zero feature entries — and is replaced in Phase 6 with the
full grammar-aware description length.
"""

from __future__ import annotations

import math

from pjepa.graphs import TypedAttributedGraph

__all__ = ["description_length"]


def description_length(graph: TypedAttributedGraph) -> float:
    """Estimate the description length of a graph in nats.

    The estimate uses three additive terms:

    * Vertex count contribution: ``log(|V| + 1)``.
    * Edge count contribution: ``log(|E| + 1)``.
    * Non-zero feature contribution: ``log(nnz(F) + 1)``.

    Args:
        graph: The graph whose description length to estimate.

    Returns:
        A non-negative float in nats.

    Example:
        >>> g = TypedAttributedGraph(torch.zeros((3, 4)), torch.zeros((2, 0), dtype=torch.long))
        >>> description_length(g) > 0.0
        True
    """
    n_vertices = graph.num_vertices()
    n_edges = graph.num_edges()
    nnz_features = int((graph.vertex_features != 0).sum().item())
    return math.log1p(n_vertices) + math.log1p(n_edges) + math.log1p(nnz_features)
