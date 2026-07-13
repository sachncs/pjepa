"""Structural graph augmentations.

* :class:`DropEdge` removes a random fraction of edges.
* :class:`DropNode` removes a random fraction of vertices.
* :class:`Subgraph` returns a randomly sampled vertex-induced subgraph.
* :class:`ConnectedSubgraph` returns a vertex-induced subgraph
  consisting of the vertices reachable from a random start vertex by a
  breadth-first walk along incident edges.
"""

from __future__ import annotations

from collections import deque

import torch

from pjepa.augmentations.base import Augmentation
from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["ConnectedSubgraph", "DropEdge", "DropNode", "Subgraph"]


class DropEdge(Augmentation):
    """Drop a fraction ``strength`` of edges at random.

    Attributes:
        strength: Fraction of edges to drop, in ``[0, 1]``.
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation.

        Returns the input graph unchanged when it has zero edges or
        when ``strength * num_edges`` rounds down to zero.
        """
        n_edges = graph.num_edges()
        if n_edges == 0:
            return graph
        n_drop = int(self.strength * n_edges)
        if n_drop == 0:
            return graph
        perm = torch.randperm(n_edges, generator=self.generator)[:n_drop]
        keep_mask = torch.ones(n_edges, dtype=torch.bool)
        keep_mask[perm] = False
        return graph.with_features(
            edge_index=graph.edge_index[:, keep_mask],
            edge_features=graph.edge_features[keep_mask],
            version=graph.version + 1,
        )


class DropNode(Augmentation):
    """Drop a fraction ``strength`` of vertices at random.

    At least one vertex is always kept so the result is non-empty
    (provided the input graph is non-empty).
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation.

        Returns the input graph unchanged when it has zero or one
        vertex.
        """
        n_vertices = graph.num_vertices()
        if n_vertices <= 1:
            return graph
        n_drop = min(int(self.strength * n_vertices), n_vertices - 1)
        if n_drop == 0:
            return graph
        perm = torch.randperm(n_vertices, generator=self.generator)[:n_drop]
        keep_mask = torch.ones(n_vertices, dtype=torch.bool)
        keep_mask[perm] = False
        return graph.subgraph(keep_mask)


class ConnectedSubgraph(Augmentation):
    """Return a vertex-induced subgraph reachable by a breadth-first walk.

    The walk starts at a vertex chosen uniformly at random and grows
    outwards, adding every newly-discovered neighbour to the frontier,
    until either ``max(1, strength * num_vertices)`` vertices have been
    collected or the frontier is exhausted. The returned subgraph is
    therefore guaranteed to be connected when the input graph is.

    The implementation uses :class:`collections.deque` for the frontier,
    giving ``O(N + E)`` per call. The ``generator`` argument (when
    provided) seeds the start-vertex choice; the rest of the BFS is
    deterministic given that vertex.

    Attributes:
        strength: Fraction of vertices to retain, in ``(0, 1]``.
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation.

        Returns the input graph unchanged when it has zero vertices
        (the BFS already starts from the only vertex when ``N == 1``).
        """
        n_vertices = graph.num_vertices()
        if n_vertices == 0:
            return graph
        target = max(1, int(self.strength * n_vertices))
        start = torch.randint(0, n_vertices, (1,), generator=self.generator).item()
        adjacency: dict[int, list[int]] = {i: [] for i in range(n_vertices)}
        if graph.edge_index.numel() > 0:
            for s, d in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()):
                adjacency[s].append(d)
        visited: set[int] = {start}
        # deque is appended on the right and popped on the left, which
        # gives breadth-first expansion from the start vertex.
        frontier: deque[int] = deque([start])
        while frontier and len(visited) < target:
            node = frontier.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    frontier.append(neighbour)
                    if len(visited) >= target:
                        break
        mask = torch.zeros(n_vertices, dtype=torch.bool)
        for v in visited:
            mask[v] = True
        return graph.subgraph(mask)


class Subgraph(Augmentation):
    """Return a randomly sampled vertex-induced subgraph.

    The ``strength`` parameter is interpreted as the fraction of
    vertices to retain, in ``(0, 1]``. At least one vertex is always
    kept so the result is non-empty.

    Attributes:
        strength: Fraction of vertices to retain, in ``(0, 1]``.

    Raises:
        GraphError: If ``strength`` is outside ``(0, 1]`` (enforced
            at call time so callers can reuse a single instance with
            different configurations).
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation.

        Returns the input graph unchanged when it has zero vertices.
        """
        n_vertices = graph.num_vertices()
        if n_vertices == 0:
            return graph
        if not 0.0 < self.strength <= 1.0:
            raise GraphError(f"Subgraph: strength must be in (0, 1]; got {self.strength}")
        target = max(1, int(self.strength * n_vertices))
        perm = torch.randperm(n_vertices, generator=self.generator)[:target]
        mask = torch.zeros(n_vertices, dtype=torch.bool)
        mask[perm] = True
        return graph.subgraph(mask)
