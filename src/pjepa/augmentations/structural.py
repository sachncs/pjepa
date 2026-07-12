"""Structural graph augmentations.

* :class:`DropEdge` removes a random fraction of edges.
* :class:`DropNode` removes a random fraction of vertices.
* :class:`RandomWalkSubgraph` returns a vertex-induced subgraph
  consisting of vertices reachable from a random start vertex.
"""

from __future__ import annotations

import torch

from pjepa.augmentations.base import Augmentation
from pjepa.graphs import TypedAttributedGraph

__all__ = ["DropEdge", "DropNode", "RandomWalkSubgraph"]


class DropEdge(Augmentation):
    """Drop a fraction ``strength`` of edges at random.

    Attributes:
        strength: Fraction of edges to drop, in [0, 1].
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation."""
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

    Attributes:
        strength: Fraction of vertices to drop, in [0, 1].
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation."""
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


class RandomWalkSubgraph(Augmentation):
    """Return a vertex-induced subgraph reachable by a random walk.

    The walk starts at a random vertex and continues until ``strength * N``
    vertices have been visited or no new vertex can be reached.

    Attributes:
        strength: Fraction of vertices to retain, in (0, 1].
    """

    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
        """Apply the augmentation."""
        n_vertices = graph.num_vertices()
        if n_vertices == 0:
            return graph
        target = max(1, int(self.strength * n_vertices))
        start = torch.randint(0, n_vertices, (1,), generator=self.generator).item()
        visited = {start}
        frontier = [start]
        edge_index = graph.edge_index
        adjacency: dict[int, list[int]] = {i: [] for i in range(n_vertices)}
        if edge_index.numel() > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                adjacency[s].append(d)
        while frontier and len(visited) < target:
            node = frontier.pop()
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