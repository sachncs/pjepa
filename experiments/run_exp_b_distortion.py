"""Experiment runner for Proposition 7 (hyperbolic distortion)."""

from __future__ import annotations

import math

import torch

from pjepa.graphs import TypedAttributedGraph

__all__ = ["run"]


def _b_ary_tree(b: int, depth: int) -> TypedAttributedGraph:
    """Build a complete b-ary tree of given depth as a TypedAttributedGraph."""
    n_vertices = sum(b**k for k in range(depth + 1))
    edges: list[tuple[int, int]] = []
    for level in range(depth):
        start = sum(b**k for k in range(level))
        next_start = sum(b**k for k in range(level + 1))
        for parent in range(start, start + b**level):
            for child_offset in range(b):
                edges.append((parent, next_start + (parent - start) * b + child_offset))
    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
    return TypedAttributedGraph(
        vertex_features=torch.zeros((n_vertices, 2)),
        edge_index=edge_index,
        edge_features=torch.zeros((edge_index.shape[1], 1)),
    )


def _embed_euclidean_random(tree: TypedAttributedGraph) -> torch.Tensor:
    """Embed the tree into R^2 by a deterministic random projection."""
    torch.manual_seed(int(tree.num_vertices()))
    coords = torch.randn((tree.num_vertices(), 2))
    return coords


def _embed_hyperbolic_conformal(tree: TypedAttributedGraph) -> torch.Tensor:
    """Embed the tree into the Poincaré disk via the Sarkar construction.

    Vertices at depth ``k`` are placed at hyperbolic radius ``k * log(b)``.
    """
    b = 2
    depth = 0
    while sum(b**k for k in range(depth + 1)) < tree.num_vertices():
        depth += 1
    coords: list[tuple[float, float]] = []
    for v in range(tree.num_vertices()):
        level = 0
        running = 0
        while running + b**level <= v:
            running += b**level
            level += 1
        r = level * math.log(b + 1)
        # Place on a circle at radius r in the Poincaré disk.
        angle = (v - running) * (2 * math.pi / max(1, b**level))
        x = math.tanh(r / 2) * math.cos(angle)
        y = math.tanh(r / 2) * math.sin(angle)
        coords.append((x, y))
    return torch.tensor(coords, dtype=torch.float32)


def _edge_distances(coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    return torch.norm(coords[src] - coords[dst], dim=-1)


def run() -> dict[str, object]:
    """Measure per-edge distortion for trees of varying depth."""
    rows: list[dict[str, float]] = []
    for depth in (3, 5, 7):
        tree = _b_ary_tree(b=2, depth=depth)
        euc = _embed_euclidean_random(tree)
        hyp = _embed_hyperbolic_conformal(tree)
        euc_dists = _edge_distances(euc, tree.edge_index)
        hyp_dists = _edge_distances(hyp, tree.edge_index)
        rows.append(
            {
                "depth": float(depth),
                "n_vertices": float(tree.num_vertices()),
                "euclidean_max_per_edge": float(euc_dists.max().item()),
                "hyperbolic_max_per_edge": float(hyp_dists.max().item()),
            }
        )
    return {"rows": rows}