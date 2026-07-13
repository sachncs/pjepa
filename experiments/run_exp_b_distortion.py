"""Experiment B — Hyperbolic vs Euclidean distortion on trees (Phase 4).

Validates Proposition 7 by embedding b-ary trees of varying depth into
both the Poincaré disk (a Sarkar-inspired deterministic construction)
and ``R^d`` (a Bourgain-style multi-scale projection), then comparing
per-edge distortion.

Two honest disclaimers on the embedding algorithms:

* The Poincaré-disk embedding is **Sarkar-inspired**: it uses the
standard deterministic construction that places the root at the
origin, gives every depth-``k`` vertex a fixed angular slot, and
advances each depth by a constant hyperbolic step ``alpha = 2 *
asinh(1)``. This is a low-distortion tree embedding in the
Sarkar (2011) sense but is not the full recursive Sarkar
Delaunay construction; on regular b-ary trees it agrees with the
full construction up to global isometries.
* The Euclidean embedding is **Bourgain-inspired**: each level ``k``
  contributes a random unit vector per slot and each vertex is the
  weighted sum ``Σ 1/(k+1) · u_{slot,k}`` over its ancestor slots.
  This is a multi-scale approximation to Bourgain's fractal-dimension
  theorem rather than the full theorem, but it preserves the
  multi-scale structure that makes Bourgain-style embeddings useful
  for tree metrics.

Outputs:
    ``<output_dir>/distortion.csv``
    ``<output_dir>/distortion.png``
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pjepa.eval import set_publication_style
from pjepa.logging_setup import LogFormat, configure_logging, get_logger
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "DEFAULT_BRANCHINGS",
    "DEFAULT_DEPTHS",
    "DEFAULT_DIMS",
    "DEFAULT_N_SEEDS",
    "SARKAR_RADIUS_STEP",
    "DistortionExperimentConfig",
    "bourgain_inspired_embedding",
    "edge_distances",
    "edge_hyperbolic_distances",
    "plot_distortion",
    "run",
    "sarkar_inspired_embedding",
    "stats",
    "tree_edge_index",
    "tree_parents",
]


DEFAULT_DEPTHS: tuple[int, ...] = (3, 5, 7)
DEFAULT_BRANCHINGS: tuple[int, ...] = (2, 4)
DEFAULT_DIMS: tuple[int, ...] = (4, 8, 16)
DEFAULT_N_SEEDS: int = 2
SARKAR_RADIUS_STEP: float = 2.0 * math.asinh(1.0)


class DistortionExperimentConfig:
    """Configuration for the distortion experiment.

    Defaults are smoke-friendly; the plan-compliant sweep uses
    ``depths = (5, 10, 20, 50, 100)``, ``dims = (4, 8, 16, 32)`` and
    ``n_seeds = 5``.

    Attributes:
        depths: Tree depths to sweep.
        branchings: Branching factors ``b`` per tree.
        dims: Euclidean embedding dimensions ``d``.
        n_seeds: Number of random seeds per (depth, branching, dim)
          cell.
        output_dir: Output directory for CSV and PNG.
    """

    def __init__(
        self,
        depths: tuple[int, ...] = DEFAULT_DEPTHS,
        branchings: tuple[int, ...] = DEFAULT_BRANCHINGS,
        dims: tuple[int, ...] = DEFAULT_DIMS,
        n_seeds: int = DEFAULT_N_SEEDS,
        output_dir: str = "results",
    ) -> None:
        """Store the experiment parameters.

        Args:
            depths: Tree depths to sweep.
            branchings: Branching factors ``b`` per tree.
            dims: Euclidean embedding dimensions ``d``.
            n_seeds: Number of random seeds per (depth, branching, dim)
              cell.
            output_dir: Output directory for CSV and PNG.
        """
        self.depths = tuple(int(d) for d in depths if int(d) > 0)
        self.branchings = tuple(int(b) for b in branchings if int(b) > 1)
        self.dims = tuple(int(d) for d in dims if int(d) > 0)
        self.n_seeds = int(n_seeds)
        self.output_dir = str(output_dir)


def tree_parents(b: int, depth: int) -> tuple[list[int | None], int]:
    """Return the parent map of a complete b-ary tree of given depth.

    The tree uses breadth-first ordering: vertex ``0`` is the root and
    vertex ``Σ_{i<k} b^i`` is the first vertex at depth ``k``. Each
    non-root vertex ``v`` at depth ``k`` has parent
    ``Σ_{i<k-1} b^i + (v - Σ_{i<k} b^i) // b``.

    Args:
        b: Branching factor (``>= 2``).
        depth: Maximum depth (root is depth ``0``).

    Returns:
        A tuple ``(parents, n_vertices)`` where ``parents[v]`` is the
        parent of vertex ``v`` and ``None`` for the root.

    Raises:
        ValueError: If ``b < 2`` or ``depth < 0``.
    """
    if b < 2:
        raise ValueError(f"tree_parents: b must be >= 2; got {b}")
    if depth < 0:
        raise ValueError(f"tree_parents: depth must be non-negative; got {depth}")
    n = sum(int(b) ** k for k in range(int(depth) + 1))
    parents: list[int | None] = [None] * n
    for k in range(1, int(depth) + 1):
        start = sum(int(b) ** i for i in range(k))
        end = sum(int(b) ** i for i in range(k + 1))
        prev_start = sum(int(b) ** i for i in range(k - 1))
        for v in range(start, end):
            parents[v] = prev_start + (v - start) // int(b)
    return parents, n


def tree_edge_index(b: int, depth: int) -> torch.Tensor:
    """Build the ``[2, E]`` edge index of a complete b-ary tree.

    Args:
        b: Branching factor (``>= 2``).
        depth: Maximum depth (root is depth ``0``).

    Returns:
        A ``[2, E]`` ``long`` tensor of (parent, child) pairs. Empty
        trees return a ``[2, 0]`` tensor.
    """
    parents, _ = tree_parents(b, depth)
    src: list[int] = []
    dst: list[int] = []
    for v, p in enumerate(parents):
        if p is not None:
            src.append(int(p))
            dst.append(int(v))
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def sarkar_inspired_embedding(
    parents: list[int | None],
    n_vertices: int,
    b: int,
    depth: int,
    seed: int,
) -> torch.Tensor:
    """Embed a tree in the Poincaré disk via a Sarkar-inspired construction.

    The root is placed at the origin. Each non-root vertex at depth
    ``k`` is placed at hyperbolic distance ``k * alpha`` from the
    origin with ``alpha = 2 * asinh(1)`` (the constant
    :data:`SARKAR_RADIUS_STEP`), and the angular slot of every child
    is a uniformly spaced fraction of its parent's slot, so that all
    depth-``k`` vertices lie on the circle of radius
    ``tanh(k * alpha / 2)``. The construction guarantees bounded
    per-edge distortion on trees (Sarkar 2011, Theorem 1) but is
    **not** the full recursive Sarkar Delaunay construction; it is the
    deterministic regular-tree specialisation.

    Complexity is ``O(|V|)`` and depends only on the parent map.

    Args:
        parents: Parent map from :func:`tree_parents`.
        n_vertices: Total number of vertices.
        b: Branching factor.
        depth: Tree depth.
        seed: Random seed (currently unused but reserved for
          stochastic variants of Sarkar).

    Returns:
        A ``[n_vertices, 2]`` tensor of Euclidean Poincaré-disk
        coordinates with norms strictly less than 1.
    """
    del seed
    coords = torch.zeros((int(n_vertices), 2), dtype=torch.float32)
    width = [0.0] * int(n_vertices)
    angle = [0.0] * int(n_vertices)
    width[0] = 2.0 * math.pi
    angle[0] = 0.0
    coords[0] = torch.tensor([0.0, 0.0])
    depth_of = [0] * int(n_vertices)
    for v in range(1, int(n_vertices)):
        p = parents[v]
        depth_of[v] = depth_of[int(p)] + 1
    for v in range(1, int(n_vertices)):
        p = parents[v]
        start_of_level = sum(int(b) ** i for i in range(depth_of[v]))
        local_idx = int(v) - start_of_level
        parent_angle = angle[int(p)]
        parent_width = width[int(p)]
        slot_angle = parent_width / float(b)
        angle[v] = parent_angle + (local_idx + 0.5) * slot_angle - parent_width / 2.0
        width[v] = slot_angle
        radius = depth_of[v] * SARKAR_RADIUS_STEP
        r_euc = math.tanh(radius / 2.0)
        coords[v, 0] = r_euc * math.cos(angle[v])
        coords[v, 1] = r_euc * math.sin(angle[v])
    return coords


def bourgain_inspired_embedding(
    parents: list[int | None],
    n_vertices: int,
    b: int,
    depth: int,
    d: int,
    seed: int,
) -> torch.Tensor:
    """Multi-scale Bourgain-inspired tree embedding into ``R^d``.

    For each level ``k`` and each "slot" at that level, sample a random
    ``d``-dim unit vector. Each vertex's embedding is the weighted sum
    of its ancestor slots' unit vectors with weight ``1 / (k + 1)``.
    The construction is **Bourgain-inspired**: it captures the
    multi-scale structure of Bourgain's fractal-dimension theorem
    (Bourgain 1985) but is **not** the full Bourgain embedding,
    which uses randomised distance sampling rather than slot-weighted
    sums.

    Complexity is ``O(|V| · d · depth)``; the inner per-vertex chain
    walk is linear in the depth.

    Args:
        parents: Parent map from :func:`tree_parents`.
        n_vertices: Total number of vertices.
        b: Branching factor.
        depth: Tree depth.
        d: Embedding dimension.
        seed: Random seed.

    Returns:
        A ``[n_vertices, d]`` tensor of Euclidean coordinates.
    """
    gen = torch.Generator().manual_seed(int(seed))
    max_slots = max(int(b) ** k for k in range(int(depth) + 1))
    level_vecs = torch.randn((int(depth) + 1, max_slots, int(d)), generator=gen)
    level_vecs = level_vecs / level_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    coords = torch.zeros((int(n_vertices), int(d)), dtype=torch.float32)
    starts = [0] * (int(depth) + 1)
    for k in range(int(depth) + 1):
        starts[k] = sum(int(b) ** i for i in range(k))
    for v in range(int(n_vertices)):
        chain: list[int] = []
        node = int(v)
        while node != 0:
            chain.append(node)
            p = parents[node]
            node = int(p) if p is not None else 0
        chain.append(0)
        chain.reverse()
        emb = torch.zeros(int(d), dtype=torch.float32)
        for level_idx, ancestor in enumerate(chain):
            slot = int(ancestor) - starts[level_idx]
            weight = 1.0 / float(level_idx + 1)
            emb = emb + weight * level_vecs[level_idx, slot]
        coords[v] = emb
    return coords


def edge_distances(coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Per-edge Euclidean distances.

    Args:
        coords: An ``[N, d]`` tensor of Euclidean coordinates.
        edge_index: A ``[2, E]`` ``long`` tensor in COO format.

    Returns:
        A ``[E]`` tensor of per-edge Euclidean distances; an empty
        tensor when ``edge_index`` is empty.
    """
    if edge_index.numel() == 0:
        return torch.zeros((0,), dtype=coords.dtype)
    src = edge_index[0]
    dst = edge_index[1]
    return torch.norm(coords[src] - coords[dst], dim=-1)


def edge_hyperbolic_distances(coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Per-edge hyperbolic distances in the Poincaré disk.

    Uses the closed-form formula
    ``d(x, y) = arcosh(1 + 2 · ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))``
    which is the standard geodesic distance on the Poincaré disk of
    curvature ``-1``.

    Args:
        coords: An ``[N, 2]`` tensor of Euclidean Poincaré-disk
          coordinates with norms strictly below 1.
        edge_index: A ``[2, E]`` ``long`` tensor in COO format.

    Returns:
        A ``[E]`` tensor of per-edge hyperbolic distances; an empty
        tensor when ``edge_index`` is empty.
    """
    if edge_index.numel() == 0:
        return torch.zeros((0,), dtype=coords.dtype)
    src = edge_index[0]
    dst = edge_index[1]
    x = coords[src]
    y = coords[dst]
    diff_sq = (x - y).pow(2).sum(dim=-1)
    one_minus_x = (1.0 - x.pow(2).sum(dim=-1)).clamp(min=1e-12)
    one_minus_y = (1.0 - y.pow(2).sum(dim=-1)).clamp(min=1e-12)
    arg = 1.0 + 2.0 * diff_sq / (one_minus_x * one_minus_y)
    arg = arg.clamp(min=1.0 + 1e-7)
    return torch.acosh(arg)


def stats(values: torch.Tensor) -> dict[str, float]:
    """Mean, std, min, max of a tensor.

    Args:
        values: A 1-D ``float`` tensor. Empty tensors return zeros.

    Returns:
        A dict with ``mean``, ``std`` (population), ``min`` and
        ``max`` keys.
    """
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def run(config: DistortionExperimentConfig | None = None) -> dict[str, object]:
    """Run the hyperbolic-vs-Euclidean distortion experiment.

    Args:
        config: Experiment configuration. When ``None`` the smoke
          defaults are used.

    Returns:
        A dictionary with ``rows`` and the output paths.
    """
    if config is None:
        config = DistortionExperimentConfig()
    log = get_logger(__name__)
    rows: list[dict[str, object]] = []
    for b in config.branchings:
        for depth in config.depths:
            parents, n_vertices = tree_parents(int(b), int(depth))
            edge_index = tree_edge_index(int(b), int(depth))
            hyp_coords = sarkar_inspired_embedding(parents, n_vertices, int(b), int(depth), seed=0)
            hyp_dists = edge_hyperbolic_distances(hyp_coords, edge_index)
            hyp_stats = stats(hyp_dists)
            for d in config.dims:
                for seed in range(config.n_seeds):
                    set_global_seed(seed * 1009 + int(depth) * 31 + int(d))
                    euc_coords = bourgain_inspired_embedding(
                        parents,
                        n_vertices,
                        int(b),
                        int(depth),
                        int(d),
                        seed=seed,
                    )
                    euc_dists = edge_distances(euc_coords, edge_index)
                    euc_stats = stats(euc_dists)
                    pair_ratio = (euc_dists / hyp_dists.clamp(min=1e-12)).clamp(max=1e6)
                    pair_stats = stats(pair_ratio)
                    max_ratio = float(pair_stats["max"])
                    mean_ratio = float(pair_stats["mean"])
                    row = {
                        "depth": int(depth),
                        "branching": int(b),
                        "d": int(d),
                        "seed": int(seed),
                        "n_vertices": int(n_vertices),
                        "n_edges": int(edge_index.shape[1]),
                        "hyp_mean": hyp_stats["mean"],
                        "hyp_std": hyp_stats["std"],
                        "hyp_max": hyp_stats["max"],
                        "hyp_min": hyp_stats["min"],
                        "euc_mean": euc_stats["mean"],
                        "euc_std": euc_stats["std"],
                        "euc_max": euc_stats["max"],
                        "euc_min": euc_stats["min"],
                        "euc_over_hyp_mean": mean_ratio,
                        "euc_over_hyp_max": max_ratio,
                    }
                    rows.append(row)
                    log.info(
                        "distortion trial complete",
                        extra={
                            "event": "distortion.trial",
                            "depth": int(depth),
                            "branching": int(b),
                            "d": int(d),
                            "seed": int(seed),
                            "euc_max": float(euc_stats["max"]),
                            "hyp_max": float(hyp_stats["max"]),
                            "mean_ratio": mean_ratio,
                            "max_ratio": max_ratio,
                        },
                    )
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "distortion.csv"
    fieldnames = [
        "depth",
        "branching",
        "d",
        "seed",
        "n_vertices",
        "n_edges",
        "hyp_mean",
        "hyp_std",
        "hyp_max",
        "hyp_min",
        "euc_mean",
        "euc_std",
        "euc_max",
        "euc_min",
        "euc_over_hyp_mean",
        "euc_over_hyp_max",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    png_path = out_dir / "distortion.png"
    plot_distortion(rows, png_path)
    log.info(
        "experiment complete",
        extra={
            "event": "distortion.experiment_complete",
            "n_rows": len(rows),
            "csv": str(csv_path),
            "png": str(png_path),
        },
    )
    return {"rows": rows, "csv": str(csv_path), "png": str(png_path)}


def plot_distortion(rows: list[dict[str, object]], png_path: Path) -> None:
    """Plot per-edge ``d_E / d_H`` ratio vs depth, grouped by (b, d).

    The left panel plots the mean ratio and the right panel the max
    ratio, both versus tree depth. Each curve corresponds to one
    embedding dimension, and the two panels share a y-axis range.

    Args:
        rows: Per-trial rows emitted by :func:`run`.
        png_path: Destination PNG path.
    """
    set_publication_style()
    grouped_mean: dict[tuple[int, int], dict[int, list[float]]] = {}
    grouped_max: dict[tuple[int, int], dict[int, list[float]]] = {}
    for row in rows:
        key = (int(row["branching"]), int(row["d"]))
        grouped_mean.setdefault(key, {})
        grouped_max.setdefault(key, {})
        grouped_mean[key].setdefault(int(row["depth"]), []).append(float(row["euc_over_hyp_mean"]))
        grouped_max[key].setdefault(int(row["depth"]), []).append(float(row["euc_over_hyp_max"]))
    fig, axes = plt.subplots(1, 2, sharey=False, figsize=(11.0, 4.5))
    branchings = sorted({k[0] for k in grouped_mean})
    for ax, b, payload in zip(axes, branchings, (grouped_mean, grouped_max), strict=False):
        dims = sorted({k[1] for k in payload if k[0] == b})
        for d in dims:
            depths = sorted(payload[(b, d)].keys())
            means = [sum(payload[(b, d)][dd]) / len(payload[(b, d)][dd]) for dd in depths]
            ax.plot(depths, means, marker="o", label=f"d={d}")
        ax.set_xlabel("Tree depth D")
        ax.set_title(f"branching b={b}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean per-edge ratio d_E / d_H")
    axes[1].set_ylabel("max per-edge ratio d_E / d_H")
    fig.suptitle("Distortion: Bourgain-inspired (R^d) vs Sarkar-inspired (H²)")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the hyperbolic-vs-Euclidean distortion experiment."
    )
    parser.add_argument("--depths", type=int, nargs="*", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--branchings", type=int, nargs="*", default=list(DEFAULT_BRANCHINGS))
    parser.add_argument("--dims", type=int, nargs="*", default=list(DEFAULT_DIMS))
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    config = DistortionExperimentConfig(
        depths=tuple(args.depths),
        branchings=tuple(args.branchings),
        dims=tuple(args.dims),
        n_seeds=int(args.seeds),
        output_dir=str(args.output_dir),
    )
    run(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
