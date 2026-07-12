"""Decoupling measurement (Phase 11 Exp G).

Validates the paper's claim that inference cost is independent of
persistent graph size. We measure wall-clock per inference step as
a function of the persistent graph size ``N`` and the working-graph
budget ``B``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from pjepa.graphs import TypedAttributedGraph
from pjepa.logging_setup import configure_logging, get_logger, LogFormat

__all__ = ["run_decoupling_measurement"]


def _build_chain_graph(n_vertices: int, feature_dim: int) -> TypedAttributedGraph:
    """Build a chain graph with ``n_vertices`` nodes."""
    if n_vertices < 2:
        edges = []
    else:
        edges = [(i, i + 1) for i in range(n_vertices - 1)]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).T
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return TypedAttributedGraph(
        vertex_features=torch.randn((n_vertices, feature_dim)),
        edge_index=edge_index,
        edge_features=torch.zeros((edge_index.shape[1], 1)),
    )


def _measure_inference(
    n_vertices: int,
    feature_dim: int,
    n_trials: int = 5,
) -> dict[str, float]:
    """Measure the average inference time on a graph of size ``n_vertices``."""
    from pjepa.encoders import EuclideanMPNN

    encoder = EuclideanMPNN(input_dim=feature_dim, hidden_dim=64, num_layers=3, output_dim=64)
    encoder.eval()
    times = []
    for _ in range(n_trials):
        g = _build_chain_graph(n_vertices, feature_dim)
        start = time.perf_counter()
        with torch.no_grad():
            _ = encoder(g)
        times.append(time.perf_counter() - start)
    return {
        "n_vertices": float(n_vertices),
        "mean_seconds": float(sum(times) / len(times)),
        "std_seconds": float((sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5),
    }


def _measure_retrieval(
    n_vertices: int,
    feature_dim: int,
    budget: int,
    n_trials: int = 5,
) -> dict[str, float]:
    """Measure the retrieval time on a graph of size ``n_vertices``."""
    from pjepa.retrieval import FacilityLocationUtility, GreedyRetrieval

    times = []
    for _ in range(n_trials):
        g = _build_chain_graph(n_vertices, feature_dim)
        utility = FacilityLocationUtility(vertex_features=g.vertex_features)
        retriever = GreedyRetrieval(budget=budget)
        start = time.perf_counter()
        retriever.select(g, torch.zeros((1, feature_dim)), utility=utility)
        times.append(time.perf_counter() - start)
    return {
        "n_vertices": float(n_vertices),
        "budget": float(budget),
        "mean_seconds": float(sum(times) / len(times)),
    }


def run_decoupling_measurement(
    output_dir: str = "results",
    n_sizes: tuple[int, ...] = (50, 100, 200, 400, 800),
    feature_dim: int = 32,
    budgets: tuple[int, ...] = (16, 32, 64),
) -> dict[str, object]:
    """Run the decoupling measurement and write a CSV report."""
    log = get_logger(__name__)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    log.info("measuring inference cost", extra={"event": "decoupling.inference_start"})
    for n in n_sizes:
        stats = _measure_inference(n, feature_dim)
        rows.append({"what": "encoder", **{k: v for k, v in stats.items() if k != "n_vertices"}, "n_vertices": n})
        log.info(
            "inference measured",
            extra={"event": "decoupling.inference", "n_vertices": n, "mean_seconds": stats["mean_seconds"]},
        )
    for n in n_sizes:
        for b in budgets:
            stats = _measure_retrieval(n, feature_dim, budget=b)
            rows.append({"what": "retrieval", **stats})
    with (out / "decoupling.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["what", "n_vertices", "budget", "mean_seconds", "std_seconds"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {"rows": rows}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the inference–storage decoupling measurement.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--feature-dim", type=int, default=32)
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    run_decoupling_measurement(output_dir=args.output_dir, feature_dim=args.feature_dim)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())