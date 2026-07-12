"""Experiment runner for the (1 - 1/e) submodular-retrieval claim."""

from __future__ import annotations

import random
from itertools import combinations

import torch

from pjepa.graphs import TypedAttributedGraph
from pjepa.retrieval import FacilityLocationUtility, GreedyRetrieval

__all__ = ["run"]


def _random_submodular(n: int, seed: int) -> FacilityLocationUtility:
    """Construct a random facility-location utility on ``n`` items."""
    torch.manual_seed(seed)
    features = torch.randn((n, 4))
    return FacilityLocationUtility(vertex_features=features)


def _brute_opt(util: FacilityLocationUtility, n: int, budget: int) -> float:
    """Compute the exact optimal facility-location value by enumeration."""
    best = 0.0
    for combo in combinations(range(n), budget):
        subset = torch.tensor(list(combo), dtype=torch.long)
        value = float(util(subset, torch.zeros((1, util.vertex_features.shape[1]))))
        if value > best:
            best = value
    return best


def run() -> dict[str, object]:
    """Run the synthetic (1 - 1/e) validation experiment."""
    threshold = 1.0 - 1.0 / torch.e
    threshold = float(threshold)
    n_seeds = 3
    budgets = (2, 3, 4)
    rows: list[dict[str, float]] = []
    for seed in range(n_seeds):
        n = 8
        util = _random_submodular(n, seed=seed)
        for budget in budgets:
            opt = _brute_opt(util, n, budget)
            graph = TypedAttributedGraph(
                vertex_features=util.vertex_features,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            retriever = GreedyRetrieval(budget=budget)
            result = retriever.select(graph, torch.zeros((1, util.vertex_features.shape[1])), utility=util)
            ratio = result.utility / opt if opt > 0 else 1.0
            rows.append(
                {
                    "seed": seed,
                    "budget": float(budget),
                    "opt": float(opt),
                    "greedy": float(result.utility),
                    "ratio": float(ratio),
                    "passes_threshold": bool(ratio >= threshold - 1e-5),
                }
            )
    return {
        "threshold": threshold,
        "rows": rows,
        "all_pass": all(r["passes_threshold"] for r in rows),
    }


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    print(run())