"""Experiment A — Submodular retrieval quality (Phase 4).

Validates Theorem 3 (Nemhauser-Wolsey-Fisher 1978) by comparing the
greedy working-graph retriever to the exact optimum on random monotone
submodular facility-location utilities. The greedy algorithm achieves
the ``(1 - 1/e) ≈ 0.6321`` approximation guarantee; this experiment
measures the empirical ratio over a sweep of vertex counts ``n`` and
cardinality budgets ``B``.

For tractable instances (``C(n, B) ≤ 10**6``) the optimum is computed by
exhaustive enumeration; for larger instances the optimum is
**approximated** by the best of :data:`PSEUDO_RESTARTS` random subsets.
The pseudo-optimum is therefore a *lower* bound on the exact optimum
``OPT``. As a consequence the ratio ``greedy / pseudo_opt`` is an
*upper* bound on the true ``greedy / OPT`` ratio: rows passing the
``(1 - 1/e)`` check with the pseudo-optimum do not, by themselves,
prove the theorem on that instance. A pseudo-row that *fails* the
check is genuinely counter-evident (because if the true ratio were
already at the threshold, the pseudo ratio could only be larger).
Rows using the exact optimum are the only ones that constitute a
strict verification of the bound.

Outputs:
    ``<output_dir>/retrieval_quality.csv``
    ``<output_dir>/retrieval_quality.png``
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pjepa.eval import set_publication_style
from pjepa.graphs import TypedAttributedGraph
from pjepa.logging_setup import LOG_FORMAT_JSON, configure_logging, get_logger
from pjepa.retrieval import FacilityLocationUtility, GreedyRetrieval
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "BRUTE_FORCE_CAP",
    "DEFAULT_BUDGETS",
    "DEFAULT_NS",
    "DEFAULT_N_SEEDS",
    "PSEUDO_RESTARTS",
    "RetrievalExperimentConfig",
    "brute_force_optimum",
    "greedy_utility",
    "plot_ratio_vs_budget",
    "pseudo_optimum",
    "random_submodular",
    "run",
]


DEFAULT_NS: tuple[int, ...] = (40, 50)
DEFAULT_BUDGETS: tuple[int, ...] = (5, 7)
DEFAULT_N_SEEDS: int = 3
BRUTE_FORCE_CAP: int = 3_000_000
PSEUDO_RESTARTS: int = 8


class RetrievalExperimentConfig:
    """Configuration for the (1 - 1/e) retrieval experiment.

    The defaults are smoke-friendly: ``ns = (40, 50)``, ``budgets = (5, 7)``
    and ``n_seeds = 3``. The full plan sweep uses ``(50, 100, 500)`` for
    ``ns`` and ``(5, 10, 20)`` for ``budgets``.

    Attributes:
        ns: Vertex counts to sweep over.
        budgets: Retrieval budgets to sweep over.
        n_seeds: Number of random submodular instances per (n, B) pair.
        observation_dim: Number of observation features used by the
          facility-location utility.
        feature_dim: Number of vertex features per graph.
        output_dir: Directory where the CSV and PNG are written.
    """

    def __init__(
        self,
        ns: tuple[int, ...] = DEFAULT_NS,
        budgets: tuple[int, ...] = DEFAULT_BUDGETS,
        n_seeds: int = DEFAULT_N_SEEDS,
        observation_dim: int = 4,
        feature_dim: int = 4,
        output_dir: str = "results",
    ) -> None:
        """Store the experiment parameters.

        Args:
            ns: Vertex counts to sweep over.
            budgets: Retrieval budgets to sweep over.
            n_seeds: Number of random submodular instances per (n, B)
              pair.
            observation_dim: Observation feature dimension.
            feature_dim: Vertex feature dimension.
            output_dir: Directory where outputs are written.
        """
        self.ns = tuple(int(n) for n in ns if int(n) > 0)
        self.budgets = tuple(int(b) for b in budgets if int(b) > 0)
        self.n_seeds = int(n_seeds)
        self.observation_dim = int(observation_dim)
        self.feature_dim = int(feature_dim)
        self.output_dir = str(output_dir)


def random_submodular(
    n: int,
    seed: int,
    feature_dim: int,
    observation_dim: int,
) -> tuple[FacilityLocationUtility, torch.Tensor]:
    """Construct a random facility-location utility and observation.

    The vertex features and observation are drawn from a standard
    normal distribution with the supplied ``seed``. The facility-location
    utility they induce is provably monotone submodular (Nemhauser-
    Wolsey-Fisher 1978).

    Args:
        n: Number of vertices.
        seed: Random seed for the local generator.
        feature_dim: Vertex feature dimension.
        observation_dim: Observation dimension.

    Returns:
        A tuple ``(utility, observation)``.
    """
    g = torch.Generator().manual_seed(int(seed))
    features = torch.randn((int(n), int(feature_dim)), generator=g)
    observation = torch.randn((int(observation_dim), int(feature_dim)), generator=g)
    return FacilityLocationUtility(vertex_features=features), observation


def brute_force_optimum(
    util: FacilityLocationUtility,
    n: int,
    budget: int,
    observation: torch.Tensor,
) -> float | None:
    """Compute the exact optimum by vectorised enumeration.

    Enumerates every ``C(n, budget)`` subset, scores each under the
    facility-location utility, and returns the maximum. Returns
    ``None`` when the combinatorial count exceeds
    :data:`BRUTE_FORCE_CAP`, in which case the caller should fall back
    to :func:`pseudo_optimum`.

    Args:
        util: The facility-location utility.
        n: Number of vertices in the persistent graph.
        budget: Cardinality budget.
        observation: The observation features.

    Returns:
        The exact optimum value, or ``None`` when ``C(n, budget)``
        exceeds :data:`BRUTE_FORCE_CAP`.
    """
    if budget <= 0 or budget > n:
        return 0.0
    combos = math.comb(n, budget)
    if combos > BRUTE_FORCE_CAP:
        return None
    subsets = torch.tensor(list(combinations(range(n), int(budget))), dtype=torch.long)
    sub_feats = util.vertex_features[subsets]
    sub_n = torch.nn.functional.normalize(sub_feats, dim=-1)
    obs_n = torch.nn.functional.normalize(observation, dim=-1)
    sims = torch.einsum("cbd,md->cmb", sub_n, obs_n)
    best_per_obs = sims.max(dim=-1).values.clamp(min=0.0)
    coverage = best_per_obs.sum(dim=-1)
    return float(coverage.max().item())


def pseudo_optimum(
    util: FacilityLocationUtility,
    n: int,
    budget: int,
    observation: torch.Tensor,
    seed: int,
    n_starts: int = PSEUDO_RESTARTS,
) -> float:
    """Approximate the optimum by the best of multiple random subsets.

    Random subsets of cardinality ``budget`` cannot exceed the optimum
    of the FL utility, so the returned value is a **lower** bound on
    ``OPT``. Consequently ``greedy / pseudo_opt`` is an **upper** bound
    on the true ``greedy / OPT`` ratio: a row passing the ``(1-1/e)``
    check on pseudo does not by itself verify the theorem, while a
    pseudo-row that fails the check would also fail on the exact
    optimum.

    Args:
        util: The facility-location utility.
        n: Number of vertices.
        budget: Cardinality budget.
        observation: The observation features.
        seed: Random seed for the random restarts.
        n_starts: Number of random-start subsets to evaluate.

    Returns:
        A non-negative lower bound on the optimum.
    """
    gen = torch.Generator().manual_seed(int(seed))
    best = 0.0
    for _ in range(int(n_starts)):
        perm = torch.randperm(int(n), generator=gen)
        subset = perm[: int(budget)]
        value = float(util(subset, observation))
        if value > best:
            best = value
    return best


def greedy_utility(
    util: FacilityLocationUtility,
    n: int,
    budget: int,
    observation: torch.Tensor,
) -> float:
    """Return the greedy utility on the given submodular problem.

    Builds an edgeless :class:`TypedAttributedGraph` from ``util``'s
    vertex features and runs :class:`GreedyRetrieval` against it.

    Args:
        util: The facility-location utility.
        n: Number of vertices.
        budget: Cardinality budget.
        observation: The observation features.

    Returns:
        The cumulative utility achieved by the greedy selection.
    """
    g = TypedAttributedGraph(
        vertex_features=util.vertex_features,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    retriever = GreedyRetrieval(budget=int(budget))
    result = retriever.select(g, observation, utility=util)
    return float(result.utility)


def run(config: RetrievalExperimentConfig | None = None) -> dict[str, object]:
    """Run the synthetic (1 - 1/e) validation experiment.

    Args:
        config: Experiment configuration. When ``None`` the smoke
          defaults are used.

    Returns:
        A dictionary with ``threshold``, ``rows``, ``all_pass``,
        ``csv`` and ``png`` keys. ``all_pass`` is ``True`` iff every
        row met the threshold; if no rows were generated, ``all_pass``
        is ``False`` to avoid reporting a vacuous pass.
    """
    if config is None:
        config = RetrievalExperimentConfig()
    log = get_logger(__name__)
    threshold = 1.0 - 1.0 / math.e
    rows: list[dict[str, object]] = []
    for n in config.ns:
        for budget in config.budgets:
            for seed in range(config.n_seeds):
                set_global_seed(seed * 1_000_003 + int(n) * 31 + int(budget))
                util, obs = random_submodular(
                    n=int(n),
                    seed=seed,
                    feature_dim=config.feature_dim,
                    observation_dim=config.observation_dim,
                )
                opt_exact = brute_force_optimum(util, int(n), int(budget), obs)
                if opt_exact is None:
                    pseudo = pseudo_optimum(util, int(n), int(budget), obs, seed=seed)
                    opt_used = pseudo
                    opt_kind = "pseudo"
                else:
                    opt_used = opt_exact
                    opt_kind = "exact"
                greedy = greedy_utility(util, int(n), int(budget), obs)
                ratio = greedy / opt_used if opt_used > 0.0 else 1.0
                passes = bool(ratio >= threshold - 1e-5)
                rows.append(
                    {
                        "n": int(n),
                        "budget": int(budget),
                        "seed": int(seed),
                        "opt": float(opt_used),
                        "opt_kind": opt_kind,
                        "greedy": float(greedy),
                        "ratio": float(ratio),
                        "passes_threshold": passes,
                    }
                )
                log.info(
                    "retrieval trial complete",
                    extra={
                        "event": "retrieval.trial",
                        "n": int(n),
                        "budget": int(budget),
                        "seed": int(seed),
                        "opt_kind": opt_kind,
                        "opt": float(opt_used),
                        "greedy": float(greedy),
                        "ratio": float(ratio),
                        "passes_threshold": passes,
                    },
                )
    all_pass = bool(rows) and all(bool(r["passes_threshold"]) for r in rows)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "retrieval_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "n",
                "budget",
                "seed",
                "opt",
                "opt_kind",
                "greedy",
                "ratio",
                "passes_threshold",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    png_path = out_dir / "retrieval_quality.png"
    plot_ratio_vs_budget(rows, threshold, png_path)
    log.info(
        "experiment complete",
        extra={
            "event": "retrieval.experiment_complete",
            "n_rows": len(rows),
            "all_pass": all_pass,
            "csv": str(csv_path),
            "png": str(png_path),
        },
    )
    return {
        "threshold": float(threshold),
        "rows": rows,
        "all_pass": all_pass,
        "csv": str(csv_path),
        "png": str(png_path),
    }


def plot_ratio_vs_budget(
    rows: list[dict[str, object]],
    threshold: float,
    png_path: Path,
) -> None:
    """Plot mean greedy/opt ratio per (n, budget); write a PNG.

    Args:
        rows: Per-trial rows emitted by :func:`run`.
        threshold: The ``(1 - 1/e) ≈ 0.6321`` reference value.
        png_path: Destination PNG path.
    """
    set_publication_style()
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        key = (int(row["n"]), int(row["budget"]))
        grouped.setdefault(key, []).append(float(row["ratio"]))
    ns_sorted = sorted({k[0] for k in grouped})
    budgets_sorted = sorted({k[1] for k in grouped})
    fig, ax = plt.subplots()
    for n in ns_sorted:
        means = []
        stds = []
        for b in budgets_sorted:
            values = grouped.get((n, b), [])
            if not values:
                means.append(float("nan"))
                stds.append(0.0)
            else:
                m = sum(values) / len(values)
                var = sum((v - m) ** 2 for v in values) / max(len(values) - 1, 1)
                means.append(m)
                stds.append(var**0.5)
        ax.errorbar(
            budgets_sorted,
            means,
            yerr=stds,
            marker="o",
            label=f"n={n}",
        )
    ax.axhline(threshold, color="black", linestyle=":", label=f"(1-1/e) ≈ {threshold:.3f}")
    ax.set_xlabel("Budget B")
    ax.set_ylabel("Greedy / OPT ratio")
    ax.set_title("Submodular retrieval quality vs budget")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.savefig(png_path)
    plt.close(fig)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the (1 - 1/e) submodular-retrieval validation experiment."
    )
    parser.add_argument("--ns", type=int, nargs="*", default=list(DEFAULT_NS))
    parser.add_argument("--budgets", type=int, nargs="*", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--observation-dim", type=int, default=4)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LOG_FORMAT_JSON)
    config = RetrievalExperimentConfig(
        ns=tuple(args.ns),
        budgets=tuple(args.budgets),
        n_seeds=int(args.seeds),
        feature_dim=int(args.feature_dim),
        observation_dim=int(args.observation_dim),
        output_dir=str(args.output_dir),
    )
    summary = run(config)
    log = get_logger(__name__)
    log.info(
        "retrieval experiment summary",
        extra={
            "event": "retrieval.summary",
            "all_pass": summary["all_pass"],
            "threshold": summary["threshold"],
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
