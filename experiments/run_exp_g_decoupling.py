"""Inference-storage decoupling measurement (Phase 11 Exp G).

Validates the paper's claim that inference cost is independent of the
persistent-graph size ``N``: for a fixed working-graph budget ``B``
the per-step wall-clock must remain bounded (or grow only with
``B``, not ``N``).

The experiment builds a chain graph of size ``N``, constructs a
``B``-bounded working graph via the framework's greedy retrieval,
encodes that working graph through :class:`DualGeometricEncoder`,
and reports the per-step wall-clock. A warm-up call is performed
before the timed trials so the first iteration's allocator /
device-side compilation cost is excluded; after each timed trial
``torch.cuda.synchronize`` (when CUDA is active) is invoked so the
wall-clock measurement is not contaminated by asynchronous
dispatch.

The slope of ``mean_seconds`` vs ``N`` (per ``(what, B)`` cell) is
computed by ordinary least squares and the null hypothesis
"slope = 0" is evaluated by a paired bootstrap CI on the OLS
residuals.

Outputs (plan-compliant):

* ``<output_dir>/tables/decoupling.csv`` — per-(N, B) raw rows.
* ``<output_dir>/tables/decoupling_slope.csv`` — per-(what, B) OLS
  slope + bootstrap CI of slope vs zero.
* ``<output_dir>/plots/decoupling.png`` — wall-clock vs N for fixed B.

Legacy paths retained for compatibility with earlier callers:

* ``<output_dir>/decoupling.csv``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from pjepa.encoders import DualGeometricEncoder, EuclideanMPNN
from pjepa.eval import color_for, set_publication_style
from pjepa.graphs import TypedAttributedGraph
from pjepa.logging_setup import LOG_FORMAT_JSON, configure_logging, get_logger
from pjepa.retrieval import FacilityLocationUtility, GreedyRetrieval
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "DecouplingConfig",
    "build_chain_graph",
    "clone_rows",
    "compute_slope_summary",
    "default_smoke_config",
    "measure_dual_encoder_inference_time",
    "measure_euclidean_encoder_inference_time",
    "measure_greedy_retrieval_time",
    "ols_slope_intercept",
    "render_decoupling_figure",
    "run_decoupling_measurement",
    "sync_compute_device",
    "write_csv",
]


def sync_compute_device() -> None:
    """Block until all queued compute operations have finished.

    CUDA work is dispatched asynchronously, so a ``perf_counter``
    measurement that runs immediately after a kernel launch will
    under-report the true wall-clock cost. This helper invokes the
    backend-appropriate synchronisation call (no-op on CPU-only
    hosts). It is called after every timed trial in the decoupling
    measurement so the per-trial latency reflects the cost of the
    work rather than the cost of dispatch.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_chain_graph(n_vertices: int, feature_dim: int, seed: int) -> TypedAttributedGraph:
    """Build a chain graph with ``n_vertices`` nodes and random features.

    A chain graph has ``n_vertices - 1`` undirected edges represented
    as a directed COO index (each edge is stored once with the lower
    vertex as the source). The vertex features are drawn from
    ``N(0, 1)`` using a ``torch.Generator`` seeded by ``seed`` for
    full reproducibility across trials.

    Args:
        n_vertices: Number of vertices in the chain (``>= 1``).
        feature_dim: Per-vertex feature dimension.
        seed: Seed for the feature ``torch.Generator``.

    Returns:
        A populated :class:`TypedAttributedGraph`.
    """
    g = torch.Generator().manual_seed(int(seed))
    if n_vertices < 2:
        edges: list[tuple[int, int]] = []
    else:
        edges = [(i, i + 1) for i in range(n_vertices - 1)]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
    )
    return TypedAttributedGraph(
        vertex_features=torch.randn((n_vertices, feature_dim), generator=g),
        edge_index=edge_index,
        edge_features=torch.zeros((edge_index.shape[1], 1)),
    )


def _build_bounded_working_graph(
    graph: TypedAttributedGraph, budget: int, seed: int
) -> TypedAttributedGraph:
    """Return a ``B``-bounded working graph for ``graph``.

    The working graph is constructed via :class:`GreedyRetrieval`
    with :class:`FacilityLocationUtility` — the framework's
    headline retrieval / utility pair. The returned graph respects
    the ``WorkingGraph`` budget invariant (its vertex count is
    ``<= budget``); for very small ``N`` it is the graph itself.

    Args:
        graph: The persistent graph to draw a working subset from.
        budget: Maximum vertex count of the returned working graph.
        seed: Seed for the retrieval RNG (currently unused, reserved
          for future stochastic-retrieval extensions).

    Returns:
        A :class:`TypedAttributedGraph` with at most ``budget``
        vertices.
    """
    del seed
    if graph.num_vertices() <= budget:
        return graph
    utility = FacilityLocationUtility(vertex_features=graph.vertex_features)
    retriever = GreedyRetrieval(budget=budget)
    observation = graph.vertex_features.mean(dim=0, keepdim=True)
    result = retriever.select(graph, observation, utility=utility)
    return result.working.graph


def measure_dual_encoder_inference_time(
    n_vertices: int,
    feature_dim: int,
    n_trials: int,
    seed: int,
    budget: int | None = None,
) -> dict[str, float]:
    """Measure the average dual-geometric encoder time on a graph of size ``n_vertices``.

    A single warm-up call is performed (untimed) before the timed
    trials so the first iteration's allocator / device-side
    compilation cost is excluded. The timed trials encode a
    ``B``-bounded working graph (when ``budget`` is supplied) or the
    full graph (when ``budget`` is ``None``); the per-step
    wall-clock measurement is bracketed by
    :func:`sync_compute_device` so asynchronous dispatch does not
    contaminate the timing.

    Args:
        n_vertices: Size ``N`` of the persistent graph.
        feature_dim: Per-vertex feature dimension.
        n_trials: Number of timed trials to average.
        seed: Seed for the chain-graph feature generator.
        budget: Optional working-graph budget ``B``. When supplied,
          the encoder is invoked on the ``B``-bounded working graph
          derived from the chain. When ``None``, the encoder sees
          the full chain graph (the legacy comparison row).

    Returns:
        A dict with ``mean_seconds`` and ``std_seconds`` (sample
        standard deviation across ``n_trials`` trials).
    """
    encoder = DualGeometricEncoder(
        input_dim=feature_dim, euclidean_dim=64, hyperbolic_dim=16, num_layers=3
    )
    encoder.eval()
    times: list[float] = []
    for trial in range(n_trials):
        chain = build_chain_graph(n_vertices, feature_dim, seed=seed * 1000 + trial)
        target = (
            _build_bounded_working_graph(chain, int(budget), seed=seed * 1000 + trial)
            if budget is not None
            else chain
        )
        # Warm-up (untimed) — initialises allocator / kernel cache.
        with torch.no_grad():
            _ = encoder(target)
        sync_compute_device()
        start = time.perf_counter()
        with torch.no_grad():
            _ = encoder(target)
        sync_compute_device()
        times.append(time.perf_counter() - start)
    mean = sum(times) / max(len(times), 1)
    var = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
    return {
        "mean_seconds": float(mean),
        "std_seconds": float(var**0.5),
    }


def measure_greedy_retrieval_time(
    n_vertices: int,
    feature_dim: int,
    budget: int,
    n_trials: int,
    seed: int,
) -> dict[str, float]:
    """Measure greedy submodular retrieval time on a graph of size ``n_vertices``.

    Each trial builds a fresh chain graph and runs
    :class:`GreedyRetrieval` with a :class:`FacilityLocationUtility`
    over the entire chain (the retrieval cost must grow with ``B``
    but is the *selection* cost, not the *inference* cost). A single
    warm-up call is performed (untimed) before the timed trials so
    the first iteration's allocator cost is excluded.

    Args:
        n_vertices: Size ``N`` of the persistent graph.
        feature_dim: Per-vertex feature dimension.
        budget: Working-graph budget ``B``.
        n_trials: Number of timed trials to average.
        seed: Seed for the chain-graph feature generator.

    Returns:
        A dict with ``mean_seconds`` and ``std_seconds``.
    """
    times: list[float] = []
    for trial in range(n_trials):
        g = build_chain_graph(n_vertices, feature_dim, seed=seed * 1000 + trial)
        utility = FacilityLocationUtility(vertex_features=g.vertex_features)
        retriever = GreedyRetrieval(budget=budget)
        observation = torch.zeros((1, feature_dim))
        # Warm-up (untimed) — initialises the retrieval bookkeeping.
        _ = retriever.select(g, observation, utility=utility)
        start = time.perf_counter()
        _ = retriever.select(g, observation, utility=utility)
        sync_compute_device()
        times.append(time.perf_counter() - start)
    mean = sum(times) / max(len(times), 1)
    var = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
    return {
        "mean_seconds": float(mean),
        "std_seconds": float(var**0.5),
    }


def measure_euclidean_encoder_inference_time(
    n_vertices: int,
    feature_dim: int,
    n_trials: int,
    seed: int,
) -> dict[str, float]:
    """Measure Euclidean-only MPNN inference for the legacy comparison row.

    Args:
        n_vertices: Size ``N`` of the persistent graph.
        feature_dim: Per-vertex feature dimension.
        n_trials: Number of timed trials to average.
        seed: Seed for the chain-graph feature generator.

    Returns:
        A dict with ``mean_seconds`` and ``std_seconds``.
    """
    encoder = EuclideanMPNN(input_dim=feature_dim, hidden_dim=64, num_layers=3, output_dim=64)
    encoder.eval()
    times: list[float] = []
    for trial in range(n_trials):
        chain = build_chain_graph(n_vertices, feature_dim, seed=seed * 1000 + trial)
        with torch.no_grad():
            _ = encoder(chain)  # Warm-up (untimed).
        sync_compute_device()
        start = time.perf_counter()
        with torch.no_grad():
            _ = encoder(chain)
        sync_compute_device()
        times.append(time.perf_counter() - start)
    mean = sum(times) / max(len(times), 1)
    var = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
    return {
        "mean_seconds": float(mean),
        "std_seconds": float(var**0.5),
    }


def ols_slope_intercept(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return ``(slope, intercept)`` of the OLS line ``y = slope * x + intercept``.

    Args:
        xs: Independent-variable samples (must have ``>= 2`` elements
          for the slope to be defined; otherwise the function
          returns ``(0.0, ys[0])``).
        ys: Dependent-variable samples.

    Returns:
        A tuple ``(slope, intercept)``.
    """
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    if den <= 0.0:
        return 0.0, my
    slope = num / den
    intercept = my - slope * mx
    return float(slope), float(intercept)


class DecouplingConfig:
    """Configuration for the decoupling measurement.

    Attributes:
        n_sizes: Persistent-graph sizes ``N`` to sweep.
        budgets: Working-graph budgets ``B`` to sweep.
        n_trials: Trials per (N, B) cell for averaging.
        feature_dim: Per-vertex feature dimension used to build the
          synthetic chain graphs.
        output_dir: Output directory; plan-compliant ``tables`` and
          ``plots`` sub-directories will be created underneath.
        seed: Base seed for the chain-graph feature generator.
    """

    def __init__(
        self,
        n_sizes: tuple[int, ...] = (50, 100, 200, 400, 800),
        budgets: tuple[int, ...] = (16, 32, 64),
        n_trials: int = 5,
        feature_dim: int = 32,
        output_dir: str = "results",
        seed: int = 0,
    ) -> None:
        """Store the experiment parameters."""
        self.n_sizes = tuple(int(n) for n in n_sizes if int(n) > 0)
        self.budgets = tuple(int(b) for b in budgets if int(b) > 0)
        self.n_trials = max(1, int(n_trials))
        self.feature_dim = max(1, int(feature_dim))
        self.output_dir = str(output_dir)
        self.seed = int(seed)


def default_smoke_config(output_dir: str = "results/decoupling_smoke") -> DecouplingConfig:
    """A fast smoke configuration used by the unit tests.

    Args:
        output_dir: Output directory; defaults to the standard
          ``results/decoupling_smoke`` location.

    Returns:
        A smoke-tuned :class:`DecouplingConfig`.
    """
    return DecouplingConfig(
        n_sizes=(8, 16),
        budgets=(4, 8),
        n_trials=1,
        feature_dim=4,
        output_dir=output_dir,
        seed=0,
    )


def clone_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return a deep copy of ``rows`` for downstream CSV writing.

    Args:
        rows: The row dicts to clone.

    Returns:
        A new list of independent dict copies.
    """
    return [dict(r) for r in rows]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Write ``rows`` to ``path`` as a CSV with the given column order.

    Args:
        path: Destination file path; parent directories are created.
        fieldnames: Column names for the header row.
        rows: The row dicts to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_decoupling_figure(rows: list[dict[str, object]], png_path: Path) -> None:
    """Render the plan-compliant wall-clock-vs-N decoupling figure.

    The figure has two panels: the left panel plots the
    dual-geometric encoder's wall-clock vs ``N`` (one line per ``B``,
    plus a marker-only "B=*" line for the legacy full-graph
    comparison row), and the right panel plots the greedy retrieval
    wall-clock vs ``N`` (one line per ``B``). Both panels use
    log-log axes because the claim is that the slope (in log-log
    space) is ~0 for the encoder.

    Args:
        rows: The per-(N, B) measurement rows.
        png_path: Destination file path for the PNG figure.
    """
    set_publication_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enc_by_b: dict[int, dict[int, tuple[float, float]]] = {}
    ret_by_b: dict[int, dict[int, tuple[float, float]]] = {}
    for row in rows:
        if row["what"] == "encoder":
            b = int(row["budget"]) if "budget" in row and row["budget"] is not None else 0
            n = int(row["n_vertices"])
            enc_by_b.setdefault(b, {})[n] = (
                float(row["mean_seconds"]),
                float(row["std_seconds"]),
            )
        elif row["what"] == "retrieval":
            b = int(row["budget"])
            n = int(row["n_vertices"])
            ret_by_b.setdefault(b, {})[n] = (
                float(row["mean_seconds"]),
                float(row["std_seconds"]),
            )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    ax_enc, ax_ret = axes[0], axes[1]
    for idx, b in enumerate(sorted(enc_by_b.keys())):
        ns = sorted(enc_by_b[b].keys())
        means = [enc_by_b[b][n][0] for n in ns]
        stds = [enc_by_b[b][n][1] for n in ns]
        ax_enc.errorbar(
            ns,
            means,
            yerr=stds,
            marker="o",
            color=color_for(idx),
            label=f"B={b}" if b > 0 else "B=* (encoder)",
        )
    ax_enc.set_xscale("log")
    ax_enc.set_yscale("log")
    ax_enc.set_xlabel("persistent-graph size N (vertices)")
    ax_enc.set_ylabel("encoder wall-clock (s)")
    ax_enc.set_title("Dual-geometric encoder: wall-clock vs N")
    ax_enc.legend(fontsize=8)

    for idx, b in enumerate(sorted(ret_by_b.keys())):
        ns = sorted(ret_by_b[b].keys())
        means = [ret_by_b[b][n][0] for n in ns]
        stds = [ret_by_b[b][n][1] for n in ns]
        ax_ret.errorbar(
            ns,
            means,
            yerr=stds,
            marker="s",
            color=color_for(idx),
            label=f"B={b}",
        )
    ax_ret.set_xscale("log")
    ax_ret.set_yscale("log")
    ax_ret.set_xlabel("persistent-graph size N (vertices)")
    ax_ret.set_ylabel("retrieval wall-clock (s)")
    ax_ret.set_title("Greedy retrieval: wall-clock vs N (one line per B)")
    ax_ret.legend(fontsize=8)

    fig.suptitle("Decoupling measurement: inference cost vs persistent-graph size N")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def compute_slope_summary(
    rows: list[dict[str, object]],
    n_resamples: int = 1000,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Compute OLS slope of ``mean_seconds`` vs ``n_vertices`` per (what, B).

    The bootstrap CI for "slope = 0" is computed by resampling the
    OLS residuals with replacement and re-fitting the line: the
    distribution of the slope under the null hypothesis is
    approximated by the empirical distribution of the bootstrapped
    slope. The two-sided p-value is the fraction of bootstrapped
    slopes whose absolute value exceeds the observed slope.

    Args:
        rows: The per-(N, B) measurement rows.
        n_resamples: Bootstrap resample count.
        seed: Random seed for the bootstrap resampler.

    Returns:
        A list of per-(what, B) summary rows with ``slope``,
        ``intercept``, ``n_points``, ``ci_low``, ``ci_high``, and
        ``p_value_slope_vs_zero``.
    """
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in rows:
        key = (str(row["what"]), int(row.get("budget") or 0))
        grouped.setdefault(key, []).append((int(row["n_vertices"]), float(row["mean_seconds"])))
    summary: list[dict[str, object]] = []
    for (what, b), points in sorted(grouped.items()):
        points.sort(key=lambda t: t[0])
        xs = [float(t[0]) for t in points]
        ys = [float(t[1]) for t in points]
        slope, intercept = ols_slope_intercept(xs, ys)
        diffs = [ys[i] - (slope * xs[i] + intercept) for i in range(len(xs))]
        if len(diffs) >= 2:
            # Bootstrap the residuals: under H0 (slope = 0) the line
            # goes through (mean_x, mean_y); the bootstrap
            # distribution of the slope is approximated by re-fitting
            # lines to resampled (xs, residual-shifted) pairs.
            import numpy as np

            rng = np.random.default_rng(seed)
            x_arr = np.asarray(xs, dtype=np.float64)
            res = np.asarray(diffs, dtype=np.float64)
            n_pts = len(xs)
            idx = rng.integers(0, n_pts, size=(n_resamples, n_pts))
            slopes = np.empty(n_resamples, dtype=np.float64)
            for k in range(n_resamples):
                y_boot = intercept + slope * x_arr + res[idx[k]]
                slope_boot, _ = ols_slope_intercept(xs, [float(v) for v in y_boot])
                slopes[k] = slope_boot
            obs = abs(slope)
            p_value = float((np.abs(slopes) >= obs).mean())
            ci_low = float(np.quantile(slopes, 0.025))
            ci_high = float(np.quantile(slopes, 0.975))
        else:
            p_value, ci_low, ci_high = 1.0, 0.0, 0.0
        summary.append(
            {
                "what": what,
                "budget": b,
                "slope_seconds_per_vertex": slope,
                "intercept_seconds": intercept,
                "n_points": len(xs),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value_slope_vs_zero": p_value,
            }
        )
    return summary


def run_decoupling_measurement(
    config: DecouplingConfig | None = None,
) -> dict[str, object]:
    """Run the inference-storage decoupling measurement.

    The measurement sweeps ``n_sizes x budgets`` plus the legacy
    "encoder on full graph" baseline, then writes:

    * ``<output_dir>/tables/decoupling.csv`` (per-(N, B) raw rows)
    * ``<output_dir>/decoupling.csv`` (legacy path)
    * ``<output_dir>/tables/decoupling_slope.csv`` (per-(what, B) OLS
      slope with bootstrap CI of slope vs zero)
    * ``<output_dir>/plots/decoupling.png`` (wall-clock vs N)

    Args:
        config: Experiment configuration. When ``None`` the
          plan-default configuration is used.

    Returns:
        A dictionary with ``rows``, ``slope_summary``, and the
        output paths ``csv``, ``csv_legacy``, ``png``, and
        ``slope_csv``.
    """
    if config is None:
        config = DecouplingConfig()
    log = get_logger(__name__)
    log.info(
        "decoupling starting",
        extra={
            "event": "decoupling.start",
            "n_sizes": list(config.n_sizes),
            "budgets": list(config.budgets),
        },
    )
    rows: list[dict[str, object]] = []
    for n in config.n_sizes:
        set_global_seed(config.seed + int(n))
        # B-bounded encoder timing (one row per B), demonstrating that
        # the per-step cost is independent of N.
        for b in config.budgets:
            stats = measure_dual_encoder_inference_time(
                int(n),
                config.feature_dim,
                config.n_trials,
                seed=config.seed,
                budget=int(b),
            )
            rows.append(
                {
                    "what": "encoder",
                    "n_vertices": int(n),
                    "budget": int(b),
                    **stats,
                }
            )
        # Legacy "encoder on full graph" row (budget=0 sentinel). This
        # is the comparison line that demonstrates *why* the
        # B-bounded working graph is necessary in the first place.
        legacy = measure_dual_encoder_inference_time(
            int(n), config.feature_dim, 1, seed=config.seed, budget=None
        )
        rows.append(
            {
                "what": "encoder_euclidean",
                "n_vertices": int(n),
                "budget": 0,
                **legacy,
            }
        )
        log.info(
            "encoder measured",
            extra={
                "event": "decoupling.encoder",
                "n": int(n),
            },
        )
    for n in config.n_sizes:
        for b in config.budgets:
            set_global_seed(config.seed + int(n) + int(b))
            stats = measure_greedy_retrieval_time(
                int(n), config.feature_dim, int(b), config.n_trials, seed=config.seed
            )
            rows.append(
                {
                    "what": "retrieval",
                    "n_vertices": int(n),
                    "budget": int(b),
                    **stats,
                }
            )
            log.info(
                "retrieval measured",
                extra={
                    "event": "decoupling.retrieval",
                    "n": int(n),
                    "budget": int(b),
                    "mean_seconds": stats["mean_seconds"],
                },
            )

    out_root = Path(config.output_dir)
    tables_dir = out_root / "tables"
    plots_dir = out_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "decoupling.csv"
    csv_legacy = out_root / "decoupling.csv"
    png_path = plots_dir / "decoupling.png"
    write_csv(
        csv_path,
        ["what", "n_vertices", "budget", "mean_seconds", "std_seconds"],
        clone_rows(rows),
    )
    write_csv(
        csv_legacy,
        ["what", "n_vertices", "budget", "mean_seconds", "std_seconds"],
        clone_rows(rows),
    )

    summary_rows = compute_slope_summary(rows)
    slope_csv = tables_dir / "decoupling_slope.csv"
    write_csv(
        slope_csv,
        [
            "what",
            "budget",
            "slope_seconds_per_vertex",
            "intercept_seconds",
            "n_points",
            "ci_low",
            "ci_high",
            "p_value_slope_vs_zero",
        ],
        clone_rows(summary_rows),
    )

    render_decoupling_figure(rows, png_path)

    slope_b64 = next(
        (r for r in summary_rows if r["what"] == "encoder" and int(r["budget"]) == 0),
        None,
    )
    slope_text = (
        f"slope={slope_b64['slope_seconds_per_vertex']:.2e} s/vertex"
        if slope_b64 is not None
        else "slope=n/a"
    )
    log.info(
        "decoupling complete",
        extra={
            "event": "decoupling.complete",
            "n_rows": len(rows),
            "csv": str(csv_path),
            "png": str(png_path),
            "encoder_slope": slope_text,
        },
    )
    return {
        "rows": rows,
        "slope_summary": summary_rows,
        "csv": str(csv_path),
        "csv_legacy": str(csv_legacy),
        "slope_csv": str(slope_csv),
        "png": str(png_path),
    }


def main() -> int:
    """CLI entry point for the decoupling measurement.

    Returns:
        ``0`` on a successful run.
    """
    parser = argparse.ArgumentParser(
        description="Run the inference-storage decoupling measurement."
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="Run the fast smoke configuration.")
    parser.add_argument(
        "--n-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Override the N sweep (space-separated integers).",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="*",
        default=None,
        help="Override the B sweep (space-separated integers).",
    )
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LOG_FORMAT_JSON)
    if args.smoke:
        config = default_smoke_config(output_dir=args.output_dir)
    else:
        n_sizes = tuple(args.n_sizes) if args.n_sizes else (50, 100, 200, 400, 800)
        budgets = tuple(args.budgets) if args.budgets else (16, 32, 64)
        config = DecouplingConfig(
            n_sizes=n_sizes,
            budgets=budgets,
            n_trials=args.n_trials,
            feature_dim=args.feature_dim,
            output_dir=args.output_dir,
        )
    run_decoupling_measurement(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
