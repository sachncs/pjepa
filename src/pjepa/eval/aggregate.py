"""Result aggregation across experiment phases.

This module is the canonical post-experiment aggregator for the
project. It ingests per-experiment metrics tables produced by the
``experiments/`` runners and emits a unified set of release-quality
artefacts.

## Architecture

* :class:`AggregatedRow` is the in-memory schema. Every output record
  is an :class:`AggregatedRow`; the row builders
  (:func:`build_rows_from_tu`, :func:`build_rows_from_cl`, ...)
  convert raw CSV rows into this schema.
* :data:`ROW_BUILDERS` is the dispatch table from experiment name to
  row builder. Adding a new experiment requires only adding a key to
  this table and a matching candidate to
  :func:`default_source_paths`.
* :func:`merge_rows` orchestrates the read-then-build pipeline.
* :func:`write_artifacts` flattens the rows into a JSONL file, a CSV
  file, and a Markdown summary grouped by
  ``(experiment, dataset, method)``.

## Outputs

The aggregator always emits the following artefacts, even when no
runs have been recorded yet (this guarantees consumer stability —
downstream notebooks can rely on the files always existing):

* ``results/all_runs.jsonl`` — every recorded run as a single JSON
  object per line.
* ``results/tables/all_runs.csv`` — flat CSV with one row per run
  containing ``experiment``, ``dataset``, ``method``, ``seed``,
  ``fold``, ``metric`` plus any extra columns preserved from the
  source.
* ``results/tables/summary.md`` — Markdown table grouped by
  ``(experiment, dataset, method)`` with ``n``, ``mean``, ``std``,
  ``median``, ``min``, ``max``.

## Tolerant sources

The aggregator reads JSONL/CSV files from the following candidate
locations (each probe is tried in order; the first existing file wins
via :func:`first_existing_path`):

* ``results/tu/tu_results.csv``
* ``results/cl/cl_results.csv``
* ``results/ogb/ogb_results.csv``
* ``results/ablation/ablation.csv`` (preferred) or
  ``results/ablation.csv`` (legacy)
* ``results/decoupling/tables/decoupling.csv`` (preferred),
  ``results/decoupling/decoupling.csv`` (fallback),
  ``results/decoupling.csv`` (legacy)
* ``results/sensitivity/tables/sensitivity_B.csv`` (preferred),
  ``results/sensitivity/sensitivity_B.csv`` (fallback),
  ``results/sensitivity_B.csv`` (legacy)
* ``results/optuna/<dataset>/best_config.yaml`` (loaded separately by
  :func:`pjepa.training.optuna_search.load_best_config`)

Each row is tagged with ``experiment`` so downstream tools can filter
by phase. Run identifiers are derived from the source columns
(``seed``, ``fold`` / ``task``, ``method``).

## Complexity

Let ``N`` be the total number of source rows across every experiment
and ``K`` the number of ``(experiment, dataset, method)`` groups.
Reading is O(N); the per-group summary reduces in O(K * n_i) over
each group; the row iteration dominates so the write is O(N). The
default implementation holds the entire row list in memory; ``N``
is bounded by the number of experiment runs (typically a few
thousand), which fits comfortably in a few MiB.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pjepa.logging_setup import get_logger

__all__ = [
    "PHASE_KEYS",
    "ROW_BUILDERS",
    "AggregatedRow",
    "AggregationResult",
    "aggregate_all",
    "build_rows_from_ablation",
    "build_rows_from_cl",
    "build_rows_from_decoupling",
    "build_rows_from_ogb",
    "build_rows_from_sensitivity",
    "build_rows_from_tu",
    "build_summary_rows",
    "coerce_float",
    "coerce_int",
    "content_hash",
    "default_source_paths",
    "first_existing_path",
    "format_metric_float",
    "merge_rows",
    "read_csv_rows",
    "render_summary_markdown",
    "write_artifacts",
]


PHASE_KEYS: tuple[str, ...] = (
    "tu",
    "cl",
    "ogb",
    "ablation",
    "decoupling",
    "sensitivity",
)
"""Order in which experiment phases are aggregated.

The order determines the order of artefact emission in the JSONL/CSV
outputs and the section ordering in :func:`render_summary_markdown`.
"""


@dataclass
class AggregatedRow:
    """A single row in the aggregated table.

    Attributes:
        experiment: The originating experiment (e.g. ``"tu"``).
        dataset: The dataset name (or the variant / measurement name).
        method: The method name (or the row key when the source has
          no method column).
        seed: Optional seed index.
        fold: Optional fold index.
        metric: The headline numeric metric (``accuracy`` by default).
        extra: Additional columns preserved verbatim.
    """

    experiment: str
    dataset: str
    method: str
    seed: int | None = None
    fold: int | None = None
    metric: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict representation, suitable for CSV/JSON."""
        out: dict[str, Any] = {
            "experiment": self.experiment,
            "dataset": self.dataset,
            "method": self.method,
        }
        if self.seed is not None:
            out["seed"] = int(self.seed)
        if self.fold is not None:
            out["fold"] = int(self.fold)
        if self.metric is not None:
            out["metric"] = float(self.metric)
        for key, value in self.extra.items():
            out.setdefault(str(key), value)
        return out


@dataclass(frozen=True)
class AggregationResult:
    """Container for the aggregated artefacts.

    Attributes:
        rows: Every collected row.
        jsonl_path: Path to the written ``all_runs.jsonl``.
        csv_path: Path to the written ``tables/all_runs.csv``.
        summary_path: Path to the written ``tables/summary.md``.
    """

    rows: tuple[AggregatedRow, ...]
    jsonl_path: Path
    csv_path: Path
    summary_path: Path


def default_source_paths(results_root: str | Path) -> dict[str, Path | None]:
    """Return the canonical ``experiment -> source CSV`` mapping.

    Args:
        results_root: The ``results/`` directory.

    Returns:
        A mapping keyed by experiment name (``tu``, ``cl``, ``ogb``,
        ``ablation``, ``decoupling``, ``sensitivity``). Missing files
        map to ``None`` so callers can still iterate the full set.
    """
    root = Path(results_root)
    candidates: dict[str, tuple[Path, ...]] = {
        "tu": (root / "tu" / "tu_results.csv", root / "tu_results.csv"),
        "cl": (root / "cl" / "cl_results.csv", root / "cl_results.csv"),
        "ogb": (root / "ogb" / "ogb_results.csv", root / "ogb_results.csv"),
        "ablation": (root / "ablation" / "ablation.csv", root / "ablation.csv"),
        "decoupling": (
            root / "decoupling" / "tables" / "decoupling.csv",
            root / "decoupling" / "decoupling.csv",
            root / "decoupling.csv",
        ),
        "sensitivity": (
            root / "sensitivity" / "tables" / "sensitivity_B.csv",
            root / "sensitivity" / "sensitivity_B.csv",
            root / "sensitivity_B.csv",
        ),
    }
    return {key: first_existing_path(paths) for key, paths in candidates.items()}


def first_existing_path(paths: Iterable[Path]) -> Path | None:
    """Return the first path in ``paths`` that exists, or ``None``.

    The helper exists because each experiment stores its outputs under
    a slightly different convention (with/without a ``tables/`` parent,
    with/without the family directory). Trying each candidate in order
    and returning the first hit keeps the aggregator tolerant of the
    historical layout.

    Args:
        paths: Candidate paths to test in order.

    Returns:
        The first existing file path, or ``None`` if none of the
        candidates point to an existing file.
    """
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def read_csv_rows(path: Path | None) -> list[dict[str, Any]]:
    """Read a CSV file into a list of dict rows.

    Args:
        path: Path to the CSV file. ``None`` yields an empty list.

    Returns:
        The list of row dictionaries; empty when the file is missing.
    """
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def coerce_float(value: Any) -> float | None:
    """Coerce ``value`` to ``float``, returning ``None`` on failure.

    Empty strings, non-numeric strings, and ``None`` are all reported
    as ``None``. Non-finite floats (``NaN``) are also mapped to
    ``None`` so downstream summary arithmetic never has to special-case
    them.

    Args:
        value: Any value (typically parsed from a CSV cell).

    Returns:
        The finite float value, or ``None`` when ``value`` cannot be
        coerced.
    """
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def coerce_int(value: Any) -> int | None:
    """Coerce ``value`` to ``int``, returning ``None`` on failure.

    The coercion goes through ``float`` first so numeric strings
    (``"42"``) and integer-valued floats (``"42.0"``) round-trip
    reliably; cells that fail to convert (empty strings, alphabetic
    content) produce ``None``.

    Args:
        value: Any value (typically parsed from a CSV cell).

    Returns:
        The integer value, or ``None`` when ``value`` cannot be
        coerced.
    """
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_rows_from_tu(rows: list[dict[str, Any]]) -> list[AggregatedRow]:
    """Build :class:`AggregatedRow` records from a TU results CSV.

    The canonical TU results file (``tu_results.csv``) has columns
    ``dataset``, ``method``, ``seed``, ``fold``, ``accuracy``. Any
    extra columns are preserved verbatim in the row's ``extra`` dict.

    Args:
        rows: Raw rows from :func:`read_csv_rows`.

    Returns:
        The mapped list of aggregated rows.
    """
    out: list[AggregatedRow] = []
    for row in rows:
        accuracy = coerce_float(row.get("accuracy"))
        out.append(
            AggregatedRow(
                experiment="tu",
                dataset=str(row.get("dataset", "unknown")),
                method=str(row.get("method", "unknown")),
                seed=coerce_int(row.get("seed")),
                fold=coerce_int(row.get("fold")),
                metric=accuracy,
                extra={
                    k: v
                    for k, v in row.items()
                    if k not in {"dataset", "method", "seed", "fold", "accuracy"}
                },
            )
        )
    return out


def build_rows_from_cl(rows: list[dict[str, Any]]) -> list[AggregatedRow]:
    """Build aggregated rows from a CL results CSV.

    The CL results file uses ``task`` instead of ``fold`` to index
    the continual-learning task; the metric column is either
    ``accuracy`` or a generic ``metric``.

    Args:
        rows: Raw rows from :func:`read_csv_rows`.

    Returns:
        The mapped list of aggregated rows.
    """
    out: list[AggregatedRow] = []
    for row in rows:
        metric = coerce_float(row.get("accuracy")) or coerce_float(row.get("metric"))
        out.append(
            AggregatedRow(
                experiment="cl",
                dataset=str(row.get("dataset", "unknown")),
                method=str(row.get("method", "unknown")),
                seed=coerce_int(row.get("seed")),
                fold=coerce_int(row.get("task")),
                metric=metric,
                extra={
                    k: v
                    for k, v in row.items()
                    if k not in {"dataset", "method", "seed", "task", "accuracy", "metric"}
                },
            )
        )
    return out


def build_rows_from_ogb(rows: list[dict[str, Any]]) -> list[AggregatedRow]:
    """Build aggregated rows from an OGB-arxiv results CSV.

    OGB-arxiv exposes a single dataset (``ogbn-arxiv``) so the
    ``dataset`` field is fixed. The metric column is either
    ``test_accuracy`` (canonical) or ``accuracy`` (legacy).

    Args:
        rows: Raw rows from :func:`read_csv_rows`.

    Returns:
        The mapped list of aggregated rows.
    """
    out: list[AggregatedRow] = []
    for row in rows:
        metric = coerce_float(row.get("test_accuracy")) or coerce_float(row.get("accuracy"))
        out.append(
            AggregatedRow(
                experiment="ogb",
                dataset="ogbn-arxiv",
                method=str(row.get("method", "unknown")),
                seed=coerce_int(row.get("seed")),
                fold=None,
                metric=metric,
                extra={
                    k: v
                    for k, v in row.items()
                    if k not in {"method", "seed", "test_accuracy", "accuracy"}
                },
            )
        )
    return out


def build_rows_from_ablation(rows: list[dict[str, Any]]) -> list[AggregatedRow]:
    """Build aggregated rows from an ablation results CSV.

    Ablation rows use ``variant`` as the canonical "method" axis and
    accept either ``dataset`` or ``variant`` as the row label. Extra
    columns (everything beyond ``variant``, ``dataset``, ``method``,
    ``seed``, ``fold``, ``accuracy``, ``metric``) are preserved.

    Args:
        rows: Raw rows from :func:`read_csv_rows`.

    Returns:
        The mapped list of aggregated rows.
    """
    out: list[AggregatedRow] = []
    for row in rows:
        metric = coerce_float(row.get("accuracy")) or coerce_float(row.get("metric"))
        out.append(
            AggregatedRow(
                experiment="ablation",
                dataset=str(row.get("dataset", row.get("variant", "unknown"))),
                method=str(row.get("variant", row.get("method", "unknown"))),
                seed=coerce_int(row.get("seed")),
                fold=coerce_int(row.get("fold")),
                metric=metric,
                extra={
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "dataset",
                        "variant",
                        "method",
                        "seed",
                        "fold",
                        "accuracy",
                        "metric",
                    }
                },
            )
        )
    return out


def build_rows_from_decoupling(rows: list[dict[str, Any]]) -> list[AggregatedRow]:
    """Build aggregated rows from a decoupling measurement CSV.

    Each row is encoded with the vertex-count ``N`` and budget ``B``
    embedded in the dataset/method fields so the summary markdown
    reads naturally. Wall-clock seconds (or a generic ``metric``)
    become the headline metric.

    Args:
        rows: Raw rows from :func:`read_csv_rows`.

    Returns:
        The mapped list of aggregated rows.
    """
    out: list[AggregatedRow] = []
    for row in rows:
        metric = (
            coerce_float(row.get("wall_clock_seconds"))
            or coerce_float(row.get("mean_seconds"))
            or coerce_float(row.get("metric"))
        )
        n_value = row.get("N") or row.get("n_vertices")
        b_value = row.get("B") or row.get("budget")
        out.append(
            AggregatedRow(
                experiment="decoupling",
                dataset=f"N={n_value if n_value is not None else 'unknown'}",
                method=f"B={b_value if b_value is not None else 'unknown'}",
                seed=coerce_int(row.get("seed")),
                fold=None,
                metric=metric,
                extra={
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "N",
                        "n_vertices",
                        "B",
                        "budget",
                        "seed",
                        "wall_clock_seconds",
                        "mean_seconds",
                        "metric",
                    }
                },
            )
        )
    return out


def build_rows_from_sensitivity(rows: list[dict[str, Any]]) -> list[AggregatedRow]:
    """Build aggregated rows from a sensitivity-sweep CSV.

    Sensitivity rows are indexed by ``dataset`` and the working-graph
    budget ``B``; the metric column is ``accuracy`` (or a generic
    ``metric`` fallback).

    Args:
        rows: Raw rows from :func:`read_csv_rows`.

    Returns:
        The mapped list of aggregated rows.
    """
    out: list[AggregatedRow] = []
    for row in rows:
        metric = coerce_float(row.get("accuracy")) or coerce_float(row.get("metric"))
        out.append(
            AggregatedRow(
                experiment="sensitivity",
                dataset=str(row.get("dataset", "unknown")),
                method=f"B={row.get('B', row.get('budget', 'unknown'))}",
                seed=coerce_int(row.get("seed")),
                fold=None,
                metric=metric,
                extra={
                    k: v
                    for k, v in row.items()
                    if k not in {"dataset", "B", "budget", "seed", "accuracy", "metric"}
                },
            )
        )
    return out


ROW_BUILDERS: dict[str, Callable[[list[dict[str, Any]]], list[AggregatedRow]]] = {
    "tu": build_rows_from_tu,
    "cl": build_rows_from_cl,
    "ogb": build_rows_from_ogb,
    "ablation": build_rows_from_ablation,
    "decoupling": build_rows_from_decoupling,
    "sensitivity": build_rows_from_sensitivity,
}
"""Mapping from an experiment name to its row-building function.

Adding a new experiment requires only adding a key here and a
matching ``default_source_paths`` candidate; :func:`merge_rows`
will pick the new builder up automatically.
"""


def merge_rows(
    sources: dict[str, Path | None],
) -> list[AggregatedRow]:
    """Build :class:`AggregatedRow` objects from every source file.

    Args:
        sources: Mapping from experiment name to source CSV path. Paths
          that are ``None`` or do not exist contribute no rows.

    Returns:
        The aggregated row list.
    """
    rows: list[AggregatedRow] = []
    for experiment, path in sources.items():
        builder = ROW_BUILDERS.get(experiment)
        if builder is None:
            continue
        raw = read_csv_rows(path)
        rows.extend(builder(raw))
    return rows


def build_summary_rows(rows: Iterable[AggregatedRow]) -> list[dict[str, Any]]:
    """Reduce per-run rows into per-group summary dictionaries.

    Each output entry is keyed by ``(experiment, dataset, method)``
    and contains ``n``, ``mean``, ``std``, ``median``, ``min``, and
    ``max`` summary statistics. Rows with a ``None`` metric are
    excluded from the grouping so a malformed row never poisons the
    summary.

    Args:
        rows: The aggregated row list.

    Returns:
        A list of summary rows sorted lexicographically by
        ``(experiment, dataset, method)``.
    """
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.metric is None:
            continue
        grouped[(row.experiment, row.dataset, row.method)].append(float(row.metric))
    summary: list[dict[str, Any]] = []
    for (experiment, dataset, method), values in sorted(grouped.items()):
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
        std = var**0.5
        sorted_vals = sorted(values)
        median = (
            sorted_vals[n // 2]
            if n % 2 == 1
            else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
        )
        summary.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "method": method,
                "n": n,
                "mean": mean,
                "std": std,
                "median": median,
                "min": min(values),
                "max": max(values),
            }
        )
    return summary


def format_metric_float(value: float) -> str:
    """Format a metric value with 4 decimal places, tolerating NaN/inf.

    Args:
        value: The floating-point metric.

    Returns:
        ``"NaN"`` for non-finite values, ``"0.1234"``-formatted
        strings otherwise.
    """
    if value != value or value in (float("inf"), float("-inf")):
        return "NaN"
    return f"{value:.4f}"


def render_summary_markdown(summary: list[dict[str, Any]]) -> str:
    """Render the Markdown summary table from the summary rows.

    Args:
        summary: Per-group summary rows from :func:`build_summary_rows`.

    Returns:
        A Markdown string with one section per experiment (alphabetic
        order) and a stat table per section. When ``summary`` is empty
        the function returns a "no results yet" message so downstream
        consumers can rely on the artefact's existence.
    """
    lines: list[str] = [
        "# Persistent-JEPA — Aggregated Results",
        "",
        "Auto-generated by `pjepa/eval/aggregate.py`. Each row aggregates",
        "per-run metrics into mean ± std with the median / min / max bounds.",
        "",
    ]
    if not summary:
        lines.append("No run results found yet. Run an experiment first;")
        lines.append("see `experiments/REPRODUCE.md` for the entry points.")
        lines.append("")
        return "\n".join(lines)

    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_experiment[row["experiment"]].append(row)

    for experiment, rows in sorted(by_experiment.items()):
        lines.append(f"## `{experiment}`")
        lines.append("")
        lines.append("| Dataset | Method | n | Mean | Std | Median | Min | Max |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            template = (
                "| {dataset} | {method} | {n} | {mean} | {std} | {median} | {minv} | {maxv} |"
            )
            lines.append(
                template.format(
                    dataset=row["dataset"],
                    method=row["method"],
                    n=int(row["n"]),
                    mean=format_metric_float(float(row["mean"])),
                    std=format_metric_float(float(row["std"])),
                    median=format_metric_float(float(row["median"])),
                    minv=format_metric_float(float(row["min"])),
                    maxv=format_metric_float(float(row["max"])),
                )
            )
        lines.append("")
    return "\n".join(lines)


def content_hash(rows: list[AggregatedRow]) -> str:
    """Return a stable SHA-256 hash of the row list.

    The digest is row-order-insensitive in spirit (it iterates rows in
    the order received) but stable for a given input list because the
    per-row JSON encoding is sorted by key.

    Args:
        rows: The aggregated row list.

    Returns:
        A hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row.to_dict(), sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_artifacts(
    rows: list[AggregatedRow],
    output_root: str | Path,
) -> AggregationResult:
    """Write the aggregated artefacts to ``output_root``.

    The artefacts are:

    * ``output_root/all_runs.jsonl`` — every recorded run, one JSON
      object per line.
    * ``output_root/tables/all_runs.csv`` — flat CSV with columns
      ``experiment``, ``dataset``, ``method``, ``seed``, ``fold``,
      ``metric``.
    * ``output_root/tables/summary.md`` — Markdown summary grouped by
      ``(experiment, dataset, method)`` with mean / std / median /
      min / max.

    Args:
        rows: The aggregated row list.
        output_root: The results root directory.

    Returns:
        A populated :class:`AggregationResult` with the resolved
        output paths.
    """
    log = get_logger(__name__)
    root = Path(output_root)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "all_runs.jsonl"
    csv_path = tables / "all_runs.csv"
    summary_path = tables / "summary.md"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True))
            fh.write("\n")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["experiment", "dataset", "method", "seed", "fold", "metric"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment": row.experiment,
                    "dataset": row.dataset,
                    "method": row.method,
                    "seed": "" if row.seed is None else int(row.seed),
                    "fold": "" if row.fold is None else int(row.fold),
                    "metric": "" if row.metric is None else f"{row.metric:.6f}",
                }
            )

    summary_rows = build_summary_rows(rows)
    summary_path.write_text(render_summary_markdown(summary_rows), encoding="utf-8")
    log.info(
        "aggregation complete",
        extra={
            "event": "aggregation.complete",
            "n_rows": len(rows),
            "summary_groups": len(summary_rows),
            "content_hash": content_hash(rows),
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
    )
    return AggregationResult(
        rows=tuple(rows),
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        summary_path=summary_path,
    )


def aggregate_all(results_root: str | Path = "results") -> AggregationResult:
    """Aggregate every supported experiment under ``results_root``.

    Args:
        results_root: The ``results/`` directory.

    Returns:
        The :class:`AggregationResult` containing the written paths.
    """
    sources = default_source_paths(results_root)
    rows = merge_rows(sources)
    return write_artifacts(rows, results_root)
