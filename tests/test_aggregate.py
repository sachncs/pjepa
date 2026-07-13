"""Tests for the Phase 12 result aggregator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pjepa.eval import (
    AggregatedRow,
    AggregationResult,
    aggregate_all,
    default_source_paths,
    merge_rows,
    write_artifacts,
)
from pjepa.eval.aggregate import (
    build_summary_rows,
    first_existing_path,
    format_metric_float,
)

__all__ = [
    "test_aggregate_all_empty_results_dir_emits_files",
    "test_aggregate_all_reads_cl_ogb_ablation",
    "test_aggregate_all_reads_tu_results",
    "test_default_source_paths_handles_missing_files",
    "test_first_existing_returns_first_match",
    "test_first_existing_returns_none_for_missing",
    "test_format_float_handles_nan_inf",
    "test_merge_rows_handles_empty_sources",
    "test_summary_rows_aggregate_per_group",
    "test_write_artifacts_emits_canonical_files",
    "test_write_artifacts_with_populated_rows",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_aggregate_all_empty_results_dir_emits_files(tmp_path: Path) -> None:
    """`aggregate_all` writes the canonical artefacts even when no runs exist."""
    result = aggregate_all(tmp_path)
    assert isinstance(result, AggregationResult)
    assert result.rows == ()
    assert result.jsonl_path.exists()
    assert result.csv_path.exists()
    assert result.summary_path.exists()
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "No run results found" in summary


def test_default_source_paths_handles_missing_files(tmp_path: Path) -> None:
    """`default_source_paths` returns None for missing source files."""
    sources = default_source_paths(tmp_path)
    assert all(value is None for value in sources.values())


def test_first_existing_returns_none_for_missing() -> None:
    """`first_existing_path` returns ``None`` when no path matches."""
    assert first_existing_path([Path("/tmp/__pjepa_missing_one__")]) is None


def test_first_existing_returns_first_match(tmp_path: Path) -> None:
    """`first_existing_path` returns the first path that exists."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    b.write_text("x\n", encoding="utf-8")
    assert first_existing_path((a, b)) == b
    a.write_text("x\n", encoding="utf-8")
    assert first_existing_path((a, b)) == a


def test_format_float_handles_nan_inf() -> None:
    """`format_metric_float` returns ``NaN`` for non-finite values."""
    assert format_metric_float(float("nan")) == "NaN"
    assert format_metric_float(float("inf")) == "NaN"
    assert format_metric_float(0.5) == "0.5000"


def test_summary_rows_aggregate_per_group() -> None:
    """`build_summary_rows` reduces per-run rows to per-group mean/std/median."""
    rows = [
        AggregatedRow(experiment="tu", dataset="MUTAG", method="GIN", metric=0.80),
        AggregatedRow(experiment="tu", dataset="MUTAG", method="GIN", metric=0.90),
        AggregatedRow(experiment="tu", dataset="MUTAG", method="GIN", metric=0.70),
    ]
    summary = build_summary_rows(rows)
    assert len(summary) == 1
    entry = summary[0]
    assert entry["experiment"] == "tu"
    assert entry["dataset"] == "MUTAG"
    assert entry["method"] == "GIN"
    assert entry["n"] == 3
    assert entry["mean"] == pytest.approx(0.8)
    assert entry["median"] == pytest.approx(0.8)
    assert entry["min"] == pytest.approx(0.7)
    assert entry["max"] == pytest.approx(0.9)


def test_merge_rows_handles_empty_sources() -> None:
    """`merge_rows` returns an empty list when every source is missing."""
    assert merge_rows({"tu": None, "cl": None, "ogb": None}) == []


def test_aggregate_all_reads_tu_results(tmp_path: Path) -> None:
    """`aggregate_all` ingests ``tu/tu_results.csv`` rows."""
    _write_csv(
        tmp_path / "tu" / "tu_results.csv",
        ["dataset", "method", "seed", "fold", "accuracy"],
        [
            {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 0, "accuracy": 0.8},
            {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 1, "accuracy": 0.9},
            {
                "dataset": "MUTAG",
                "method": "PersistentJEPA",
                "seed": 0,
                "fold": 0,
                "accuracy": 0.85,
            },
        ],
    )
    result = aggregate_all(tmp_path)
    assert len(result.rows) == 3
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "MUTAG" in summary
    assert "GIN" in summary
    assert "PersistentJEPA" in summary


def test_aggregate_all_reads_cl_ogb_ablation(tmp_path: Path) -> None:
    """`aggregate_all` ingests CL, OGB, ablation, decoupling, sensitivity."""
    _write_csv(
        tmp_path / "cl" / "cl_results.csv",
        ["dataset", "method", "seed", "task", "accuracy"],
        [
            {
                "dataset": "PROTEINS",
                "method": "PersistentJEPA",
                "seed": 0,
                "task": 0,
                "accuracy": 0.7,
            }
        ],
    )
    _write_csv(
        tmp_path / "ogb" / "ogb_results.csv",
        ["method", "seed", "test_accuracy"],
        [{"method": "Persistent-JEPA", "seed": 0, "test_accuracy": 0.72}],
    )
    _write_csv(
        tmp_path / "ablation" / "ablation.csv",
        ["dataset", "variant", "seed", "fold", "accuracy"],
        [
            {
                "dataset": "PROTEINS",
                "variant": "full",
                "seed": 0,
                "fold": 0,
                "accuracy": 0.82,
            }
        ],
    )
    _write_csv(
        tmp_path / "decoupling" / "tables" / "decoupling.csv",
        ["N", "B", "seed", "wall_clock_seconds"],
        [{"N": 100, "B": 8, "seed": 0, "wall_clock_seconds": 0.001}],
    )
    _write_csv(
        tmp_path / "sensitivity" / "tables" / "sensitivity_B.csv",
        ["dataset", "B", "seed", "accuracy"],
        [{"dataset": "PROTEINS", "B": 64, "seed": 0, "accuracy": 0.80}],
    )
    result = aggregate_all(tmp_path)
    experiments = {row.experiment for row in result.rows}
    assert experiments == {"cl", "ogb", "ablation", "decoupling", "sensitivity"}
    jsonl_lines = result.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 5
    decoded = [json.loads(line) for line in jsonl_lines]
    assert any(d["experiment"] == "ogb" and d["dataset"] == "ogbn-arxiv" for d in decoded)


def test_write_artifacts_emits_canonical_files(tmp_path: Path) -> None:
    """`write_artifacts` always emits the three artefacts."""
    rows = [
        AggregatedRow(experiment="tu", dataset="MUTAG", method="GIN", metric=0.8),
    ]
    result = write_artifacts(rows, tmp_path)
    assert result.jsonl_path.exists()
    assert result.csv_path.exists()
    assert result.summary_path.exists()
    csv_content = result.csv_path.read_text(encoding="utf-8")
    assert "experiment,dataset,method" in csv_content


def test_write_artifacts_with_populated_rows(tmp_path: Path) -> None:
    """`write_artifacts` writes per-row entries to JSONL and CSV."""
    rows = [
        AggregatedRow(
            experiment="tu",
            dataset="MUTAG",
            method="GIN",
            seed=0,
            fold=0,
            metric=0.85,
            extra={"extra_field": "abc"},
        ),
    ]
    result = write_artifacts(rows, tmp_path)
    jsonl_lines = result.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(jsonl_lines[0])
    assert payload["experiment"] == "tu"
    assert payload["metric"] == pytest.approx(0.85)
    assert payload["extra_field"] == "abc"


def test_aggregated_row_to_dict_omits_none() -> None:
    """`AggregatedRow.to_dict` omits missing optional fields."""
    row = AggregatedRow(experiment="tu", dataset="MUTAG", method="GIN")
    payload = row.to_dict()
    assert "seed" not in payload
    assert "fold" not in payload
    assert "metric" not in payload
