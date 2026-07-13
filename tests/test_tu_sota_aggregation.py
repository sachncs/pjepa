"""Tests for Phase 8 TU aggregation, plots, best-config loading, and the
Persistent-JEPA path that exercises JEPA / FreeEnergy / retrieval machinery."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pjepa.graphs import TypedAttributedGraph

__all__ = [
    "test_aggregate_results_includes_extended_stats",
    "test_build_plan_tables_significance_bonferroni",
    "test_build_plan_tables_summary_columns",
    "test_load_best_config_for_dataset_returns_empty_when_missing",
    "test_load_best_config_for_dataset_returns_params",
    "test_optuna_search_config_defaults",
    "test_persistent_jepa_runs_smoke",
    "test_plan_tables_respect_no_pretraining_flag",
    "test_write_plan_tables_creates_csvs",
    "test_write_plots_creates_pngs",
]


def _fake_pair(
    label: int, input_dim: int = 3, num_vertices: int = 5
) -> tuple[TypedAttributedGraph, int]:
    """Build a small synthetic graph/label pair used by the JEPA smoke tests."""
    edges = [[i, (i + 1) % num_vertices] for i in range(num_vertices)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    g = TypedAttributedGraph(
        vertex_features=torch.randn((num_vertices, input_dim)),
        edge_index=edge_index,
        edge_features=torch.zeros((edge_index.shape[1], 1)),
    )
    return g, label


# ============================== AGGREGATION ==============================


def test_aggregate_results_includes_extended_stats() -> None:
    """aggregate_results emits mean/std/n_folds/median/min/max."""
    from experiments.run_exp_d_tu_sota import aggregate_results

    rows = [
        {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 0, "accuracy": 0.8},
        {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 1, "accuracy": 0.9},
        {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 2, "accuracy": 0.7},
    ]
    summary = aggregate_results(rows)
    entry = summary["MUTAG|GIN"]
    assert entry["mean"] == pytest.approx(0.8)
    assert entry["std"] > 0.0
    assert entry["n_folds"] == 3
    assert "median" in entry and "min" in entry and "max" in entry


def test_build_plan_tables_summary_columns() -> None:
    """build_plan_tables emits mean/std/n_folds per (dataset, method)."""
    from experiments.run_exp_d_tu_sota import build_plan_tables

    rows = [
        {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 0, "accuracy": 0.8},
        {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 1, "accuracy": 0.9},
        {"dataset": "MUTAG", "method": "PersistentJEPA", "seed": 0, "fold": 0, "accuracy": 0.85},
        {"dataset": "MUTAG", "method": "PersistentJEPA", "seed": 0, "fold": 1, "accuracy": 0.75},
    ]
    plan = build_plan_tables(rows, n_resamples=300, seed=0)
    summary = plan.summary_rows
    assert {(r["method"]) for r in summary} == {"GIN", "PersistentJEPA"}
    assert all("mean" in r and "std" in r and "n_folds" in r for r in summary)
    assert any(r["dataset"] == "MUTAG" for r in plan.bootstrap_rows)
    assert any(r["dataset"] == "MUTAG" for r in plan.significance_rows)
    for row in plan.bootstrap_rows:
        assert "mean_diff_vs_reference" in row


def test_build_plan_tables_significance_bonferroni() -> None:
    """Bonferroni-corrected p-values are >= raw Wilcoxon p-values."""
    from experiments.run_exp_d_tu_sota import build_plan_tables

    rows = []
    for i in range(8):
        rows.append(
            {
                "dataset": "D1",
                "method": "PersistentJEPA",
                "seed": i,
                "fold": 0,
                "accuracy": 0.7 + 0.01 * i,
            }
        )
        rows.append(
            {
                "dataset": "D1",
                "method": "GIN",
                "seed": i,
                "fold": 0,
                "accuracy": 0.5 + 0.01 * i,
            }
        )
    plan = build_plan_tables(rows, n_resamples=200, seed=0)
    sig = [r for r in plan.significance_rows if r["method"] == "GIN"]
    assert sig
    for row in sig:
        assert row["p_value_bonferroni"] >= row["p_value"]


def test_build_plan_tables_handles_empty_rows() -> None:
    """build_plan_tables returns an empty PlanTables for empty rows."""
    from experiments.run_exp_d_tu_sota import build_plan_tables

    plan = build_plan_tables([])
    assert plan.summary_rows == []
    assert plan.bootstrap_rows == []
    assert plan.significance_rows == []
    assert plan.per_dataset_methods == {}


def test_build_plan_tables_aligns_with_custom_reference() -> None:
    """The reference_method argument is reflected in the significance table."""
    from experiments.run_exp_d_tu_sota import build_plan_tables

    rows = [
        {"dataset": "D1", "method": "A", "seed": 0, "fold": 0, "accuracy": 0.7},
        {"dataset": "D1", "method": "A", "seed": 0, "fold": 1, "accuracy": 0.75},
        {"dataset": "D1", "method": "B", "seed": 0, "fold": 0, "accuracy": 0.8},
        {"dataset": "D1", "method": "B", "seed": 0, "fold": 1, "accuracy": 0.85},
    ]
    plan = build_plan_tables(rows, n_resamples=100, reference_method="B", seed=0)
    for row in plan.significance_rows:
        assert row["reference"] == "B"


def test_write_plan_tables_creates_csvs(tmp_path: Path) -> None:
    """write_plan_tables creates the three plan-compliant CSVs."""
    from experiments.run_exp_d_tu_sota import build_plan_tables, write_plan_tables

    rows = [
        {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 0, "accuracy": 0.8},
        {"dataset": "MUTAG", "method": "PersistentJEPA", "seed": 0, "fold": 0, "accuracy": 0.85},
    ]
    plan = build_plan_tables(rows, n_resamples=100, seed=0)
    paths = write_plan_tables(plan, tmp_path)
    assert paths["summary"].exists()
    assert paths["bootstrap"].exists()
    assert paths["significance"].exists()
    summary_text = paths["summary"].read_text()
    assert "mean" in summary_text
    bootstrap_text = paths["bootstrap"].read_text()
    assert "mean_diff_vs_reference" in bootstrap_text


def test_write_plots_creates_pngs(tmp_path: Path) -> None:
    """write_plots emits dataset-aligned radar and heatmap PNGs."""
    from experiments.run_exp_d_tu_sota import build_plan_tables, write_plots

    rows = [
        {"dataset": "D1", "method": "A", "seed": 0, "fold": 0, "accuracy": 0.8},
        {"dataset": "D1", "method": "B", "seed": 0, "fold": 0, "accuracy": 0.7},
        {"dataset": "D2", "method": "A", "seed": 0, "fold": 0, "accuracy": 0.6},
        {"dataset": "D2", "method": "B", "seed": 0, "fold": 0, "accuracy": 0.55},
        {"dataset": "D3", "method": "A", "seed": 0, "fold": 0, "accuracy": 0.5},
        {"dataset": "D3", "method": "B", "seed": 0, "fold": 0, "accuracy": 0.45},
    ]
    plan = build_plan_tables(rows, n_resamples=100, seed=0)
    paths = write_plots(plan, tmp_path / "plots")
    assert paths["radar"].exists()
    assert paths["heatmap"].exists()
    assert paths["radar"].stat().st_size > 0
    assert paths["heatmap"].stat().st_size > 0


def test_write_plots_pads_missing_cells(tmp_path: Path) -> None:
    """write_plots aligns every method to the full dataset list."""
    from experiments.run_exp_d_tu_sota import build_plan_tables, write_plots

    rows = [
        {"dataset": "D1", "method": "A", "seed": 0, "fold": 0, "accuracy": 0.8},
        {"dataset": "D2", "method": "B", "seed": 0, "fold": 0, "accuracy": 0.7},
    ]
    plan = build_plan_tables(rows, n_resamples=100, seed=0)
    paths = write_plots(plan, tmp_path / "plots")
    assert paths["radar"].exists()
    assert paths["heatmap"].exists()


# ============================== BEST CONFIG LOADING ==============================


def test_load_best_config_for_dataset_returns_empty_when_missing(tmp_path: Path) -> None:
    """load_best_config_for_dataset returns an empty dict for missing files."""
    from experiments.run_exp_d_tu_sota import load_best_config_for_dataset

    assert load_best_config_for_dataset("MUTAG", tmp_path) == {}
    assert load_best_config_for_dataset("MUTAG", None) == {}


def test_load_best_config_for_dataset_returns_params(tmp_path: Path) -> None:
    """load_best_config_for_dataset returns the parsed ``best_params`` dict."""
    from experiments.run_exp_d_tu_sota import load_best_config_for_dataset
    from pjepa.training import load_best_config

    cfg_dir = tmp_path / "MUTAG"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "best_config.yaml"
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("PyYAML not installed")
    cfg_path.write_text(
        yaml.safe_dump({"dataset": "MUTAG", "best_params": {"lr": 0.005}, "best_value": 0.9}),
        encoding="utf-8",
    )
    params = load_best_config_for_dataset("MUTAG", tmp_path)
    assert params["lr"] == 0.005
    assert load_best_config(cfg_path)["best_value"] == pytest.approx(0.9)


# ============================== PERSISTENT-JEPA PATH ==============================


def test_persistent_jepa_runs_smoke() -> None:
    """train_persistent_jepa returns a finite accuracy in [0, 1]."""
    from experiments.run_exp_d_tu_sota import TUExperimentConfig, train_persistent_jepa

    config = TUExperimentConfig(
        datasets=(),
        methods=("PersistentJEPA",),
        n_seeds=1,
        n_folds=2,
        epochs=2,
        batch_size=2,
        budget=4,
        output_dir="results/tu_smoke",
        run_jepa_pretraining=True,
    )
    train_pairs = [_fake_pair(label=i % 2) for i in range(6)]
    test_pairs = [_fake_pair(label=(i + 1) % 2) for i in range(3)]
    accuracy = train_persistent_jepa(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        config=config,
        best_params={"hidden_dim": 16, "num_layers": 1, "lr": 1e-2, "weight_decay": 1e-4},
    )
    assert 0.0 <= accuracy <= 1.0


def test_persistent_jepa_with_empty_train_returns_zero() -> None:
    """train_persistent_jepa returns 0.0 when no training pairs are provided."""
    from experiments.run_exp_d_tu_sota import TUExperimentConfig, train_persistent_jepa

    config = TUExperimentConfig()
    accuracy = train_persistent_jepa(
        train_pairs=[],
        test_pairs=[_fake_pair(label=0)],
        config=config,
    )
    assert accuracy == 0.0


def test_plan_tables_respect_no_pretraining_flag() -> None:
    """The ``run_jepa_pretraining=False`` path still produces a finite accuracy."""
    from experiments.run_exp_d_tu_sota import TUExperimentConfig, train_persistent_jepa

    config = TUExperimentConfig(
        datasets=(),
        methods=("PersistentJEPA",),
        n_seeds=1,
        n_folds=2,
        epochs=2,
        batch_size=2,
        budget=4,
        run_jepa_pretraining=False,
    )
    train_pairs = [_fake_pair(label=i % 2) for i in range(6)]
    test_pairs = [_fake_pair(label=(i + 1) % 2) for i in range(3)]
    accuracy = train_persistent_jepa(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        config=config,
        best_params={"hidden_dim": 16, "num_layers": 1},
    )
    assert 0.0 <= accuracy <= 1.0


def test_optuna_search_config_defaults() -> None:
    """OptunaSearchConfig defaults match the Phase 8 plan."""
    from pjepa.training import OptunaSearchConfig

    cfg = OptunaSearchConfig()
    assert cfg.min_resource == 20
    assert cfg.max_resource == 200
    assert cfg.reduction_factor == 3
    assert "lr" in cfg.search_space
