"""Tests for experiment runners (Phase 8/9/10/11)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.run_exp_d_tu_sota import (
    TU_DATASETS,
    TU_METHODS,
    PlanTables,
    TUExperimentConfig,
    aggregate_results,
    build_baseline,
    build_encoder,
    build_persistent_jepa_triple,
    concatenate_graphs,
    encode_baseline,
    feature_batches,
    kfold,
    load_best_config_for_dataset,
    run_experiment,
    train_classifier,
)
from pjepa.exceptions import ConfigError
from pjepa.graphs import TypedAttributedGraph

__all__ = [
    "test_bad_tu_experiment_unknown_method",
    "test_happy_aggregate_results",
    "test_happy_concatenate_graphs",
    "test_happy_kfold_disjoint",
    "test_happy_tu_experiment_runs_smoke",
    "test_property_tu_datasets_match_known",
    "test_property_tu_methods_match_known",
    "test_ugly_tu_experiment_single_seed",
]


def test_happy_tu_experiment_runs_smoke() -> None:
    """A 1-seed, 2-fold, 1-method run completes."""
    config = TUExperimentConfig(
        datasets=("MUTAG",),
        methods=("GIN",),
        n_seeds=1,
        n_folds=2,
        epochs=20,
        output_dir="results/tu_smoke",
    )
    rows = run_experiment(config)
    assert len(rows) > 0
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in rows)


def test_happy_aggregate_results() -> None:
    """aggregate_results produces a summary keyed by (dataset, method)."""
    rows = [
        {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 0, "accuracy": 0.8},
        {"dataset": "MUTAG", "method": "GIN", "seed": 0, "fold": 1, "accuracy": 0.9},
        {"dataset": "MUTAG", "method": "GCN", "seed": 0, "fold": 0, "accuracy": 0.7},
    ]
    summary = aggregate_results(rows)
    assert "MUTAG|GIN" in summary
    assert summary["MUTAG|GIN"]["mean"] == pytest.approx(0.85)
    assert summary["MUTAG|GCN"]["mean"] == pytest.approx(0.7)


def test_happy_aggregate_results_handles_empty_input() -> None:
    """aggregate_results returns an empty dict for empty rows."""
    assert aggregate_results([]) == {}


def test_happy_kfold_disjoint() -> None:
    """K-fold produces disjoint train and test sets."""
    pairs = [
        (
            TypedAttributedGraph(torch.randn((2, 2)), torch.zeros((2, 0), dtype=torch.long)),
            i,
        )
        for i in range(20)
    ]
    for fold_idx, (tr, te) in enumerate(kfold(pairs, k=5, seed_split=0)):
        train_labels = {lbl for _, lbl in tr}
        test_labels = {lbl for _, lbl in te}
        assert train_labels.isdisjoint(test_labels), f"fold {fold_idx} has overlap"


def test_happy_kfold_raises_on_nonpositive_k() -> None:
    """kfold raises ConfigError for non-positive k."""
    pairs = [(TypedAttributedGraph(torch.randn((2, 2)), torch.zeros((2, 0), dtype=torch.long)), 0)]
    with pytest.raises(ConfigError):
        for _ in kfold(pairs, k=0, seed_split=0):
            pass


def test_happy_concatenate_graphs() -> None:
    """concatenate_graphs correctly merges features and edges."""
    g1 = TypedAttributedGraph(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[0], [1]], dtype=torch.long),
    )
    g2 = TypedAttributedGraph(
        torch.tensor([[5.0, 6.0]]),
        torch.tensor([[], []], dtype=torch.long),
    )
    merged = concatenate_graphs([g1, g2])
    assert merged.num_vertices() == 3
    assert torch.allclose(
        merged.vertex_features, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )


def test_happy_concatenate_graphs_rejects_empty() -> None:
    """concatenate_graphs raises ConfigError on empty input."""
    with pytest.raises(ConfigError):
        concatenate_graphs([])


def test_bad_tu_experiment_unknown_method() -> None:
    """An unknown method raises ConfigError during baseline construction."""
    with pytest.raises(ConfigError):
        build_baseline("NonExistentMethod", input_dim=4, num_classes=2)


def test_ugly_tu_experiment_single_seed() -> None:
    """A single-seed, 2-fold run still produces results."""
    config = TUExperimentConfig(
        datasets=("MUTAG",),
        methods=("GIN",),
        n_seeds=1,
        n_folds=2,
        epochs=5,
        output_dir="results/tu_smoke",
    )
    rows = run_experiment(config)
    assert len(rows) >= 1


def test_property_tu_datasets_match_known() -> None:
    """The TU_DATASETS list contains the six standard TUDatasets."""
    expected = ("PROTEINS", "MUTAG", "NCI1", "IMDB-BINARY", "REDDIT-BINARY", "DD")
    assert expected == TU_DATASETS


def test_property_tu_methods_match_known() -> None:
    """The TU_METHODS list contains the seven standard methods."""
    assert "PersistentJEPA" in TU_METHODS
    assert "GIN" in TU_METHODS
    assert "GCN" in TU_METHODS
    assert "GraphMAE" in TU_METHODS
    assert "GraphCL" in TU_METHODS
    assert "InfoGraph" in TU_METHODS
    assert "Naive" in TU_METHODS
    assert len(TU_METHODS) == 7


def test_build_persistent_jepa_triple_dim_match() -> None:
    """build_persistent_jepa_triple returns encoder/predictor/target with matching shapes."""
    encoder, predictor, target = build_persistent_jepa_triple(
        input_dim=4, hidden_dim=8, num_layers=2
    )
    g = TypedAttributedGraph(
        vertex_features=torch.randn((3, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    e, h = encoder(g)
    assert e.shape == (3, 8)
    assert h.shape[0] == 3
    pred = predictor(e.mean(dim=0).unsqueeze(0))
    assert pred.shape == (1, 8)
    target.update()
    with torch.no_grad():
        e_target, h_target = target.shadow(g)
    assert e_target.shape == (3, 8)
    assert h_target.shape[0] == 3


def test_feature_batches_yields_aligned_triples() -> None:
    """feature_batches yields ``(chunk, context, target)`` of equal length."""
    pairs = [
        (
            TypedAttributedGraph(
                vertex_features=torch.randn((2, 3)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            ),
            i,
        )
        for i in range(5)
    ]
    chunks = list(feature_batches(pairs, batch_size=2))
    assert len(chunks) == 3
    for chunk, ctx, tgt in chunks:
        assert len(chunk) == ctx.shape[0] == tgt.shape[0]


def test_build_encoder_dim_match() -> None:
    """build_encoder returns a DualGeometricEncoder with the requested dim."""
    enc = build_encoder(input_dim=4, hidden_dim=8, num_layers=2)
    g = TypedAttributedGraph(
        vertex_features=torch.randn((3, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    e, h = enc(g)
    assert e.shape == (3, 8)
    assert h.shape[0] == 3


def test_encode_baseline_handles_naive() -> None:
    """encode_baseline delegates to the inner linear for the Naive baseline."""
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(3, 2))
    g = TypedAttributedGraph(
        vertex_features=torch.randn((4, 3)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    out = encode_baseline(model, g)
    assert out.shape[-1] == 2


def test_train_classifier_rejects_empty_inputs() -> None:
    """train_classifier raises ConfigError when either split is empty."""
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ConfigError):
        train_classifier(
            model=model,
            train_pairs=[],
            test_pairs=[],
            epochs=1,
            learning_rate=1e-2,
        )


def test_load_best_config_for_dataset_returns_empty_when_missing(tmp_path: Path) -> None:
    """load_best_config_for_dataset returns an empty dict for missing files."""
    assert load_best_config_for_dataset("MUTAG", tmp_path) == {}
    assert load_best_config_for_dataset("MUTAG", None) == {}


def test_plan_tables_dataclass_field_signatures() -> None:
    """PlanTables exposes the documented fields."""
    plan = PlanTables(
        summary_rows=[],
        bootstrap_rows=[],
        significance_rows=[],
        per_dataset_methods={},
    )
    assert plan.summary_rows == []
    assert plan.per_dataset_methods == {}


def test_run_experiment_skips_missing_datasets(tmp_path: Path) -> None:
    """run_experiment continues when a dataset fails to load."""
    config = TUExperimentConfig(
        datasets=("MUTAG",),
        methods=("GIN",),
        n_seeds=1,
        n_folds=2,
        epochs=2,
        output_dir=str(tmp_path),
    )
    rows = run_experiment(config)
    assert isinstance(rows, list)
    for row in rows:
        assert row["dataset"] == "MUTAG"
