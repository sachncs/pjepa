"""Tests for experiment runners (Phase 8/9/10/11)."""

from __future__ import annotations

import pytest
import torch

from experiments.run_exp_d_tu_sota import (
    TU_DATASETS,
    TU_METHODS,
    TUExperimentConfig,
    _concatenate_graphs,
    _kfold,
    aggregate_results,
)
from experiments.run_exp_h_ablations import ABLATION_VARIANTS, _build_variant
from pjepa.graphs import TypedAttributedGraph

__all__ = [
    "test_bad_tu_experiment_unknown_method",
    "test_happy_ablation_builds_each_variant",
    "test_happy_aggregate_results",
    "test_happy_concatenate_graphs",
    "test_happy_kfold_disjoint",
    "test_happy_tu_experiment_runs_smoke",
    "test_property_ablation_variants_match_known",
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
    from experiments.run_exp_d_tu_sota import run_experiment

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


def test_happy_kfold_disjoint() -> None:
    """K-fold produces disjoint train and test sets."""
    pairs = [
        (TypedAttributedGraph(torch.randn((2, 2)), torch.zeros((2, 0), dtype=torch.long)), i)
        for i in range(20)
    ]
    for fold_idx, (tr, te) in enumerate(_kfold(pairs, k=5, seed_split=0)):
        train_labels = {lbl for _, lbl in tr}
        test_labels = {lbl for _, lbl in te}
        assert train_labels.isdisjoint(test_labels), f"fold {fold_idx} has overlap"


def test_happy_concatenate_graphs() -> None:
    """_concatenate_graphs correctly merges features and edges."""
    g1 = TypedAttributedGraph(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[0], [1]], dtype=torch.long)
    )
    g2 = TypedAttributedGraph(torch.tensor([[5.0, 6.0]]), torch.tensor([[], []], dtype=torch.long))
    merged = _concatenate_graphs([g1, g2])
    assert merged.num_vertices() == 3
    assert torch.allclose(
        merged.vertex_features, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )


def test_happy_ablation_builds_each_variant() -> None:
    """Every ablation variant constructs a model."""
    for variant in ABLATION_VARIANTS:
        model = _build_variant(variant, input_dim=7, num_classes=2)
        assert model is not None


def test_bad_tu_experiment_unknown_method() -> None:
    """An unknown method raises ConfigError during baseline construction."""

    from experiments.run_exp_d_tu_sota import _build_baseline

    with pytest.raises(Exception):
        _build_baseline("NonExistentMethod", input_dim=4, num_classes=2)


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
    from experiments.run_exp_d_tu_sota import run_experiment

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


def test_property_ablation_variants_match_known() -> None:
    """The ABLATION_VARIANTS list contains the seven expected variants."""
    expected = {
        "full",
        "minus_hyperbolic",
        "minus_persistent",
        "minus_four_conditions",
        "minus_ema",
        "minus_jepa_loss",
        "random_encoder",
    }
    assert set(ABLATION_VARIANTS) == expected
