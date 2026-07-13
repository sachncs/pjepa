"""Tests for the Phase 10 OGB-arxiv experiment.

The tests use a synthetic 30-node graph with three classes so they run
without network access. They cover:

* the OGB-arxiv loader's leakage-assertion contract;
* the neighbour-sampling / induced-subgraph helpers;
* the GraphSAGE and BGRL baselines;
* the experiment dispatch over all five methods;
* the aggregation to ``results/tables/ogb_summary.csv``;
* the optional prediction artifact writer (including the
  ``node_ids`` path that emits official OGB test indices);
* the practical RSS cap and checkpoint sharding utilities.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from experiments.run_exp_f_ogb_arxiv import (
    OGB_METHODS,
    OGBConfig,
    OGBRunResult,
    PersistentJEPAClassifier,
    aggregate_ogb_results,
    build_predictions_artifact,
    default_smoke_config,
    run_ogb_experiment,
)
from pjepa.baselines import BGRL, GraphMAE, GraphSAGE
from pjepa.data.ogb import (
    OGBArxiv,
    induce_subgraph,
    neighbor_sample,
)
from pjepa.exceptions import BackendError, DataError
from pjepa.graphs import TypedAttributedGraph
from pjepa.perf import (
    ShardedCheckpoint,
    assert_rss_cap,
    current_rss_mb,
    load_sharded_state_dict,
    shard_state_dict,
)

__all__ = [
    "test_aggregate_ogb_results_columns",
    "test_assert_rss_cap_raises_when_exceeded",
    "test_bgri_encoder_disable_classifier",
    "test_bgri_loss_finite",
    "test_bgri_target_update_changes_parameters",
    "test_build_predictions_artifact_subdir_created",
    "test_build_predictions_artifact_uses_official_node_ids",
    "test_build_predictions_artifact_writes_csv",
    "test_graphmae_node_level_encode",
    "test_graphsage_node_classifier_forward_shape",
    "test_induce_subgraph_empty",
    "test_induce_subgraph_round_trip",
    "test_load_sharded_state_dict_missing_manifest",
    "test_neighbor_sample_empty_seeds_returns_empty",
    "test_neighbor_sample_includes_seed",
    "test_neighbor_sample_reduces_for_higher_neighbors",
    "test_ogb_arxiv_assert_no_leakage_raises_when_test_labels_in_tensor",
    "test_ogb_arxiv_assert_no_leakage_succeeds_when_masked",
    "test_ogb_arxiv_load_test_labels_records_access",
    "test_ogb_arxiv_test_labels_private_by_default",
    "test_ogb_arxiv_to_round_trips_indices",
    "test_ogb_methods_includes_all_five",
    "test_persistent_jepa_classifier_forward_shape",
    "test_run_ogb_experiment_emits_official_node_ids",
    "test_run_ogb_experiment_smoke_all_five_methods",
    "test_run_ogb_experiment_smoke_writes_outputs",
    "test_run_ogb_experiment_writes_summary_csv",
    "test_shard_state_dict_cap_size",
    "test_shard_state_dict_round_trip",
    "test_smoke_config_is_fast",
    "test_smoke_rss_cap_works_when_unset",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_ogb(
    num_nodes: int = 30,
    feature_dim: int = 4,
    num_classes: int = 3,
) -> OGBArxiv:
    """Build a small synthetic OGB-shaped dataset for tests.

    Args:
        num_nodes: Number of vertices in the synthetic ring graph.
        feature_dim: Per-vertex feature dimension.
        num_classes: Number of distinct labels.

    Returns:
        A populated :class:`OGBArxiv`.
    """
    torch.manual_seed(0)
    features = torch.randn((num_nodes, feature_dim))
    labels = torch.tensor([i % num_classes for i in range(num_nodes)], dtype=torch.long)
    edges_src = torch.arange(num_nodes, dtype=torch.long)
    edges_dst = torch.roll(edges_src, shifts=-1)
    edges = torch.stack([edges_src, edges_dst], dim=0)
    graph = TypedAttributedGraph(
        vertex_features=features,
        edge_index=edges,
        edge_features=torch.zeros((edges.shape[1], 1)),
        vertex_labels=labels,
    )
    train_idx = list(range(num_nodes // 2))
    val_idx = list(range(num_nodes // 2, num_nodes // 2 + 5))
    test_idx = list(range(num_nodes // 2 + 5, num_nodes))
    return OGBArxiv(
        graph=graph,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        feature_dim=feature_dim,
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# OGB loader + leakage assertion
# ---------------------------------------------------------------------------


def test_ogb_arxiv_test_labels_private_by_default() -> None:
    """Test labels are not part of the public attributes."""
    dataset = _make_synthetic_ogb()
    assert hasattr(dataset, "test_indices")
    assert not dataset.test_labels_accessed


def test_ogb_arxiv_load_test_labels_records_access() -> None:
    """load_test_labels flips the access flag and returns the labels."""
    dataset = _make_synthetic_ogb()
    assert not dataset.test_labels_accessed
    test_labels = dataset.load_test_labels()
    assert dataset.test_labels_accessed
    assert test_labels.shape[0] == len(dataset.test_indices)
    expected = dataset.graph.vertex_labels[torch.tensor(dataset.test_indices, dtype=torch.long)]
    assert torch.equal(test_labels, expected)


def test_ogb_arxiv_assert_no_leakage_raises_when_test_labels_in_tensor() -> None:
    """assert_no_test_leakage raises DataError if a test label is non-zero."""
    dataset = _make_synthetic_ogb()
    leaked = dataset.graph.vertex_labels.clone()
    leaked[torch.tensor(dataset.test_indices, dtype=torch.long)] = 1
    with pytest.raises(DataError):
        dataset.assert_no_test_leakage(training_labels=leaked)


def test_ogb_arxiv_assert_no_leakage_succeeds_when_masked() -> None:
    """assert_no_test_leakage is silent when test labels are zeroed."""
    dataset = _make_synthetic_ogb()
    masked = dataset.graph.vertex_labels.clone()
    masked[torch.tensor(dataset.test_indices, dtype=torch.long)] = 0
    dataset.assert_no_test_leakage(training_labels=masked)


def test_ogb_arxiv_assert_no_leakage_raises_after_load_test_labels() -> None:
    """Calling load_test_labels then assert_no_test_leakage raises."""
    dataset = _make_synthetic_ogb()
    dataset.load_test_labels()
    with pytest.raises(DataError):
        dataset.assert_no_test_leakage(training_labels=dataset.graph.vertex_labels.clone())


def test_ogb_arxiv_to_round_trips_indices() -> None:
    """The device move preserves splits and feature metadata."""
    dataset = _make_synthetic_ogb()
    target_device = torch.device("cpu")
    moved = dataset.to(target_device)
    assert moved.train_indices == dataset.train_indices
    assert moved.val_indices == dataset.val_indices
    assert moved.test_indices == dataset.test_indices
    assert moved.feature_dim == dataset.feature_dim
    assert moved.num_classes == dataset.num_classes


# ---------------------------------------------------------------------------
# Neighbour sampling + induced subgraph
# ---------------------------------------------------------------------------


def test_neighbor_sample_includes_seed() -> None:
    """The seed nodes are always present in the sampled subgraph."""
    dataset = _make_synthetic_ogb(num_nodes=20)
    seeds = torch.tensor([0, 5, 10], dtype=torch.long)
    sample = neighbor_sample(
        dataset.graph.edge_index,
        seeds,
        num_hops=2,
        num_neighbors=-1,
        num_total_nodes=dataset.graph.num_vertices(),
    )
    for seed in seeds.tolist():
        assert seed in sample.node_ids.tolist()


def test_neighbor_sample_reduces_for_higher_neighbors() -> None:
    """Larger neighbour budgets never produce smaller subgraphs."""
    dataset = _make_synthetic_ogb(num_nodes=40)
    seeds = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    small = neighbor_sample(
        dataset.graph.edge_index, seeds, num_hops=2, num_neighbors=1, num_total_nodes=40
    )
    large = neighbor_sample(
        dataset.graph.edge_index, seeds, num_hops=2, num_neighbors=-1, num_total_nodes=40
    )
    assert large.node_ids.numel() >= small.node_ids.numel()


def test_neighbor_sample_empty_seeds_returns_empty() -> None:
    """Empty seed tensors produce empty samples (and no exception)."""
    empty = torch.empty((0,), dtype=torch.long)
    sample = neighbor_sample(
        torch.zeros((2, 0), dtype=torch.long),
        empty,
        num_hops=2,
        num_neighbors=4,
        num_total_nodes=0,
    )
    assert sample.node_ids.numel() == 0
    assert sample.edge_index.shape == (2, 0)


def test_induce_subgraph_round_trip() -> None:
    """The induced subgraph's edges are restricted to the requested nodes."""
    edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    keep = torch.tensor([0, 2, 4], dtype=torch.long)
    sub = induce_subgraph(edges, keep, num_total_nodes=6)
    src, dst = sub[0].tolist(), sub[1].tolist()
    for s, d in zip(src, dst):
        assert s in (0, 1, 2)
        assert d in (0, 1, 2)
        assert s != d or s == 0


def test_induce_subgraph_empty() -> None:
    """An empty ``nodes`` tensor returns an empty edge-index."""
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    sub = induce_subgraph(edges, torch.empty((0,), dtype=torch.long), num_total_nodes=2)
    assert sub.shape == (2, 0)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def test_graphsage_node_classifier_forward_shape() -> None:
    """GraphSAGE produces per-vertex logits of shape ``[N, num_classes]``."""
    graph = TypedAttributedGraph(
        vertex_features=torch.randn((6, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long),
    )
    model = GraphSAGE(input_dim=4, hidden_dim=8, num_layers=2, num_classes=3)
    logits = model.node_logits(graph)
    assert logits.shape == (6, 3)


def test_graphmae_node_level_encode() -> None:
    """GraphMAE.encode returns per-vertex embeddings of the right shape."""
    graph = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
    )
    enc = GraphMAE(input_dim=4, hidden_dim=6, num_layers=2)
    h = enc.encode(graph)
    assert h.shape == (5, 6)


def test_bgri_loss_finite() -> None:
    """BGRL loss is finite on two augmented views of the same graph."""
    graph = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
    )
    model = BGRL(input_dim=4, hidden_dim=8, num_layers=2)
    other = TypedAttributedGraph(
        vertex_features=graph.vertex_features + 0.1,
        edge_index=graph.edge_index,
        edge_features=graph.edge_features,
    )
    loss = model.loss(graph, other)
    assert torch.isfinite(loss)


def test_bgri_target_update_changes_parameters() -> None:
    """BGRL.update_target mutates the target parameters."""
    model = BGRL(input_dim=4, hidden_dim=8, num_layers=1)
    target_params = next(iter(model.target_encoder.parameters()))
    online_params = next(iter(model.online_encoder.parameters()))
    before = target_params.detach().clone()
    with torch.no_grad():
        online_params.add_(0.5)
    model.update_target()
    after = target_params.detach().clone()
    assert not torch.equal(before, after)


def test_bgri_encoder_disable_classifier() -> None:
    """BGRL with num_classes=0 disables the classifier head."""
    model = BGRL(input_dim=4, hidden_dim=8, num_layers=1, num_classes=0)
    assert model.classifier is None
    assert model.embed(
        TypedAttributedGraph(
            vertex_features=torch.randn((3, 4)),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        )
    ).shape == (1, 8)


def test_persistent_jepa_classifier_forward_shape() -> None:
    """PersistentJEPAClassifier maps ``[B, H]`` to ``[B, num_classes]``."""
    head = PersistentJEPAClassifier(hidden_dim=16, num_classes=3)
    feats = torch.randn((4, 16))
    out = head(feats)
    assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# Experiment dispatch + aggregation
# ---------------------------------------------------------------------------


def test_ogb_methods_includes_all_five() -> None:
    """OGB_METHODS contains all five Phase 10 methods."""
    assert set(OGB_METHODS) == {"GCN", "GraphSAGE", "BGRL", "GraphMAE", "PersistentJEPA"}


def test_smoke_config_is_fast() -> None:
    """default_smoke_config produces a one-epoch, one-seed configuration."""
    base = OGBConfig()
    smoke = default_smoke_config(base)
    assert smoke.smoke is True
    assert smoke.epochs == 1
    assert smoke.n_seeds == 1
    assert smoke.hidden_dim == 8
    assert smoke.num_layers == 1


def test_run_ogb_experiment_smoke_all_five_methods() -> None:
    """Smoke mode trains and evaluates every method end-to-end."""
    dataset = _make_synthetic_ogb()
    config = OGBConfig(
        smoke=True,
        output_dir="/tmp/phase10_smoke_all",
        rss_cap_mb=0.0,
        emit_predictions=False,
    )
    results = run_ogb_experiment(config, dataset=dataset)
    methods_seen = {r.method for r in results}
    assert methods_seen == set(OGB_METHODS)
    for r in results:
        assert 0.0 <= r.val_acc <= 1.0
        assert 0.0 <= r.test_acc <= 1.0
        assert r.elapsed_seconds >= 0.0


def test_run_ogb_experiment_smoke_writes_outputs(tmp_path: Path) -> None:
    """Smoke mode writes the long CSV and the summary table."""
    dataset = _make_synthetic_ogb()
    config = OGBConfig(
        smoke=True,
        output_dir=str(tmp_path),
        rss_cap_mb=0.0,
        emit_predictions=True,
    )
    run_ogb_experiment(config, dataset=dataset)
    assert (tmp_path / "ogb_results.csv").exists()
    assert (tmp_path / "tables" / "ogb_summary.csv").exists()
    assert (tmp_path / "predictions").is_dir()


def test_run_ogb_experiment_writes_summary_csv(tmp_path: Path) -> None:
    """ogb_summary.csv contains the expected columns."""
    dataset = _make_synthetic_ogb()
    config = OGBConfig(smoke=True, output_dir=str(tmp_path), rss_cap_mb=0.0)
    run_ogb_experiment(config, dataset=dataset)
    with (tmp_path / "tables" / "ogb_summary.csv").open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    expected_cols = {
        "method",
        "n_seeds",
        "mean_test_acc",
        "std_test_acc",
        "mean_val_acc",
        "mean_elapsed_seconds",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_p_value",
        "wilcoxon_p",
        "wilcoxon_p_bonferroni",
    }
    assert expected_cols.issubset(set(header))


def test_run_ogb_experiment_emits_official_node_ids(tmp_path: Path) -> None:
    """Emitted prediction CSVs use the official OGB test indices."""
    dataset = _make_synthetic_ogb()
    config = OGBConfig(smoke=True, output_dir=str(tmp_path), rss_cap_mb=0.0, emit_predictions=True)
    run_ogb_experiment(config, dataset=dataset)
    artifact = tmp_path / "predictions" / "GCN_seed0.csv"
    assert artifact.exists()
    with artifact.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    assert header == ["node_id", "prediction"]
    node_ids = [int(r[0]) for r in rows]
    # The official OGB test indices must match ``dataset.test_indices``.
    assert node_ids == list(dataset.test_indices)


def test_aggregate_ogb_results_columns() -> None:
    """aggregate_ogb_results emits bootstrap CI and Bonferroni-adjusted p-values."""
    rows = [
        OGBRunResult(
            method="GCN",
            seed=0,
            val_acc=0.7,
            test_acc=0.71,
            elapsed_seconds=1.0,
            n_train_nodes=10,
            n_val_nodes=5,
            n_test_nodes=5,
        ),
        OGBRunResult(
            method="GCN",
            seed=1,
            val_acc=0.72,
            test_acc=0.73,
            elapsed_seconds=1.0,
            n_train_nodes=10,
            n_val_nodes=5,
            n_test_nodes=5,
        ),
        OGBRunResult(
            method="PersistentJEPA",
            seed=0,
            val_acc=0.75,
            test_acc=0.78,
            elapsed_seconds=1.0,
            n_train_nodes=10,
            n_val_nodes=5,
            n_test_nodes=5,
        ),
        OGBRunResult(
            method="PersistentJEPA",
            seed=1,
            val_acc=0.76,
            test_acc=0.79,
            elapsed_seconds=1.0,
            n_train_nodes=10,
            n_val_nodes=5,
            n_test_nodes=5,
        ),
    ]
    summary = aggregate_ogb_results(rows, reference="GCN", n_resamples=200, seed=0)
    assert "GCN" in summary and "PersistentJEPA" in summary
    for entry in summary.values():
        assert "bootstrap_ci_low" in entry
        assert "bootstrap_ci_high" in entry
        assert "wilcoxon_p_bonferroni" in entry
    assert summary["PersistentJEPA"]["wilcoxon_p_bonferroni"] >= 0.0


def test_aggregate_ogb_results_pairs_by_seed() -> None:
    """The CI is computed over seeds present in both methods only."""
    rows = [
        OGBRunResult(
            method="GCN",
            seed=0,
            val_acc=0.7,
            test_acc=0.71,
            elapsed_seconds=1.0,
            n_train_nodes=10,
            n_val_nodes=5,
            n_test_nodes=5,
        ),
        OGBRunResult(
            method="GCN",
            seed=1,
            val_acc=0.72,
            test_acc=0.73,
            elapsed_seconds=1.0,
            n_train_nodes=10,
            n_val_nodes=5,
            n_test_nodes=5,
        ),
        OGBRunResult(
            method="PersistentJEPA",
            seed=1,
            val_acc=0.75,
            test_acc=0.78,
            elapsed_seconds=1.0,
            n_train_nodes=10,
            n_val_nodes=5,
            n_test_nodes=5,
        ),
    ]
    summary = aggregate_ogb_results(rows, reference="GCN", n_resamples=200, seed=0)
    # Only seed=1 is in common; the CI / Wilcoxon test compare exactly
    # one matched observation per method.
    assert summary["PersistentJEPA"]["wilcoxon_p_bonferroni"] >= 0.0


# ---------------------------------------------------------------------------
# Prediction artifacts
# ---------------------------------------------------------------------------


def test_build_predictions_artifact_writes_csv(tmp_path: Path) -> None:
    """build_predictions_artifact writes a CSV with node_id, prediction rows."""
    preds = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    path = build_predictions_artifact(preds, tmp_path, "GCN", 0)
    assert path.exists()
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    assert header == ["node_id", "prediction"]
    assert len(rows) == preds.numel()


def test_build_predictions_artifact_subdir_created(tmp_path: Path) -> None:
    """The predictions/ subdirectory is created if it does not yet exist."""
    preds = torch.tensor([1, 0], dtype=torch.long)
    path = build_predictions_artifact(preds, tmp_path, "GraphSAGE", 1)
    assert path.parent.is_dir()
    assert path.exists()


def test_build_predictions_artifact_uses_official_node_ids(tmp_path: Path) -> None:
    """When ``node_ids`` is supplied, the ``node_id`` column carries it."""
    preds = torch.tensor([0, 1, 2], dtype=torch.long)
    node_ids = [12345, 12346, 12347]
    path = build_predictions_artifact(preds, tmp_path, "PersistentJEPA", 0, node_ids=node_ids)
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    assert header == ["node_id", "prediction"]
    assert [int(r[0]) for r in rows] == node_ids


# ---------------------------------------------------------------------------
# Sharding + RSS cap
# ---------------------------------------------------------------------------


def test_shard_state_dict_round_trip(tmp_path: Path) -> None:
    """Sharding and loading round-trips every tensor exactly."""
    state = {
        "w": torch.randn((4, 4)),
        "b": torch.randn((4,)),
        "small": torch.zeros((1,)),
    }
    manifest = shard_state_dict(state, tmp_path, max_shard_bytes=64)
    assert isinstance(manifest, ShardedCheckpoint)
    loaded = load_sharded_state_dict(tmp_path)
    assert set(loaded) == set(state)
    for name, tensor in state.items():
        assert torch.equal(loaded[name], tensor)


def test_shard_state_dict_cap_size(tmp_path: Path) -> None:
    """A single oversized tensor raises CheckpointError when above the cap."""
    from pjepa.exceptions import CheckpointError

    huge = torch.zeros((256, 256))
    with pytest.raises(CheckpointError):
        shard_state_dict({"big": huge}, tmp_path, max_shard_bytes=1024)


def test_load_sharded_state_dict_missing_manifest(tmp_path: Path) -> None:
    """Loading from a missing directory raises CheckpointError."""
    from pjepa.exceptions import CheckpointError

    with pytest.raises(CheckpointError):
        load_sharded_state_dict(tmp_path)


def test_assert_rss_cap_raises_when_exceeded() -> None:
    """assert_rss_cap raises when the configured cap is too small."""
    cap = max(current_rss_mb() / 2.0, 1.0)
    with pytest.raises(BackendError):
        assert_rss_cap(cap)


def test_smoke_rss_cap_works_when_unset(tmp_path: Path) -> None:
    """A zero RSS cap disables the assertion entirely."""
    dataset = _make_synthetic_ogb()
    config = OGBConfig(smoke=True, output_dir=str(tmp_path), rss_cap_mb=0.0, emit_predictions=False)
    run_ogb_experiment(config, dataset=dataset)
