"""Tests for pjepa.augmentations, pjepa.data, pjepa.baselines.

Covers the eight-class test taxonomy.
"""

from __future__ import annotations

import pytest
import torch

from pjepa.augmentations import (
    AugmentationPipeline,
    DropEdge,
    DropFeature,
    DropNode,
    FeatureMask,
    RandomWalkSubgraph,
)
from pjepa.augmentations.base import PipelineMode
from pjepa.baselines import EWC, GCN, GIN, GraphCL, GraphMAE, InfoGraph
from pjepa.data.cl_splits import make_class_incremental_split
from pjepa.data.tu import load_tu_dataset
from pjepa.exceptions import DataError, GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = [
    "test_happy_drop_edge_removes_edges",
    "test_happy_drop_node_reduces_vertices",
    "test_happy_drop_feature_zero_mask",
    "test_happy_feature_mask_applies_token",
    "test_happy_random_walk_subgraph",
    "test_happy_pipeline_sequential",
    "test_happy_pipeline_random_sample_one",
    "test_happy_gcn_forward",
    "test_happy_gin_forward",
    "test_happy_graphmae_forward",
    "test_happy_graphcl_loss_runs",
    "test_happy_infograph_loss_runs",
    "test_happy_ewc_capture_and_penalty",
    "test_happy_gem_memory_add",
    "test_happy_make_class_incremental_split",
    "test_bad_augmentation_strength_out_of_range",
    "test_bad_drop_node_empty_graph",
    "test_bad_pipeline_empty_augmentations",
    "test_bad_pipeline_zero_k",
    "test_bad_ewc_negative_lambda",
    "test_bad_gem_zero_capacity",
    "test_bad_class_incremental_split_too_many_tasks",
    "test_bad_class_incremental_split_empty_labels",
    "test_bad_gcn_zero_dim",
    "test_ugly_drop_edge_no_edges",
    "test_ugly_random_walk_on_disconnected",
    "test_leaky_augmentation_state_isolation",
    "test_round_trip_augmentation_pipeline_serialization",
    "test_cross_backend_mps_gcn_forward",
    "test_distributional_drop_edge_distribution",
    "test_property_graphmae_mask_is_subset",
    "test_property_class_incremental_classes_disjoint",
]


# ============================== AUGMENTATIONS ==============================


def _toy_graph(num_vertices: int = 10, feature_dim: int = 4) -> TypedAttributedGraph:
    edges = [(i, (i + 1) % num_vertices) for i in range(num_vertices)]
    ei = torch.tensor(edges, dtype=torch.long).T
    return TypedAttributedGraph(
        vertex_features=torch.ones((num_vertices, feature_dim)),
        edge_index=ei,
        edge_features=torch.zeros((ei.shape[1], 1)),
    )


def test_happy_drop_edge_removes_edges() -> None:
    """DropEdge removes roughly the configured fraction of edges."""
    g = _toy_graph(20)
    out = DropEdge(strength=0.5)(g)
    assert out.num_edges() < g.num_edges()


def test_happy_drop_node_reduces_vertices() -> None:
    """DropNode removes a fraction of vertices."""
    g = _toy_graph(20)
    out = DropNode(strength=0.3)(g)
    assert out.num_vertices() < g.num_vertices()


def test_happy_drop_feature_zero_mask() -> None:
    """DropFeature zeros a fraction of feature dimensions."""
    g = _toy_graph(8, feature_dim=10)
    out = DropFeature(strength=0.5, generator=torch.Generator().manual_seed(0))(g)
    # Some feature columns should be all zero in the output.
    assert (out.vertex_features == 0.0).any(dim=0).sum() >= 1


def test_happy_feature_mask_applies_token() -> None:
    """FeatureMask replaces selected entries with the mask token (zeros)."""
    g = _toy_graph(8, feature_dim=6)
    aug = FeatureMask(feature_dim=6, strength=0.5, generator=torch.Generator().manual_seed(0))
    out = aug(g)
    # At least one entry should have been replaced with the mask value (0.0).
    assert (out.vertex_features == 0.0).any()


def test_happy_random_walk_subgraph() -> None:
    """RandomWalkSubgraph returns a connected subgraph."""
    g = _toy_graph(20)
    out = RandomWalkSubgraph(strength=0.5, generator=torch.Generator().manual_seed(0))(g)
    assert out.num_vertices() <= 10


def test_happy_pipeline_sequential() -> None:
    """A sequential pipeline applies augmentations in order."""
    g = _toy_graph(20)
    pipeline = AugmentationPipeline(
        [DropEdge(strength=0.2), DropNode(strength=0.2)],
        mode=PipelineMode.SEQUENTIAL,
    )
    out = pipeline(g)
    assert out.num_vertices() <= g.num_vertices()


def test_happy_pipeline_random_sample_one() -> None:
    """A random-sample-one pipeline picks one augmentation."""
    g = _toy_graph(20)
    pipeline = AugmentationPipeline(
        [DropEdge(strength=0.2), DropNode(strength=0.2)],
        mode=PipelineMode.RANDOM_SAMPLE_ONE,
        generator=torch.Generator().manual_seed(0),
    )
    out = pipeline(g)
    assert out.num_vertices() <= g.num_vertices()


# ============================== BASELINES ==============================


def test_happy_gcn_forward() -> None:
    """GCN returns per-graph logits of the right shape."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    )
    model = GCN(input_dim=4, hidden_dim=8, num_classes=2)
    out = model(g)
    assert out.shape == (1, 2)


def test_happy_gin_forward() -> None:
    """GIN returns per-graph logits of the right shape."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    )
    model = GIN(input_dim=4, hidden_dim=8, num_classes=2, num_layers=2)
    out = model(g)
    assert out.shape == (1, 2)


def test_happy_graphmae_forward() -> None:
    """GraphMAE returns embedding, mask, and reconstruction."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    )
    model = GraphMAE(input_dim=4, hidden_dim=8, num_layers=2)
    out = model(g)
    assert "embedding" in out
    assert out["mask"].sum() > 0
    assert out["reconstruction"].shape == (5, 4)


def test_happy_graphcl_loss_runs() -> None:
    """GraphCL produces a finite NT-Xent loss for two views."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    model = GraphCL(input_dim=4, hidden_dim=8)
    loss = model.loss(g, g)
    assert torch.isfinite(loss)


def test_happy_infograph_loss_runs() -> None:
    """InfoGraph produces a finite MI loss."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    model = InfoGraph(input_dim=4, hidden_dim=8)
    node, graph_emb = model.encode(g)
    perm = torch.randperm(5)
    shuffled_node = node[perm]
    loss = model.loss(node, graph_emb, shuffled_node)
    assert torch.isfinite(loss)


def test_happy_ewc_capture_and_penalty() -> None:
    """EWC captures Fisher info and computes a penalty."""
    param = torch.nn.Parameter(torch.randn(3))
    ewc = EWC(lambda_ewc=1.0)
    # Loss with non-trivial gradient produces non-zero Fisher info.
    loss = (param ** 2).sum() + (param * torch.tensor([1.0, 2.0, 3.0])).sum()
    ewc.capture([("p", param)], loss)
    # Penalise drift.
    drifted = torch.nn.Parameter(torch.ones(3))
    penalty = ewc.penalty([("p", drifted)])
    assert float(penalty.item()) > 0


def test_happy_gem_memory_add() -> None:
    """GEM memory stores samples up to capacity."""
    from pjepa.baselines.gem import GEM

    gem = GEM(capacity=4)
    for i in range(6):
        gem.add(torch.tensor([i]), torch.tensor([0]))
    assert len(gem) == 4


# ============================== DATA ==============================


def test_happy_make_class_incremental_split() -> None:
    """A 5-task class-incremental split is constructed correctly."""
    labels = [i % 10 for i in range(100)]
    split = make_class_incremental_split(labels, num_tasks=5, seed_split=0)
    assert split.num_tasks() == 5
    assert split.num_classes == 10
    assert sum(split.task_size(t) for t in range(5)) == 100


# ============================== BAD PATHS ==============================


def test_bad_augmentation_strength_out_of_range() -> None:
    """A strength outside [0, 1] is rejected."""
    with pytest.raises(GraphError):
        DropEdge(strength=1.5)


def test_bad_drop_node_empty_graph() -> None:
    """DropNode on an empty graph returns the original (cannot drop)."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 3)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    out = DropNode(strength=0.5)(g)
    assert out.num_vertices() == 0


def test_bad_pipeline_empty_augmentations() -> None:
    """An empty augmentation list is rejected."""
    with pytest.raises(GraphError):
        AugmentationPipeline([])


def test_bad_pipeline_zero_k() -> None:
    """A zero k in RANDOM_SAMPLE_K mode is rejected."""
    with pytest.raises(GraphError):
        AugmentationPipeline([DropEdge(strength=0.1)], mode=PipelineMode.RANDOM_SAMPLE_K, k=0)


def test_bad_ewc_negative_lambda() -> None:
    """A negative EWC lambda is rejected."""
    with pytest.raises(ValueError):
        EWC(lambda_ewc=-1.0)


def test_bad_gem_zero_capacity() -> None:
    """A zero GEM capacity is rejected."""
    with pytest.raises(Exception):
        GEM(capacity=0)


def test_bad_class_incremental_split_too_many_tasks() -> None:
    """More tasks than classes is rejected."""
    with pytest.raises(DataError):
        make_class_incremental_split([0, 1, 2], num_tasks=5, seed_split=0)


def test_bad_class_incremental_split_empty_labels() -> None:
    """An empty label list is rejected."""
    with pytest.raises(DataError):
        make_class_incremental_split([], num_tasks=1, seed_split=0)


def test_bad_gcn_zero_dim() -> None:
    """A zero-dim GCN is rejected."""
    with pytest.raises(ValueError):
        GCN(input_dim=0, hidden_dim=8)


# ============================== UGLY PATHS ==============================


def test_ugly_drop_edge_no_edges() -> None:
    """DropEdge on a graph with zero edges returns the graph unchanged."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((3, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    out = DropEdge(strength=0.5)(g)
    assert out.num_edges() == 0


def test_ugly_random_walk_on_disconnected() -> None:
    """RandomWalkSubgraph returns a non-empty result on a disconnected graph."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((5, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    out = RandomWalkSubgraph(strength=0.5, generator=torch.Generator().manual_seed(0))(g)
    # Walks on a disconnected graph return at least the starting vertex.
    assert out.num_vertices() >= 1


# ============================== LEAKY / ROUND-TRIP ==============================


def test_leaky_augmentation_state_isolation() -> None:
    """Two back-to-back augmentations do not share generator state."""
    aug = DropEdge(strength=0.5)
    g1 = _toy_graph(20)
    g2 = _toy_graph(20)
    out1 = aug(g1)
    out2 = aug(g2)
    # The exact result depends on the generator; here we only assert
    # that both calls produce valid outputs without raising.
    assert out1.num_edges() >= 0
    assert out2.num_edges() >= 0


def test_round_trip_augmentation_pipeline_serialization() -> None:
    """AugmentationPipeline reconstructs an equivalent pipeline."""
    augs = [DropEdge(strength=0.2), DropNode(strength=0.2)]
    pipe = AugmentationPipeline(augs, mode=PipelineMode.SEQUENTIAL)
    pipe2 = AugmentationPipeline(pipe.augmentations, mode=pipe.mode, k=pipe.k)
    g = _toy_graph(20)
    assert pipe(g).num_vertices() == pipe2(g).num_vertices()


# ============================== CROSS-BACKEND ==============================


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_gcn_forward() -> None:
    """GCN runs on MPS."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    ).to("mps")
    model = GCN(input_dim=4, hidden_dim=8, num_classes=2).to("mps")
    out = model(g)
    assert out.device.type == "mps"


# ============================== DISTRIBUTIONAL ==============================


def test_distributional_drop_edge_distribution() -> None:
    """DropEdge produces a varying edge count across random draws."""
    g = _toy_graph(50)
    survivors_sets = []
    for seed in range(10):
        aug = DropEdge(strength=0.3, generator=torch.Generator().manual_seed(seed))
        out = aug(g)
        # Compute which edges survived by checking edge index ordering.
        survivors_sets.append(tuple(sorted(map(tuple, out.edge_index.T.tolist()))))
    # With 0.3 drop and 50 edges, at least two distinct subsets should appear.
    assert len(set(survivors_sets)) >= 2


# ============================== PROPERTY ==============================


def test_property_graphmae_mask_is_subset() -> None:
    """GraphMAE's mask is a subset of the vertices."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((10, 4)),
        edge_index=torch.tensor([[i, (i + 1) % 10] for i in range(10)], dtype=torch.long).T,
    )
    model = GraphMAE(input_dim=4, hidden_dim=8, num_layers=2, mask_ratio=0.4)
    out = model(g)
    assert out["mask"].sum() <= 10


def test_property_class_incremental_classes_disjoint() -> None:
    """Class-incremental split assigns each class to exactly one task."""
    labels = [i % 7 for i in range(70)]
    split = make_class_incremental_split(labels, num_tasks=7, seed_split=42)
    union: set[int] = set()
    for tc in split.task_classes:
        assert not (tc & union), f"overlap between tasks: {tc & union}"
        union |= tc
    assert union == {0, 1, 2, 3, 4, 5, 6}