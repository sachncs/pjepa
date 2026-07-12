"""Tests for pjepa.graphs."""

from __future__ import annotations

import pytest
import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import PersistentState, TypedAttributedGraph, WorkingGraph
from pjepa.graphs.persistent_state import CommitRejected
from pjepa.graphs.typed_graph import graph_from_edge_index

__all__ = [
    "test_bad_edge_index_dtype",
    "test_bad_edge_index_out_of_range",
    "test_bad_edge_index_wrong_shape",
    "test_cross_backend_mps_subgraph",
    "test_distributional_random_graphs",
    "test_happy_graph_construct",
    "test_leaky_repeated_construction_no_state",
    "test_persistent_commit_increments_version",
    "test_persistent_commit_rejects_non_negative_delta_j",
    "test_persistent_reject_records_reason",
    "test_property_with_features_increments_version",
    "test_round_trip_versioning",
    "test_subgraph_round_trip",
    "test_ugly_empty_graph",
    "test_ugly_single_vertex_no_edges",
    "test_working_graph_budget_enforced",
]


def test_happy_graph_construct() -> None:
    """A well-formed graph constructs without error."""
    v = torch.randn((3, 4))
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    e = torch.randn((3, 2))
    g = TypedAttributedGraph(v, ei, e)
    assert g.num_vertices() == 3
    assert g.num_edges() == 3


def test_bad_edge_index_wrong_shape() -> None:
    """Edge index with shape [3, E] is rejected."""
    v = torch.zeros((3, 4))
    ei = torch.zeros((3, 2), dtype=torch.long)
    with pytest.raises(GraphError):
        TypedAttributedGraph(v, ei, torch.zeros((2, 0)))


def test_bad_edge_index_dtype() -> None:
    """Edge index with float dtype is rejected."""
    v = torch.zeros((3, 4))
    ei = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(GraphError):
        TypedAttributedGraph(v, ei, torch.zeros((2, 0)))


def test_bad_edge_index_out_of_range() -> None:
    """Edge index pointing past the vertex count is rejected."""
    v = torch.zeros((2, 4))
    ei = torch.tensor([[0, 1], [5, 0]], dtype=torch.long)
    with pytest.raises(GraphError):
        TypedAttributedGraph(v, ei, torch.zeros((2, 2)))


def test_ugly_empty_graph() -> None:
    """A graph with zero vertices and zero edges constructs successfully."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_features=torch.zeros((0, 3)),
    )
    assert g.num_vertices() == 0
    assert g.num_edges() == 0


def test_ugly_single_vertex_no_edges() -> None:
    """A single-vertex, zero-edge graph constructs successfully."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((1, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_features=torch.zeros((0, 3)),
    )
    sub = g.subgraph(torch.tensor([True]))
    assert sub.num_vertices() == 1


def test_leaky_repeated_construction_no_state() -> None:
    """Repeated construction does not leak module-level state."""
    from pjepa.graphs.typed_graph import TypedAttributedGraph as Tag

    a = Tag(torch.zeros((2, 3)), torch.zeros((2, 0), dtype=torch.long))
    b = Tag(torch.zeros((5, 7)), torch.zeros((2, 0), dtype=torch.long))
    assert a.num_vertices() == 2
    assert b.num_vertices() == 5


def test_round_trip_versioning() -> None:
    """Every functional update increments the version counter."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((2, 3)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    assert g.version == 0
    g2 = g.with_features(global_features=torch.zeros((4,)))
    assert g2.version == 1


def test_subgraph_round_trip() -> None:
    """A subgraph induced by selecting all vertices equals the original."""
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    e = torch.tensor([[10.0], [20.0], [30.0]])
    g = TypedAttributedGraph(v, ei, e)
    mask = torch.tensor([True, True, True])
    sub = g.subgraph(mask)
    assert sub.num_vertices() == 3
    assert sub.num_edges() == 3
    assert torch.allclose(sub.vertex_features, v)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_subgraph() -> None:
    """Subgraph computation is consistent on MPS."""
    v = torch.randn((4, 2), device="mps")
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long, device="mps")
    e = torch.randn((3, 1), device="mps")
    g = TypedAttributedGraph(v, ei, e)
    mask = torch.tensor([True, False, True, True], device="mps")
    sub = g.subgraph(mask)
    assert sub.vertex_features.device.type == "mps"
    assert sub.num_vertices() == 3


def test_distributional_random_graphs() -> None:
    """Random graphs of various sizes construct without error."""
    for n in [1, 5, 50, 200]:
        ei = torch.randint(0, max(n, 1), (2, n * 2), dtype=torch.long)
        ei = ei[:, (ei[0] != ei[1]).nonzero(as_tuple=True)[0]]
        ei[:, ei[0] >= n] = 0
        g = TypedAttributedGraph(
            vertex_features=torch.zeros((n, 3)),
            edge_index=ei,
            edge_features=torch.zeros((ei.shape[1], 1)),
        )
        assert g.num_vertices() == n


def test_property_with_features_increments_version() -> None:
    """with_features always bumps the version, never decreases it."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((1, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    last = g.version
    for _ in range(10):
        g = g.with_features(global_features=torch.zeros((2,)))
        assert g.version > last
        last = g.version


def test_persistent_commit_increments_version() -> None:
    """A successful commit replaces the graph and records the commit."""
    g0 = TypedAttributedGraph(
        vertex_features=torch.zeros((1, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    state = PersistentState(graph=g0)
    g1 = g0.with_features(global_features=torch.zeros((3,)))
    state2 = state.commit(g1, cost=0.1, timestamp=1.0, delta_j=-0.5)
    assert state2.num_commits() == 1
    assert state2.graph.version == 1


def test_persistent_commit_rejects_non_negative_delta_j() -> None:
    """A commit with Δ𝒥 ≥ 0 is rejected."""
    g0 = TypedAttributedGraph(
        vertex_features=torch.zeros((1, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    state = PersistentState(graph=g0)
    g1 = g0.with_features(global_features=torch.zeros((3,)))
    with pytest.raises(GraphError):
        state.commit(g1, cost=0.0, timestamp=1.0, delta_j=0.0)
    with pytest.raises(GraphError):
        state.commit(g1, cost=0.0, timestamp=1.0, delta_j=0.5)


def test_persistent_reject_records_reason() -> None:
    """A rejection appends to the rejections audit trail."""
    g0 = TypedAttributedGraph(
        vertex_features=torch.zeros((1, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    state = PersistentState(graph=g0)
    state2 = state.reject("bisimilarity violated", cost=0.3)
    assert state2.num_rejections() == 1
    rejection = state2.rejections[0]
    assert isinstance(rejection, CommitRejected)
    assert "bisimilarity" in rejection.reason


def test_working_graph_budget_enforced() -> None:
    """A working graph with too many vertices is rejected at construction."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((10, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    with pytest.raises(GraphError):
        WorkingGraph(graph=g, budget=5)


def test_working_graph_utilisation() -> None:
    """Utilisation equals vertex count / budget."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((3, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    working = WorkingGraph(graph=g, budget=10)
    assert working.utilisation() == pytest.approx(0.3)


def test_graph_from_edge_index_synthesises_features() -> None:
    """graph_from_edge_index builds a graph with optional zero features."""
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    g = graph_from_edge_index(ei, num_vertices=2, vertex_dim=4, edge_dim=2)
    assert g.num_vertices() == 2
    assert g.num_edges() == 2
    assert g.vertex_features.shape == (2, 4)


def test_graph_from_edge_index_rejects_negative_vertices() -> None:
    """A negative vertex count is rejected."""
    with pytest.raises(GraphError):
        graph_from_edge_index(torch.zeros((2, 0), dtype=torch.long), num_vertices=-1)


def test_persistent_to_device_round_trip() -> None:
    """PersistentState.to moves the graph to a new device."""
    g0 = TypedAttributedGraph(
        vertex_features=torch.zeros((1, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    state = PersistentState(graph=g0)
    state_cpu = state.to(torch.device("cpu"))
    assert state_cpu.graph.vertex_features.device.type == "cpu"
