"""Tests for pjepa.augmentations registry, Identity, and Subgraph."""

from __future__ import annotations

import pytest
import torch

from pjepa.augmentations import (
    Identity,
    Subgraph,
    available_augmentations,
    evict_augmentation,
    get_augmentation,
    register,
)
from pjepa.augmentations.base import Augmentation
from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = [
    "test_bad_registry_duplicate_name",
    "test_bad_registry_unknown_name",
    "test_bad_subgraph_zero_strength",
    "test_happy_identity_is_noop",
    "test_happy_registry_lists_builtins",
    "test_happy_registry_lookup_builtin",
    "test_happy_registry_user_registration",
    "test_happy_subgraph_reduces_vertices",
    "test_round_trip_subgraph_seed_determinism",
    "test_ugly_subgraph_empty_graph",
]


def _toy_graph(num_vertices: int = 16, feature_dim: int = 4) -> TypedAttributedGraph:
    edges = [(i, (i + 1) % num_vertices) for i in range(num_vertices)]
    ei = torch.tensor(edges, dtype=torch.long).T
    return TypedAttributedGraph(
        vertex_features=torch.ones((num_vertices, feature_dim)),
        edge_index=ei,
        edge_features=torch.zeros((ei.shape[1], 1)),
    )


def test_happy_identity_is_noop() -> None:
    """Identity returns the same graph unchanged."""
    g = _toy_graph()
    out = Identity()(g)
    assert out is g or (
        out.num_vertices() == g.num_vertices()
        and out.num_edges() == g.num_edges()
        and torch.equal(out.vertex_features, g.vertex_features)
        and torch.equal(out.edge_index, g.edge_index)
    )


def test_happy_subgraph_reduces_vertices() -> None:
    """Subgraph returns a vertex-induced subgraph of the configured size."""
    g = _toy_graph(num_vertices=20)
    aug = Subgraph(strength=0.5, generator=torch.Generator().manual_seed(0))
    out = aug(g)
    assert 1 <= out.num_vertices() <= 10


def test_ugly_subgraph_empty_graph() -> None:
    """Subgraph on an empty graph returns the empty graph."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 3)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    out = Subgraph(strength=0.5)(g)
    assert out.num_vertices() == 0


def test_bad_subgraph_zero_strength() -> None:
    """A zero strength is rejected."""
    with pytest.raises(GraphError):
        Subgraph(strength=0.0)(_toy_graph())


def test_round_trip_subgraph_seed_determinism() -> None:
    """Two Subgraph calls with the same seed produce the same graph."""
    g = _toy_graph(num_vertices=24)
    out1 = Subgraph(strength=0.5, generator=torch.Generator().manual_seed(7))(g)
    out2 = Subgraph(strength=0.5, generator=torch.Generator().manual_seed(7))(g)
    assert torch.equal(out1.edge_index, out2.edge_index)
    assert torch.equal(out1.vertex_features, out2.vertex_features)


def test_happy_registry_lists_builtins() -> None:
    """The built-in augmentations are listed by the registry."""
    names = set(available_augmentations())
    assert "drop_edge" in names
    assert "drop_node" in names
    assert "drop_feature" in names
    assert "feature_mask" in names
    assert "identity" in names
    assert "subgraph" in names
    assert "connected_subgraph" in names


def test_happy_registry_lookup_builtin() -> None:
    """Lookups return the registered class."""
    assert get_augmentation("drop_edge") is not None
    assert get_augmentation("identity") is Identity


def test_bad_registry_unknown_name() -> None:
    """An unknown augmentation name raises a GraphError."""
    with pytest.raises(GraphError):
        get_augmentation("not-a-real-augmentation")


def test_happy_registry_user_registration() -> None:
    """User-registered augmentations are reachable by name."""

    @register("custom-test-aug")
    class _Custom(Augmentation):
        def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
            return graph

    try:
        assert get_augmentation("custom-test-aug") is _Custom
        assert "custom-test-aug" in available_augmentations()
    finally:
        evict_augmentation("custom-test-aug")


def test_bad_registry_duplicate_name() -> None:
    """Registering the same name twice raises a GraphError."""

    @register("dup-aug")
    class _A(Augmentation):
        def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
            return graph

    try:
        with pytest.raises(GraphError):

            @register("dup-aug")
            class _B(Augmentation):
                def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph:
                    return graph
    finally:
        evict_augmentation("dup-aug")
