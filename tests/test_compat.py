"""Tests for pjepa.compat documented aliases."""

from __future__ import annotations

import torch

import pjepa
from pjepa.compat import (
    Graph,
    GraphState,
    PersistentGraph,
    PJEPAAugmentation,
    PJEPAEncoder,
    make_typed_graph,
)
from pjepa.graphs import (
    PersistentState,
    TypedAttributedGraph,
    WorkingGraph,
)

__all__ = [
    "test_happy_alias_graph",
    "test_happy_alias_persistent_graph",
    "test_happy_alias_working_graph",
    "test_happy_aliases_at_top_level",
    "test_happy_make_typed_graph_basic",
    "test_happy_make_typed_graph_with_edges",
]


def test_happy_alias_graph() -> None:
    """``Graph`` is an alias for ``TypedAttributedGraph``."""
    assert Graph is TypedAttributedGraph


def test_happy_alias_persistent_graph() -> None:
    """``PersistentGraph`` is an alias for ``PersistentState``."""
    assert PersistentGraph is PersistentState


def test_happy_alias_working_graph() -> None:
    """``GraphState`` is an alias for ``WorkingGraph``."""
    assert GraphState is WorkingGraph


def test_happy_aliases_at_top_level() -> None:
    """Compatibility aliases are re-exported from the top-level package."""
    assert pjepa.Graph is TypedAttributedGraph
    assert pjepa.PersistentGraph is PersistentState
    assert pjepa.GraphState is WorkingGraph
    assert pjepa.make_typed_graph is make_typed_graph
    assert pjepa.PJEPAEncoder is PJEPAEncoder
    assert pjepa.PJEPAAugmentation is PJEPAAugmentation


def test_happy_make_typed_graph_basic() -> None:
    """make_typed_graph builds a TypedAttributedGraph from positional tensors."""
    vf = torch.ones((3, 2))
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    g = make_typed_graph(vf, ei)
    assert isinstance(g, TypedAttributedGraph)
    assert g.num_vertices() == 3
    assert g.num_edges() == 2


def test_happy_make_typed_graph_with_edges() -> None:
    """make_typed_graph forwards edge_features when provided."""
    vf = torch.ones((3, 2))
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    ef = torch.zeros((2, 1))
    g = make_typed_graph(vf, ei, ef)
    assert torch.equal(g.edge_features, ef)
