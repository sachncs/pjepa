"""Tests for pjepa.encoders.registry."""

from __future__ import annotations

import pytest
import torch

from pjepa.encoders import (
    DualGeometricEncoder,
    EuclideanMPNN,
    HyperbolicProjection,
    JEPAPredictor,
    available_encoders,
    evict_encoder,
    get_encoder,
    register,
)
from pjepa.encoders.base import Encoder
from pjepa.exceptions import ContractError
from pjepa.graphs import TypedAttributedGraph

__all__ = [
    "test_bad_registry_duplicate_encoder_name",
    "test_bad_registry_unknown_encoder",
    "test_happy_registry_lists_builtins",
    "test_happy_registry_lookup_builtin",
    "test_happy_registry_user_registration",
]


def _toy_graph() -> TypedAttributedGraph:
    return TypedAttributedGraph(
        vertex_features=torch.randn((4, 3)),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
    )


def test_happy_registry_lists_builtins() -> None:
    """All built-in encoders are registered on import."""
    names = set(available_encoders())
    assert "euclidean_mpnn" in names
    assert "hyperbolic" in names
    assert "dual_geometric" in names
    assert "jepa_predictor" in names


def test_happy_registry_lookup_builtin() -> None:
    """Lookups return the registered subclass."""
    assert get_encoder("euclidean_mpnn") is EuclideanMPNN
    assert get_encoder("hyperbolic") is HyperbolicProjection
    assert get_encoder("dual_geometric") is DualGeometricEncoder
    assert get_encoder("jepa_predictor") is JEPAPredictor


def test_bad_registry_unknown_encoder() -> None:
    """An unknown encoder name raises a ContractError."""
    with pytest.raises(ContractError):
        get_encoder("not-a-real-encoder")


def test_bad_registry_duplicate_encoder_name() -> None:
    """Registering the same name twice raises a ContractError."""

    @register("dup-encoder")
    class _A(Encoder, torch.nn.Module):
        output_dim = 4

        def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
            return torch.zeros((graph.num_vertices(), self.output_dim))

        def to(self, device):  # type: ignore[override]
            return super().to(device)

    try:
        with pytest.raises(ContractError):

            @register("dup-encoder")
            class _B(Encoder, torch.nn.Module):
                output_dim = 4

                def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
                    return torch.zeros((graph.num_vertices(), self.output_dim))

                def to(self, device):  # type: ignore[override]
                    return super().to(device)
    finally:
        evict_encoder("dup-encoder")


def test_happy_registry_user_registration() -> None:
    """A user-registered encoder is reachable by name."""

    @register("user-test-encoder")
    class _User(Encoder, torch.nn.Module):
        output_dim = 4

        def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
            return torch.zeros((graph.num_vertices(), self.output_dim))

        def to(self, device):  # type: ignore[override]
            return super().to(device)

    try:
        assert get_encoder("user-test-encoder") is _User
        assert "user-test-encoder" in available_encoders()
    finally:
        evict_encoder("user-test-encoder")
