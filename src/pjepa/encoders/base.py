"""Encoder protocol definition."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from pjepa.graphs import TypedAttributedGraph

__all__ = ["Encoder", "EncoderProtocol"]


@runtime_checkable
class Encoder(Protocol):
    """Protocol every encoder must satisfy.

    An encoder maps a :class:`TypedAttributedGraph` to an embedding
    tensor. Implementations are typically :class:`torch.nn.Module`
    subclasses, but the protocol only requires the two methods below.
    """

    output_dim: int

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Encode the graph and return a per-vertex or graph-level embedding."""
        ...

    def to(self, device: torch.device) -> "Encoder":
        """Move the encoder's parameters to the given device."""
        ...


EncoderProtocol = Encoder  # Alias for documentation.