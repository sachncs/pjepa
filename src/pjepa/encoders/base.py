"""Encoder protocol definition.

The encoder protocol is the only contract the rest of the library
relies on. It allows new architectures to be plugged in without
modifying the encoder registry.
"""

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

    Attributes:
        output_dim: The dimensionality of the produced embedding; the
            same value is also reported as the last dimension of the
            output tensor of :meth:`forward`.

    Example:
        >>> class Tiny(Encoder, torch.nn.Module):
        ...     output_dim = 4
        ...     def forward(self, graph):
        ...         return torch.zeros((graph.num_vertices(), self.output_dim))
        ...     def to(self, device):
        ...         return super().to(device)
        >>> isinstance(Tiny(), Encoder)
        True
    """

    output_dim: int

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Encode the graph and return an embedding tensor.

        Args:
            graph: The input graph.

        Returns:
            A ``[N, output_dim]`` per-vertex embedding tensor (or a
            graph-level tensor of shape ``[output_dim]`` when the
            encoder produces a graph summary).
        """

    def to(self, device: torch.device) -> Encoder:
        """Move the encoder's parameters to the given device."""


EncoderProtocol = Encoder
"""Alias for :class:`Encoder`, kept for documentation symmetry."""
