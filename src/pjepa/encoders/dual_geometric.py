"""Dual-geometric encoder: Euclidean + hyperbolic components.

The composition of a :class:`Euclidean` and a :class:`Hyperbolic`
projection is the canonical encoder of the framework. It produces
a per-vertex representation that captures both locality and
hierarchical structure, which Proposition 3 of the paper justifies.

The class is a polymorphic :class:`Encoder` subclass. Its
:attr:`output_dim` is the sum of the Euclidean and hyperbolic
widths (the size of a concatenated representation). The forward
call returns a tuple ``(euclidean, hyperbolic)`` for callers that
need each component separately; the inherited :meth:`encode`
concatenates them along the feature axis for callers that want a
single tensor.

Example:
    >>> import torch
    >>> from pjepa.encoders.dual_geometric import DualGeometric
    >>> from pjepa.graphs import Graph
    >>> v = torch.randn((4, 8))
    >>> ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    >>> g = Graph(v, ei, torch.zeros((3, 2)))
    >>> enc = DualGeometric(input_dim=8, euclidean_dim=16, hyperbolic_dim=4, num_layers=2)
    >>> e, h = enc(g)
    >>> tuple(e.shape), tuple(h.shape)
    ((4, 16), (4, 4))
    >>> encoded = enc.encode(g)
    >>> tuple(encoded.shape)
    (4, 20)
"""

from __future__ import annotations

import torch

from pjepa.encoders.base import Encoder
from pjepa.encoders.euclidean_mpnn import Euclidean
from pjepa.encoders.hyperbolic import Hyperbolic
from pjepa.graphs import Graph

__all__ = ["DualGeometric"]


class DualGeometric(Encoder):
    """Euclidean + hyperbolic encoder.

    The forward pass returns a tuple ``(euclidean, hyperbolic)``
    of two per-vertex tensors. The inherited :meth:`encode`
    method concatenates them along the feature axis so callers
    that only need a single tensor can use the standard
    :class:`Encoder` protocol.

    Attributes:
        euclidean: The underlying :class:`Euclidean` encoder.
        hyperbolic: The :class:`Hyperbolic` projection applied
            to the Euclidean output.
        euclidean_dim: Width of the Euclidean representation.
        hyperbolic_dim: Width of the hyperbolic representation.

    Args:
        input_dim: Vertex feature dimension.
        euclidean_dim: Width of the Euclidean representation.
        hyperbolic_dim: Width of the hyperbolic representation.
        num_layers: Number of message-passing layers in the
            Euclidean encoder.
        curvature: Curvature ``-c`` of the Poincaré ball.

    Raises:
        ValueError: At construction if any dimension is
            non-positive.

    Example:
        >>> from pjepa.encoders.dual_geometric import DualGeometric
        >>> enc = DualGeometric(input_dim=4, euclidean_dim=8, hyperbolic_dim=4, num_layers=2)
        >>> enc.output_dim
        12
    """

    def __init__(
        self,
        input_dim: int,
        euclidean_dim: int = 128,
        hyperbolic_dim: int = 32,
        num_layers: int = 4,
        curvature: float = 1.0,
    ) -> None:
        """Initialise the encoder.

        Args:
            input_dim: Vertex feature dimension.
            euclidean_dim: Width of the Euclidean representation.
            hyperbolic_dim: Width of the hyperbolic representation.
            num_layers: Number of message-passing layers in the
                Euclidean encoder.
            curvature: Curvature ``-c`` of the Poincaré ball.

        Raises:
            ValueError: If any dimension is non-positive.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"DualGeometric: input_dim must be positive; got {input_dim}")
        if euclidean_dim <= 0:
            raise ValueError(f"DualGeometric: euclidean_dim must be positive; got {euclidean_dim}")
        if hyperbolic_dim <= 0:
            raise ValueError(
                f"DualGeometric: hyperbolic_dim must be positive; got {hyperbolic_dim}"
            )
        self.euclidean = Euclidean(
            input_dim=int(input_dim),
            hidden_dim=int(euclidean_dim),
            num_layers=int(num_layers),
            output_dim=int(euclidean_dim),
        )
        self.hyperbolic = Hyperbolic(
            input_dim=int(euclidean_dim),
            output_dim=int(hyperbolic_dim),
            curvature=float(curvature),
        )
        self.euclidean_dim = int(euclidean_dim)
        self.hyperbolic_dim = int(hyperbolic_dim)
        self.output_width = int(euclidean_dim) + int(hyperbolic_dim)

    def forward(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the graph into Euclidean and hyperbolic components.

        Args:
            graph: The input graph.

        Returns:
            A tuple ``(euclidean, hyperbolic)`` where
            ``euclidean`` is ``[N, euclidean_dim]`` and
            ``hyperbolic`` is ``[N, hyperbolic_dim]``.
        """
        e = self.euclidean(graph)
        h = self.hyperbolic(e)
        return e, h

    def encode(self, graph: Graph) -> torch.Tensor:
        """Encode the graph and concatenate the two components.

        Overrides the base :meth:`encode` to return a single
        ``[N, output_dim]`` tensor instead of a tuple.

        Args:
            graph: The input graph.

        Returns:
            A ``[N, euclidean_dim + hyperbolic_dim]`` tensor.
        """
        e, h = self.forward(graph)
        return torch.cat([e, h], dim=-1)

    def summary(self) -> dict[str, int | str]:
        """Return a JSON-serialisable description of the encoder.

        Returns:
            A dictionary with keys ``class``, ``output_dim``,
            ``euclidean_dim``, and ``hyperbolic_dim``.
        """
        return {
            "class": type(self).__name__,
            "output_dim": self.output_dim,
            "euclidean_dim": self.euclidean_dim,
            "hyperbolic_dim": self.hyperbolic_dim,
        }
