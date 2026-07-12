"""Dual-geometric encoder: Euclidean + hyperbolic components.

The dual-geometric encoder combines an :class:`EuclideanMPNN` with a
:class:`HyperbolicProjection` to produce a per-vertex representation
that captures both locality and hierarchical structure. The
combination is justified by Proposition 3 in the paper.
"""

from __future__ import annotations

import torch
from torch import nn

from pjepa.encoders.euclidean_mpnn import EuclideanMPNN
from pjepa.encoders.hyperbolic import HyperbolicProjection
from pjepa.graphs import TypedAttributedGraph

__all__ = ["DualGeometricEncoder"]


class DualGeometricEncoder(nn.Module):
    """Encode a graph into a dual-geometric per-vertex representation.

    The forward pass returns a tuple ``(e, h)`` of Euclidean and
    hyperbolic components. Downstream code can either concatenate
    them or process each component separately.
    """

    def __init__(
        self,
        input_dim: int,
        euclidean_dim: int = 128,
        hyperbolic_dim: int = 32,
        num_layers: int = 4,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.euclidean = EuclideanMPNN(
            input_dim=input_dim,
            hidden_dim=euclidean_dim,
            num_layers=num_layers,
            output_dim=euclidean_dim,
        )
        self.hyperbolic = HyperbolicProjection(
            input_dim=euclidean_dim,
            output_dim=hyperbolic_dim,
            curvature=curvature,
        )
        self.euclidean_dim = euclidean_dim
        self.hyperbolic_dim = hyperbolic_dim

    def forward(self, graph: TypedAttributedGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the graph into Euclidean and hyperbolic components.

        Args:
            graph: The input graph.

        Returns:
            A tuple ``(e, h)`` where ``e`` is ``[N, euclidean_dim]``
            and ``h`` is ``[N, hyperbolic_dim]``.
        """
        e = self.euclidean(graph)
        h = self.hyperbolic(e)
        return e, h
