"""Hyperbolic projection of Euclidean features into the Poincaré ball.

The projector wraps a linear projection followed by a ``tanh``-based
radial rescaling that maps every output onto the open unit ball
``B^d = {x ∈ ℝ^d : ‖x‖ < 1}``. Numerical stability is enforced by
clamping norms just inside ``max_norm``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pjepa.exceptions import NumericalError

__all__ = ["HyperbolicProjection"]


class HyperbolicProjection(nn.Module):
    """Project Euclidean features into the Poincaré ball of curvature ``-c``.

    The forward pass applies a linear map ``ℝ^{input_dim} → ℝ^{output_dim}``
    and then a two-step radial rescaling:

    1. ``u = project / ‖project‖`` followed by
       ``r = tanh(‖project‖ * sqrt(c))`` so the result lies on the
       hyperbolic ball of curvature ``-c``.
    2. The norm is clamped to ``max_norm`` to defend against
       floating-point drift when downstream code adds or subtracts
       small perturbations.

    Attributes:
        input_dim: Dimension of the input Euclidean features.
        output_dim: Dimension of the output hyperbolic features.
        curvature: A positive float controlling the curvature ``-c``.
        max_norm: Hyperbolic norms are clamped below this value to
            maintain numerical stability.

    Raises:
        ValueError: At construction if any dimension is non-positive,
            ``curvature <= 0``, or ``max_norm`` is outside ``(0, 1)``.
        NumericalError: At forward time if the output is not finite.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        curvature: float = 1.0,
        max_norm: float = 1.0 - 1e-5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("HyperbolicProjection: dims must be positive")
        if curvature <= 0:
            raise ValueError(f"HyperbolicProjection: curvature must be positive; got {curvature}")
        if not 0.0 < max_norm < 1.0:
            raise ValueError(f"HyperbolicProjection: max_norm must be in (0, 1); got {max_norm}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.curvature = curvature
        self.max_norm = max_norm
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project Euclidean features into the Poincaré ball.

        Args:
            features: A ``[..., input_dim]`` tensor of Euclidean
                features.

        Returns:
            A ``[..., output_dim]`` tensor of hyperbolic features
            with norms strictly below ``max_norm``.
        """
        projected = self.proj(features)
        norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        # Radial rescaling via ``tanh``: maps Euclidean direction onto the ball.
        scaled = projected / norms * torch.tanh(norms * math.sqrt(self.curvature))
        norms = scaled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        scaled = scaled / norms * norms.clamp(max=self.max_norm)
        if not torch.isfinite(scaled).all():
            raise NumericalError("HyperbolicProjection: produced non-finite values")
        return scaled
