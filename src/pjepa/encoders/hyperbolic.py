"""Hyperbolic projection of Euclidean features into the Poincaré ball.

The :class:`Hyperbolic` class wraps a linear projection followed by
a ``tanh``-based radial rescaling that maps every output onto the
open unit ball ``B^d = {x ∈ ℝ^d : ‖x‖ < 1}``. Numerical stability
is enforced by clamping norms just inside ``max_norm``.

The class is a standalone :class:`torch.nn.Module`, not a
:class:`Encoder` subclass, because it operates on tensors rather
than on :class:`Graph` instances. The dual-geometric encoder
(``dual_geometric.DualGeometric``) composes a :class:`Euclidean`
encoder with a :class:`Hyperbolic` projection to produce both
Euclidean and hyperbolic components.

Example:
    >>> import torch
    >>> from pjepa.encoders.hyperbolic import Hyperbolic
    >>> proj = Hyperbolic(input_dim=4, output_dim=3)
    >>> x = torch.randn((5, 4))
    >>> y = proj(x)
    >>> tuple(y.shape)
    (5, 3)
    >>> float(y.norm(dim=-1).max()) < 1.0
    True
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pjepa.exceptions import NumericalError

__all__ = ["Hyperbolic"]


class Hyperbolic(nn.Module):
    """Project Euclidean features into the Poincaré ball of curvature ``-c``.

    The forward pass applies a linear map
    ``ℝ^{input_dim} → ℝ^{output_dim}`` and then a two-step radial
    rescaling:

    1. ``u = project / ‖project‖`` followed by
       ``r = tanh(‖project‖ * sqrt(c))`` so the result lies on the
       hyperbolic ball of curvature ``-c``.
    2. The norm is clamped to ``max_norm`` to defend against
       floating-point drift when downstream code adds or subtracts
       small perturbations.

    Attributes:
        input_dim: Dimension of the input Euclidean features.
        output_width: Dimension of the output hyperbolic features.
        curvature: A positive float controlling the curvature ``-c``.
        max_norm: Hyperbolic norms are clamped below this value to
            maintain numerical stability.

    Args:
        input_dim: Dimension of the input Euclidean features.
        output_dim: Dimension of the output hyperbolic features.
        curvature: A positive float controlling the curvature ``-c``.
        max_norm: Hyperbolic norms are clamped below this value to
            maintain numerical stability.

    Raises:
        ValueError: At construction if any dimension is non-positive,
            ``curvature <= 0``, or ``max_norm`` is outside ``(0, 1)``.
        NumericalError: At forward time if the output is not finite.

    Example:
        >>> import torch
        >>> from pjepa.encoders.hyperbolic import Hyperbolic
        >>> proj = Hyperbolic(input_dim=2, output_dim=2)
        >>> x = torch.randn((3, 2))
        >>> torch.isfinite(proj(x)).all().item()
        True
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        curvature: float = 1.0,
        max_norm: float = 1.0 - 1e-5,
    ) -> None:
        """Initialise the projector.

        Args:
            input_dim: Dimension of the input Euclidean features.
            output_dim: Dimension of the output hyperbolic features.
            curvature: A positive float controlling the curvature ``-c``.
            max_norm: Hyperbolic norms are clamped below this value to
                maintain numerical stability.

        Raises:
            ValueError: If any dimension is non-positive,
                ``curvature <= 0``, or ``max_norm`` is outside
                ``(0, 1)``.
        """
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Hyperbolic: dims must be positive")
        if curvature <= 0:
            raise ValueError(f"Hyperbolic: curvature must be positive; got {curvature}")
        if not 0.0 < max_norm < 1.0:
            raise ValueError(f"Hyperbolic: max_norm must be in (0, 1); got {max_norm}")
        self.input_dim = int(input_dim)
        self.output_width = int(output_dim)
        self.curvature = float(curvature)
        self.max_norm = float(max_norm)
        self.proj = nn.Linear(int(input_dim), int(output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project Euclidean features into the Poincaré ball.

        Args:
            features: A ``[..., input_dim]`` tensor of Euclidean
                features.

        Returns:
            A ``[..., output_dim]`` tensor of hyperbolic features
            with norms strictly below ``max_norm``.

        Raises:
            NumericalError: If the produced tensor is not finite.
        """
        projected = self.proj(features)
        norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        # Radial rescaling via ``tanh``: maps Euclidean direction onto the ball.
        scaled = projected / norms * torch.tanh(norms * math.sqrt(self.curvature))
        norms = scaled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        scaled = scaled / norms * norms.clamp(max=self.max_norm)
        if not torch.isfinite(scaled).all():
            raise NumericalError("Hyperbolic: produced non-finite values")
        return scaled
