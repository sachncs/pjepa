"""JEPA predictor with EMA target encoder (BYOL-style).

The online predictor produces predicted target embeddings from
context embeddings plus a positional encoding. The target encoder
is an exponential moving average of the online encoder (Grill et al.
2020) and is updated by the trainer.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from pjepa.exceptions import NumericalError

__all__ = ["JEPAPredictor", "TargetEncoder"]


class JEPAPredictor(nn.Module):
    """Predictor head that maps context features to predicted target features.

    Args:
        input_dim: Dimension of context features.
        hidden_dim: Hidden width of the predictor MLP.
        output_dim: Dimension of predicted target features.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("JEPAPredictor: dims must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict the target embedding from the context embedding.

        Args:
            context: A ``[..., input_dim]`` context tensor.

        Returns:
            A ``[..., output_dim]`` predicted target tensor.
        """
        return self.net(context)


class TargetEncoder:
    """Exponential moving average of an online encoder.

    The target encoder is a delayed copy of an online encoder whose
    parameters are updated after each training step as
    ``theta_target = tau * theta_target + (1 - tau) * theta_online``.
    """

    def __init__(self, online: nn.Module, momentum: float = 0.996) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"TargetEncoder: momentum must be in [0, 1]; got {momentum}")
        self.online = online
        self.momentum = momentum
        self.shadow = copy.deepcopy(online)
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self) -> None:
        """Update the target parameters via EMA."""
        for online_param, shadow_param in zip(self.online.parameters(), self.shadow.parameters()):
            new_value = (
                self.momentum * shadow_param.data + (1.0 - self.momentum) * online_param.data
            )
            if not torch.isfinite(new_value).all():
                raise NumericalError("TargetEncoder.update: produced non-finite parameters")
            shadow_param.data.copy_(new_value)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Forward through the target encoder without gradients."""
        with torch.no_grad():
            return self.shadow(*args, **kwargs)
