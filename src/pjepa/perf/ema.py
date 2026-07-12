"""Exponential Moving Average target encoder.

This is a thin wrapper around the BYOL-style EMA mechanism that keeps
a delayed copy of an online encoder's parameters. The schedule can be
fixed or cosine-annealed.
"""

from __future__ import annotations

import copy
import math

import torch

from pjepa.exceptions import NumericalError

__all__ = ["EMATarget"]


class EMATarget:
    """Maintain a delayed copy of an online module's parameters.

    Attributes:
        online: The online module whose parameters are tracked.
        momentum: The EMA momentum; ``1.0`` means no update, ``0.0``
          means instant copy.
        schedule: Either ``"constant"`` (fixed momentum) or
          ``"cosine"`` (cosine annealed from the initial value to
          the final value).
        final_momentum: For cosine schedule, the target momentum at
          the end of training.
        total_steps: For cosine schedule, the total number of updates
          over which to anneal.
    """

    def __init__(
        self,
        online: torch.nn.Module,
        momentum: float = 0.996,
        schedule: str = "constant",
        final_momentum: float = 0.999,
        total_steps: int = 1000,
    ) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise NumericalError(
                f"EMATarget: momentum must be in [0, 1]; got {momentum}"
            )
        if not 0.0 <= final_momentum <= 1.0:
            raise NumericalError(
                f"EMATarget: final_momentum must be in [0, 1]; got {final_momentum}"
            )
        if schedule not in ("constant", "cosine"):
            raise NumericalError(
                f"EMATarget: schedule must be 'constant' or 'cosine'; got {schedule!r}"
            )
        if total_steps <= 0:
            raise NumericalError(
                f"EMATarget: total_steps must be positive; got {total_steps}"
            )
        self.online = online
        self.momentum = momentum
        self.schedule = schedule
        self.final_momentum = final_momentum
        self.total_steps = total_steps
        self.step = 0
        self.shadow = copy.deepcopy(online)
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    def current_momentum(self) -> float:
        """Return the momentum that will be used for the next update."""
        if self.schedule == "constant":
            return self.momentum
        progress = min(1.0, self.step / self.total_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_momentum - (self.final_momentum - self.momentum) * cosine

    @torch.no_grad()
    def update(self) -> None:
        """Update the target parameters via EMA."""
        m = self.current_momentum()
        for online_param, shadow_param in zip(
            self.online.parameters(), self.shadow.parameters()
        ):
            new_value = m * shadow_param.data + (1.0 - m) * online_param.data
            if not torch.isfinite(new_value).all():
                raise NumericalError("EMATarget.update: produced non-finite parameters")
            shadow_param.data.copy_(new_value)
        self.step += 1

    def forward(self, *args: object, **kwargs: object) -> object:
        """Forward through the target encoder without gradients."""
        with torch.no_grad():
            return self.shadow(*args, **kwargs)
