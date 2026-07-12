"""Stochastic Weight Averaging wrapper.

SWA averages model parameters across multiple snapshots, starting
from a configurable epoch. The averaged weights tend to generalise
better than any single snapshot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from pjepa.exceptions import ConfigError

__all__ = ["SWAConfig", "SWAWrapper"]


@dataclass(frozen=True)
class SWAConfig:
    """Configuration for the SWA wrapper.

    Attributes:
        start_epoch: First epoch at which to begin averaging.
        swa_lr: Optional learning rate to use after SWA is activated
          (often lower than the base learning rate).
    """

    start_epoch: int = 0
    swa_lr: float | None = None


class SWAWrapper:
    """Maintain a running average of model parameters.

    Attributes:
        model: The model being averaged.
        config: The SWA configuration.
    """

    def __init__(self, model: torch.nn.Module, config: SWAConfig | None = None) -> None:
        self.config = config or SWAConfig()
        if self.config.start_epoch < 0:
            raise ConfigError(
                f"SWAConfig.start_epoch must be non-negative; got {self.config.start_epoch}"
            )
        self.model = model
        self.snapshot_count = 0
        self.averaged_state: dict[str, torch.Tensor] = {}
        self.snapshots: deque = deque(maxlen=64)

    def should_snapshot(self, epoch: int) -> bool:
        """Return whether the current epoch should contribute a snapshot."""
        return epoch >= self.config.start_epoch

    def update(self, epoch: int) -> None:
        """Record a snapshot of the model parameters at the given epoch.

        Args:
            epoch: The current training epoch.
        """
        if not self.should_snapshot(epoch):
            return
        snapshot = {name: param.detach().clone() for name, param in self.model.named_parameters()}
        self.snapshots.append(snapshot)
        self.snapshot_count += 1
        # Maintain a running average for efficiency.
        if not self.averaged_state:
            self.averaged_state = {name: tensor.clone() for name, tensor in snapshot.items()}
        else:
            n = float(self.snapshot_count)
            for name, tensor in snapshot.items():
                self.averaged_state[name] = (n - 1.0) / n * self.averaged_state[name] + (
                    1.0 / n
                ) * tensor

    def averaged_parameters(self) -> dict[str, torch.Tensor]:
        """Return the current averaged parameters."""
        return dict(self.averaged_state)

    def apply_to(self) -> None:
        """Copy the averaged parameters into the live model.

        Call this after training to load the averaged weights into
        ``self.model`` for inference.
        """
        if not self.averaged_state:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.averaged_state:
                    param.data.copy_(self.averaged_state[name])
