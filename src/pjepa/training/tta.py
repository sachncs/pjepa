"""Test-Time Augmentation wrapper.

At inference time the input is passed through ``n_aug`` augmentations;
the predictions are averaged. Improves accuracy at 2-3x the
inference cost.
"""

from __future__ import annotations

import torch

from pjepa.augmentations import Augmentation
from pjepa.exceptions import ConfigError

__all__ = ["TTAConfig", "TTAWrapper"]


class TTAConfig:
    """Configuration for TTA.

    Attributes:
        n_aug: Number of augmented passes per inference call.
        include_original: Whether the original (unaugmented) input is
          included in the average.
    """

    def __init__(self, n_aug: int = 5, include_original: bool = True) -> None:
        if n_aug < 1:
            raise ConfigError(f"TTAConfig.n_aug must be >= 1; got {n_aug}")
        self.n_aug = n_aug
        self.include_original = include_original


class TTAWrapper(torch.nn.Module):
    """Wrap a model with test-time augmentation.

    Attributes:
        model: The model to wrap.
        augmentation: The augmentation to apply ``n_aug`` times.
        config: The TTA configuration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        augmentation: Augmentation,
        config: TTAConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.augmentation = augmentation
        self.config = config or TTAConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run TTA inference.

        Args:
            x: An input tensor. The augmentation operates on the
              tensor directly (e.g., a feature tensor); for graph
              inputs, pass the underlying feature matrix.

        Returns:
            The mean of the predictions across augmented passes.
        """
        outputs = []
        if self.config.include_original:
            outputs.append(self.model(x))
        for _ in range(self.config.n_aug):
            x_aug = self.augmentation(x)
            outputs.append(self.model(x_aug))
        return torch.stack(outputs, dim=0).mean(dim=0)
