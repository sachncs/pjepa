"""Supervised training loop.

A simple loop that alternates forward passes, loss computation, and
optimizer steps on a stream of ``(input, target)`` pairs. The loop is
deliberately generic; continual-learning extensions wrap it with
strategy-specific update rules (EWC penalty, GEM projection, etc.).

## Architecture

```
  for epoch in 1..config.epochs:
      for x, y in batches:
          logits = model(x)
          loss = loss_fn(logits, y)
          loss.backward()
          optimizer.step()
```

## Complexity

Let ``N`` be the total number of ``(input, target)`` batches and
``E`` the configured epoch count. The loop visits every batch exactly
``E`` times, so the wall-clock cost is ``O(E * N)`` forward / backward
passes; memory is bounded by the per-batch activation cache of the
model.

## Exceptions

A non-positive epoch count is the only configuration-time validation;
runtime errors in the underlying torch operations are propagated
unchanged so callers can decide how to handle them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from pjepa.exceptions import ConfigError
from pjepa.logging_setup import get_logger

__all__ = ["SupervisedConfig", "supervised_train_loop"]


@dataclass(frozen=True)
class SupervisedConfig:
    """Configuration for the supervised loop.

    Attributes:
        epochs: Number of training epochs.
        learning_rate: Optimiser learning rate. The default
          ``5e-4`` is a common AdamW starting point for graph
          classification baselines (GCN, GIN, GraphSAGE).
        weight_decay: AdamW weight decay. The default ``1e-5``
          matches the pretraining-loop default.
        checkpoint_dir: Directory to write per-epoch checkpoints
          into. (The loop itself does not currently consume this
          attribute; it is preserved for experiment-runner
          symmetry.)
    """

    epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    checkpoint_dir: str = "results/checkpoints"


def supervised_train_loop(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: SupervisedConfig,
) -> list[float]:
    """Run the supervised loop and return per-epoch mean loss values.

    Args:
        model: The model to train.
        loss_fn: The loss function.
        optimizer: The optimiser.
        batches: Iterable of ``(input, target)`` batches.
        config: Supervised configuration.

    Returns:
        List of per-epoch mean loss values, one per epoch.

    Raises:
        ConfigError: When ``config.epochs`` is non-positive.
    """
    if config.epochs <= 0:
        raise ConfigError(f"supervised_train_loop: epochs must be positive; got {config.epochs}")
    log = get_logger(__name__)
    losses: list[float] = []
    for epoch in range(1, config.epochs + 1):
        epoch_losses: list[float] = []
        for x, y in batches:
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(mean_loss)
        log.info("supervised epoch=%d mean_loss=%.4f", epoch, mean_loss)
    return losses
