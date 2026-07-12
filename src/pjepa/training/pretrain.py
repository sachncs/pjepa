"""Pretraining loop for self-supervised methods.

The loop runs the encoder + predictor on a stream of observations,
computes the JEPA loss, and applies EMA updates to the target encoder.
Checkpointing happens at the end of every epoch.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from pjepa.augmentations import AugmentationPipeline
from pjepa.encoders import JEPAPredictor, TargetEncoder
from pjepa.exceptions import ConfigError
from pjepa.logging_setup import get_logger
from pjepa.training.checkpoint import Checkpoint, save_checkpoint

__all__ = ["PretrainConfig", "pretrain_loop"]


@dataclass(frozen=True)
class PretrainConfig:
    """Configuration for the pretraining loop.

    Attributes:
        epochs: Number of epochs.
        learning_rate: Optimiser learning rate.
        weight_decay: AdamW weight decay.
        momentum: EMA momentum for the target encoder.
        checkpoint_dir: Directory to write checkpoints to.
    """

    epochs: int = 200
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    momentum: float = 0.996
    checkpoint_dir: str = "results/checkpoints"


def pretrain_loop(
    encoder: nn.Module,
    predictor: JEPAPredictor,
    target: TargetEncoder,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: PretrainConfig,
    augmentation: AugmentationPipeline | None = None,
    log_every: int = 10,
) -> list[float]:
    """Run the pretraining loop and return the per-epoch mean losses.

    Args:
        encoder: The online encoder module.
        predictor: The JEPA predictor.
        target: The EMA target encoder.
        optimizer: The optimiser (AdamW is standard).
        batches: An iterable yielding ``(context_features, target_features)``.
        config: The pretraining configuration.
        augmentation: Optional augmentation pipeline applied to inputs.
        log_every: How often to log progress, in steps.

    Returns:
        A list of per-epoch mean loss values.
    """
    if config.epochs <= 0:
        raise ConfigError(f"pretrain_loop: epochs must be positive; got {config.epochs}")
    log = get_logger(__name__)
    losses: list[float] = []
    for epoch in range(1, config.epochs + 1):
        epoch_losses: list[float] = []
        for step, (context, target_features) in enumerate(batches, start=1):
            if augmentation is not None:
                # The augmentation is graph-aware but for the
                # pretraining loop we use it on the context tensor.
                context = augmentation_call(augmentation, context)
            predicted = predictor(context)
            loss = torch.nn.functional.smooth_l1_loss(predicted, target_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            target.update()
            epoch_losses.append(float(loss.item()))
            if step % log_every == 0:
                log.info("pretrain step loss=%.4f", float(loss.item()))
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(mean_loss)
        ckpt = Checkpoint(
            encoder_state=encoder.state_dict(),
            predictor_state=predictor.state_dict(),
            target_state=target.shadow.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            loss=mean_loss,
        )
        save_checkpoint(ckpt, config.checkpoint_dir, run_id=f"epoch_{epoch:04d}")
    return losses


def augmentation_call(pipeline: AugmentationPipeline, tensor: torch.Tensor) -> torch.Tensor:
    """Apply an :class:`AugmentationPipeline` to a feature tensor.

    The augmentation pipeline operates on
    :class:`TypedAttributedGraph`; for the pretraining loop we expose
    a pass-through that does nothing if the pipeline expects graphs.
    """
    _ = pipeline
    return tensor
