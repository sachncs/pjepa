"""Pretraining loop for self-supervised methods.

This module implements the canonical JEPA-style pretraining loop:

* Run :class:`pjepa.encoders.JEPAPredictor` on a context tensor.
* Compute the Smooth-L1 loss against the target tensor (typically
  produced by a frozen target encoder).
* Optimise with AdamW.
* Update the target encoder as an EMA of the online encoder after
  every optimiser step.

The loop also supports a pluggable tensor-level augmentation, an
optional validation callback fired every ``val_every`` epochs, and a
sleep-cadence object that can stop the run early.

## Architecture

```
                          ┌─────────────┐
   context ──► encoder ──►│ predictor    │──► predicted
                          └─────────────┘
                          ┌─────────────┐
   target  ──► target ──►│ (EMA copy)   │──► reference
                          └─────────────┘
                          loss = SmoothL1(predicted, reference)
```

The target encoder is :class:`pjepa.perf.EMATarget` (or the
config-equivalent wrapper in :class:`pjepa.encoders.TargetEncoder`).

## Complexity

Let ``N`` be the number of batches per epoch and ``D`` be the
feature dimension. Each step is ``O(B²)`` in the predictor (a
small MLP) plus the cost of the online / target encoder forward
passes. End-to-end the loop is ``O(E * N)`` for ``E`` epochs; memory
is bounded by the batch size plus one copy of the four model state
dicts (encoder, predictor, target, optimiser).

## Exceptions

All configuration-time validation failures raise
:class:`pjepa.exceptions.ConfigError`. Runtime errors in the
underlying torch operations are left to propagate unchanged so the
caller can decide how to handle them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from pjepa.augmentations import AugmentationPipeline, TensorDropFeature
from pjepa.encoders import JEPAPredictor, TargetEncoder
from pjepa.exceptions import ConfigError
from pjepa.logging_setup import get_logger
from pjepa.training.checkpoint import Checkpoint, save_checkpoint

__all__ = [
    "PretrainConfig",
    "SleepCadence",
    "ValidationCallback",
    "augmentation_call",
    "build_tensor_augmentation",
    "pretrain_loop",
]


@runtime_checkable
class SleepCadence(Protocol):
    """Protocol for sleep-cadence objects usable by :func:`pretrain_loop`.

    The framework's :class:`pjepa.scheduler.SleepCadence` satisfies
    this protocol out of the box; tests and minimal adapters can
    implement it directly by defining a no-arg ``should_sleep``
    method.

    Implementations are consulted once per epoch: returning ``True``
    triggers an early stop after the current epoch's checkpoint has
    been written.
    """

    def should_sleep(self) -> bool:
        """Return whether a sleep cycle should fire."""
        ...


@runtime_checkable
class ValidationCallback(Protocol):
    """Protocol for the validation callback used by :func:`pretrain_loop`.

    The callback receives the live encoder, predictor, and the
    current epoch number and must return a finite float (typically a
    validation loss or a metric). Returning a non-finite value will
    not crash the loop (it gets stored verbatim in the checkpoint
    metadata) but downstream consumers should treat non-finite
    metrics as missing data.
    """

    def __call__(self, encoder: nn.Module, predictor: nn.Module, epoch: int) -> float:
        """Compute the validation metric for the current epoch."""
        ...


@dataclass
class PretrainConfig:
    """Configuration for the pretraining loop.

    Attributes:
        epochs: Number of training epochs. Must be positive.
        learning_rate: Optimiser learning rate. Standard AdamW
          range: ``[1e-5, 1e-3]``.
        weight_decay: AdamW weight decay. Standard range:
          ``[0, 1e-2]``.
        momentum: EMA momentum for the target encoder. Range: ``[0, 1)``;
          ``0.99`` to ``0.999`` are the values reported in the JEPA
          literature.
        checkpoint_dir: Directory to write per-epoch checkpoints
          into. Each epoch gets its own subdirectory called
          ``epoch_<NNNN>``.
        val_every: How often (in epochs) to invoke the validation
          callback. ``0`` disables validation.
        log_every: How often (in steps) to log the per-step loss.
          ``0`` silences the per-step logger.
        augmentation: Optional name of an augmentation to use. One
          of ``"none"``, ``"dropfeat"``, or ``"composite"``.
          ``"none"`` disables augmentation; ``"dropfeat"`` applies a
          tensor-level feature-drop; ``"composite"`` selects a
          random tensor-level drop. This mirrors the categorical
          search dimension in ``configs/tu.yaml``.
        augmentation_strength: Strength of the tensor augmentation
          (fraction of features to drop). Range: ``[0, 1]``.
        seed: Optional seed applied to the augmentation generator so
          trials are reproducible.
        cadence: Optional :class:`SleepCadence` object consulted
          once per epoch; when it returns ``True`` the loop stops
          early.
        extras: Additional state to persist in every checkpoint's
          ``metadata.json``. Useful for downstream filtering by
          experiment phase.
    """

    epochs: int = 200
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    momentum: float = 0.996
    checkpoint_dir: str = "results/checkpoints"
    val_every: int = 0
    log_every: int = 10
    augmentation: str = "none"
    augmentation_strength: float = 0.2
    seed: int | None = None
    cadence: SleepCadence | None = None
    extras: dict[str, object] = field(default_factory=dict)


def augmentation_call(
    augmentation: TensorDropFeature | AugmentationPipeline | Callable[[torch.Tensor], torch.Tensor],
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply a tensor-compatible augmentation to ``tensor``.

    The pretraining loop operates on 2-D feature tensors, not on
    :class:`pjepa.graphs.TypedAttributedGraph`, so the augmentation
    factories that produce :class:`AugmentationPipeline` instances are
    rejected with a helpful error message. :class:`TensorDropFeature`
    and arbitrary ``Tensor -> Tensor`` callables work as expected.

    Args:
        augmentation: A callable that accepts a 2-D tensor and
          returns a 2-D tensor.
        tensor: The input feature tensor.

    Returns:
        The augmented tensor.

    Raises:
        ConfigError: If ``augmentation`` is an :class:`AugmentationPipeline`
          because the pretraining loop is tensor-level.
        ConfigError: If the augmentation does not return a
          :class:`torch.Tensor`.
        ConfigError: If the augmented tensor has a different shape
          than the input.
    """
    if isinstance(augmentation, AugmentationPipeline):
        raise ConfigError(
            "augmentation_call: AugmentationPipeline expects TypedAttributedGraph inputs; "
            "pretrain_loop operates on tensors, use TensorDropFeature instead"
        )
    result = augmentation(tensor)
    if not isinstance(result, torch.Tensor):
        raise ConfigError(
            f"augmentation_call: augmentation must return a Tensor; got {type(result).__name__}"
        )
    if result.shape != tensor.shape:
        raise ConfigError(
            f"augmentation_call: augmentation must preserve shape; "
            f"got {tuple(result.shape)} vs {tuple(tensor.shape)}"
        )
    return result


def build_tensor_augmentation(
    name: str,
    strength: float,
    seed: int | None,
) -> TensorDropFeature | None:
    """Construct a tensor augmentation from a config-style name.

    The accepted names are ``"none"``, ``"dropfeat"``, and
    ``"composite"`` (mirrors the categorical dimension of the
    Optuna search space in ``configs/tu.yaml``). When ``name`` is
    ``None`` or empty the function returns ``None`` so callers can
    disable augmentation explicitly. The ``strength`` value is the
    fraction of feature columns to drop; ``seed`` is forwarded to a
    fresh :class:`torch.Generator` for reproducibility.

    Args:
        name: ``"none"`` | ``"dropfeat"`` | ``"composite"``.
        strength: Drop strength in ``[0, 1]``.
        seed: Optional seed for the augmentation's RNG.

    Returns:
        A :class:`TensorDropFeature` instance, or ``None`` for
        ``"none"``.

    Raises:
        ConfigError: If ``name`` does not match any known
          augmentation.
    """
    if name in (None, "", "none"):
        return None
    if name not in ("dropfeat", "composite"):
        raise ConfigError(
            f"build_tensor_augmentation: unknown augmentation name {name!r}; "
            "expected one of 'none', 'dropfeat', 'composite'"
        )
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    return TensorDropFeature(strength=float(strength), generator=generator)


def pretrain_loop(
    encoder: nn.Module,
    predictor: JEPAPredictor,
    target: TargetEncoder,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: PretrainConfig | None = None,
    augmentation: TensorDropFeature
    | AugmentationPipeline
    | Callable[[torch.Tensor], torch.Tensor]
    | None = None,
    val_fn: ValidationCallback | None = None,
    log_every: int | None = None,
) -> list[float]:
    """Run the pretraining loop and return the per-epoch mean losses.

    The loop runs ``config.epochs`` passes over the ``batches``
    iterable. At every step we

    1. optionally apply the augmentation to the context tensor via
       :func:`augmentation_call`,
    2. compute the predictor output ``predicted``,
    3. compute the Smooth-L1 loss ``SmoothL1(predicted, target)``,
    4. backpropagate and update the encoder / predictor parameters,
    5. update the target encoder via EMA
       (:meth:`TargetEncoder.update`).

    A checkpoint is written at the end of every epoch via
    :func:`pjepa.training.checkpoint.save_checkpoint`; the directory
    layout is one subdirectory per epoch (``epoch_<NNNN>``) so the
    checkpoints can be played back sequentially or selectively.

    Args:
        encoder: The online encoder module.
        predictor: The JEPA predictor.
        target: The EMA target encoder wrapper.
        optimizer: The optimiser (AdamW is standard).
        batches: An iterable yielding ``(context_features, target_features)``.
          Both must be ``[B, D]`` tensors.
        config: The pretraining configuration. When ``None``, a default
          :class:`PretrainConfig` is used.
        augmentation: Optional augmentation applied to the context
          tensor before the predictor. When provided, it overrides
          ``config.augmentation``.
        val_fn: Optional validation callback. When supplied and
          ``config.val_every > 0``, the callback is invoked every
          ``config.val_every`` epochs. The callback receives the
          live ``encoder``, ``predictor``, and the current ``epoch``;
          its return value is stored as ``val_metric`` in the
          checkpoint metadata and in :attr:`PretrainConfig.extras`.
        log_every: Override for ``config.log_every``.

    Returns:
        A list of per-epoch mean loss values (one float per epoch).
        If ``SleepCadence.should_sleep()`` returns ``True`` mid-loop
        the returned list is truncated accordingly.

    Raises:
        ConfigError: If ``config.epochs <= 0``,
          ``config.val_every < 0``, or ``config.augmentation_strength``
          is outside ``[0, 1]``.
    """
    cfg = config or PretrainConfig()
    if cfg.epochs <= 0:
        raise ConfigError(f"pretrain_loop: epochs must be positive; got {cfg.epochs}")
    if cfg.val_every < 0:
        raise ConfigError(f"pretrain_loop: val_every must be non-negative; got {cfg.val_every}")
    if not (0.0 <= cfg.augmentation_strength <= 1.0):
        raise ConfigError(
            "pretrain_loop: augmentation_strength must be in [0, 1]; "
            f"got {cfg.augmentation_strength}"
        )
    if augmentation is None:
        augmentation = build_tensor_augmentation(
            cfg.augmentation, cfg.augmentation_strength, cfg.seed
        )
    step_log_every = int(log_every) if log_every is not None else int(cfg.log_every)
    log = get_logger(__name__)
    losses: list[float] = []
    val_metrics: list[float] = []
    best_val: float = float("inf")
    best_epoch: int = 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_losses: list[float] = []
        for step, (context, target_features) in enumerate(batches, start=1):
            if augmentation is not None:
                context = augmentation_call(augmentation, context)
            predicted = predictor(context)
            loss = torch.nn.functional.smooth_l1_loss(predicted, target_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            target.update()
            epoch_losses.append(float(loss.item()))
            if step_log_every > 0 and step % step_log_every == 0:
                log.info("pretrain step loss=%.4f", float(loss.item()))
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(mean_loss)
        val_metric: float | None = None
        if val_fn is not None and cfg.val_every > 0 and epoch % cfg.val_every == 0:
            val_metric = float(val_fn(encoder, predictor, epoch))
            val_metrics.append(val_metric)
            if val_metric < best_val:
                best_val = val_metric
                best_epoch = epoch
            log.info(
                "pretrain val epoch=%d metric=%.4f",
                epoch,
                val_metric,
            )
        extras = dict(cfg.extras)
        extras["epoch_loss"] = mean_loss
        if val_metric is not None:
            extras["val_metric"] = val_metric
        ckpt = Checkpoint(
            encoder_state=encoder.state_dict(),
            predictor_state=predictor.state_dict(),
            target_state=target.shadow.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            loss=mean_loss,
            extras=extras,
        )
        save_checkpoint(ckpt, cfg.checkpoint_dir, run_id=f"epoch_{epoch:04d}")
        if cfg.cadence is not None and cfg.cadence.should_sleep():
            log.info(
                "pretrain cadence-triggered early stop",
                extra={"event": "pretrain.cadence_stop", "epoch": epoch},
            )
            break
    if val_fn is not None and cfg.val_every > 0:
        log.info(
            "pretrain best val",
            extra={"event": "pretrain.best_val", "best_val": best_val, "best_epoch": best_epoch},
        )
    return losses
