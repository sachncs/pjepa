"""Training infrastructure: pretrain, train, eval, checkpointing, and wrappers.

## Layout

* :mod:`pjepa.training.pretrain` — JEPA-style pretraining loop with
  optional tensor augmentation, validation callback, and sleep-cadence
  early stop.
* :mod:`pjepa.training.train` — generic supervised training loop
  suitable for downstream classification baselines.
* :mod:`pjepa.training.eval` — linear-probe evaluation for
  self-supervised encoders.
* :mod:`pjepa.training.checkpoint` — save/load with sharded
  persistent-graph support.
* :mod:`pjepa.training.swa` — Stochastic Weight Averaging wrapper.
* :mod:`pjepa.training.tta` — Test-Time Augmentation wrapper.
* :mod:`pjepa.training.ensemble` — k-model ensemble with three
  aggregation strategies.
* :mod:`pjepa.training.distillation` — Hinton-style knowledge
  distillation loss.
* :mod:`pjepa.training.optuna_search` — packaged Optuna integration
  with SQLite/WAL backend.
"""

from __future__ import annotations

from pjepa.training.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from pjepa.training.distillation import DistillationConfig, DistillationLoss, distill_kl
from pjepa.training.ensemble import Aggregator, Ensemble
from pjepa.training.eval import LinearProbeResult, linear_probe_eval
from pjepa.training.optuna_search import (
    OPTUNA_SEARCH_SPACE,
    OptunaSearch,
    OptunaSearchConfig,
    TrialResult,
    build_augmentation_from_name,
    enable_sqlite_wal,
    load_best_config,
    suggest_hyperparameters,
)
from pjepa.training.pretrain import (
    PretrainConfig,
    SleepCadence,
    ValidationCallback,
    augmentation_call,
    build_tensor_augmentation,
    pretrain_loop,
)
from pjepa.training.swa import SWAConfig, SWAWrapper
from pjepa.training.train import SupervisedConfig, supervised_train_loop
from pjepa.training.tta import TTAConfig, TTAWrapper

__all__ = [
    "OPTUNA_SEARCH_SPACE",
    "Aggregator",
    "Checkpoint",
    "DistillationConfig",
    "DistillationLoss",
    "Ensemble",
    "LinearProbeResult",
    "OptunaSearch",
    "OptunaSearchConfig",
    "PretrainConfig",
    "SWAConfig",
    "SWAWrapper",
    "SleepCadence",
    "SupervisedConfig",
    "TTAConfig",
    "TTAWrapper",
    "TrialResult",
    "ValidationCallback",
    "augmentation_call",
    "build_augmentation_from_name",
    "build_tensor_augmentation",
    "distill_kl",
    "enable_sqlite_wal",
    "linear_probe_eval",
    "load_best_config",
    "load_checkpoint",
    "pretrain_loop",
    "save_checkpoint",
    "suggest_hyperparameters",
    "supervised_train_loop",
]
