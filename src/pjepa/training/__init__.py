"""Training infrastructure: pretrain, train, eval, checkpointing, and wrappers.

Modules:
- pretrain.py: JEPA-style pretraining loop.
- train.py: Supervised training loop.
- eval.py: Linear-probe evaluation.
- checkpoint.py: Save/load with sharded persistent-graph support.
- swa.py: Stochastic Weight Averaging wrapper.
- tta.py: Test-Time Augmentation wrapper.
- ensemble.py: k-model ensemble with three aggregation strategies.
- distillation.py: Hinton-style knowledge distillation loss.
"""

from __future__ import annotations

from pjepa.training.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from pjepa.training.distillation import DistillationConfig, DistillationLoss, distill_kl
from pjepa.training.ensemble import Aggregator, Ensemble
from pjepa.training.eval import LinearProbeResult, linear_probe_eval
from pjepa.training.pretrain import PretrainConfig, pretrain_loop
from pjepa.training.swa import SWAConfig, SWAWrapper
from pjepa.training.train import SupervisedConfig, supervised_train_loop
from pjepa.training.tta import TTAConfig, TTAWrapper

__all__ = [
    "Aggregator",
    "Checkpoint",
    "DistillationConfig",
    "DistillationLoss",
    "Ensemble",
    "LinearProbeResult",
    "PretrainConfig",
    "SWAConfig",
    "SWAWrapper",
    "SupervisedConfig",
    "TTAConfig",
    "TTAWrapper",
    "distill_kl",
    "linear_probe_eval",
    "load_checkpoint",
    "pretrain_loop",
    "save_checkpoint",
    "supervised_train_loop",
]
