"""Evaluation utilities: metrics, bootstrap CI, statistical tests."""

from __future__ import annotations

from pjepa.eval.bootstrap import paired_bootstrap_ci
from pjepa.eval.metrics import accuracy, forgetting_rate, mean_per_class_accuracy
from pjepa.eval.stats import bonferroni_correction, wilcoxon_signed_rank

__all__ = [
    "accuracy",
    "bonferroni_correction",
    "forgetting_rate",
    "mean_per_class_accuracy",
    "paired_bootstrap_ci",
    "wilcoxon_signed_rank",
]
