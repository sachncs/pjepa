"""Evaluation utilities: metrics, bootstrap CI, statistical tests."""

from __future__ import annotations

from pjepa.eval.metrics import accuracy, mean_per_class_accuracy, forgetting_rate
from pjepa.eval.bootstrap import paired_bootstrap_ci
from pjepa.eval.stats import wilcoxon_signed_rank, bonferroni_correction

__all__ = [
    "accuracy",
    "mean_per_class_accuracy",
    "forgetting_rate",
    "paired_bootstrap_ci",
    "wilcoxon_signed_rank",
    "bonferroni_correction",
]