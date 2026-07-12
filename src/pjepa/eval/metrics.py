"""Classification metrics for continual and standard evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

__all__ = ["accuracy", "mean_per_class_accuracy", "forgetting_rate"]


def accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Return the fraction of correct predictions.

    Args:
        predictions: Predicted labels.
        targets: Ground-truth labels.

    Returns:
        The accuracy in [0, 1].
    """
    if not predictions or not targets or len(predictions) != len(targets):
        raise ValueError("accuracy: predictions and targets must be non-empty and equal length")
    correct = sum(int(p == t) for p, t in zip(predictions, targets))
    return correct / len(targets)


def mean_per_class_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Return the mean per-class accuracy.

    Args:
        predictions: Predicted labels.
        targets: Ground-truth labels.

    Returns:
        The mean accuracy across classes, in [0, 1].
    """
    correct: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)
    for p, t in zip(predictions, targets):
        total[int(t)] += 1
        if int(p) == int(t):
            correct[int(t)] += 1
    if not total:
        raise ValueError("mean_per_class_accuracy: empty input")
    return float(np.mean([correct[c] / total[c] for c in total]))


def forgetting_rate(per_task_accuracies: Sequence[Sequence[float]]) -> float:
    """Compute the average forgetting rate across continual-learning tasks.

    Forgetting for task ``i`` is the difference between the maximum
    accuracy achieved on ``i`` after training on ``i`` and the final
    accuracy on ``i`` after all tasks are seen.

    Args:
        per_task_accuracies: ``[num_tasks][num_tasks]`` matrix where
          entry ``(i, j)`` is the accuracy on task ``i`` after training
          on task ``j``.

    Returns:
        The average forgetting rate in [-1, 1]; negative values
          indicate positive transfer (the model improved on task
          ``i`` after training on later tasks).
    """
    if not per_task_accuracies:
        raise ValueError("forgetting_rate: empty matrix")
    num_tasks = len(per_task_accuracies)
    forgettings: list[float] = []
    for i in range(num_tasks):
        max_acc = max(per_task_accuracies[i][: i + 1])
        final_acc = per_task_accuracies[i][-1]
        forgettings.append(max_acc - final_acc)
    return float(np.mean(forgettings))