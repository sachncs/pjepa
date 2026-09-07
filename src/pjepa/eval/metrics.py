"""Classification metrics for continual and standard evaluation.

This module exposes the five metrics used by the experiment runners
and the aggregator:

* :func:`accuracy` — micro-averaged accuracy.
* :func:`mean_per_class_accuracy` — macro-averaged accuracy.
* :func:`forgetting_rate` — average forgetting per task (CL setting).
* :func:`backward_transfer` — alias of ``-forgetting_rate`` with the
  standard CL sign convention.
* :func:`forward_transfer` — average accuracy on task ``j``
  immediately after training on task ``j-1``, minus a per-task baseline.

## Definitions

The continual-learning metrics follow the notation of Lopez-Paz &
Ranzato (2017). Given ``T`` tasks and the evaluation matrix
``R ∈ [0, 1]^{T × T}`` where ``R[j, i]`` is the accuracy on task
``i`` after training on task ``j``:

* ``forgetting = (1/T) Σ_i R[i, i] - R[T, i]`` — *signed*
  ``R[i, i] - R[T, i]``. Positive values indicate forgetting; negative
  values indicate positive transfer.
* ``backward_transfer = -forgetting`` (canonical CL sign).
* ``forward_transfer = (1/(T-1)) Σ_{j=2..T} R[j-1, j] - b_j``.

Where ``b_j`` is the per-task baseline accuracy (random-init
accuracy on task ``j``). ``None`` defaults the baseline to zero,
which makes the metric equal the raw first-encounter accuracy.

## Complexity

Every metric is ``O(T²)`` over a ``T``-task matrix, ``O(N)`` over
``N`` predictions, or ``O(T)`` over the per-task accuracy list.
Memory is ``O(1)``; there is no allocation larger than a small
dictionary keyed by class id.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

__all__ = [
    "accuracy",
    "backward_transfer",
    "forgetting_rate",
    "forward_transfer",
    "mean_per_class_accuracy",
]


def accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Return the fraction of correct predictions (micro-averaged).

    Args:
        predictions: Predicted labels in ``[0, n_classes)``.
        targets: Ground-truth labels with the same length as
          ``predictions``.

    Returns:
        The accuracy in ``[0, 1]`` (1.0 = perfect).

    Raises:
        ValueError: When the inputs are empty, or the two
          sequences have different lengths.
    """
    if not predictions or not targets or len(predictions) != len(targets):
        raise ValueError("accuracy: predictions and targets must be non-empty and equal length")
    correct = sum(int(p == t) for p, t in zip(predictions, targets))
    return correct / len(targets)


def mean_per_class_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Return the mean per-class accuracy (macro-averaged).

    The metric computes the per-class correct / total ratio and
    averages across classes. It is the line-probe / SOTA comparison
    protocol used by every TU experiment in this codebase; we
    prefer it over micro-averaged :func:`accuracy` because
    class-imbalanced datasets (e.g. ``PROTEINS``) can otherwise
    hide silent regressions.

    Args:
        predictions: Predicted labels in ``[0, n_classes)``.
        targets: Ground-truth labels with the same length as
          ``predictions``.

    Returns:
        The macro-averaged accuracy in ``[0, 1]``.

    Raises:
        ValueError: When the inputs are empty.
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

    For task ``i`` we define the forgetting as ``max R[i, j]`` for
    ``j <= i`` (the best accuracy on ``i`` while still training on a
    task that contains ``i``) minus the final accuracy on ``i``,
    ``R[T, i]``. The metric averages this quantity across tasks.
    Following Lopez-Paz & Ranzato (2017) we report the **signed**
    difference, so forgetting is positive when the accuracy drops and
    negative when the model somehow *improved* on ``i`` after later
    tasks (positive transfer).

    Args:
        per_task_accuracies: ``[num_tasks][num_tasks]`` matrix where
          entry ``(i, j)`` is the accuracy on task ``i`` after training
          on task ``j``. The matrix must be square.

    Returns:
        The average forgetting rate in ``[-1, 1]``. Positive values
        indicate forgetting; negative values indicate positive
        backward transfer.

    Raises:
        ValueError: When the input matrix is empty.
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


def backward_transfer(per_task_accuracies: Sequence[Sequence[float]]) -> float:
    r"""Compute the backward transfer (``BWT``) across continual-learning tasks.

    ``BWT`` is the standard CL sign convention: a positive value
    indicates positive backward transfer (the model improves on task
    ``i`` after training on later tasks), and a negative value
    indicates catastrophic forgetting.

    .. math::

        \text{BWT} = \frac{1}{T} \sum_{i=0}^{T-1} R_{T, i} - R_{i, i}

    where ``R_{j, i}`` is the accuracy on task ``i`` after training
    on task ``j``.

    Args:
        per_task_accuracies: ``[num_tasks][num_tasks]`` matrix where
          entry ``(i, j)`` is the accuracy on task ``i`` after training
          on task ``j``.

    Returns:
        The average backward transfer in ``[-1, 1]``.
    """
    return -forgetting_rate(per_task_accuracies)


def forward_transfer(
    per_task_accuracies: Sequence[Sequence[float]],
    baseline_per_task: Sequence[float] | None = None,
) -> float:
    r"""Compute the forward transfer (``FWT``) across continual-learning tasks.

    ``FWT`` measures whether earlier tasks prime the model for later
    tasks. The standard CL definition is:

    .. math::

        \text{FWT} = \frac{1}{T-1} \sum_{j=1}^{T-1}
            (R_{j-1, j} - b_j)

    where ``R_{j-1, j}`` is the accuracy on task ``j`` *before*
    training on ``j`` (i.e. right after training on ``j-1``) and
    ``b_j`` is the per-task baseline (the random-init accuracy on
    task ``j``).

    Args:
        per_task_accuracies: ``[num_tasks][num_tasks]`` matrix where
          entry ``(i, j)`` is the accuracy on task ``i`` after training
          on task ``j``.
        baseline_per_task: Optional ``[num_tasks]`` sequence of
          per-task baseline accuracies. When omitted, ``b_j`` is taken
          to be ``0.0``, in which case the function returns the
          average raw "first encounter" accuracy on each task.

    Returns:
        The average forward transfer in ``[-1, 1]``. Higher values
        (less negative) indicate stronger forward transfer.

    Raises:
        ValueError: When the input matrix is empty.
    """
    if not per_task_accuracies:
        raise ValueError("forward_transfer: empty matrix")
    num_tasks = len(per_task_accuracies)
    if num_tasks < 2:
        return 0.0
    deltas: list[float] = []
    for j in range(1, num_tasks):
        acc_on_j_before_training = float(per_task_accuracies[j][j - 1])
        baseline = float(baseline_per_task[j]) if baseline_per_task is not None else 0.0
        deltas.append(acc_on_j_before_training - baseline)
    return float(np.mean(deltas))
