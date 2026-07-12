"""Continual-learning data splits.

Constructs class-incremental splits where each task contains a
disjoint subset of classes. The split is deterministic given a seed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch

from pjepa.exceptions import DataError

__all__ = ["ClassIncrementalSplit", "make_class_incremental_split"]


class ClassIncrementalSplit:
    """A class-incremental split of a labelled dataset.

    Attributes:
        tasks: A list of lists of integer indices; ``tasks[t]`` holds
          the indices that belong to task ``t``.
        num_classes: The total number of distinct classes in the split.
        task_classes: A list of sets; ``task_classes[t]`` is the set
          of class labels assigned to task ``t``.
    """

    def __init__(
        self,
        tasks: Sequence[Sequence[int]],
        task_classes: Sequence[Sequence[int]],
    ) -> None:
        if len(tasks) != len(task_classes):
            raise DataError(
                "ClassIncrementalSplit: tasks and task_classes must have equal length"
            )
        if not tasks:
            raise DataError("ClassIncrementalSplit: at least one task is required")
        seen: set[int] = set()
        for t_idx, (task, classes) in enumerate(zip(tasks, task_classes)):
            if not task:
                raise DataError(f"ClassIncrementalSplit: task {t_idx} is empty")
            if set(classes) & seen:
                raise DataError(
                    f"ClassIncrementalSplit: task {t_idx} classes overlap a previous task"
                )
            seen.update(classes)
        self.tasks = [list(t) for t in tasks]
        self.task_classes = [set(c) for c in task_classes]
        self.num_classes = len(seen)

    def num_tasks(self) -> int:
        """Return the number of tasks in the split."""
        return len(self.tasks)

    def task_size(self, task_index: int) -> int:
        """Return the number of samples in ``tasks[task_index]``."""
        return len(self.tasks[task_index])


def make_class_incremental_split(
    labels: Sequence[int],
    num_tasks: int,
    seed_split: int,
) -> ClassIncrementalSplit:
    """Construct a class-incremental split.

    Each class is assigned to exactly one task; classes are distributed
    roughly evenly across the tasks. Within a task, samples are
    shuffled deterministically using ``seed_split``.

    Args:
        labels: The label for every sample in the dataset.
        num_tasks: The number of tasks to produce.
        seed_split: The split seed; same seed yields same split.

    Returns:
        A :class:`ClassIncrementalSplit`.

    Raises:
        DataError: If ``num_tasks`` exceeds the number of distinct
          classes or ``labels`` is empty.
    """
    if not labels:
        raise DataError("make_class_incremental_split: labels is empty")
    if num_tasks <= 0:
        raise DataError(
            f"make_class_incremental_split: num_tasks must be positive; got {num_tasks}"
        )
    classes = sorted(set(int(l) for l in labels))
    if num_tasks > len(classes):
        raise DataError(
            f"make_class_incremental_split: num_tasks {num_tasks} exceeds "
            f"number of distinct classes {len(classes)}"
        )

    rng = torch.Generator().manual_seed(int(seed_split))
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        indices_by_class[int(lbl)].append(i)

    tasks_indices: list[list[int]] = [[] for _ in range(num_tasks)]
    tasks_classes: list[set[int]] = [set() for _ in range(num_tasks)]
    for class_idx, cls in enumerate(classes):
        target_task = class_idx % num_tasks
        order = indices_by_class[cls][:]
        perm = torch.randperm(len(order), generator=rng).tolist()
        for p in perm:
            tasks_indices[target_task].append(order[p])
        tasks_classes[target_task].add(cls)

    return ClassIncrementalSplit(tasks=tasks_indices, task_classes=tasks_classes)