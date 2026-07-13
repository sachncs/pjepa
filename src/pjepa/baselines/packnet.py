"""PackNet-style continual-learning baseline (Mallya & Lazebnik, 2018).

Allocates a disjoint slice of each parameter tensor to every task,
trains only the slice owned by the current task, and freezes the
slice once the task is done. The mask is binary (``0``/``1``),
kept per parameter-name, and applied both to gradients (so frozen
weights do not receive updates) and to the parameter values
themselves (so the classifier cannot accidentally route through
frozen slots via optimiser momentum).

The implementation deliberately stays small and dependency-light:
it relies only on :mod:`torch` and :class:`dict` so it can be
reused by the experiment runner without dragging in any
encoder-specific code.

## Algorithm

```
   task t_slice = base + 0..slice_size, modulo n
   task_mask[:, t_slice] = 1

   before step():
       grad *= task_mask
       grad *= 1 - frozen_mask

   begin_task(t):  allocate t's slice, refresh frozen_mask
   freeze_current_task():  fold current task's slice into frozen_mask
```

## Complexity

* :meth:`begin_task` — ``O(P * slice_size)`` per task to write the
  binary masks.
* :meth:`apply_grad_mask` — ``O(P)`` per step (a single multiplication
  per parameter).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from pjepa.exceptions import ConfigError

__all__ = ["PackNet"]


class PackNet:
    """Per-task parameter-mask continual-learning baseline.

    Attributes:
        num_tasks: The total number of CL tasks in the run.
        slice_fraction: Fraction of each parameter's entries
          allocated to every task. ``1 / num_tasks`` is the canonical
          default.
        seed: Seed for the deterministic mask initialisation.
        task_masks: Per-task name→mask map. Exposed for inspection
          and test introspection; treated as read-only outside of
          :meth:`begin_task`.
        frozen_mask: Combined frozen mask (every prior task's mask
          clamped to ``[0, 1]``). Exposed for inspection; updated
          only via :meth:`begin_task` and
          :meth:`freeze_current_task`.
    """

    def __init__(self, num_tasks: int, slice_fraction: float | None = None, seed: int = 0) -> None:
        if num_tasks <= 0:
            raise ConfigError(f"PackNet: num_tasks must be positive; got {num_tasks}")
        if slice_fraction is not None and not (0.0 < slice_fraction <= 1.0):
            raise ConfigError(f"PackNet: slice_fraction must be in (0, 1]; got {slice_fraction}")
        self.num_tasks = int(num_tasks)
        self.slice_fraction = (
            float(slice_fraction) if slice_fraction is not None else 1.0 / self.num_tasks
        )
        self.seed = int(seed)
        self.task_masks: dict[int, dict[str, torch.Tensor]] = {}
        self.frozen_mask: dict[str, torch.Tensor] = {}

    @property
    def current_task_mask(self) -> dict[str, torch.Tensor]:
        """Return the mask of the most recently begun task (or empty)."""
        if not self.task_masks:
            return {}
        last_task = max(self.task_masks)
        return self.task_masks[last_task]

    def begin_task(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        task_idx: int,
    ) -> None:
        """Allocate the task's parameter slice and freeze every prior slice.

        Args:
            named_parameters: Iterable of ``(name, parameter)``
              pairs from the model to be trained.
            task_idx: Zero-based index of the task being begun.
              Must be unique and increasing across calls and satisfy
              ``0 <= task_idx < num_tasks``.

        Raises:
            ConfigError: When ``task_idx`` is out of range or has
              already been begun.
        """
        if task_idx in self.task_masks:
            raise ConfigError(f"PackNet.begin_task: task {task_idx} already begun")
        if task_idx < 0 or task_idx >= self.num_tasks:
            raise ConfigError(
                f"PackNet.begin_task: task_idx {task_idx} out of range [0, {self.num_tasks})"
            )
        task_mask: dict[str, torch.Tensor] = {}
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            n = param.numel()
            slice_size = max(1, round(self.slice_fraction * n))
            base = (task_idx * slice_size) % n
            indices = torch.tensor([(base + i) % n for i in range(slice_size)], dtype=torch.long)
            mask = torch.zeros(n, dtype=torch.float32)
            mask[indices] = 1.0
            mask = mask.view_as(param.data).to(param.data.device)
            task_mask[name] = mask
        self.task_masks[task_idx] = task_mask
        self.refresh_frozen_mask()

    def refresh_frozen_mask(self) -> None:
        """Combine all but the current task's mask into the frozen mask."""
        if not self.task_masks:
            self.frozen_mask = {}
            return
        current = max(self.task_masks)
        combined: dict[str, torch.Tensor] = {}
        for task_idx, task_mask in self.task_masks.items():
            if task_idx == current:
                continue
            for name, mask in task_mask.items():
                if name not in combined:
                    combined[name] = mask.clone()
                else:
                    combined[name] = (combined[name] + mask).clamp(max=1.0)
        self.frozen_mask = combined

    def apply_grad_mask(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        """Zero out gradients for parameters outside the current task's slice.

        Should be called *after* ``loss.backward()`` and *before*
        ``optimizer.step()``. Frozen-task weights retain their
        previous values because their gradient is zeroed.
        """
        if not self.task_masks:
            return
        current = max(self.task_masks)
        task_mask = self.task_masks[current]
        frozen_mask = self.frozen_mask
        for name, param in named_parameters:
            if param.grad is None:
                continue
            grad_mask = task_mask.get(name)
            frozen = frozen_mask.get(name)
            if grad_mask is None and frozen is None:
                continue
            mask = torch.zeros_like(param.grad)
            if grad_mask is not None:
                mask = mask + grad_mask
            if frozen is not None:
                mask = mask * (1.0 - frozen)
            param.grad.mul_(mask)

    def freeze_current_task(self) -> None:
        """Fold the current task's slice into the frozen mask.

        Raises:
            ConfigError: When no task has been begun yet.
        """
        if not self.task_masks:
            raise ConfigError("PackNet.freeze_current_task: no task has been begun")
        self.refresh_frozen_mask()

    def active_parameter_count(self) -> int:
        """Return the number of currently trainable parameter entries."""
        if not self.task_masks:
            return 0
        current = max(self.task_masks)
        return int(sum(int(m.sum().item()) for m in self.task_masks[current].values()))

    def frozen_parameter_count(self) -> int:
        """Return the number of frozen parameter entries across all prior tasks."""
        return int(sum(int(m.sum().item()) for m in self.frozen_mask.values()))
