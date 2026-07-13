r"""Knowledge distillation loss.

A student model is trained to match a teacher model's predictions.
The temperature-scaled KL divergence of Hinton et al. (2015) is used.

## Math

For each logit vector ``z`` we compute the teacher and student
softmax-distributions at temperature ``T``:

.. math::

    p_T(z) = \mathrm{softmax}(z / T)

and the KL divergence is

.. math::

    \mathrm{KL}(p_T^T \| p_T^S) = T^2 \cdot
        \mathrm{KL}(\mathrm{softmax}(z^T / T)
                   \| \mathrm{softmax}(z^S / T))

The ``T^2`` rescaling (Hinton's trick) compensates for the
``1/T^2`` shrinkage of the gradient magnitudes under the softened
softmax.

## Complexity

Each forward call performs one ``softmax``, one ``log_softmax``, and
one ``F.kl_div`` per element; that is ``O(B * C)`` for ``B`` batch
size and ``C`` class count.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pjepa.exceptions import ConfigError

__all__ = ["DistillationConfig", "DistillationLoss", "distill_kl"]


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for distillation.

    Attributes:
        temperature: Softmax temperature for the KL divergence.
          Higher values produce softer distributions. The Hinton et
          al. recipe uses ``T = 4``.
        alpha: Weight on the distillation loss (``alpha * distill +
          (1 - alpha) * task``). ``0.0`` reduces to pure task loss;
          ``1.0`` reduces to pure distillation.
    """

    temperature: float = 4.0
    alpha: float = 0.5


def distill_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Compute the temperature-scaled KL divergence between student and teacher.

    Args:
        student_logits: ``[B, C]`` student logits.
        teacher_logits: ``[B, C]`` teacher logits (detached).
        temperature: Softmax temperature.

    Returns:
        The mean KL divergence across the batch, rescaled by
        ``temperature**2``.

    Raises:
        ConfigError: When ``temperature`` is non-positive or the
          two tensors have different shapes.
    """
    if temperature <= 0:
        raise ConfigError(f"distill_kl: temperature must be positive; got {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ConfigError(
            f"distill_kl: student {tuple(student_logits.shape)} and teacher "
            f"{tuple(teacher_logits.shape)} must share shape"
        )
    p_teacher = torch.softmax(teacher_logits / temperature, dim=-1)
    log_p_student = torch.log_softmax(student_logits / temperature, dim=-1)
    kl = torch.nn.functional.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return kl * (temperature**2)


class DistillationLoss(torch.nn.Module):
    """Combined task + distillation loss.

    The module exposes a single :meth:`forward` returning
    ``alpha * distill + (1 - alpha) * task``, where ``task`` is
    computed by the configured task-loss module
    (default :class:`torch.nn.CrossEntropyLoss`).
    """

    def __init__(self, config: DistillationConfig | None = None) -> None:
        super().__init__()
        self.config = config or DistillationConfig()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        task_loss_fn: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            student_logits: ``[B, C]`` student logits.
            teacher_logits: ``[B, C]`` teacher logits (detached).
            targets: ``[B]`` ground-truth labels.
            task_loss_fn: Optional task loss module. Defaults to
              :class:`torch.nn.CrossEntropyLoss`.

        Returns:
            The combined loss ``alpha * distill + (1 - alpha) * task``.
        """
        loss_fn = task_loss_fn if task_loss_fn is not None else torch.nn.CrossEntropyLoss()
        task_loss = loss_fn(student_logits, targets)
        distill = distill_kl(
            student_logits,
            teacher_logits.detach(),
            temperature=self.config.temperature,
        )
        return self.config.alpha * distill + (1.0 - self.config.alpha) * task_loss
