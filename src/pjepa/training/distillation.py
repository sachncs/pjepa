"""Knowledge distillation loss.

A student model is trained to match a teacher model's predictions.
The temperature-scaled KL divergence of Hinton et al. (2015) is used.
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
        alpha: Weight on the distillation loss (vs. the task loss).
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
        The mean KL divergence across the batch.
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
    """Combined task + distillation loss."""

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
