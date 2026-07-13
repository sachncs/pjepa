"""Direct Preference Optimisation (DPO) loss for the rewriting engine.

The rewriting engine uses DPO to train the candidate generator on
preference data: pairs of (winner, loser) candidate rewrites, where
the winner scored higher under the unified objective 𝒥. The loss is
the standard log-sigmoid objective from Rafailov et al. (NeurIPS 2023)
with optional symmetric label smoothing.

The implementation is a literal translation of the reference equation::

    L_DPO = -E[ log σ( beta * (Δθ_w - Δθ_l)) ]

where ``Δθ_w = log π_w - log π_w^ref`` and similarly for the loser.
Label smoothing replaces the inner ``log σ`` with a convex combination
of ``log σ(margin)`` and ``log σ(-margin)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["DPOConfig", "dpo_loss"]


@dataclass(frozen=True)
class DPOConfig:
    """Configuration for the DPO objective.

    Attributes:
        beta: Temperature parameter; larger values push the policy
            further from the reference.
        label_smoothing: Optional label smoothing coefficient in
            ``[0, 0.5)``. ``0.0`` recovers the canonical DPO loss.
    """

    beta: float = 0.1
    label_smoothing: float = 0.0


def dpo_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    reference_chosen_logprob: torch.Tensor,
    reference_rejected_logprob: torch.Tensor,
    config: DPOConfig | None = None,
) -> torch.Tensor:
    """Compute the DPO loss for a batch of preference pairs.

    Args:
        chosen_logprob: ``[B]`` log-probabilities of the chosen
            action under the policy being trained.
        rejected_logprob: ``[B]`` log-probabilities of the rejected
            action under the policy being trained.
        reference_chosen_logprob: ``[B]`` log-probabilities of the
            chosen action under the reference policy.
        reference_rejected_logprob: ``[B]`` log-probabilities of the
            rejected action under the reference policy.
        config: Optional DPO configuration; defaults to
            :class:`DPOConfig`.

    Returns:
        A scalar tensor holding the mean DPO loss for the batch.

    Raises:
        ValueError: If the input tensors do not share the same shape,
            or if ``label_smoothing`` is outside ``[0, 0.5)``.

    Example:
        >>> loss = dpo_loss(c_lp, r_lp, c_ref, r_ref)
        >>> loss.requires_grad
        True
    """
    cfg = config or DPOConfig()
    if not (
        chosen_logprob.shape
        == rejected_logprob.shape
        == reference_chosen_logprob.shape
        == reference_rejected_logprob.shape
    ):
        raise ValueError("dpo_loss: all four log-probability tensors must share the same shape")
    if not 0.0 <= cfg.label_smoothing < 0.5:
        raise ValueError(
            f"dpo_loss: label_smoothing must be in [0, 0.5); got {cfg.label_smoothing}"
        )
    chosen_logratio = chosen_logprob - reference_chosen_logprob
    rejected_logratio = rejected_logprob - reference_rejected_logprob
    margin = cfg.beta * (chosen_logratio - rejected_logratio)
    if cfg.label_smoothing > 0.0:
        loss = -cfg.label_smoothing * torch.nn.functional.logsigmoid(-margin) - (
            1.0 - cfg.label_smoothing
        ) * torch.nn.functional.logsigmoid(margin)
    else:
        loss = -torch.nn.functional.logsigmoid(margin)
    return loss.mean()
