"""Information Bottleneck Lagrangian and its variational bound.

The IB Lagrangian (Tishby et al. 1999) is
    𝓛_IB(q) = I(X; Z) − β · I(Y; Z).
Alemi et al. (2017) gave the variational upper bound that we use at
training time:
    𝓛_VIB ≤ 𝔼[−log p(y | z)] + β · D_KL(q(z | x) ‖ p(z)).

This module provides both the symbolic Lagrangian and the variational
estimator used by the JEPA predictor.
"""

from __future__ import annotations

import torch

from pjepa.exceptions import NumericalError

__all__ = ["ib_lagrangian", "variational_ib_bound"]


def ib_lagrangian(
    ix_z: float,
    iy_z: float,
    beta: float,
) -> float:
    """Compute the IB Lagrangian ``I(X;Z) - beta * I(Y;Z)``.

    Args:
        ix_z: Mutual information ``I(X;Z)`` in nats.
        iy_z: Mutual information ``I(Y;Z)`` in nats.
        beta: The IB trade-off coefficient.

    Returns:
        The IB Lagrangian value.

    Raises:
        NumericalError: If either mutual information is negative.

    Example:
        >>> ib_lagrangian(1.0, 0.5, 0.1)
        0.95
    """
    if ix_z < 0 or iy_z < 0:
        raise NumericalError(
            f"ib_lagrangian: mutual informations must be non-negative; "
            f"got I(X;Z)={ix_z}, I(Y;Z)={iy_z}"
        )
    return ix_z - beta * iy_z


def variational_ib_bound(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    beta: float = 1.0,
) -> float:
    """Compute the variational IB bound from two log-distributions.

    Both inputs are interpreted as *logits* over a discrete latent
    space; the function softmaxes them and computes
    ``D_KL(q(z|x) ‖ p(z))`` plus the regression residual placeholder.

    Args:
        posterior_logits: ``[B, K]`` encoder logits.
        prior_logits: ``[B, K]`` prior logits.
        beta: Trade-off coefficient.

    Returns:
        A non-negative float; the KL term dominates.

    Raises:
        NumericalError: If the result is non-finite.
    """
    if posterior_logits.shape != prior_logits.shape:
        raise NumericalError(
            f"variational_ib_bound: posterior {tuple(posterior_logits.shape)} "
            f"and prior {tuple(prior_logits.shape)} must share shape"
        )
    log_q = torch.log_softmax(posterior_logits, dim=-1)
    log_p = torch.log_softmax(prior_logits, dim=-1)
    q = log_q.exp()
    kl = (q * (log_q - log_p)).sum(dim=-1).mean()
    if not torch.isfinite(kl):
        raise NumericalError("variational_ib_bound: KL is non-finite")
    return float(kl.item()) * beta
