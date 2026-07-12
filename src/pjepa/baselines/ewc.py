"""Elastic Weight Consolidation baseline (Kirkpatrick et al., 2017).

Penalises changes to parameters identified as important by the
diagonal of the Fisher information matrix. Used as the continual-
learning baseline in the SOTA comparison.
"""

from __future__ import annotations

import torch

__all__ = ["EWC"]


class EWC:
    """Elastic Weight Consolidation regulariser.

    Attributes:
        lambda_ewc: Strength of the consolidation penalty.
    """

    def __init__(self, lambda_ewc: float = 1000.0) -> None:
        if lambda_ewc < 0:
            raise ValueError(f"EWC: lambda_ewc must be non-negative; got {lambda_ewc}")
        self.lambda_ewc = lambda_ewc
        self._fisher: dict[str, torch.Tensor] = {}
        self._star: dict[str, torch.Tensor] = {}

    def capture(self, named_parameters, loss: torch.Tensor) -> None:
        """Compute and cache the diagonal Fisher information and the parameters.

        Args:
            named_parameters: An iterable of ``(name, parameter)`` pairs.
            loss: A scalar tensor from which gradients are computed
              via backprop.
        """
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            self._star[name] = param.detach().clone()
        grads = torch.autograd.grad(
            loss,
            [p for _, p in named_parameters if p.requires_grad],
            retain_graph=False,
            allow_unused=True,
        )
        for (name, param), grad in zip(
            [(n, p) for n, p in named_parameters if p.requires_grad], grads
        ):
            if grad is None:
                self._fisher[name] = torch.zeros_like(param.detach())
            else:
                self._fisher[name] = grad.detach() ** 2

    def penalty(self, named_parameters) -> torch.Tensor:
        """Compute the EWC penalty for the current parameters."""
        loss = torch.zeros(1, dtype=torch.float32)
        for name, param in named_parameters:
            if name not in self._fisher:
                continue
            loss = loss + (self._fisher[name] * (param - self._star[name]) ** 2).sum()
        return self.lambda_ewc * loss.squeeze()

    def reset(self) -> None:
        """Clear the cached Fisher information and reference parameters."""
        self._fisher.clear()
        self._star.clear()
