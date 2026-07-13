r"""Elastic Weight Consolidation baseline (Kirkpatrick et al., 2017).

Penalises changes to parameters identified as important by the
diagonal of the Fisher information matrix. Used as the
continual-learning baseline in the SOTA comparison.

## Algorithm

The penalty added to the task loss is

.. math::

    \mathcal{L}_{\text{EWC}} = \lambda \sum_i F_i (\theta_i -
                                                  \theta_i^*)^2

where ``F_i`` is the diagonal Fisher information for parameter
``i`` and ``\theta_i^*`` is the parameter snapshot from after the
previous task.

## Complexity

:meth:`capture` evaluates per-parameter gradients in ``O(P)``;
:meth:`penalty` is ``O(P)`` per call. Memory is dominated by two
``O(P)`` caches (``fisher_information`` and ``reference_parameters``).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["EWC"]


class EWC:
    """Elastic Weight Consolidation regulariser.

    Attributes:
        lambda_ewc: Strength of the consolidation penalty. Non-negative.
        fisher_information: The cached diagonal Fisher map (name →
          ``Tensor``). Exposed via :attr:`fisher_state`; treated as
          read-only outside of the class.
        reference_parameters: The cached "star" parameter map
          (name → ``Tensor``). Exposed via :attr:`fisher_state`;
          treated as read-only outside of the class.
    """

    def __init__(self, lambda_ewc: float = 1000.0) -> None:
        if lambda_ewc < 0:
            raise ValueError(f"EWC: lambda_ewc must be non-negative; got {lambda_ewc}")
        self.lambda_ewc = lambda_ewc
        self.fisher_information: dict[str, torch.Tensor] = {}
        self.reference_parameters: dict[str, torch.Tensor] = {}

    def capture(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        loss: torch.Tensor,
    ) -> None:
        """Compute and cache the diagonal Fisher information and the parameters.

        Args:
            named_parameters: An iterable of ``(name, parameter)`` pairs.
            loss: A scalar tensor from which gradients are computed
              via backprop.
        """
        params_list = list(named_parameters)
        trainable = [(name, param) for name, param in params_list if param.requires_grad]
        for name, param in trainable:
            self.reference_parameters[name] = param.detach().clone()
        grads = torch.autograd.grad(
            loss,
            [param for _, param in trainable],
            retain_graph=False,
            allow_unused=True,
        )
        for (name, param), grad in zip(trainable, grads):
            if grad is None:
                self.fisher_information[name] = torch.zeros_like(param.detach())
            else:
                self.fisher_information[name] = grad.detach() ** 2

    def penalty(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> torch.Tensor:
        """Compute the EWC penalty for the current parameters.

        The penalty is ``λ Σ F_i (θ_i - θ_i*)²``; parameters not
        present in the cached Fisher map contribute nothing.

        Args:
            named_parameters: An iterable of ``(name, parameter)`` pairs.

        Returns:
            A scalar tensor equal to ``λ Σ F_i (θ_i - θ_i*)²``.
        """
        loss = torch.zeros(1, dtype=torch.float32)
        for name, param in named_parameters:
            if name not in self.fisher_information:
                continue
            loss = (
                loss
                + (
                    self.fisher_information[name] * (param - self.reference_parameters[name]) ** 2
                ).sum()
            )
        return self.lambda_ewc * loss.squeeze()

    def fisher_state(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return copies of the cached Fisher and reference parameters.

        Returns:
            A mapping with two keys: ``"fisher"`` and ``"star"``.
            Each value is a name-to-tensor mapping whose tensors are
            detached clones (safe to inspect and to compare across
            iterations).
        """
        return {
            "fisher": {
                name: tensor.detach().clone() for name, tensor in self.fisher_information.items()
            },
            "star": {
                name: tensor.detach().clone() for name, tensor in self.reference_parameters.items()
            },
        }

    def set_fisher_state(
        self,
        fisher: dict[str, torch.Tensor],
        star: dict[str, torch.Tensor],
    ) -> None:
        """Replace the cached Fisher information and reference parameters.

        Useful when the Fisher is computed incrementally (e.g.
        accumulated across mini-batches) and only set at the end of
        a task.

        Args:
            fisher: A name-to-tensor mapping that becomes the new
              diagonal Fisher information.
            star: A name-to-tensor mapping that becomes the new
              reference parameters.
        """
        self.fisher_information = {name: tensor.detach().clone() for name, tensor in fisher.items()}
        self.reference_parameters = {name: tensor.detach().clone() for name, tensor in star.items()}

    def reset(self) -> None:
        """Clear the cached Fisher information and reference parameters."""
        self.fisher_information.clear()
        self.reference_parameters.clear()
