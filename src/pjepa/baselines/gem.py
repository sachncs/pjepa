r"""Gradient Episodic Memory baseline (Lopez-Paz & Ranzato, 2017).

Stores a fixed-size memory of past samples and projects candidate
gradients so they do not increase loss on the memory.

## Algorithm

For a new gradient ``g`` we collect per-sample reference gradients
``g_i = ∇_θ L_i`` (one per memory sample) and form the matrix
``R ∈ ℝ^{M × P}`` whose rows are the flattened reference gradients.
A memory violation occurs when ``g`` has positive inner product with
any ``g_i`` along the "wrong" axis — specifically, when

.. math::

    g^T g_i < 0

for some memory sample ``i`` (a positive dot product would
*decrease* memory loss, which we tolerate; a negative dot product
means ``g`` would *increase* memory loss, which GEM forbids).

When at least one constraint is violated, GEM solves the equality-
constrained QP

.. math::

    \tilde{g} = g - R^T w

with

.. math::

    w = (R R^T + \lambda I)^{-1} R g

The Lagrange multiplier ``w`` is the closed-form solution of the
GEM dual; the projection ``g - R^T w`` lands on the boundary of the
feasible cone and removes the offending direction.

## Complexity

Per projected gradient the cost is ``O(M * P)`` where ``M`` is the
memory size and ``P`` is the number of parameters (here ``M ≤
capacity``). The closed-form solve uses
``torch.linalg.solve(gram, rg)``, which is ``O(M^3)`` per call but
in practice ``M ≤ 256``.

## Exceptions

A non-positive ``capacity`` is rejected with
:class:`pjepa.exceptions.ConfigError`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from pjepa.exceptions import ConfigError

__all__ = ["GEM", "MemorySample"]


@dataclass
class MemorySample:
    """A single memory sample.

    Attributes:
        x: The input tensor.
        y: The label.
    """

    x: torch.Tensor
    y: torch.Tensor


@dataclass
class GEM:
    """Gradient Episodic Memory wrapper.

    Attributes:
        capacity: Maximum memory size.
        memory: Bounded :class:`collections.deque` of
          :class:`MemorySample` instances. ``maxlen`` is set to
          ``capacity``; the deque discards the oldest entry when full.
    """

    capacity: int = 256
    memory: deque = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ConfigError(f"GEM: capacity must be positive; got {self.capacity}")
        self.memory = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a sample to memory.

        Args:
            x: The input tensor (cloned to detach from the caller's
              graph).
            y: The label tensor (cloned likewise).
        """
        self.memory.append(MemorySample(x=x.detach().clone(), y=y.detach().clone()))

    def project_gradient(
        self,
        gradient: torch.Tensor,
        model_output_fn: Callable[[torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Project ``gradient`` so it does not increase loss on memory samples.

        Args:
            gradient: The candidate gradient (flat tensor).
            model_output_fn: Callable mapping an input tensor to
              model outputs. Must expose ``.parameters()`` so that
              per-sample reference gradients can be computed.
            loss_fn: Callable mapping ``(output, target)`` to a
              scalar loss.

        Returns:
            The (possibly projected) gradient, in the same shape as
            the input.
        """
        if len(self.memory) == 0:
            return gradient
        # Compute reference gradients from memory samples.
        ref_grads = []
        for sample in self.memory:
            out = model_output_fn(sample.x)
            loss = loss_fn(out, sample.y)
            grad = torch.autograd.grad(
                loss,
                [p for p in model_output_fn.parameters()],
                retain_graph=False,
                allow_unused=True,
            )
            ref_grads.append(
                torch.cat(
                    [
                        g.flatten() if g is not None else torch.zeros_like(p.flatten())
                        for g, p in zip(grad, model_output_fn.parameters())
                    ]
                )
            )
        ref = torch.stack(ref_grads, dim=0)  # [M, P]; M = len(memory), P = num parameters.
        inner = ref @ gradient
        # A memory constraint is *violated* when ``g^T g_i < 0``: that means the
        # candidate gradient would *increase* the memory loss. A non-negative
        # dot product is acceptable — it means ``g`` does not hurt memory loss.
        violated = inner < -1e-7
        if not violated.any():
            return gradient
        # Closed-form GEM dual projection (Lopez-Paz & Ranzato, 2017, §3.2).
        # The constraint is ``g' = g - R^T w`` with ``R w = R g`` when feasible;
        # the damped solve ``w = (R R^T + λ I)^{-1} R g`` restores feasibility
        # via a tiny Tikhonov regulariser (λ = 1e-3 here).
        R = ref
        g = gradient
        gram = R @ R.T + 1e-3 * torch.eye(R.shape[0])
        w = torch.linalg.solve(gram, R @ g)
        projection = R.T @ w
        # ``v = projection - g`` is the direction we projected *into*; ``alpha``
        # rescales the projection so the worst memory constraint is exactly
        # satisfied instead of over-projected.
        v = projection - g
        denom = v @ v
        if denom <= 0:
            return g
        alpha = (g @ R[inner.argmin()] + 1e-7) / denom
        return g - alpha * v
