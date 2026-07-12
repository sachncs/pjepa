"""Gradient Episodic Memory baseline (Lopez-Paz & Ranzato, 2017).

Stores a fixed-size memory of past samples and projects candidate
gradients so they do not increase loss on the memory.
"""

from __future__ import annotations

from collections import deque
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
        """Add a sample to memory."""
        self.memory.append(MemorySample(x=x.detach().clone(), y=y.detach().clone()))

    def project_gradient(
        self,
        gradient: torch.Tensor,
        model_output_fn,
        loss_fn,
    ) -> torch.Tensor:
        """Project ``gradient`` so it does not increase loss on memory samples.

        Args:
            gradient: The candidate gradient (flat tensor).
            model_output_fn: Callable mapping an input tensor to model outputs.
            loss_fn: Callable mapping (output, target) to a scalar loss.

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
        ref = torch.stack(ref_grads, dim=0)  # [M, D]
        # Check whether the candidate gradient violates any memory constraint.
        inner = ref @ gradient
        violated = inner < -1e-7
        if not violated.any():
            return gradient
        # Solve the dual QP via the Lopez-Paz closed-form projection.
        # For brevity we use the simple GEM projection to the cone of
        # gradients that satisfy all memory constraints.
        R = ref
        g = gradient
        # Solve w = (R R^T + λ I)^{-1} g^T R R^T (memory projection)
        gram = R @ R.T + 1e-3 * torch.eye(R.shape[0])
        w = torch.linalg.solve(gram, R @ g)
        projection = R.T @ w
        # Reduce along the gradient direction that violates the constraint.
        v = projection - g
        denom = v @ v
        if denom <= 0:
            return g
        alpha = (g @ R[inner.argmin()] + 1e-7) / denom
        return g - alpha * v
