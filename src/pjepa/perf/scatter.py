"""Fused scatter operations.

A thin wrapper around ``torch.Tensor.scatter_add_`` that is
device-aware and always returns the updated tensor in place.
"""

from __future__ import annotations

import torch

from pjepa.hardware import detect_backend
from pjepa.hardware import sync_if_mps as _sync_mps

__all__ = ["fused_scatter_add", "fused_scatter_mean"]


def fused_scatter_add(
    out: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """In-place scatter-add using the most efficient kernel for the active backend.

    Args:
        out: The output tensor (modified in place).
        index: The index tensor.
        src: The source tensor.
        dim: The dimension along which to scatter.

    Returns:
        The ``out`` tensor with the scatter-add applied.

    Raises:
        ValueError: If shapes are incompatible.
    """
    if out.shape[dim] < int(index.max().item()) + 1:
        raise ValueError(
            f"fused_scatter_add: index max {int(index.max().item())} exceeds "
            f"out size {out.shape[dim]} on dim {dim}"
        )
    backend = detect_backend()
    if backend.value == "mps":
        # MPS scatter can lag; force a sync before/after.
        _sync_mps()
    # Broadcast the index to match src's dimensionality (PyTorch scatter
    # requires this even when broadcasting is otherwise permitted).
    index = _expand_index(index, src)
    out.scatter_add_(dim, index, src)
    if backend.value == "mps":
        _sync_mps()
    return out


def fused_scatter_mean(
    out: torch.Tensor,
    count: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Compute the segment-wise mean via scatter-add then divide by counts.

    Args:
        out: Pre-allocated output tensor (modified in place).
        count: Pre-allocated count tensor (modified in place). When
          ``src`` is multi-dimensional and ``out`` has matching shape,
          ``count`` may be 1-D and is reshaped internally for the
          scatter.
        index: The index tensor.
        src: The source tensor.
        dim: The dimension along which to scatter.

    Returns:
        The ``out`` tensor with the mean values.
    """
    out.zero_()
    count.zero_()
    fused_scatter_add(out, index, src, dim=dim)
    # Count scatter: use a 1-D ones source so that count's shape matches
    # the index's shape (both 1-D).
    ones_1d = torch.ones((index.shape[0],), dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, ones_1d)
    if out.ndim == 2 and count.ndim == 1:
        count = count.unsqueeze(-1)
    count.clamp_(min=1)
    out.div_(count)
    return out


def _expand_index(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Broadcast a 1-D index to match ``src.ndim``."""
    if index.ndim < src.ndim:
        view_shape = list(index.shape) + [1] * (src.ndim - index.ndim)
        return index.view(view_shape).expand_as(src)
    return index
