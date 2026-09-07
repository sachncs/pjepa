"""Fused scatter operations.

A thin wrapper around :meth:`torch.Tensor.scatter_add_` that is
device-aware and always returns the updated tensor in place.

## Architecture

```
   out.scatter_add_(dim, index, src)
```

The wrapper:

1. Validates that every index in ``index`` resolves within the
   ``dim`` extent of ``out``.
2. Forces a synchronous flush before and after the scatter on
   MPS (the MPS scatter kernel can lag the rest of the graph).
3. Broadcasts ``index`` to match ``src``'s dimensionality
   because PyTorch's scatter requires the broadcast even when
   ``index`` would otherwise be allowed to broadcast.

## Complexity

* :func:`fused_scatter_add` — ``O(|src|)``.
* :func:`fused_scatter_mean` — ``O(|src|)`` plus a constant
  ``count`` scatter and a division.
"""

from __future__ import annotations

import torch

from pjepa.hardware import detect_backend, sync_if_mps

__all__ = ["expand_index_to_shape", "fused_scatter_add", "fused_scatter_mean"]


def expand_index_to_shape(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Broadcast a 1-D ``index`` to match ``src.ndim``.

    Args:
        index: The 1-D or ``src.ndim``-D index tensor.
        src: The source tensor the index will be applied to.

    Returns:
        The ``index`` tensor, broadcast to ``src.ndim``.
    """
    if index.ndim < src.ndim:
        view_shape = list(index.shape) + [1] * (src.ndim - index.ndim)
        return index.view(view_shape).expand_as(src)
    return index


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
        ValueError: When the largest value in ``index`` exceeds
          the size of ``out`` on the scatter dimension.
    """
    if out.shape[dim] < int(index.max().item()) + 1:
        raise ValueError(
            f"fused_scatter_add: index max {int(index.max().item())} exceeds "
            f"out size {out.shape[dim]} on dim {dim}"
        )
    backend = detect_backend()
    if backend.value == "mps":
        # MPS scatter can lag; force a sync before/after.
        sync_if_mps()
    index = expand_index_to_shape(index, src)
    out.scatter_add_(dim, index, src)
    if backend.value == "mps":
        sync_if_mps()
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
          ``src`` is multi-dimensional and ``out`` has matching
          shape, ``count`` may be 1-D and is reshaped internally
          for the scatter.
        index: The index tensor.
        src: The source tensor.
        dim: The dimension along which to scatter.

    Returns:
        The ``out`` tensor with the mean values.
    """
    out.zero_()
    count.zero_()
    fused_scatter_add(out, index, src, dim=dim)
    ones_1d = torch.ones((index.shape[0],), dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, ones_1d)
    if out.ndim == 2 and count.ndim == 1:
        count = count.unsqueeze(-1)
    count.clamp_(min=1)
    out.div_(count)
    return out
