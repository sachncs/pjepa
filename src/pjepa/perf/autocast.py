"""Backend-aware mixed-precision autocast context manager.

The wrapper returns a context manager that:

* enables :func:`torch.cuda.amp.autocast` on CUDA backends;
* enables :func:`torch.autocast` with ``device_type="mps"`` on MPS;
* returns a no-op context manager on CPU.

For the bisimulation metric and the MDL evaluation the caller is
responsible for staying in ``fp64``; the autocast wrapper is only
intended for the encoder and predictor forward passes.

## Exceptions

The function never raises; the no-op context manager is always a
safe fallback so callers do not need to guard the call site.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch

from pjepa.hardware import detect_backend

__all__ = ["NullAutocastContext", "autocast_context", "null_autocast"]


class NullAutocastContext(AbstractContextManager[None]):
    """A no-op context manager used when autocast is unavailable.

    The class is exposed publicly so test fixtures and
    backend-agnostic helpers can construct one directly without
    pulling in the parent factory function.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def autocast_context(
    enabled: bool = True,
    dtype: torch.dtype | None = None,
) -> AbstractContextManager[None]:
    """Return a backend-appropriate autocast context manager.

    Args:
        enabled: When ``False``, returns a no-op context manager.
        dtype: Optional explicit autocast dtype; defaults to
          ``float16`` on CUDA and MPS. ``float32`` is rejected by
          PyTorch's autocast implementation so we silently fall
          through to the no-op context when that happens.

    Returns:
        A context manager that enables autocast on entry and
        disables it on exit. ``NullAutocastContext`` on CPU or
        when ``enabled=False``.
    """
    if not enabled:
        return NullAutocastContext()
    backend = detect_backend()
    if backend.value == "cuda":
        target_dtype = dtype or torch.float16
        return torch.cuda.amp.autocast(dtype=target_dtype)
    if backend.value == "mps":
        target_dtype = dtype or torch.float16
        return torch.autocast(device_type="mps", dtype=target_dtype)
    return NullAutocastContext()


def null_autocast() -> AbstractContextManager[None]:
    """Convenience accessor returning the no-op context manager.

    Returns:
        A :class:`NullAutocastContext` ready to use as a context
        manager. Used by tests and by callers that want to
        explicitly disable autocast without inspecting the backend.
    """
    return NullAutocastContext()
