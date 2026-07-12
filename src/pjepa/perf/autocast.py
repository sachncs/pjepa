"""Backend-aware mixed-precision autocast context manager.

The wrapper returns a context manager that:

* Enables ``torch.cuda.amp.autocast`` on CUDA.
* Enables ``torch.autocast(device_type="mps", dtype=torch.float16)`` on MPS.
* Returns ``contextlib.nullcontext`` on CPU.

For the bisimulation metric and the MDL evaluation the caller is
responsible for staying in fp64; the autocast wrapper is only intended
for the encoder and predictor forward passes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any

import torch

from pjepa.hardware import detect_backend

__all__ = ["autocast_context"]


class _NullContext(AbstractContextManager[None]):
    """A no-op context manager used when autocast is unavailable."""

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
        dtype: Optional explicit autocast dtype; defaults to ``float16``
          on CUDA and MPS.

    Returns:
        A context manager that enables autocast on entry and disables
        it on exit.
    """
    if not enabled:
        return _NullContext()
    backend = detect_backend()
    if backend.value == "cuda":
        target_dtype = dtype or torch.float16
        return torch.cuda.amp.autocast(dtype=target_dtype)
    if backend.value == "mps":
        target_dtype = dtype or torch.float16
        return torch.autocast(device_type="mps", dtype=target_dtype)
    return _NullContext()


def null_autocast() -> Iterator[None]:
    """Convenience iterator for the null context (used in tests)."""
    with _NullContext():
        yield None
