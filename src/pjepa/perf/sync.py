"""Synchronisation helpers for the MPS backend.

MPS operations are asynchronous by default. CPU code that reads MPS
tensors triggers implicit synchronisation, which can introduce hidden
stalls. The :func:`sync_mps` helper forces an explicit synchronisation
when needed.
"""

from __future__ import annotations

import torch

__all__ = ["sync_mps"]


def sync_mps() -> None:
    """Block until all pending MPS operations have completed.

    Returns:
        None. No-op on non-MPS backends.
    """
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        torch.mps.synchronize()
