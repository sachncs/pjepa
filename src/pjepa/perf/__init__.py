"""Performance infrastructure for the pjepa package.

This subpackage provides capability-aware performance primitives:

* :class:`EMATarget` — BYOL-style exponential moving average for
  target encoders.
* :func:`safe_compile` — backend-aware ``torch.compile`` wrapper
  with graceful fallback.
* :func:`autocast_context` — backend-aware mixed-precision context
  manager.
* :func:`fused_scatter_add` — fused ``scatter_add_`` with a CPU fallback
  for backends where the native kernel is slow.
* :func:`sync_mps` — explicit MPS synchronisation helper.
"""

from __future__ import annotations

from pjepa.perf.autocast import autocast_context
from pjepa.perf.compile import safe_compile
from pjepa.perf.ema import EMATarget
from pjepa.perf.scatter import fused_scatter_add, fused_scatter_mean
from pjepa.perf.sync import sync_mps

__all__ = [
    "EMATarget",
    "autocast_context",
    "fused_scatter_add",
    "fused_scatter_mean",
    "safe_compile",
    "sync_mps",
]
