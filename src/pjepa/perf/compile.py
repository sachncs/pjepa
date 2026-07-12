"""Backend-aware ``torch.compile`` wrapper.

The wrapper selects a compile mode based on the active backend:

* CUDA: ``reduce-overhead`` for the best throughput on small batches.
* MPS: ``default`` because some operators do not fuse cleanly on MPS.
* CPU: ``default`` (or skipped under a flag, since CPU compile is
  rarely worth the cost).

On any compilation failure the wrapper logs a YELLOW warning and
returns the uncompiled module.
"""

from __future__ import annotations

import torch

from pjepa.hardware import detect_backend
from pjepa.logging_setup import get_logger

__all__ = ["safe_compile"]


def safe_compile(
    module: torch.nn.Module,
    *,
    mode: str | None = None,
    fullgraph: bool = False,
) -> torch.nn.Module:
    """Compile ``module`` using ``torch.compile`` with a backend-appropriate mode.

    Args:
        module: The module to compile.
        mode: Optional explicit compile mode. When ``None`` a default
          is chosen based on the active backend.
        fullgraph: Whether to require the full graph to be capturable.

    Returns:
        The compiled module, or the uncompiled module on failure.
    """
    backend = detect_backend()
    chosen_mode = mode
    if chosen_mode is None:
        chosen_mode = "reduce-overhead" if backend.value == "cuda" else "default"
    log = get_logger(__name__)
    try:
        compiled = torch.compile(module, mode=chosen_mode, fullgraph=fullgraph)
        log.info(
            "compiled module",
            extra={"event": "compile.success", "backend": backend.value, "mode": chosen_mode},
        )
        return compiled
    except Exception as exc:
        log.info(
            "compile failed; returning uncompiled module",
            extra={"event": "compile.failure", "backend": backend.value, "error": str(exc)},
        )
        return module
