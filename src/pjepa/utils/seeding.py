"""Deterministic seeding for the pjepa package.

The library never calls ``torch.manual_seed`` or ``numpy.random.seed``
directly; instead it routes every seed through :func:`set_global_seed`,
which records the seed on a context-local object so that runs can be
reproduced exactly. Tests and benchmarks use this entry point as well.
"""

from __future__ import annotations

import os
import random
from contextvars import ContextVar
from typing import Final

import numpy as np
import torch

from pjepa.exceptions import ConfigError

__all__ = ["set_global_seed", "get_global_seed", "seed_for", "current_seed"]

_DEFAULT_SEED: Final[int] = 0
_VALIDATE_DETERMINISM_ENV: Final[str] = "PJEPA_DETERMINISTIC"


_current_seed: ContextVar[int] = ContextVar("pjepa_current_seed", default=_DEFAULT_SEED)


def get_global_seed() -> int:
    """Return the seed currently in force for the calling context.

    Returns:
        The integer seed last set via :func:`set_global_seed`. If
        :func:`set_global_seed` has never been invoked, returns the
        default seed (``0``).

    Example:
        >>> set_global_seed(42)
        >>> get_global_seed()
        42
    """
    return _current_seed.get()


def set_global_seed(seed: int) -> int:
    """Seed every random source used by the library.

    The seed is stored in a context variable and applied to the Python
    ``random`` module, NumPy, and PyTorch (CPU and any active CUDA or
    MPS device). Deterministic algorithm selection is left to the
    caller; this function only seeds the underlying generators.

    Args:
        seed: A non-negative 32-bit integer. Values outside the valid
          range raise :class:`ConfigError`.

    Returns:
        The seed that was applied, identical to the input.

    Raises:
        ConfigError: If ``seed`` is negative or larger than ``2**32 - 1``.

    Example:
        >>> set_global_seed(7)
        7
        >>> get_global_seed()
        7
    """
    if seed < 0 or seed >= 2**32:
        raise ConfigError(
            f"set_global_seed: seed must be in [0, 2**32); got {seed}"
        )
    _current_seed.set(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS has no separate generator; torch.manual_seed already covered it.
        pass

    os.environ.setdefault(_VALIDATE_DETERMINISM_ENV, "0")
    return seed


def seed_for(component: str, base: int | None = None) -> int:
    """Derive a deterministic sub-seed for a named component.

    This helper produces a stable, distinct seed for every component of
    the library (``encoder``, ``retrieval``, ``scheduler``, ...). The
    derivation uses :func:`numpy.random.SeedSequence.spawn` so that
    sub-seeds are independent but reproducible from the global seed.

    Args:
        component: A short identifier such as ``"encoder"`` or
          ``"replay_buffer"``.
        base: Optional override for the global seed. When ``None`` the
          current global seed is used.

    Returns:
        A 32-bit non-negative integer that can be passed to
        :func:`torch.Generator.manual_seed` or equivalent.

    Example:
        >>> set_global_seed(123)
        >>> seed_for("encoder")
        2519165639
    """
    if not isinstance(component, str) or not component:
        raise ConfigError("seed_for: component must be a non-empty string")
    root_seed = _DEFAULT_SEED if base is None else base
    ss = np.random.SeedSequence([root_seed, hash(component) & 0xFFFFFFFF])
    child = ss.spawn(1)[0]
    return int(child.generate_state(1)[0])


def current_seed() -> int:
    """Alias for :func:`get_global_seed` kept for ergonomic call sites."""
    return _current_seed.get()