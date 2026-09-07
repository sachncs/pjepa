"""pytest configuration for the pjepa project.

Adds the in-tree ``src/`` directory to ``sys.path`` so that tests can
import ``pjepa`` without an editable install, primes the doctest
namespace with the common imports every module's ``>>>`` examples
rely on, and auto-skips marker-gated tests when their gate is not
satisfied:

* ``slow`` tests are skipped unless the user passes ``-m slow`` or
  removes ``-m "not slow"``.
* ``mps_sequential`` tests are skipped when MPS is not available.
"""

from __future__ import annotations

import os
import sys

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency.
    torch = None  # type: ignore[assignment]


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(autouse=True)
def doctest_namespace(doctest_namespace: dict[str, object]) -> dict[str, object]:
    """Make ``torch`` (and friends) available to ``>>>`` doctest blocks."""
    doctest_namespace.setdefault("torch", torch)
    return doctest_namespace


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests with environment-gated markers when the gate is unmet."""
    mps_available = bool(
        torch is not None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    skip_mps = pytest.mark.skip(reason="MPS not available on this host")
    skip_slow = pytest.mark.skip(
        reason="slow test; pass -m slow or remove -m 'not slow' to run"
    )
    for item in items:
        if "mps_sequential" in item.keywords and not mps_available:
            item.add_marker(skip_mps)
        if "slow" in item.keywords and "-m" in config.invocation_params.args:
            selection = marker_selection(config)
            if selection is not None and "slow" not in selection:
                item.add_marker(skip_slow)


def marker_selection(config: pytest.Config) -> str | None:
    """Return the ``-m`` expression string passed on the CLI, if any."""
    args = config.invocation_params.args
    for index, arg in enumerate(args):
        if arg == "-m" and index + 1 < len(args):
            return args[index + 1]
    return None
