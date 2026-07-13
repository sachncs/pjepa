"""Tests for the ``pjepa.version`` module and the PEP 561 marker file."""

from __future__ import annotations

import os

import pjepa
from pjepa import version

__all__ = [
    "test_happy_py_typed_marker_present",
    "test_happy_version_exported",
    "test_happy_version_is_string",
    "test_round_trip_version_module_and_package",
]


def test_happy_version_exported() -> None:
    """The top-level package re-exports ``__version__``."""
    assert hasattr(pjepa, "__version__")
    assert pjepa.__version__ == version.__version__


def test_happy_version_is_string() -> None:
    """The version string is a non-empty string."""
    assert isinstance(pjepa.__version__, str)
    assert len(pjepa.__version__) > 0


def test_round_trip_version_module_and_package() -> None:
    """The package-level version equals the canonical module version."""
    assert version.__version__ == pjepa.__version__


def test_happy_py_typed_marker_present() -> None:
    """The ``py.typed`` marker file exists alongside the package."""
    package_dir = os.path.dirname(pjepa.__file__)
    marker = os.path.join(package_dir, "py.typed")
    assert os.path.isfile(marker), f"py.typed missing at {marker}"
    assert os.path.getsize(marker) >= 0
