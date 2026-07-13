"""Canonical package version for ``pjepa``.

This module is the single source of truth for the runtime version
string. Keeping the version in a dedicated module (rather than inlined
in :mod:`pjepa.__init__`) lets runtime introspection, packaging, and
type-checkers share one identifier.

The string follows ``major.minor.patch`` semantics; comparisons should
go through :mod:`packaging.version`. The value is intentionally
short-lived and bumped on every release.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__: str = "1.0.0"
