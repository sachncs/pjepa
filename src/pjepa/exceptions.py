"""Custom exception hierarchy for the pjepa package.

Every error raised from inside the library is an instance of
:class:`PJEPAError` or one of its subclasses. Tests can therefore
catch the base class to assert "any pjepa failure" or catch a specific
subclass to assert "this particular kind of failure". Bare ``except``
clauses in library code are forbidden by the project's quality
standards (see ``plans/04_quality_standards.md``).
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "CheckpointError",
    "ConfigError",
    "ContractError",
    "DataError",
    "GraphError",
    "NumericalError",
    "PJEPAError",
]


class PJEPAError(Exception):
    """Base class for every error raised by the pjepa library."""


class ConfigError(PJEPAError):
    """Raised when a configuration is missing, malformed, or invalid."""


class DataError(PJEPAError):
    """Raised when a dataset cannot be loaded, parsed, or validated."""


class GraphError(PJEPAError):
    """Raised when a graph violates a structural invariant of the framework."""


class NumericalError(PJEPAError):
    """Raised when a numerical operation produces non-finite or unstable values."""


class ContractError(PJEPAError):
    """Raised when a Protocol is not satisfied by a supposed implementation."""


class CheckpointError(PJEPAError):
    """Raised when a checkpoint cannot be saved, loaded, or resumed."""


class BackendError(PJEPAError):
    """Raised when the active compute backend cannot perform a required operation."""
