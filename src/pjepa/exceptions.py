"""Custom exception hierarchy for the pjepa package.

Every error raised from inside the library is an instance of
:class:`PJEPAError` or one of its subclasses. Tests can therefore catch
the base class to assert "any pjepa failure" or catch a specific
subclass to assert "this particular kind of failure". Bare ``except:``
clauses in library code are forbidden by the project's quality standards
(see ``plans/04_quality_standards.md``); callers should catch the
narrowest applicable subclass.

The hierarchy below is intentionally flat — every subclass inherits
directly from :class:`PJEPAError`. This keeps ``except`` ordering
predictable and avoids the "diamond" surprises that deeper trees create.
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
    "SearchError",
]


class PJEPAError(Exception):
    """Base class for every error raised by the ``pjepa`` library.

    Catching this class also catches every subclass listed in
    :data:`__all__`; subclasses are mutually independent and never
    shadow one another in ``except`` chains.
    """


class SearchError(PJEPAError):
    """Raised when a hyperparameter-search run cannot make progress.

    Examples include an empty training/test split supplied to
    :meth:`pjepa.training.OptunaSearch.evaluate`, an unknown
    distribution kind reported by
    :func:`pjepa.training.optuna_search.suggest_hyperparameters`, or
    a serialisation failure inside
    :meth:`pjepa.training.OptunaSearch.save_best_config` when the
    YAML fallback path cannot represent the value.

    The exception is intentionally narrow: regular training failures
    (``ValueError``, ``RuntimeError``) are caught locally and reported
    per-trial in :class:`pjepa.training.TrialResult` rather than
    terminating the whole search.
    """


class ConfigError(PJEPAError):
    """Raised when a configuration is missing, malformed, or invalid.

    Typical triggers are a missing YAML file, a missing required
    section reported by :func:`pjepa.config.load_config`, an invalid
    section identifier rejected by
    :class:`pjepa.config.ConfigSchema`, and seeding arguments outside
    the documented range.
    """


class DataError(PJEPAError):
    """Raised when a dataset cannot be loaded, parsed, or validated.

    Examples include missing TUDataset / OGB downloads, mismatched
    feature dimensions across batches, and label-shape inconsistencies
    discovered during supervised training.
    """


class GraphError(PJEPAError):
    """Raised when a graph violates a structural invariant of the framework.

    Typical triggers are edges that reference non-existent vertices,
    vertex or edge label arrays of inconsistent length, edge index
    tensors with the wrong dtype (``torch.long`` is required), or a
    working-graph vertex count that exceeds its declared budget.
    """


class NumericalError(PJEPAError):
    """Raised when a numerical operation produces non-finite or unstable values.

    The library prefers raising over returning silently corrupted
    tensors. Callers who genuinely want to recover should catch this
    exception at module boundaries rather than at every loss site.
    """


class ContractError(PJEPAError):
    """Raised when a Protocol is not satisfied by a supposed implementation.

    The runtime-checkable decorators on :class:`pjepa.encoders.base.Encoder`
    and :class:`pjepa.retrieval.utility.RetrievalUtility` raise this
    class when ``isinstance`` checks fail unexpectedly.
    """


class CheckpointError(PJEPAError):
    """Raised when a checkpoint cannot be saved, loaded, or resumed.

    The class distinguishes I/O failures (``OSError``) and shape or
    state-dict mismatches from a missing file — all three are reported
    uniformly so the training loop can react in one place.
    """


class BackendError(PJEPAError):
    """Raised when the active compute backend cannot perform a required operation.

    Most often this is requested through
    :func:`pjepa.hardware.current_device` when ``Backend.CUDA`` or
    :data:`Backend.MPS` is requested on a host where the corresponding
    runtime is unavailable.
    """
