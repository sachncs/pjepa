"""Tests for pjepa.exceptions."""

from __future__ import annotations

import pytest

from pjepa.exceptions import (
    BackendError,
    CheckpointError,
    ConfigError,
    ContractError,
    DataError,
    GraphError,
    NumericalError,
    PJEPAError,
)

__all__ = [
    "test_happy_raise_and_catch",
    "test_bad_unknown_subclass_caught_as_pjepa",
    "test_property_hierarchy",
]


def test_happy_raise_and_catch() -> None:
    """Raising a subclass and catching by its own type works."""
    with pytest.raises(ConfigError):
        raise ConfigError("bad config")


def test_bad_unknown_subclass_caught_as_pjepa() -> None:
    """Any PJEPAError is catchable via the base class."""
    with pytest.raises(PJEPAError):
        raise NumericalError("nans in computation")


def test_property_hierarchy() -> None:
    """Each subclass inherits from PJEPAError, not from arbitrary Exception."""
    for cls in (
        ConfigError,
        DataError,
        GraphError,
        NumericalError,
        ContractError,
        CheckpointError,
        BackendError,
    ):
        assert issubclass(cls, PJEPAError)
        assert issubclass(cls, Exception)


def test_message_is_preserved() -> None:
    """The error message round-trips through str(exc)."""
    err = GraphError("invariant violated")
    assert "invariant violated" in str(err)


def test_each_error_can_be_raised_independently() -> None:
    """Each error type can be raised and re-raised through a base handler."""
    for cls in (DataError, CheckpointError, BackendError, ContractError):
        with pytest.raises(PJEPAError):
            raise cls(f"{cls.__name__} message")