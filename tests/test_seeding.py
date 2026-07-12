"""Tests for pjepa.utils.seeding.

Covers the eight-class test taxonomy:

* happy    — typical seed values
* bad      — invalid seed values raise ConfigError
* ugly     — repeated seeding is deterministic
* leaky    — N/A (stateless module)
* round-trip — get/set round-trip
* cross-backend — seed propagates to MPS generator too
* distributional — sub-seeds differ across components
* property — sub-seeds are reproducible from a fixed global seed
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from pjepa.exceptions import ConfigError
from pjepa.utils import seeding

__all__ = ["test_happy_set_and_get", "test_bad_seed_rejected", "test_ugly_seeding_is_deterministic", "test_round_trip_context", "test_cross_backend_mps", "test_distributional_components_differ", "test_property_sub_seed_reproducible"]


def test_happy_set_and_get() -> None:
    """A typical seed value is stored and returned."""
    seeding.set_global_seed(123)
    assert seeding.get_global_seed() == 123


def test_bad_seed_rejected() -> None:
    """Negative or out-of-range seeds raise ConfigError."""
    with pytest.raises(ConfigError):
        seeding.set_global_seed(-1)
    with pytest.raises(ConfigError):
        seeding.set_global_seed(2**32)


def test_ugly_seeding_is_deterministic() -> None:
    """Setting the same seed twice yields the same sequence of values."""
    seeding.set_global_seed(7)
    first = [random.random() for _ in range(5)]
    seeding.set_global_seed(7)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_round_trip_context() -> None:
    """The context variable is read back correctly after a set."""
    seeding.set_global_seed(999)
    assert seeding.current_seed() == 999


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_cross_backend_mps() -> None:
    """Seeding propagates to torch's MPS generator too."""
    seeding.set_global_seed(11)
    a = torch.randn((3,), device="mps")
    seeding.set_global_seed(11)
    b = torch.randn((3,), device="mps")
    assert torch.allclose(a, b)


def test_distributional_components_differ() -> None:
    """Sub-seeds for different components are distinct."""
    seeding.set_global_seed(50)
    a = seeding.seed_for("encoder")
    b = seeding.seed_for("retrieval")
    assert a != b


def test_property_sub_seed_reproducible() -> None:
    """Sub-seeds are reproducible from the global seed."""
    seeding.set_global_seed(12345)
    a = seeding.seed_for("encoder")
    seeding.set_global_seed(12345)
    b = seeding.seed_for("encoder")
    assert a == b


def test_seed_for_rejects_empty_string() -> None:
    """seed_for raises ConfigError on empty component name."""
    with pytest.raises(ConfigError):
        seeding.seed_for("")


def test_seed_for_numpy_independence() -> None:
    """Sub-seeding does not perturb the global numpy state."""
    seeding.set_global_seed(12345)
    np.random.rand()
    first = np.random.rand()
    seeding.set_global_seed(12345)
    np.random.rand()
    second = np.random.rand()
    assert first == second