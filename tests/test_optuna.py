"""Tests for the Optuna search runner.

Covers the public helpers re-exported by
:mod:`experiments.run_optuna_search` so that downstream callers can
replace the historical private names with the new public API.
"""

from __future__ import annotations

import pytest

from experiments.run_optuna_search import OPTUNA_SEARCH_SPACE, suggest_config, train_one_trial

__all__ = [
    "test_happy_suggest_config_returns_dict",
    "test_happy_train_one_trial_returns_accuracy",
    "test_search_space_contains_canonical_keys",
    "test_ugly_train_one_trial_with_tiny_dataset",
]


@pytest.fixture
def small_tu_dataset():
    """Create a tiny synthetic dataset for testing.

    Returns:
        A tuple ``(train_pairs, test_pairs, num_classes)`` sliced
        from the cached MUTAG dataset.
    """
    from pjepa.data.tu import load_tu_dataset

    graphs, num_classes = load_tu_dataset("MUTAG")
    pairs = [(g.graph, g.label) for g in graphs]
    return pairs[: int(0.8 * len(pairs))], pairs[int(0.8 * len(pairs)) :], num_classes


class FakeTrial:
    """Minimal stub of an Optuna trial for unit testing.

    Args:
        params: A name-to-value mapping that overrides the default
            (the midpoint of the trial range).
    """

    def __init__(self, params: dict[str, object]) -> None:
        self._params = params

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        """Return the configured value or the range midpoint."""
        return float(self._params.get(name, (low + high) / 2))

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """Return the configured value or the integer midpoint."""
        return int(self._params.get(name, (low + high) // 2))

    def suggest_categorical(self, name: str, choices: list) -> object:
        """Return the configured value or the first choice."""
        return self._params.get(name, choices[0])


def test_happy_suggest_config_returns_dict() -> None:
    """``suggest_config`` returns a configuration dictionary."""
    trial = FakeTrial({})
    config = suggest_config(trial, dataset="MUTAG")
    assert "lr" in config
    assert "hidden_dim" in config
    assert config["hidden_dim"] in (64, 128, 256)


def test_search_space_contains_canonical_keys() -> None:
    """The canonical search-space keys are present."""
    expected = {"lr", "weight_decay", "hidden_dim", "num_layers", "dropout", "B", "beta_ib"}
    assert expected.issubset(set(OPTUNA_SEARCH_SPACE.keys()))


def test_happy_train_one_trial_returns_accuracy() -> None:
    """A training trial returns an accuracy in [0, 1]."""
    from pjepa.data.tu import load_tu_dataset

    graphs, num_classes = load_tu_dataset("MUTAG")
    pairs = [(g.graph, g.label) for g in graphs]
    train_pairs = pairs[:100]
    test_pairs = pairs[100:120]
    config = {
        "lr": 1e-2,
        "weight_decay": 1e-4,
        "hidden_dim": 64,
        "num_layers": 2,
        "label_smoothing": 0.0,
    }
    accuracy = train_one_trial(config, train_pairs, test_pairs, num_classes, epochs=5)
    assert 0.0 <= accuracy <= 1.0


def test_ugly_train_one_trial_with_tiny_dataset() -> None:
    """A single-pair train and a single-pair test does not crash."""
    from pjepa.data.tu import load_tu_dataset

    graphs, num_classes = load_tu_dataset("MUTAG")
    pairs = [(g.graph, g.label) for g in graphs[:4]]
    train_pairs = pairs[:2]
    test_pairs = pairs[2:]
    config = {
        "lr": 1e-2,
        "weight_decay": 1e-4,
        "hidden_dim": 32,
        "num_layers": 1,
        "label_smoothing": 0.0,
    }
    accuracy = train_one_trial(config, train_pairs, test_pairs, num_classes, epochs=2)
    assert 0.0 <= accuracy <= 1.0
