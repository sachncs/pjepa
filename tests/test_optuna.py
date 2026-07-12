"""Tests for the Optuna search runner."""

from __future__ import annotations

import pytest

from experiments.run_optuna_search import _suggest_config, _train_one_trial


@pytest.fixture
def small_tu_dataset() -> tuple[list, list, int]:
    """Create a tiny synthetic dataset for testing."""
    from pjepa.data.tu import load_tu_dataset

    graphs, num_classes = load_tu_dataset("MUTAG")
    pairs = [(g.graph, g.label) for g in graphs]
    return pairs[: int(0.8 * len(pairs))], pairs[int(0.8 * len(pairs)) :], num_classes


class _FakeTrial:
    """Minimal stub of an Optuna trial for unit testing."""

    def __init__(self, params: dict[str, object]) -> None:
        self._params = params

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        return float(self._params.get(name, (low + high) / 2))

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return int(self._params.get(name, (low + high) // 2))

    def suggest_categorical(self, name: str, choices: list) -> object:
        return self._params.get(name, choices[0])


__all__ = [
    "test_happy_suggest_config_returns_dict",
    "test_happy_train_one_trial_returns_accuracy",
    "test_ugly_train_one_trial_with_tiny_dataset",
]


def test_happy_suggest_config_returns_dict() -> None:
    """_suggest_config returns a configuration dictionary."""
    trial = _FakeTrial({})
    config = _suggest_config(trial, dataset="MUTAG")
    assert "lr" in config
    assert "hidden_dim" in config
    assert config["hidden_dim"] in (64, 128, 256)


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
    accuracy = _train_one_trial(config, train_pairs, test_pairs, num_classes, epochs=5)
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
    accuracy = _train_one_trial(config, train_pairs, test_pairs, num_classes, epochs=2)
    assert 0.0 <= accuracy <= 1.0
