"""Tests for Naive, DatasetCache, Microbenchmark, and EWC iterable handling."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from pjepa.baselines import Naive
from pjepa.baselines.ewc import EWC
from pjepa.exceptions import DataError
from pjepa.graphs import TypedAttributedGraph
from pjepa.perf import (
    DatasetCache,
    Microbenchmark,
    MicrobenchmarkResult,
    cache_key,
    compare_benchmarks,
)

__all__ = [
    "test_bad_cache_get_missing_key",
    "test_bad_naive_zero_input_dim",
    "test_bad_naive_zero_num_classes",
    "test_distributional_cache_key_distinct",
    "test_happy_cache_get_or_compute",
    "test_happy_cache_put_get",
    "test_happy_compare_benchmarks_speedup",
    "test_happy_ewc_with_generator_iterable",
    "test_happy_microbenchmark_run",
    "test_happy_naive_forward_shape",
    "test_happy_naive_with_hidden_dim",
    "test_round_trip_cache_survives_clear",
]


# ============================== NAIVE BASELINE ==============================


def _toy_graph() -> TypedAttributedGraph:
    return TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    )


def test_happy_naive_forward_shape() -> None:
    """Naive returns per-graph logits of the right shape."""
    model = Naive(input_dim=4, num_classes=3)
    out = model(_toy_graph())
    assert out.shape == (1, 3)


def test_happy_naive_with_hidden_dim() -> None:
    """Naive with a non-zero hidden_dim still produces valid logits."""
    model = Naive(input_dim=4, hidden_dim=8, num_classes=2)
    out = model(_toy_graph())
    assert out.shape == (1, 2)


def test_bad_naive_zero_input_dim() -> None:
    """A zero input_dim is rejected."""
    with pytest.raises(ValueError):
        Naive(input_dim=0)


def test_bad_naive_zero_num_classes() -> None:
    """A zero num_classes is rejected."""
    with pytest.raises(ValueError):
        Naive(input_dim=4, num_classes=0)


# ============================== EWC iterable fix ==============================


def _gen() -> Iterator[tuple[str, torch.nn.Parameter]]:
    pairs = [
        ("w1", torch.nn.Parameter(torch.randn(3, requires_grad=True))),
        ("w2", torch.nn.Parameter(torch.randn(3, requires_grad=True))),
    ]
    yield from pairs


def test_happy_ewc_with_generator_iterable() -> None:
    """EWC.capture tolerates a one-shot generator from ``model.named_parameters()``."""
    ewc = EWC(lambda_ewc=1.0)
    params = list(_gen())
    loss = sum((p**2).sum() for _, p in params)
    ewc.capture(iter(params), loss)
    state = ewc.fisher_state()
    assert set(state["fisher"]) == {"w1", "w2"}
    drift_params = [(n, torch.nn.Parameter(torch.ones_like(p.detach()))) for n, p in params]
    penalty = ewc.penalty(drift_params)
    assert float(penalty.item()) >= 0.0


# ============================== DATASET CACHE ==============================


def test_happy_cache_put_get(tmp_path: Path) -> None:
    """put then get returns the same object."""
    cache = DatasetCache(root=tmp_path)
    payload = {"x": [1, 2, 3], "y": 7}
    cache.put("abc", payload)
    assert cache.has("abc")
    loaded = cache.get("abc")
    assert loaded == payload


def test_happy_cache_get_or_compute(tmp_path: Path) -> None:
    """get_or_compute calls the supplied callable only on a miss."""
    cache = DatasetCache(root=tmp_path)
    calls = {"n": 0}

    def compute() -> object:
        calls["n"] += 1
        return 42

    first = cache.get_or_compute("k", compute)
    second = cache.get_or_compute("k", compute)
    assert first == 42 and second == 42
    assert calls["n"] == 1


def test_bad_cache_get_missing_key(tmp_path: Path) -> None:
    """get() raises DataError on a missing key."""
    cache = DatasetCache(root=tmp_path)
    with pytest.raises(DataError):
        cache.get("does-not-exist")


def test_distributional_cache_key_distinct() -> None:
    """Distinct input combinations produce distinct cache keys."""
    assert cache_key([1, 2, 3]) != cache_key([1, 2, 4])
    assert cache_key(["a", "b"]) != cache_key(["a", "c"])


def test_round_trip_cache_survives_clear(tmp_path: Path) -> None:
    """Eviction removes the cached entry."""
    cache = DatasetCache(root=tmp_path)
    cache.put("xx", 7)
    assert cache.has("xx")
    cache.evict("xx")
    assert not cache.has("xx")


# ============================== MICROBENCHMARK ==============================


def test_happy_microbenchmark_run() -> None:
    """Microbenchmark produces a summary with positive mean time."""
    bench = Microbenchmark(name="noop", n_warmup=2, n_iter=5)

    def fn() -> None:
        x = torch.randn((8, 8))
        _ = x @ x

    result = bench.run(fn)
    assert isinstance(result, MicrobenchmarkResult)
    assert result.n_iter == 5
    assert result.mean_s > 0.0
    assert result.min_s <= result.mean_s <= result.max_s + 1e-9


def test_happy_compare_benchmarks_speedup() -> None:
    """compare_benchmarks returns a speedup ratio greater than one when faster."""
    baseline = MicrobenchmarkResult(
        name="slow", n_iter=2, mean_s=1e-2, std_s=0.0, min_s=1e-2, max_s=1e-2
    )
    candidate = MicrobenchmarkResult(
        name="fast", n_iter=2, mean_s=5e-3, std_s=0.0, min_s=5e-3, max_s=5e-3
    )
    delta = compare_benchmarks(baseline, candidate)
    assert delta["speedup"] == pytest.approx(2.0)
