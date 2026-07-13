"""Microbenchmark suite.

Lightweight benchmarking utilities for measuring the wall-clock cost
of small operations. The primary entry point is
:class:`Microbenchmark` which records the time spent in repeated
invocations of a callable and reports summary statistics.

The reports are honest: every measurement is over ``n_iter`` runs
after ``n_warmup`` warm-up iterations to amortise lazy
initialisation (JIT caches, ``torch.compile`` graphs, etc.).

## Complexity

Per measurement the wall-clock cost is dominated by ``fn()``. The
benchmark overhead is ``O(n_warmup + n_iter)`` calls to
:func:`time.perf_counter`; with ``n_iter=10`` this is sub-millisecond.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Microbenchmark",
    "MicrobenchmarkResult",
    "compare_benchmarks",
    "time_once",
]


@dataclass(frozen=True)
class MicrobenchmarkResult:
    """Summary statistics from a benchmark run.

    Attributes:
        name: The benchmark name (mirrors ``Microbenchmark.name``).
        n_iter: The number of timed iterations.
        mean_s: Mean of the timed iterations (seconds).
        std_s: Population standard deviation (seconds).
        min_s: Minimum (seconds).
        max_s: Maximum (seconds).
    """

    name: str
    n_iter: int
    mean_s: float
    std_s: float
    min_s: float
    max_s: float

    def as_dict(self) -> dict[str, float | int | str]:
        """Return the result as a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "n_iter": self.n_iter,
            "mean_s": self.mean_s,
            "std_s": self.std_s,
            "min_s": self.min_s,
            "max_s": self.max_s,
        }


@dataclass
class Microbenchmark:
    """Run repeated timing measurements of a zero-argument callable.

    Attributes:
        name: A human-readable name used in reports.
        n_warmup: Warm-up iterations discarded before measurement.
        n_iter: Timed iterations.
    """

    name: str = "operation"
    n_warmup: int = 3
    n_iter: int = 10
    samples: list[float] = field(default_factory=list, init=False)

    def run(self, fn: Callable[[], Any]) -> MicrobenchmarkResult:
        """Run the benchmark and return the summary statistics.

        Args:
            fn: A zero-argument callable to time.

        Returns:
            A populated :class:`MicrobenchmarkResult`. The raw
            per-iteration samples are stored on the instance and
            accessible via :attr:`samples`.
        """
        for _ in range(self.n_warmup):
            time_once(fn)
        samples = [time_once(fn) for _ in range(self.n_iter)]

        mean = statistics.fmean(samples)
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        result = MicrobenchmarkResult(
            name=self.name,
            n_iter=self.n_iter,
            mean_s=mean,
            std_s=std,
            min_s=min(samples),
            max_s=max(samples),
        )
        self.samples = samples
        return result

    def latest_samples(self) -> list[float]:
        """Return a copy of the raw per-iteration timings from the most recent run."""
        return list(self.samples)


def time_once(fn: Callable[[], Any]) -> float:
    """Time a single invocation of ``fn`` in seconds.

    Exposed publicly so callers building their own benchmark can
    reuse the exact timing helper :class:`Microbenchmark`
    internally uses.

    Args:
        fn: The callable.

    Returns:
        The wall-clock duration in seconds.
    """
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def compare_benchmarks(
    baseline: MicrobenchmarkResult,
    candidate: MicrobenchmarkResult,
) -> dict[str, float]:
    """Compute the speedup of ``candidate`` over ``baseline``.

    Args:
        baseline: The reference benchmark result.
        candidate: The new benchmark result.

    Returns:
        A dict with ``speedup``, ``delta_mean_s``, and the two
        means. ``speedup`` is ``+inf`` when ``baseline.mean_s`` is
        zero (caller should treat that as "candidate is infinitely
        faster", typically a degenerate measurement).
    """
    if baseline.mean_s == 0.0:
        speedup = float("inf")
    else:
        speedup = baseline.mean_s / candidate.mean_s
    return {
        "baseline_mean_s": baseline.mean_s,
        "candidate_mean_s": candidate.mean_s,
        "delta_mean_s": candidate.mean_s - baseline.mean_s,
        "speedup": speedup,
    }
