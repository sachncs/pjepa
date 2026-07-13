"""Performance infrastructure for the pjepa package.

This subpackage provides capability-aware performance primitives:

* :class:`EMATarget` — BYOL-style exponential moving average for
  target encoders.
* :func:`safe_compile` — backend-aware ``torch.compile`` wrapper
  with graceful fallback.
* :func:`autocast_context` — backend-aware mixed-precision context
  manager.
* :func:`fused_scatter_add` — fused ``scatter_add_`` with a CPU fallback
  for backends where the native kernel is slow.
* :func:`sync_mps` — explicit MPS synchronisation helper.
* :class:`DatasetCache` — disk-backed dataset cache.
* :class:`Microbenchmark` — microbenchmark utilities.
* :func:`shard_state_dict` / :func:`load_sharded_state_dict` —
  checkpoint sharding for memory-bound runs (Phase 10).
* :func:`assert_rss_cap` — practical RSS ceiling for OGB-scale runs.
"""

from __future__ import annotations

from pjepa.perf.autocast import autocast_context
from pjepa.perf.benchmark import (
    Microbenchmark,
    MicrobenchmarkResult,
    compare_benchmarks,
)
from pjepa.perf.cache import DatasetCache, cache_key, memmap_array
from pjepa.perf.compile import safe_compile
from pjepa.perf.ema import EMATarget
from pjepa.perf.scatter import fused_scatter_add, fused_scatter_mean
from pjepa.perf.sharding import (
    ShardedCheckpoint,
    assert_rss_cap,
    current_rss_mb,
    load_sharded_state_dict,
    shard_state_dict,
)
from pjepa.perf.sync import sync_mps

__all__ = [
    "DatasetCache",
    "EMATarget",
    "Microbenchmark",
    "MicrobenchmarkResult",
    "ShardedCheckpoint",
    "assert_rss_cap",
    "autocast_context",
    "cache_key",
    "compare_benchmarks",
    "current_rss_mb",
    "fused_scatter_add",
    "fused_scatter_mean",
    "load_sharded_state_dict",
    "memmap_array",
    "safe_compile",
    "shard_state_dict",
    "sync_mps",
]
