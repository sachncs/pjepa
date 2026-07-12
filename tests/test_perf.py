"""Tests for pjepa.perf."""

from __future__ import annotations

import pytest
import torch

from pjepa.perf import (
    EMATarget,
    autocast_context,
    fused_scatter_add,
    fused_scatter_mean,
    safe_compile,
)
from pjepa.perf.sync import sync_mps

__all__ = [
    "test_bad_ema_bad_schedule",
    "test_bad_ema_negative_momentum",
    "test_bad_ema_zero_total_steps",
    "test_bad_fused_scatter_add_index_out_of_range",
    "test_cross_backend_safe_compile_cpu",
    "test_happy_autocast_disabled_is_noop",
    "test_happy_ema_cosine_schedule",
    "test_happy_ema_target_update",
    "test_happy_fused_scatter_add_basic",
    "test_happy_fused_scatter_mean_basic",
    "test_happy_safe_compile_runs",
    "test_happy_sync_mps_noop",
    "test_round_trip_ema_shadow_starts_as_copy",
    "test_ugly_ema_single_step",
]


def test_happy_safe_compile_runs() -> None:
    """safe_compile returns a module that can be invoked."""
    mod = torch.nn.Linear(4, 4)
    out = safe_compile(mod)
    assert out is not None


def test_happy_autocast_disabled_is_noop() -> None:
    """autocast_context(enabled=False) yields a no-op context manager."""
    with autocast_context(enabled=False):
        x = torch.randn((2, 2))
        assert x.dtype == torch.float32


def test_happy_fused_scatter_add_basic() -> None:
    """fused_scatter_add produces the same result as torch.scatter_add_."""
    out = torch.zeros((3, 2))
    index = torch.tensor([0, 1, 0, 1, 2])
    src = torch.ones((5, 2))
    out2 = out.clone()
    fused_scatter_add(out2, index, src)
    expected = torch.zeros((3, 2))
    index2 = index.view(5, 1).expand(5, 2)
    expected.scatter_add_(0, index2, src)
    assert torch.allclose(out2, expected)


def test_happy_fused_scatter_mean_basic() -> None:
    """fused_scatter_mean produces correct per-group means."""
    index = torch.tensor([0, 0, 1, 1, 1])
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    out = torch.zeros((2, 2))
    count = torch.zeros((2,))
    fused_scatter_mean(out, count, index, src)
    # Group 0: mean of [1,2] and [3,4] = [2, 3]
    # Group 1: mean of [5,6], [7,8], [9,10] = [7, 8]
    assert torch.allclose(out[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(out[1], torch.tensor([7.0, 8.0]))


def test_happy_ema_target_update() -> None:
    """EMA target update moves shadow toward online."""
    online = torch.nn.Linear(4, 4)
    ema = EMATarget(online, momentum=0.5)
    initial = ema.shadow.weight.clone()
    with torch.no_grad():
        online.weight.fill_(2.0)
    ema.update()
    assert not torch.allclose(ema.shadow.weight, initial)


def test_happy_ema_cosine_schedule() -> None:
    """Cosine schedule moves momentum toward final_momentum."""
    online = torch.nn.Linear(4, 4)
    ema = EMATarget(online, momentum=0.99, schedule="cosine", final_momentum=0.999, total_steps=10)
    # After several steps the momentum should be closer to 0.999.
    for _ in range(5):
        ema.update()
    assert ema.current_momentum() > 0.99


def test_happy_sync_mps_noop() -> None:
    """sync_mps is a safe no-op when MPS is unavailable or active."""
    sync_mps()


def test_bad_ema_negative_momentum() -> None:
    """Negative momentum is rejected."""
    with pytest.raises(Exception):
        EMATarget(torch.nn.Linear(4, 4), momentum=-0.1)


def test_bad_ema_bad_schedule() -> None:
    """An unknown schedule is rejected."""
    with pytest.raises(Exception):
        EMATarget(torch.nn.Linear(4, 4), schedule="bogus")


def test_bad_ema_zero_total_steps() -> None:
    """Zero total_steps is rejected for cosine schedule."""
    with pytest.raises(Exception):
        EMATarget(torch.nn.Linear(4, 4), schedule="cosine", total_steps=0)


def test_bad_fused_scatter_add_index_out_of_range() -> None:
    """An out-of-range index raises ValueError."""
    out = torch.zeros((3,))
    index = torch.tensor([0, 5])
    src = torch.ones((2,))
    with pytest.raises(ValueError):
        fused_scatter_add(out, index, src)


def test_ugly_ema_single_step() -> None:
    """A single EMA step does not raise."""
    online = torch.nn.Linear(4, 4)
    ema = EMATarget(online, momentum=0.5)
    ema.update()
    assert ema.step == 1


def test_round_trip_ema_shadow_starts_as_copy() -> None:
    """EMA shadow is an initial copy of the online parameters."""
    online = torch.nn.Linear(4, 4)
    with torch.no_grad():
        online.weight.fill_(3.5)
    ema = EMATarget(online, momentum=0.5)
    assert torch.allclose(ema.shadow.weight, online.weight)


def test_cross_backend_safe_compile_cpu() -> None:
    """safe_compile is callable on any backend."""
    mod = torch.nn.Linear(4, 4)
    out = safe_compile(mod)
    out(torch.randn((2, 4)))
