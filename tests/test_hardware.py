"""Tests for pjepa.hardware.

Covers the eight-class test taxonomy.
"""

from __future__ import annotations

import pytest
import torch

from pjepa.exceptions import BackendError
from pjepa.hardware import (
    Backend,
    CapabilityReport,
    ProbeResult,
    ProbeStatus,
    capabilities_as_dict,
    current_device,
    detect_backend,
    detect_capabilities,
    sync_if_mps,
)

__all__ = [
    "test_bad_device_raises_when_cuda_missing",
    "test_cross_backend_mps_matmul",
    "test_distributional_probe_status_values",
    "test_happy_backend_detected",
    "test_property_cpu_fallback_always_green",
    "test_round_trip_capabilities_as_dict",
    "test_ugly_empty_caps_report",
]


def test_happy_backend_detected() -> None:
    """detect_backend returns a known enum value."""
    backend = detect_backend()
    assert backend in (Backend.CUDA, Backend.MPS, Backend.CPU)


def test_bad_device_raises_when_cuda_missing() -> None:
    """Requesting CUDA when unavailable raises BackendError."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    with pytest.raises(BackendError):
        current_device(Backend.CUDA)


def test_ugly_empty_caps_report() -> None:
    """A CapabilityReport with no probes is not green; no probes is not 'all pass'."""
    report = CapabilityReport(
        backend=Backend.CPU,
        device_name="test",
        python_version="3.12",
        torch_version="2.13",
        platform="test",
        cpu_count=4,
        probes=(),
    )
    assert report.is_green() is False
    assert report.has_red() is False


def test_round_trip_capabilities_as_dict() -> None:
    """capabilities_as_dict yields a JSON-friendly mapping."""
    report = detect_capabilities()
    mapping = capabilities_as_dict(report)
    assert mapping["backend"] in {"cuda", "mps", "cpu"}
    assert "matmul" in mapping["probes"]


def test_cross_backend_mps_matmul() -> None:
    """Probe matrix multiplication on MPS and verify shape."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    a = torch.randn((16, 16), device="mps")
    b = torch.randn((16, 16), device="mps")
    c = a @ b
    sync_if_mps()
    assert c.shape == (16, 16)


def test_distributional_probe_status_values() -> None:
    """Every probe result has a valid ProbeStatus enum value."""
    report = detect_capabilities()
    for probe in report.probes:
        assert isinstance(probe, ProbeResult)
        assert probe.status in {ProbeStatus.GREEN, ProbeStatus.YELLOW, ProbeStatus.RED}


def test_property_cpu_fallback_always_green() -> None:
    """The cpu_fallback probe is always GREEN."""
    report = detect_capabilities()
    cpu = next(p for p in report.probes if p.name == "cpu_fallback")
    assert cpu.status is ProbeStatus.GREEN


def test_property_render_contains_backend() -> None:
    """Rendering a report includes the backend line."""
    report = detect_capabilities()
    rendered = report.render()
    assert "Backend:" in rendered
    assert "Capability probes:" in rendered


def test_sync_if_mps_is_noop_off_mps() -> None:
    """sync_if_mps is a no-op when MPS is unavailable."""
    sync_if_mps()  # Must not raise.
