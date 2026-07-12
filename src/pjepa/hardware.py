"""Hardware detection and capability reporting.

The library runs on Apple Silicon (MPS), NVIDIA (CUDA), and CPU backends.
A :class:`CapabilityReport` is built at start-up so that downstream
modules can adapt their behaviour (for example, disabling
``torch.compile`` on MPS where some operators do not fuse cleanly).
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import torch

from pjepa.exceptions import BackendError

__all__ = [
    "Backend",
    "ProbeStatus",
    "ProbeResult",
    "CapabilityReport",
    "detect_backend",
    "detect_capabilities",
    "current_device",
    "sync_if_mps",
]


class Backend(str, Enum):
    """Compute backends supported by the library."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class ProbeStatus(str, Enum):
    """Outcome of a single capability probe."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True)
class ProbeResult:
    """A single capability probe and its outcome."""

    name: str
    status: ProbeStatus
    detail: str = ""

    def render(self) -> str:
        """Return a human-readable one-line rendering of the result."""
        suffix = f" — {self.detail}" if self.detail else ""
        return f"[{self.status.value:<6}] {self.name}{suffix}"


@dataclass(frozen=True)
class CapabilityReport:
    """Aggregated capability report for the current host."""

    backend: Backend
    device_name: str
    python_version: str
    torch_version: str
    platform: str
    cpu_count: int
    probes: tuple[ProbeResult, ...] = field(default_factory=tuple)

    def is_green(self) -> bool:
        """Return ``True`` when every probe is GREEN."""
        return all(p.status is ProbeStatus.GREEN for p in self.probes)

    def has_red(self) -> bool:
        """Return ``True`` when at least one probe is RED."""
        return any(p.status is ProbeStatus.RED for p in self.probes)

    def render(self) -> str:
        """Return a multi-line rendering of the full report."""
        lines = [
            f"Backend:    {self.backend.value}",
            f"Device:     {self.device_name}",
            f"Python:     {self.python_version}",
            f"PyTorch:    {self.torch_version}",
            f"Platform:   {self.platform}",
            f"CPU count:  {self.cpu_count}",
            "",
            "Capability probes:",
        ]
        lines.extend(f"  {probe.render()}" for probe in self.probes)
        return "\n".join(lines)


def detect_backend() -> Backend:
    """Return the most capable backend available on the current host.

    Preference order: CUDA, MPS, CPU. A :class:`BackendError` is never
    raised: when no accelerator is available, the function returns
    :attr:`Backend.CPU`.

    Returns:
        The selected backend.

    Example:
        >>> detect_backend().value
        'mps'
    """
    if torch.cuda.is_available():
        return Backend.CUDA
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return Backend.MPS
    return Backend.CPU


def current_device(backend: Backend | None = None) -> torch.device:
    """Return the default :class:`torch.device` for the active backend.

    Args:
        backend: Explicit backend to use; if ``None``, the result of
          :func:`detect_backend` is used.

    Returns:
        The corresponding PyTorch device object.

    Raises:
        BackendError: If ``backend`` is CUDA but no CUDA device is
          available.

    Example:
        >>> current_device().type
        'mps'
    """
    active = backend or detect_backend()
    if active is Backend.CUDA and not torch.cuda.is_available():
        raise BackendError("current_device: CUDA requested but unavailable")
    if active is Backend.MPS and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise BackendError("current_device: MPS requested but unavailable")
    return torch.device(active.value)


def sync_if_mps() -> None:
    """Block until all pending MPS operations have completed.

    Reads of MPS tensors from CPU code implicitly synchronise, which
    can introduce hidden stalls. Callers that read MPS tensors in
    performance-sensitive code should invoke this helper explicitly.

    Returns:
        None.

    Example:
        >>> sync_if_mps()
    """
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        torch.mps.synchronize()


def _probe_matmul() -> ProbeResult:
    """Probe matrix multiplication on the active backend."""
    backend = detect_backend()
    try:
        device = current_device(backend)
        a = torch.randn((64, 64), device=device)
        b = torch.randn((64, 64), device=device)
        c = a @ b
        if backend is Backend.MPS:
            sync_if_mps()
        if not torch.isfinite(c).all():
            return ProbeResult(
                name="matmul",
                status=ProbeStatus.RED,
                detail="produced non-finite values",
            )
        return ProbeResult(name="matmul", status=ProbeStatus.GREEN)
    except Exception as exc:
        return ProbeResult(
            name="matmul",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _probe_scatter_add() -> ProbeResult:
    """Probe ``scatter_add_`` on the active backend."""
    backend = detect_backend()
    try:
        device = current_device(backend)
        idx = torch.tensor([0, 1, 0, 1, 2], device=device)
        src = torch.ones((5,), device=device)
        out = torch.zeros((3,), device=device)
        out.scatter_add_(0, idx, src)
        if backend is Backend.MPS:
            sync_if_mps()
        if not torch.allclose(out, torch.tensor([2.0, 2.0, 1.0], device=device)):
            return ProbeResult(
                name="scatter_add",
                status=ProbeStatus.RED,
                detail="unexpected result",
            )
        return ProbeResult(name="scatter_add", status=ProbeStatus.GREEN)
    except Exception as exc:
        return ProbeResult(
            name="scatter_add",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _probe_compile() -> ProbeResult:
    """Probe ``torch.compile`` on a small module."""
    backend = detect_backend()
    if backend is Backend.CPU and not os.environ.get("PJEPA_TRY_CPU_COMPILE"):
        return ProbeResult(
            name="torch.compile",
            status=ProbeStatus.YELLOW,
            detail="skipped on CPU; set PJEPA_TRY_CPU_COMPILE=1 to force",
        )
    try:
        def square(x: torch.Tensor) -> torch.Tensor:
            return x * x

        compiled = torch.compile(square)
        x = torch.randn((8,), device=current_device(backend))
        y = compiled(x)
        if backend is Backend.MPS:
            sync_if_mps()
        if not torch.allclose(y, x * x):
            return ProbeResult(
                name="torch.compile",
                status=ProbeStatus.RED,
                detail="compiled output differs",
            )
        return ProbeResult(name="torch.compile", status=ProbeStatus.GREEN)
    except Exception as exc:
        return ProbeResult(
            name="torch.compile",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _probe_hyperbolic() -> ProbeResult:
    """Probe Poincare ball operations via geoopt."""
    backend = detect_backend()
    try:
        import geoopt  # type: ignore[import-not-found]

        ball = geoopt.PoincareBallExact()
        x = ball.random((4, 3)).to(current_device(backend))
        if not torch.all(x.norm(dim=-1) < 1.0):
            return ProbeResult(
                name="hyperbolic",
                status=ProbeStatus.RED,
                detail="point outside ball",
            )
        return ProbeResult(name="hyperbolic", status=ProbeStatus.GREEN)
    except ImportError:
        return ProbeResult(
            name="hyperbolic",
            status=ProbeStatus.YELLOW,
            detail="geoopt not installed",
        )
    except Exception as exc:
        return ProbeResult(
            name="hyperbolic",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _probe_pyg_scatter() -> ProbeResult:
    """Probe PyTorch-Geometric scatter operations."""
    backend = detect_backend()
    try:
        import torch_geometric  # type: ignore[import-not-found]  # noqa: F401

        from torch_geometric.utils import scatter  # type: ignore[import-not-found]

        idx = torch.tensor([0, 1, 0, 1], device=current_device(backend))
        src = torch.ones((4, 2), device=current_device(backend))
        out = scatter(src, idx, dim=0, dim_size=2, reduce="sum")
        if backend is Backend.MPS:
            sync_if_mps()
        if out.shape != (2, 2):
            return ProbeResult(
                name="pyg_scatter",
                status=ProbeStatus.RED,
                detail=f"unexpected shape {tuple(out.shape)}",
            )
        return ProbeResult(name="pyg_scatter", status=ProbeStatus.GREEN)
    except ImportError:
        return ProbeResult(
            name="pyg_scatter",
            status=ProbeStatus.YELLOW,
            detail="torch_geometric not installed",
        )
    except Exception as exc:
        return ProbeResult(
            name="pyg_scatter",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _probe_cpu_fallback() -> ProbeResult:
    """CPU fallback is always available."""
    return ProbeResult(name="cpu_fallback", status=ProbeStatus.GREEN)


def detect_capabilities() -> CapabilityReport:
    """Build a :class:`CapabilityReport` for the current host.

    Each probe exercises a small operation and reports GREEN, YELLOW, or
    RED with a one-line detail. The function is cheap (under a second
    on a typical laptop) and is intended to be called at the start of
    every interactive session.

    Returns:
        The populated capability report.

    Example:
        >>> report = detect_capabilities()
        >>> report.has_red()
        False
    """
    backend = detect_backend()
    device_name = _device_name(backend)
    probes = (
        _probe_matmul(),
        _probe_scatter_add(),
        _probe_compile(),
        _probe_hyperbolic(),
        _probe_pyg_scatter(),
        _probe_cpu_fallback(),
    )
    return CapabilityReport(
        backend=backend,
        device_name=device_name,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        probes=probes,
    )


def _device_name(backend: Backend) -> str:
    """Return a human-readable device name for the backend."""
    if backend is Backend.CUDA and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if backend is Backend.MPS:
        return "Apple Silicon (MPS)"
    return "CPU"


def capabilities_as_dict(report: CapabilityReport) -> Mapping[str, str]:
    """Convert a :class:`CapabilityReport` to a JSON-friendly mapping."""
    return {
        "backend": report.backend.value,
        "device_name": report.device_name,
        "python_version": report.python_version,
        "torch_version": report.torch_version,
        "platform": report.platform,
        "cpu_count": str(report.cpu_count),
        "probes": {probe.name: probe.status.value for probe in report.probes},
    }