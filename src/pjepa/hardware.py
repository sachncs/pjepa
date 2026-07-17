"""Hardware detection and capability reporting.

The library runs on Apple Silicon (MPS), NVIDIA (CUDA), and CPU
backends. A :class:`CapabilityReport` is built at start-up so that
downstream modules can adapt their behaviour — for example, disabling
:func:`torch.compile` on MPS where some operators do not fuse cleanly.

All public functions and probe helpers are pure with respect to
module globals apart from the implicit dependency on the active
PyTorch / CUDA / MPS runtimes. Probes allocate small tensors on the
target device; calling them concurrently from multiple threads on the
same CUDA device is **not** safe.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import torch

from pjepa.exceptions import BackendError

__all__ = [
    "Backend",
    "CapabilityReport",
    "ProbeResult",
    "ProbeStatus",
    "capabilities_as_dict",
    "current_device",
    "detect_backend",
    "detect_capabilities",
    "sync_if_mps",
]


class Backend(str, Enum):
    """Compute backends supported by the library.

    Values are lowercase strings so they serialise naturally to JSON
    and survive ``str.lower`` round-trips through command-line
    arguments.
    """

    CUDA = "cuda"
    """NVIDIA CUDA backend."""
    MPS = "mps"
    """Apple Silicon MPS backend."""
    CPU = "cpu"
    """CPU fallback — always available."""


class ProbeStatus(str, Enum):
    """Outcome of a single capability probe.

    The three values form an ordered quality scale:

    * :data:`GREEN` — the probe ran successfully.
    * :data:`YELLOW` — the probe was skipped or ran in a degraded mode.
    * :data:`RED` — the probe failed.
    """

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True)
class ProbeResult:
    """A single capability probe and its outcome.

    Attributes:
        name: Short identifier suitable for indexing in
            :func:`capabilities_as_dict` (e.g. ``"matmul"``).
        status: The outcome of the probe.
        detail: Free-form description, typically the exception type
            and message for :data:`ProbeStatus.RED`. May be empty.

    Example:
        >>> ProbeResult(name="matmul", status=ProbeStatus.GREEN).render()
        '[GREEN ] matmul'
    """

    name: str
    status: ProbeStatus
    detail: str = ""

    def render(self) -> str:
        """Return a human-readable one-line rendering of the result.

        The status column is left-padded to a fixed width so multiple
        ``ProbeResult`` lines line up neatly when concatenated.
        """
        suffix = f" — {self.detail}" if self.detail else ""
        return f"[{self.status.value:<6}] {self.name}{suffix}"


@dataclass(frozen=True)
class CapabilityReport:
    """Aggregated capability report for the current host.

    The report is immutable; passing it around is cheap because every
    field is either a scalar, a string, or a tuple.

    Attributes:
        backend: The selected :class:`Backend`.
        device_name: A human-readable device name (``"Apple Silicon
            (MPS)"`` for MPS, ``"CPU"`` for CPU, the CUDA device name
            for CUDA).
        python_version: The interpreter version as reported by
            ``sys.version.split()[0]``.
        torch_version: ``torch.__version__``.
        platform: The string returned by :func:`platform.platform`.
        cpu_count: The number of visible CPUs as reported by
            :func:`os.cpu_count`, falling back to ``1``.
        probes: Tuple of :class:`ProbeResult` produced by
            :func:`detect_capabilities`.

    Example:
        >>> report = detect_capabilities()
        >>> report.has_red()
        False
    """

    backend: Backend
    device_name: str
    python_version: str
    torch_version: str
    platform: str
    cpu_count: int
    probes: tuple[ProbeResult, ...] = field(default_factory=tuple)

    def is_green(self) -> bool:
        """Return ``True`` when every probe is GREEN.

        An empty probe list returns ``False`` because "no probes run"
        is not the same as "every probe passed". Callers that want to
        treat a report as uninformative should check ``len(probes)``
        before calling :meth:`is_green`.
        """
        return len(self.probes) > 0 and all(
            p.status is ProbeStatus.GREEN for p in self.probes
        )

    def has_red(self) -> bool:
        """Return ``True`` when at least one probe is RED."""
        return any(p.status is ProbeStatus.RED for p in self.probes)

    def render(self) -> str:
        """Return a multi-line rendering of the full report.

        The output is suitable for direct printing at the start of
        every interactive session.
        """
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

    Preference order: CUDA, MPS, CPU. The function never raises
    :class:`BackendError`; when no accelerator is available, it
    returns :attr:`Backend.CPU`.

    The detection is performed by inspecting the global PyTorch
    runtime; importing this module performs no warm-up. The result
    reflects the runtime state at call time and may change between
    calls if the user toggles ``CUDA_VISIBLE_DEVICES`` or similar.

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
        backend: Explicit backend to use; ``None`` selects the result
            of :func:`detect_backend`.

    Returns:
        The corresponding PyTorch device object.

    Raises:
        BackendError: If ``backend`` is :attr:`Backend.CUDA` but no
            CUDA device is available, or :attr:`Backend.MPS` is
            requested on a host without an Apple Silicon runtime.

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


def probe_matmul() -> ProbeResult:
    """Probe matrix multiplication on the active backend.

    Returns:
        A :class:`ProbeResult` whose status is :data:`ProbeStatus.GREEN`
        when the matmul completes and produces finite values; otherwise
        a RED result with an explanatory detail string.
    """
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
    except (RuntimeError, NotImplementedError, ValueError, TypeError) as exc:
        return ProbeResult(
            name="matmul",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def probe_scatter_add() -> ProbeResult:
    """Probe ``Tensor.scatter_add_`` on the active backend."""
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
    except (RuntimeError, NotImplementedError, ValueError, TypeError) as exc:
        return ProbeResult(
            name="scatter_add",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def probe_compile() -> ProbeResult:
    """Probe ``torch.compile`` on a tiny square module.

    On CPU the probe is skipped by default to keep startup fast; set
    ``PJEPA_TRY_CPU_COMPILE=1`` in the environment to force it.
    """
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
    except (RuntimeError, NotImplementedError, ValueError, TypeError) as exc:
        return ProbeResult(
            name="torch.compile",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def probe_hyperbolic() -> ProbeResult:
    """Probe Poincare ball operations via ``geoopt``.

    Returns:
        A :class:`ProbeResult` that is GREEN when random points
        sampled on the ball satisfy the unit-radius constraint and
        RED otherwise.
    """
    backend = detect_backend()
    try:
        import geoopt

        ball = geoopt.PoincareBallExact()
        x = ball.random((4, 3)).to(current_device(backend))
        if not torch.all(x.norm(dim=-1) < 1.0):
            return ProbeResult(
                name="hyperbolic",
                status=ProbeStatus.RED,
                detail="point outside ball",
            )
        return ProbeResult(name="hyperbolic", status=ProbeStatus.GREEN)
    except (RuntimeError, NotImplementedError, ValueError, TypeError) as exc:
        return ProbeResult(
            name="hyperbolic",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def probe_pyg_scatter() -> ProbeResult:
    """Probe PyTorch-Geometric ``scatter`` on the active backend.

    Returns:
        GREEN when the scatter yields the expected shape, RED otherwise.
    """
    backend = detect_backend()
    try:
        from torch_geometric.utils import scatter

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
    except (RuntimeError, NotImplementedError, ValueError, TypeError) as exc:
        return ProbeResult(
            name="pyg_scatter",
            status=ProbeStatus.RED,
            detail=f"{type(exc).__name__}: {exc}",
        )


def probe_cpu_fallback() -> ProbeResult:
    """CPU fallback is always available."""
    return ProbeResult(name="cpu_fallback", status=ProbeStatus.GREEN)


def detect_capabilities() -> CapabilityReport:
    """Build a :class:`CapabilityReport` for the current host.

    Each probe exercises a small operation and reports GREEN, YELLOW,
    or RED with a one-line detail. The total cost is bounded by the
    slowest probe (typically under a second on a laptop). The
    function is intended to be called at the start of every
    interactive session, not on the hot path.

    Returns:
        The populated capability report.

    Example:
        >>> report = detect_capabilities()
        >>> report.has_red()
        False
    """
    backend = detect_backend()
    device_name = device_name_for(backend)
    probes = (
        probe_matmul(),
        probe_scatter_add(),
        probe_compile(),
        probe_hyperbolic(),
        probe_pyg_scatter(),
        probe_cpu_fallback(),
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


def device_name_for(backend: Backend) -> str:
    """Return a human-readable device name for the backend.

    Args:
        backend: The backend whose device name should be reported.

    Returns:
        For CUDA the value of ``torch.cuda.get_device_name(0)``; for
        MPS the literal ``"Apple Silicon (MPS)"``; for CPU the literal
        ``"CPU"``.
    """
    if backend is Backend.CUDA:
        return torch.cuda.get_device_name(0)
    if backend is Backend.MPS:
        return "Apple Silicon (MPS)"
    return "CPU"


def capabilities_as_dict(report: CapabilityReport) -> Mapping[str, object]:
    """Convert a :class:`CapabilityReport` to a JSON-friendly mapping.

    The ``cpu_count`` field is rendered as a string to preserve
    portability with JSON encoders that cannot disambiguate integers
    from booleans on parse. Every probe name becomes a key mapping to
    its status string.

    Args:
        report: The report to serialise.

    Returns:
        A mapping containing ``backend``, ``device_name``,
        ``python_version``, ``torch_version``, ``platform``,
        ``cpu_count``, and ``probes``.
    """
    return {
        "backend": report.backend.value,
        "device_name": report.device_name,
        "python_version": report.python_version,
        "torch_version": report.torch_version,
        "platform": report.platform,
        "cpu_count": str(report.cpu_count),
        "probes": {probe.name: probe.status.value for probe in report.probes},
    }
