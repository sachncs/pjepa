"""Checkpoint sharding and RSS (resident set size) helpers.

Phase 10 trains and evaluates on OGB-arxiv (169K nodes, ~1.1M
edges), which pushes memory into the "careful" regime. Two
utilities are exposed here:

* :func:`shard_state_dict` / :func:`load_sharded_state_dict`
  split a large :class:`torch.Tensor` state dict into
  fixed-byte chunks on disk; :func:`load_sharded_state_dict`
  reverses the operation. The shards are portable (plain
  ``torch.save``) and survive process restarts.
* :func:`current_rss_mb` returns the resident-set size of the
  current process in MiB; :func:`assert_rss_cap` raises
  :class:`BackendError` when the RSS exceeds the configured
  ceiling.

## Architecture

```
   state_dict ──► shard_state_dict ──► shard_00000.pt
                                      shard_00001.pt
                                      manifest.json

   shard_*.pt + manifest.json ──► load_sharded_state_dict ──► state_dict
```

## Complexity

* :func:`shard_state_dict` — ``O(P)`` over ``P`` named tensors.
* :func:`load_sharded_state_dict` — ``O(P)`` tensor loads.
* :func:`current_rss_mb` — ``O(1)`` system call.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from pjepa.exceptions import BackendError, CheckpointError

__all__ = [
    "ShardedCheckpoint",
    "assert_rss_cap",
    "current_rss_mb",
    "list_shards",
    "load_sharded_state_dict",
    "read_rss_bytes",
    "shard_state_dict",
    "tensor_size_bytes",
]


@dataclass(frozen=True)
class ShardedCheckpoint:
    """Description of a sharded checkpoint on disk.

    Attributes:
        directory: Directory containing ``manifest.json`` and the
          ``shard_*.pt`` files.
        manifest: A serialised mapping of tensor name →
          ``{shard, size_bytes, shape, dtype}``.
        shard_size_bytes: Maximum size of each shard in bytes.
    """

    directory: Path
    manifest: dict[str, dict[str, object]]
    shard_size_bytes: int


def read_rss_bytes() -> int:
    """Return the resident-set size of the current process in bytes.

    Uses :mod:`psutil` when available; falls back to
    :func:`resource.getrusage` on POSIX. The fallback scales the
    reported kilobyte count by ``1024`` when ``ru_maxrss`` is small
    (most Linux systems) and by ``1`` when the value is suspiciously
    large (macOS and BSD, which report bytes directly).

    Returns:
        RSS in bytes. ``0`` when neither backend is available.
    """
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return int(usage.ru_maxrss) * (1024 if usage.ru_maxrss < 10**12 else 1)
        except ImportError:
            return 0


def current_rss_mb() -> float:
    """Return the resident-set size of the current process in MiB."""
    return read_rss_bytes() / (1024.0 * 1024.0)


def assert_rss_cap(cap_mb: float) -> float:
    """Raise :class:`BackendError` if the RSS exceeds ``cap_mb``.

    Args:
        cap_mb: The maximum allowed RSS in MiB.

    Returns:
        The current RSS in MiB.

    Raises:
        BackendError: When the current RSS exceeds ``cap_mb`` or
          when the platform cannot report RSS.
    """
    if cap_mb <= 0:
        raise BackendError(f"assert_rss_cap: cap must be positive; got {cap_mb}")
    rss = current_rss_mb()
    if rss <= 0.0:
        raise BackendError(
            "assert_rss_cap: cannot read RSS on this platform "
            "(neither psutil nor resource.getrusage is available)"
        )
    if rss > cap_mb:
        raise BackendError(f"assert_rss_cap: RSS {rss:.1f} MiB exceeds cap {cap_mb:.1f} MiB")
    return rss


def tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Return the byte size of ``tensor`` after dtype conversion."""
    return int(tensor.numel() * tensor.element_size())


def shard_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    directory: str | os.PathLike[str],
    max_shard_bytes: int = 100 * 1024 * 1024,
    metadata: Mapping[str, object] | None = None,
) -> ShardedCheckpoint:
    """Split ``state_dict`` into multiple shard files under ``directory``.

    Each tensor is byte-packed into a contiguous buffer; tensors
    are placed greedily into shards whose total byte size does
    not exceed ``max_shard_bytes``. A ``manifest.json`` records
    the per-tensor location so
    :func:`load_sharded_state_dict` can reassemble the state.

    Args:
        state_dict: The mapping of tensor name → tensor.
        directory: Target directory; created if missing.
        max_shard_bytes: Maximum bytes per shard. Default is
          ``100 MiB``, matching the Phase-10 plan.
        metadata: Optional serialisable metadata to embed in the
          manifest.

    Returns:
        A :class:`ShardedCheckpoint` describing the on-disk layout.

    Raises:
        CheckpointError: If ``max_shard_bytes`` is non-positive or
          the directory cannot be created, or if a single tensor
          exceeds the per-shard budget.
    """
    if max_shard_bytes <= 0:
        raise CheckpointError(
            f"shard_state_dict: max_shard_bytes must be positive; got {max_shard_bytes}"
        )
    target = Path(directory)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CheckpointError(f"shard_state_dict: cannot create {target}: {exc}") from exc

    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes: list[int] = [0]
    manifest: dict[str, dict[str, object]] = {}

    for name, tensor in state_dict.items():
        contiguous = tensor.detach().contiguous().cpu()
        size_bytes = tensor_size_bytes(contiguous)
        if size_bytes > max_shard_bytes:
            raise CheckpointError(
                f"shard_state_dict: tensor {name!r} ({size_bytes} bytes) exceeds "
                f"max_shard_bytes={max_shard_bytes}"
            )
        if sizes[-1] + size_bytes > max_shard_bytes and shards[-1]:
            shards.append({})
            sizes.append(0)
        shard_idx = len(shards) - 1
        shards[shard_idx][name] = contiguous
        sizes[-1] += size_bytes
        manifest[name] = {
            "shard": shard_idx,
            "size_bytes": size_bytes,
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
        }

    for idx, payload in enumerate(shards):
        if not payload:
            continue
        torch.save(payload, target / f"shard_{idx:05d}.pt")

    full_manifest = {
        "manifest": manifest,
        "metadata": dict(metadata) if metadata else {},
        "n_shards": len(shards),
        "shard_size_bytes": int(max_shard_bytes),
    }
    (target / "manifest.json").write_text(json.dumps(full_manifest, indent=2), encoding="utf-8")
    return ShardedCheckpoint(
        directory=target,
        manifest=manifest,
        shard_size_bytes=int(max_shard_bytes),
    )


def load_sharded_state_dict(
    directory: str | os.PathLike[str],
    map_location: object | None = None,
) -> dict[str, torch.Tensor]:
    """Load a sharded state dict previously written by :func:`shard_state_dict`.

    Args:
        directory: Directory holding ``manifest.json`` and
          ``shard_*.pt`` files.
        map_location: Optional ``map_location`` passed through to
          :func:`torch.load`.

    Returns:
        A ``{tensor_name: tensor}`` mapping identical to the input
        of :func:`shard_state_dict`.

    Raises:
        CheckpointError: If the directory is missing or malformed.
    """
    target = Path(directory)
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise CheckpointError(f"load_sharded_state_dict: manifest not found at {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest", {})
    if not isinstance(manifest, dict):
        raise CheckpointError("load_sharded_state_dict: manifest is malformed")

    shard_cache: dict[int, dict[str, torch.Tensor]] = {}
    result: dict[str, torch.Tensor] = {}
    for name, info in manifest.items():
        shard_idx = int(info["shard"])
        if shard_idx not in shard_cache:
            shard_file = target / f"shard_{shard_idx:05d}.pt"
            if not shard_file.is_file():
                raise CheckpointError(f"load_sharded_state_dict: missing shard file {shard_file}")
            shard_cache[shard_idx] = torch.load(
                shard_file, map_location=map_location, weights_only=True
            )
        tensor = shard_cache[shard_idx].get(name)
        if tensor is None:
            raise CheckpointError(
                f"load_sharded_state_dict: tensor {name!r} not found in shard {shard_idx}"
            )
        result[name] = tensor
    return result


def list_shards(directory: str | os.PathLike[str]) -> list[Path]:
    """Return a sorted list of shard file paths under ``directory``.

    Args:
        directory: Candidate directory.

    Returns:
        Sorted list of shard paths (sorted by index). Empty list
        when ``directory`` is not a directory.
    """
    target = Path(directory)
    if not target.is_dir():
        return []
    return sorted(target.glob("shard_*.pt"))
