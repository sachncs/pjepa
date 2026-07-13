"""Memory-mapped dataset cache.

A simple content-addressed cache that stores pickled Python objects
on disk under a SHA-256 directory name. Repeated calls with the same
payload return the cached copy without recomputation.

For numerical arrays this module also provides :func:`memmap_array`
which memory-maps a ``.npy``-style file using :mod:`numpy.memmap`.

## Layout

```
   <root>/<key-prefix>/<key>/<file>
   e.g. ~/.cache/pjepa/ab/abcdef.../cache.pkl
```

The two-level prefix follows the convention of ``pip``'s wheel
cache (and others): each ``key`` is split into its first two hex
characters followed by the full key. This keeps any single
directory shallow (``O(1)`` entries per directory) so the
filesystem does not slow down on large caches.

## Complexity

* :func:`cache_key` — ``O(sum(len(p)))`` for the parts iterable.
* :meth:`DatasetCache.put` / :meth:`get` — ``O(|value|)`` to
  pickle / unpickle the object.
* :meth:`DatasetCache.clear` — ``O(N)`` to walk the cache.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from pjepa.exceptions import DataError

__all__ = ["DatasetCache", "cache_key", "memmap_array"]


def cache_key(parts: Iterable[object]) -> str:
    """Compute a deterministic cache key from an iterable of objects.

    Args:
        parts: Components that uniquely identify the cached payload.

    Returns:
        A hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def memmap_array(path: str | os.PathLike[str], shape: tuple[int, ...], dtype: str) -> np.memmap:
    """Memory-map a numerical array stored on disk.

    Args:
        path: Path to the (existing) raw binary file.
        shape: Shape of the array to map.
        dtype: NumPy-compatible dtype string.

    Returns:
        A writable :class:`numpy.memmap` view of the file.

    Raises:
        DataError: When ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataError(f"memmap_array: file {file_path} does not exist")
    return np.memmap(str(file_path), dtype=dtype, mode="r+", shape=shape)


class DatasetCache:
    """Disk-backed cache for pickled Python objects.

    Items are stored under a SHA-256 directory based on the cache
    key so the cache is portable across hosts. Eviction is the
    user's responsibility; this class is intentionally stateless
    beyond the root directory.

    Attributes:
        root: Cache root directory; created if missing. Defaults to
          ``${PJEPA_CACHE_ROOT:-~/.cache/pjepa}``.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(
            root or os.environ.get("PJEPA_CACHE_ROOT") or Path.home() / ".cache" / "pjepa"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        """Return the on-disk path for ``key`` (creates the parent).

        Args:
            key: The cache key (typically a SHA-256 hex digest).

        Returns:
            The :class:`Path` where the cached entry lives.

        Raises:
            DataError: When ``key`` is empty.
        """
        if not key:
            raise DataError("DatasetCache: cache key must be non-empty")
        sub = self.root / key[:2] / key
        sub.parent.mkdir(parents=True, exist_ok=True)
        return sub

    def has(self, key: str) -> bool:
        """Return ``True`` when ``key`` is cached on disk."""
        return self.path_for(key).exists()

    def put(self, key: str, value: Any) -> Path:
        """Store ``value`` under ``key`` and return the on-disk path."""
        path = self.path_for(key)
        with path.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def get(self, key: str) -> Any:
        """Load and return the object stored under ``key``.

        Raises:
            DataError: If the key is not in the cache.
        """
        path = self.path_for(key)
        if not path.exists():
            raise DataError(f"DatasetCache: key {key!r} not in cache")
        with path.open("rb") as fh:
            return pickle.load(fh)

    def get_or_compute(self, key: str, compute: Any) -> Any:
        """Return the cached object for ``key`` or store and return ``compute()``."""
        if self.has(key):
            return self.get(key)
        value = compute()
        self.put(key, value)
        return value

    def evict(self, key: str) -> bool:
        """Remove ``key`` from the cache; return whether it was present."""
        path = self.path_for(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Remove every cached item and return the number deleted."""
        count = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                path.unlink()
                count += 1
        return count
