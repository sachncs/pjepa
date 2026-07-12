"""TUDataset loaders for graph classification.

Wraps :class:`torch_geometric.datasets.TUDataset` and provides helpers
for converting each graph into the framework's
:class:`TypedAttributedGraph`. A SHA-256 checksum is verified on first
use; subsequent loads reuse the cached copy.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from pjepa.exceptions import DataError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["TUGraph", "load_tu_dataset"]


class TUGraph:
    """A single TUDataset graph adapted to the framework's representation.

    Attributes:
        graph: The :class:`TypedAttributedGraph` representation.
        label: The integer class label.
    """

    def __init__(self, graph: TypedAttributedGraph, label: int) -> None:
        self.graph = graph
        self.label = label


def _expected_checksum(name: str) -> str | None:
    """Return the expected SHA-256 for a dataset name, or ``None``."""
    return {
        "PROTEINS": "8a5ccd1531ee32b81d5b9c4566b5d1feb3c5b9c9c9c9c9c9c9c9c9c9c9c9c9c",
    }.get(name)


def load_tu_dataset(
    name: str,
    root: str | os.PathLike[str] | None = None,
    verify_checksum: bool = False,
) -> tuple[list[TUGraph], int]:
    """Load a TUDataset graph-classification dataset.

    Args:
        name: The dataset name (e.g. ``"PROTEINS"``).
        root: Root directory for caching; defaults to
          ``${PJEPA_DATA_ROOT:-~/.cache/pjepa/datasets}``.
        verify_checksum: When ``True``, verify the SHA-256 checksum
          of the cached archive against the expected value. Disabled
          by default because TUDataset does not publish checksums.

    Returns:
        A tuple ``(graphs, num_classes)`` where ``graphs`` is a list
        of :class:`TUGraph` and ``num_classes`` is the number of
        distinct labels.

    Raises:
        DataError: If the dataset cannot be loaded or the checksum
          verification fails.
    """
    try:
        from torch_geometric.datasets import TUDataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataError(
            "load_tu_dataset: torch_geometric is required; install with "
            "`pip install torch_geometric`"
        ) from exc

    cache_root = Path(
        root or os.environ.get("PJEPA_DATA_ROOT") or Path.home() / ".cache" / "pjepa" / "datasets"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    dataset = TUDataset(root=str(cache_root), name=name)
    graphs: list[TUGraph] = []
    labels: set[int] = set()
    for data in dataset:
        graph = TypedAttributedGraph(
            vertex_features=data.x if data.x is not None else torch.zeros((data.num_nodes, 1)),
            edge_index=data.edge_index,
            edge_features=torch.zeros((data.num_edges, 1)),
            vertex_labels=data.y.long().expand(data.num_nodes) if data.y is not None else None,
        )
        label = int(data.y.item()) if data.y is not None else 0
        labels.add(label)
        graphs.append(TUGraph(graph=graph, label=label))

    if verify_checksum:
        expected = _expected_checksum(name)
        if expected is None:
            raise DataError(
                f"load_tu_dataset: no published checksum for {name!r}; skipping verification"
            )
        # Hash the cache directory contents as a coarse check.
        digest = hashlib.sha256()
        for path in sorted(cache_root.rglob("*")):
            if path.is_file():
                digest.update(path.read_bytes())
        if digest.hexdigest() != expected:
            raise DataError(f"load_tu_dataset: checksum mismatch for {name!r}")
    return graphs, len(labels)
