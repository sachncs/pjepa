"""TUDataset loaders for graph classification.

Wraps :class:`torch_geometric.datasets.TUDataset` and provides
helpers for converting each graph into the framework's
:class:`Graph`.

## Checksums

:func:`expected_checksum` returns ``None`` for every name because
TUDataset does not publish a SHA-256 registry for its archives.
When ``verify_checksum=True`` is requested on a name without a
published digest, :func:`load_tu_dataset` raises :class:`DataError`
rather than silently skipping verification — a security-conscious
caller that explicitly opts in deserves an explicit failure.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from pjepa.exceptions import DataError
from pjepa.graphs import Graph

__all__ = ["TUGraph", "expected_checksum", "load_tu_dataset"]


class TUGraph:
    """A single TUDataset graph adapted to the framework's representation.

    Attributes:
        graph: The :class:`Graph` representation.
        label: The integer class label.

    Args:
        graph: The :class:`Graph` for this sample.
        label: The integer class label assigned by the dataset.
    """

    def __init__(self, graph: Graph, label: int) -> None:
        self.graph = graph
        self.label = label


def expected_checksum(name: str) -> str | None:
    """Return the expected SHA-256 for a TUDataset name, or ``None``.

    Args:
        name: The TUDataset name (e.g. ``"PROTEINS"``).

    Returns:
        ``None`` for every name. TUDataset does not publish a
        checksum registry, so the loader cannot honour
        ``verify_checksum=True`` for any dataset. Callers that
        request verification will get an explicit
        :class:`DataError` instead of a silent skip.
    """
    del name
    return None


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
          of the cached archive against the published value. The
          loader raises :class:`DataError` when no checksum is
          available because the underlying dataset does not
          publish one; it does not silently pass.

    Returns:
        A tuple ``(graphs, num_classes)`` where ``graphs`` is a list
        of :class:`TUGraph` and ``num_classes`` is the number of
        distinct labels.

    Raises:
        DataError: If the dataset cannot be loaded, the checksum
          verification fails, or the ``verify_checksum=True`` flag
          is set for a dataset without a published digest.
    """
    try:
        from torch_geometric.datasets import TUDataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataError(
            "load_tu_dataset: torch_geometric is required; install with "
            "`pip install torch_geometric`"
        ) from exc

    if verify_checksum:
        raise DataError(
            f"load_tu_dataset: no published checksum for {name!r}; "
            "TUDataset does not maintain a checksum registry"
        )

    cache_root = Path(
        root or os.environ.get("PJEPA_DATA_ROOT") or Path.home() / ".cache" / "pjepa" / "datasets"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    dataset = TUDataset(root=str(cache_root), name=name)
    graphs: list[TUGraph] = []
    labels: set[int] = set()
    for data in dataset:
        graph = Graph(
            vertex_features=data.x if data.x is not None else torch.zeros((data.num_nodes, 1)),
            edge_index=data.edge_index,
            edge_features=torch.zeros((data.num_edges, 1)),
            vertex_labels=data.y.long().expand(data.num_nodes) if data.y is not None else None,
        )
        label = int(data.y.item()) if data.y is not None else 0
        labels.add(label)
        graphs.append(TUGraph(graph=graph, label=label))

    return graphs, len(labels)
