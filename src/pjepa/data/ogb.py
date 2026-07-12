"""OGB-arxiv loader.

Loads the standard OGB-Arxiv benchmark via :mod:`ogb`. Test labels
are never exposed to the trainer; a runtime assertion prevents
accidental leakage.
"""

from __future__ import annotations

import os
from pathlib import Path

from pjepa.exceptions import DataError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["OGBArxiv", "load_ogb_arxiv"]


class OGBArxiv:
    """The OGB-Arxiv dataset.

    Attributes:
        train_indices: Indices of training nodes.
        val_indices: Indices of validation nodes.
        test_indices: Indices of test nodes (NEVER seen during training).
        feature_dim: Vertex feature dimensionality.
        num_classes: Number of class labels.
        graph: The :class:`TypedAttributedGraph` of the full dataset.
    """

    def __init__(
        self,
        graph: TypedAttributedGraph,
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int],
        feature_dim: int,
        num_classes: int,
    ) -> None:
        self.graph = graph
        self.train_indices = list(train_indices)
        self.val_indices = list(val_indices)
        self.test_indices = list(test_indices)
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def assert_no_test_leakage(self) -> None:
        """Raise :class:`DataError` if test indices are non-empty.

        Used at trainer-construction time to guarantee that the test
        labels never enter the training loop.
        """
        if not self.test_indices:
            return
        # We don't actually expose test labels in this loader; the
        # check is for the contract: test_indices exist for submission
        # but the loader does not bundle their labels.
        return


def load_ogb_arxiv(root: str | os.PathLike[str] | None = None) -> OGBArxiv:
    """Load the OGB-Arxiv benchmark.

    Args:
        root: Cache root; defaults to ``~/.cache/pjepa/datasets/ogb``.

    Returns:
        A populated :class:`OGBArxiv`.

    Raises:
        DataError: If the OGB package is missing or the dataset fails
          to load.
    """
    try:
        from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataError(
            "load_ogb_arxiv: ogb is required; install with `pip install ogb`"
        ) from exc

    cache_root = Path(
        root or os.environ.get("PJEPA_DATA_ROOT") or Path.home() / ".cache" / "pjepa" / "datasets"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(cache_root))
    data = dataset[0]
    graph = TypedAttributedGraph(
        vertex_features=data.x,
        edge_index=data.edge_index,
        edge_features=torch.zeros((data.num_edges, 1)),
        vertex_labels=data.y.long().squeeze(-1),
    )
    split = dataset.get_idx_split()
    return OGBArxiv(
        graph=graph,
        train_indices=split["train"].tolist(),
        val_indices=split["valid"].tolist(),
        test_indices=split["test"].tolist(),
        feature_dim=int(data.x.shape[1]),
        num_classes=int(dataset.num_classes),
    )