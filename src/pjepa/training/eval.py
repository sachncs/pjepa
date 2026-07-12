"""Linear-probe evaluation for self-supervised encoders.

A logistic regression head is fit on the frozen encoder features; the
resulting accuracy is the linear-probe score. The implementation
follows the protocol of Assran et al. (2023).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.linear_model import LogisticRegression

from pjepa.exceptions import DataError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["LinearProbeResult", "linear_probe_eval"]


@dataclass(frozen=True)
class LinearProbeResult:
    """Result of a linear-probe evaluation.

    Attributes:
        accuracy: Mean per-class accuracy on the test split.
        num_classes: Number of distinct classes seen.
        train_size: Number of training samples used.
        test_size: Number of test samples used.
    """

    accuracy: float
    num_classes: int
    train_size: int
    test_size: int


def _graph_to_embedding(graph: TypedAttributedGraph, encoder: torch.nn.Module) -> torch.Tensor:
    """Encode a single graph to a 1-D embedding tensor."""
    with torch.no_grad():
        out = encoder(graph)
    if hasattr(out, "shape") and out.ndim == 2 and out.shape[0] == 1:
        return out.squeeze(0)
    return out


def linear_probe_eval(
    encoder: torch.nn.Module,
    train_graphs: list[tuple[TypedAttributedGraph, int]],
    test_graphs: list[tuple[TypedAttributedGraph, int]],
) -> LinearProbeResult:
    """Run a linear-probe evaluation.

    Args:
        encoder: The frozen encoder.
        train_graphs: List of ``(graph, label)`` pairs for training.
        test_graphs: List of ``(graph, label)`` pairs for testing.

    Returns:
        A populated :class:`LinearProbeResult`.

    Raises:
        DataError: If the dataset is empty.
    """
    if not train_graphs or not test_graphs:
        raise DataError("linear_probe_eval: train and test sets must both be non-empty")
    train_x = torch.stack([_graph_to_embedding(g, encoder) for g, _ in train_graphs]).numpy()
    train_y = torch.tensor([lbl for _, lbl in train_graphs], dtype=torch.long).numpy()
    test_x = torch.stack([_graph_to_embedding(g, encoder) for g, _ in test_graphs]).numpy()
    test_y = torch.tensor([lbl for _, lbl in test_graphs], dtype=torch.long).numpy()
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(train_x, train_y)
    accuracy = float(clf.score(test_x, test_y))
    num_classes = int(max(train_y.max(), test_y.max()) + 1)
    return LinearProbeResult(
        accuracy=accuracy,
        num_classes=num_classes,
        train_size=len(train_graphs),
        test_size=len(test_graphs),
    )
