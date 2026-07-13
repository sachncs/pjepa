"""Linear-probe evaluation for self-supervised encoders.

A logistic-regression head is fit on the frozen encoder features; the
resulting accuracy on the held-out test split is the linear-probe
score. The implementation follows the protocol of Assran et al.
(2023), which is the canonical reference for the BGRL/MAE-style
graph-evaluation literature.

## Algorithm

1. Encode every training and test graph with the frozen encoder,
   producing a 1-D feature vector per graph.
2. Fit a multinomial :class:`sklearn.linear_model.LogisticRegression`
   on the training features.
3. Evaluate the trained head on the test features; report the
   ``score`` method's accuracy.

## Complexity

Encoding is ``O(|train| + |test|)`` forward passes; the logistic
regression fit is ``O(|train| * D * iterations)`` where ``D`` is the
feature dimension. With the default ``max_iter=1000`` the scikit-learn
implementation converges in a few seconds for TU-scale datasets.

## Exceptions

A non-empty training and test set is required;
:class:`pjepa.exceptions.DataError` is raised when either is empty.
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
        accuracy: Mean accuracy on the test split in ``[0, 1]``.
        num_classes: Number of distinct classes seen across the
          train + test sets (taken as ``max(y) + 1``).
        train_size: Number of training samples used.
        test_size: Number of test samples used.
    """

    accuracy: float
    num_classes: int
    train_size: int
    test_size: int


def encode_graph(graph: TypedAttributedGraph, encoder: torch.nn.Module) -> torch.Tensor:
    """Encode a single graph to a 1-D embedding tensor.

    The encoder's output is expected to be ``[1, D]`` (batch of
    one graph); the leading singleton dimension is squeezed out so
    downstream stacking yields ``[N, D]``. The function falls back
    to the original output when the encoder does not produce a
    ``[1, D]`` tensor.

    Args:
        graph: The input graph.
        encoder: The frozen encoder module.

    Returns:
        The graph embedding as a 1-D tensor.
    """
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
    train_x = torch.stack([encode_graph(g, encoder) for g, _ in train_graphs]).numpy()
    train_y = torch.tensor([lbl for _, lbl in train_graphs], dtype=torch.long).numpy()
    test_x = torch.stack([encode_graph(g, encoder) for g, _ in test_graphs]).numpy()
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
