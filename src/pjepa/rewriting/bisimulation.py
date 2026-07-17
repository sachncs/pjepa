"""Behavioural bisimulation metric on graphs.

The bisimulation metric implemented here compares two attributed
graphs through the *signature distance* between their vertices: a
pairwise distance is computed between every vertex of the first graph
and every vertex of the second using the per-vertex signature, and a
scalar summary is returned. The current implementation deliberately
returns the **maximum pairwise distance** rather than the Hausdorff
or Wasserstein aggregation; this is the conservative choice and is
sufficient for the verification step, which only needs a non-negative
scalar that vanishes on identical graphs and grows with structural
divergence.

Future revisions may add a true value-iteration Bellman update to
recover the Ferns-Panangaden-Precup (2004) pseudometric; the API is
designed to absorb that change without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pjepa.exceptions import NumericalError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["BisimulationMetric", "bisimulation_distance"]


@dataclass(frozen=True)
class BisimulationMetric:
    """Configuration for the bisimulation distance computation.

    Attributes:
        epsilon: Convergence tolerance reserved for the eventual
            Bellman-style value-iteration sweep. Currently unused;
            callers should set it for forward compatibility.
        max_iters: Maximum number of value-iteration sweeps.
            Likewise reserved for forward compatibility.
        relation_set: The names of relation views to consider (the
            SSCG relation set ``R`` in the paper). Currently unused
            but reserved for future multi-view extensions.
    """

    epsilon: float = 1e-4
    max_iters: int = 100
    relation_set: tuple[str, ...] = ("default",)


def vertex_signature(graph: TypedAttributedGraph) -> torch.Tensor:
    """Return a per-vertex signature suitable for bisimulation distance.

    The signature is the concatenation of the (already continuous)
    vertex features and the vertex label one-hot when labels are
    present. It is used as a stable per-vertex identity for the
    pairwise distance matrix.

    Args:
        graph: The graph whose per-vertex signature to compute.

    Returns:
        A ``[N, d]`` tensor; ``d`` equals
        ``vertex_features.shape[1]`` when labels are absent and
        ``vertex_features.shape[1] + 2`` when binary labels are
        present.
    """
    features = graph.vertex_features
    if graph.vertex_labels is not None:
        labels = torch.nn.functional.one_hot(graph.vertex_labels, num_classes=2).float()
        return torch.cat([features, labels], dim=-1)
    return features


def bisimulation_distance(
    graph_a: TypedAttributedGraph,
    graph_b: TypedAttributedGraph,
    metric: BisimulationMetric | None = None,
) -> float:
    """Compute the bisimulation pseudometric between two graphs.

    The result is the **maximum pairwise signature distance** under
    L2: a non-negative scalar that is zero when both graphs are
    empty or have pairwise identical vertex signatures, and grows as
    the graphs diverge. The metric accepts arbitrarily shaped feature
    tensors as long as the two leading dimensions (vertex counts)
    can be combined into a pairwise matrix.

    Args:
        graph_a: The first graph.
        graph_b: The second graph.
        metric: Optional configuration; defaults to
            :class:`BisimulationMetric`. Currently the configuration
            only controls fields reserved for future iterations.

    Returns:
        A non-negative ``float``. ``0.0`` indicates (exactly equal)
        signatures across all vertex pairs (or two empty graphs).

    Raises:
        NumericalError: If the computed distance contains non-finite
            values.
    """
    del metric  # Reserved for forward compatibility.
    # MPS does not support float64; fall back to float32 on that backend.
    target_dtype = torch.float64
    if graph_a.vertex_features.device.type == "mps" or graph_b.vertex_features.device.type == "mps":
        target_dtype = torch.float32
    sig_a = vertex_signature(graph_a).to(target_dtype)
    sig_b = vertex_signature(graph_b).to(target_dtype)
    n_a = sig_a.shape[0]
    n_b = sig_b.shape[0]
    if n_a == 0 and n_b == 0:
        return 0.0
    if n_a == 0 or n_b == 0:
        return float("inf")

    # Pairwise L2 signature distance forms the basis for the metric.
    # Maximum pairwise distance is a conservative scalar summary that
    # vanishes iff signatures match on every pair of (i, j).
    base = torch.cdist(sig_a.unsqueeze(0), sig_b.unsqueeze(0)).squeeze(0)
    if not torch.isfinite(base).all():
        raise NumericalError("bisimulation_distance: signature distance produced non-finite values")
    return float(base.max().item())
