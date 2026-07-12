"""Working graph wrapper.

The working graph is the only object on which inference executes. It
is a fixed-budget vertex-induced subgraph of the persistent graph,
constructed by the retrieval operator and discarded at the end of
each developmental step. The wrapper enforces the budget invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pjepa.exceptions import GraphError
from pjepa.graphs.typed_graph import TypedAttributedGraph

__all__ = ["WorkingGraph"]


@dataclass(frozen=True)
class WorkingGraph:
    """A bounded working subgraph derived from the persistent graph.

    Attributes:
        graph: The underlying :class:`TypedAttributedGraph`.
        budget: The maximum number of vertices allowed.
        parent_version: The version of the persistent graph from which
          the working graph was derived.

    Example:
        >>> working = WorkingGraph(graph=sub, budget=64, parent_version=1)
        >>> working.is_within_budget()
        True
    """

    graph: TypedAttributedGraph
    budget: int
    parent_version: int = 0

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise GraphError(f"WorkingGraph: budget must be non-negative; got {self.budget}")
        if self.budget > 0 and self.graph.num_vertices() > self.budget:
            raise GraphError(
                f"WorkingGraph: vertex count {self.graph.num_vertices()} "
                f"exceeds budget {self.budget}"
            )

    def num_vertices(self) -> int:
        """Return the number of vertices in the working graph."""
        return self.graph.num_vertices()

    def num_edges(self) -> int:
        """Return the number of edges in the working graph."""
        return self.graph.num_edges()

    def is_within_budget(self) -> bool:
        """Return whether the working graph respects its budget."""
        return self.num_vertices() <= self.budget

    def utilisation(self) -> float:
        """Return the fraction of the budget consumed."""
        if self.budget == 0:
            return 0.0
        return float(self.num_vertices()) / float(self.budget)

    def to(self, device: torch.device) -> WorkingGraph:
        """Move every tensor of the working graph to ``device``."""
        return WorkingGraph(
            graph=self.graph.to(device),
            budget=self.budget,
            parent_version=self.parent_version,
        )
