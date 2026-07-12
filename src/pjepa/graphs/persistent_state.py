"""Persistent graph state.

The persistent graph is the only object on which learning is deposited
in the framework. It is the evolved sufficient statistic of the
observation history. The wrapper here enforces three invariants:

1. The persistent graph only grows via :meth:`commit`, which validates
   a candidate rewrite against the framework's four-conditions
   acceptance criterion.
2. The wrapper exposes read-only views; the underlying
   :class:`TypedAttributedGraph` is already immutable, so the wrapper
   just enforces the commit interface.
3. The wrapper records the version number of every commit so that
   audit trails can be reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pjepa.exceptions import GraphError
from pjepa.graphs.typed_graph import TypedAttributedGraph

__all__ = ["CommitRecord", "CommitRejected", "PersistentState"]


@dataclass(frozen=True)
class CommitRecord:
    """A single successful commit to the persistent graph.

    Attributes:
        version: The graph version produced by the commit (>= 1).
        timestamp: An arbitrary monotonic counter or wall-clock value
          supplied by the caller; the framework does not interpret it.
        cost: The rewrite cost recorded by the verification step.

    Example:
        >>> record = CommitRecord(version=1, timestamp=0.0, cost=0.1)
    """

    version: int
    timestamp: float
    cost: float


@dataclass(frozen=True)
class CommitRejected:
    """A candidate rewrite rejected by the four-conditions criterion.

    Attributes:
        reason: A human-readable explanation of why the candidate was
          rejected.
        cost: The cost the verification step computed.

    Example:
        >>> rejection = CommitRejected(reason="bisimilarity violated", cost=0.5)
    """

    reason: str
    cost: float


@dataclass
class PersistentState:
    """Wrapper around the persistent graph ``G_t``.

    Attributes:
        graph: The current :class:`TypedAttributedGraph`.
        history: A tuple of :class:`CommitRecord` (successful commits).
        rejections: A tuple of :class:`CommitRejected` (rejected
          candidates).

    Example:
        >>> state = PersistentState(graph=g0)
        >>> state.commit(candidate, cost=0.0, timestamp=1.0)
    """

    graph: TypedAttributedGraph
    history: tuple[CommitRecord, ...] = field(default_factory=tuple)
    rejections: tuple[CommitRejected, ...] = field(default_factory=tuple)

    def num_vertices(self) -> int:
        """Return the number of vertices in the persistent graph."""
        return self.graph.num_vertices()

    def num_edges(self) -> int:
        """Return the number of edges in the persistent graph."""
        return self.graph.num_edges()

    def num_commits(self) -> int:
        """Return the number of accepted commits recorded so far."""
        return len(self.history)

    def num_rejections(self) -> int:
        """Return the number of rejected candidates recorded so far."""
        return len(self.rejections)

    def commit(
        self,
        candidate: TypedAttributedGraph,
        cost: float,
        timestamp: float,
        delta_j: float | None = None,
    ) -> PersistentState:
        """Replace the persistent graph with ``candidate`` and record the commit.

        The acceptance criterion is supplied by the caller via
        ``delta_j``: when ``delta_j`` is provided, the commit is
        accepted only if ``delta_j < 0``. When ``delta_j`` is ``None``,
        the caller is asserting that the candidate has already been
        verified; this is the path taken by the rewriting engine after
        all four conditions have been satisfied.

        Args:
            candidate: The candidate next-state graph.
            cost: The rewrite cost (must be non-negative).
            timestamp: A monotonic counter or wall-clock timestamp.
            delta_j: Optional ``Δ𝒥`` value; when provided, the commit
              is accepted iff ``delta_j < 0``.

        Returns:
            A new :class:`PersistentState` reflecting the accepted
            commit. The caller is responsible for replacing the live
            state with the returned value; the wrapper does not
            mutate in place.

        Raises:
            GraphError: If ``cost`` is negative or ``delta_j`` is
              non-negative.

        Example:
            >>> state2 = state.commit(candidate, cost=0.1, timestamp=1.0)
        """
        if cost < 0:
            raise GraphError(f"PersistentState.commit: cost must be non-negative; got {cost}")
        if delta_j is not None and delta_j >= 0:
            raise GraphError(f"PersistentState.commit: delta_j must be < 0; got {delta_j}")
        record = CommitRecord(
            version=self.graph.version + 1,
            timestamp=timestamp,
            cost=cost,
        )
        return PersistentState(
            graph=candidate,
            history=self.history + (record,),
            rejections=self.rejections,
        )

    def reject(self, reason: str, cost: float) -> PersistentState:
        """Record a rejected candidate without modifying the graph.

        Args:
            reason: A human-readable rejection explanation.
            cost: The cost the verification step computed.

        Returns:
            A new :class:`PersistentState` with the rejection appended
            to its audit trail.

        Raises:
            GraphError: If ``cost`` is negative or ``reason`` is empty.

        Example:
            >>> state2 = state.reject("bisimilarity violated", cost=0.5)
        """
        if cost < 0:
            raise GraphError(f"PersistentState.reject: cost must be non-negative; got {cost}")
        if not reason:
            raise GraphError("PersistentState.reject: reason must be a non-empty string")
        rejection = CommitRejected(reason=reason, cost=cost)
        return PersistentState(
            graph=self.graph,
            history=self.history,
            rejections=self.rejections + (rejection,),
        )

    def to(self, device: torch.device) -> PersistentState:
        """Move every tensor of the persistent graph to ``device``."""
        return PersistentState(
            graph=self.graph.to(device),
            history=self.history,
            rejections=self.rejections,
        )
