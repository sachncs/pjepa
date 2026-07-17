"""Persistent graph state ``G_t``.

The persistent graph is the only object on which learning is
deposited in the framework. It is the evolved sufficient statistic
of the observation history. The wrapper here enforces three invariants:

1. The persistent graph only grows via :meth:`PersistentState.commit`,
   which validates a candidate rewrite against the framework's
   four-conditions acceptance criterion.
2. The wrapper exposes read-only views; the underlying
   :class:`TypedAttributedGraph` is already immutable, so the wrapper
   just enforces the commit interface.
3. The wrapper records the version number of every commit so that
   audit trails can be reconstructed. The version of the head graph
   is therefore ``initial_version + len(history)`` once every commit
   has been accepted.

The class is intentionally cheap to copy: every field except ``graph``
is a tuple of dataclass instances, and ``dataclasses.replace``-style
updates avoid mutating the underlying tensors.
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
        version: The graph version produced by the commit. After the
            ``k``-th commit ``version == initial_version + k``; the
            framework guarantees strict monotonicity, so this number
            is a reliable audit order key.
        timestamp: An arbitrary monotonic counter or wall-clock value
            supplied by the caller; the framework does not interpret
            it. Epoch seconds are the conventional choice.
        cost: The rewrite cost recorded by the verification step.
            Non-negative by construction.
    """

    version: int
    timestamp: float
    cost: float


@dataclass(frozen=True)
class CommitRejected:
    """A candidate rewrite rejected by the four-conditions criterion.

    Attributes:
        reason: A human-readable explanation of why the candidate was
            rejected — for example ``"bisimilarity violated"`` or
            ``"delta_j is non-negative"``.
        cost: The cost the verification step computed. Non-negative
            by construction.
    """

    reason: str
    cost: float


@dataclass
class PersistentState:
    """Wrapper around the persistent graph ``G_t``.

    Each method that produces a new state (``commit``, ``reject``,
    ``to``) returns a fresh instance rather than mutating in place.
    Holding a reference to an older :class:`PersistentState` therefore
    pins the configuration at that point in time, which lets training
    loops snapshot and roll back without extra bookkeeping.

    Attributes:
        graph: The current :class:`TypedAttributedGraph`.
        history: A tuple of :class:`CommitRecord` for accepted
            commits, in order.
        rejections: A tuple of :class:`CommitRejected` for rejected
            candidates, in order.
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
        accepted only if ``delta_j < 0`` (i.e. the objective strictly
        decreases). When ``delta_j`` is ``None``, the caller asserts
        that the candidate has already been verified — typically by
        :func:`pjepa.rewriting.four_conditions.accept_candidate` —
        and the wrapper accepts the commit unconditionally.

        Args:
            candidate: The candidate next-state graph.
            cost: The rewrite cost (must be non-negative).
            timestamp: A monotonic counter or wall-clock timestamp.
            delta_j: Optional ``Δ𝒥`` value; when provided, the commit
                is accepted iff ``delta_j < 0``.

        Returns:
            A new :class:`PersistentState` whose ``graph`` is
            ``candidate`` and whose ``history`` has the new
            :class:`CommitRecord` appended.

        Raises:
            GraphError: If ``cost`` is negative, or if ``delta_j``
                is non-negative when supplied.
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
            A new :class:`PersistentState` with the rejection
            appended to its audit trail.

        Raises:
            GraphError: If ``cost`` is negative or ``reason`` is empty.
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
        """Move every tensor of the persistent graph to ``device``.

        The ``history`` and ``rejections`` tuples are pure data and do
        not need to be moved.
        """
        return PersistentState(
            graph=self.graph.to(device),
            history=self.history,
            rejections=self.rejections,
        )
