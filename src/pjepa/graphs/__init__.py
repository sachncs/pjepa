"""Graph primitives for the ``pjepa`` package.

This subpackage defines the immutable :class:`TypedAttributedGraph`
dataclass — the substrate for both the persistent graph ``G_t`` and
the working graph ``W_t`` — plus the :class:`PersistentState` and
:class:`WorkingGraph` wrappers that enforce the framework's
*no-direct-execution* invariant: the persistent graph only changes
through :meth:`PersistentState.commit`, and the working graph is
always derived from the latest commit of the persistent graph.

Classes:
    TypedAttributedGraph: Immutable structural container.
    PersistentState: Wrapper around the persistent graph ``G_t``.
    WorkingGraph: Bounded vertex-induced subgraph with a budget.
"""

from __future__ import annotations

from pjepa.graphs.persistent_state import PersistentState
from pjepa.graphs.typed_graph import TypedAttributedGraph
from pjepa.graphs.working_graph import WorkingGraph

__all__ = ["PersistentState", "TypedAttributedGraph", "WorkingGraph"]
