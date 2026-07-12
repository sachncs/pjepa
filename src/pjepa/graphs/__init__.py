"""Graph primitives for the pjepa package.

This subpackage defines the immutable :class:`TypedAttributedGraph`
dataclass (the substrate for both the persistent graph ``G_t`` and the
working graph ``W_t``) and the :class:`PersistentState` /
:class:`WorkingGraph` wrappers that enforce the framework's
*no-direct-execution* invariant.
"""

from __future__ import annotations

from pjepa.graphs.persistent_state import PersistentState
from pjepa.graphs.typed_graph import TypedAttributedGraph
from pjepa.graphs.working_graph import WorkingGraph

__all__ = ["PersistentState", "TypedAttributedGraph", "WorkingGraph"]
