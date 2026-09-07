"""Graph primitives for the ``pjepa`` package.

This subpackage defines the immutable :class:`Graph` dataclass —
the substrate for both the persistent graph ``G_t`` and the
working graph ``W_t`` — plus the :class:`State` and
:class:`Working` wrappers that enforce the framework's
*no-direct-execution* invariant: the persistent graph only changes
through :meth:`State.commit`, and the working graph is always
derived from the latest commit of the persistent graph.

Classes:
    Graph: Immutable structural container.
    State: Wrapper around the persistent graph ``G_t``.
    Working: Bounded vertex-induced subgraph with a budget.
"""

from __future__ import annotations

from pjepa.graphs.graph import Graph
from pjepa.graphs.state import State
from pjepa.graphs.working import Working

__all__ = ["Graph", "State", "Working"]
