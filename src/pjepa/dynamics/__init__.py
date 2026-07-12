"""Dynamics of the evolution operator F.

The evolution operator
    F(G, O) = Φ(ℛ(G, O), G)
advances the persistent graph by one developmental step. This
subpackage provides value-iteration utilities for the analysis of
``F`` as a discrete dynamical system (Propositions 4–6 in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["EvolutionOperator", "contractivity_bound", "fixed_point_iteration"]


@dataclass(frozen=True)
class EvolutionOperator:
    """Configuration for analysing the evolution operator F.

    Attributes:
        eta_g: Lipschitz constant of F in the graph state.
        eta_o: Lipschitz constant of F in the observation.
        bisimulation_eps: Bisimulation threshold used for state-space
          discretisation in the contraction analysis.
    """

    eta_g: float = 0.5
    eta_o: float = 0.1
    bisimulation_eps: float = 1e-3

    def is_contraction(self) -> bool:
        """Return True iff ``eta_g < 1``."""
        return self.eta_g < 1.0


def contractivity_bound(eta_g: float, eta_o: float, epsilon: float, t: int) -> float:
    """Upper bound on ``d(G_t, G'_t)`` under bounded observation perturbation.

    Args:
        eta_g: Lipschitz constant in the graph state.
        eta_o: Lipschitz constant in the observation.
        epsilon: Per-step observation perturbation bound.
        t: Number of steps.

    Returns:
        The bound ``eta_g^t * d(G_0, G'_0) + eta_o * epsilon * (1 - eta_g^t) / (1 - eta_g)``.
        When ``eta_g == 1`` the function returns the trivial linear
        upper bound ``d(G_0, G'_0) + eta_o * epsilon * t``.

    Raises:
        GraphError: If any input is negative.

    Example:
        >>> contractivity_bound(0.5, 0.1, 0.05, 10)
        0.01099...
    """
    for name, value in (("eta_g", eta_g), ("eta_o", eta_o), ("epsilon", epsilon), ("t", t)):
        if value < 0:
            raise GraphError(f"contractivity_bound: {name} must be non-negative; got {value}")
    if eta_g >= 1.0:
        # Degenerate; bound grows linearly.
        return eta_o * epsilon * t
    eta_g_t = eta_g**t
    return eta_g_t + eta_o * epsilon * (1.0 - eta_g_t) / (1.0 - eta_g)


def fixed_point_iteration(
    state: TypedAttributedGraph,
    operator,
    max_steps: int = 256,
    epsilon: float = 1e-3,
) -> tuple[TypedAttributedGraph, int]:
    """Iterate ``operator`` until a fixed point is reached or ``max_steps``.

    The iteration is treated as a discrete dynamical system; the loop
    terminates when two successive states agree within ``epsilon`` on
    the description-length pseudo-metric.

    Args:
        state: Initial graph state.
        operator: A callable mapping a graph to its successor.
        max_steps: Upper bound on iterations.
        epsilon: Termination tolerance.

    Returns:
        A tuple ``(final_state, steps)`` where ``steps`` is the number
        of iterations actually performed.

    Raises:
        GraphError: If ``max_steps`` is non-positive.

    Example:
        >>> op = lambda g: g  # identity operator has the input as fixed point
        >>> fixed_point_iteration(g, op)[1]
        1
    """
    if max_steps <= 0:
        raise GraphError(f"fixed_point_iteration: max_steps must be positive; got {max_steps}")
    current = state
    for step in range(1, max_steps + 1):
        nxt = operator(current)
        # Termination is measured on the description-length proxy.
        from pjepa.objectives.mdl import description_length

        if abs(description_length(nxt) - description_length(current)) < epsilon:
            return nxt, step
        current = nxt
    return current, max_steps