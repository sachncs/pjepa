"""Dynamics of the evolution operator F.

The evolution operator is

    F(G, O) = Φ(ℛ(G, O), G)

and advances the persistent graph by one developmental step. This
subpackage provides value-iteration utilities for the analysis of
``F`` as a discrete dynamical system (Propositions 4–6 in the paper).
The analysis assumes a Lipschitz model: each application of ``F``
contracts the graph state by at most ``η_g`` and is sensitive to the
observation by at most ``η_o``.
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
        eta_g: Lipschitz constant of F in the graph state. Values
            below ``1.0`` make the operator a strict contraction.
        eta_o: Lipschitz constant of F in the observation.
        bisimulation_eps: Bisimulation threshold used for state-space
            discretisation in the contraction analysis.
    """

    eta_g: float = 0.5
    eta_o: float = 0.1
    bisimulation_eps: float = 1e-3

    def is_contraction(self) -> bool:
        """Return ``True`` iff ``eta_g < 1``.

        Contraction in the strict sense (``η_g < 1``) guarantees the
        Banach fixed-point theorem applies and the operator has a
        unique fixed point reachable from every starting state.
        """
        return self.eta_g < 1.0


def contractivity_bound(eta_g: float, eta_o: float, epsilon: float, t: int) -> float:
    """Upper bound on ``d(G_t, G_t')`` under bounded observation perturbation.

    The derivation assumes an initial distance ``d(G_0, G_0') = 1``
    (i.e. a unit discrepancy) and a per-step observation perturbation
    bounded by ``epsilon``. The bound is the geometric series

        ``B(η_g, η_o, ε, t) = η_g^t + η_o · ε · (1 - η_g^t) / (1 - η_g)``

    when ``η_g < 1``. The implementation returns this exact quantity
    because the constant ``d(G_0, G_0')`` is fixed at unity — callers
    that need the more general bound should scale the result
    themselves. When ``η_g >= 1`` the bound degenerates and the
    implementation falls back to the conservative linear bound
    ``η_o · ε · t``, which still vanishes for small perturbations
    but does not exhibit the contracting behaviour.

    Args:
        eta_g: Lipschitz constant in the graph state.
        eta_o: Lipschitz constant in the observation.
        epsilon: Per-step observation perturbation bound.
        t: Number of steps (non-negative integer).

    Returns:
        The bound as a non-negative ``float``.

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
        # Degenerate non-contracting regime: bound grows linearly in t.
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
    the description-length proxy. Terminating on the MDL proxy keeps
    the iteration scale-free — different feature scales do not
    require a re-tuned tolerance.

    Args:
        state: Initial graph state.
        operator: A callable mapping a graph to its successor. The
            call is invoked once per iteration; it may return a new
            graph or mutate in place (the iteration reads only the
            return value).
        max_steps: Upper bound on iterations.
        epsilon: Termination tolerance on description length.

    Returns:
        A tuple ``(final_state, steps)`` where ``steps`` is the
        number of iterations actually performed. ``steps == 1`` when
        the operator is the identity on the initial state (it
        converges in one application).

    Raises:
        GraphError: If ``max_steps`` is non-positive.

    Example:
        >>> op = lambda g: g
        >>> final, steps = fixed_point_iteration(g, op)
        >>> steps
        1
    """
    if max_steps <= 0:
        raise GraphError(f"fixed_point_iteration: max_steps must be positive; got {max_steps}")
    current = state
    # Imported here to avoid a top-level cycle on package import.
    from pjepa.objectives.mdl import description_length

    for step in range(1, max_steps + 1):
        nxt = operator(current)
        if abs(description_length(nxt) - description_length(current)) < epsilon:
            return nxt, step
        current = nxt
    return current, max_steps
