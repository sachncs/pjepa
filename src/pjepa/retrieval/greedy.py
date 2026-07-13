"""Greedy working-graph retrieval (Algorithm 1 of the paper).

The greedy algorithm iteratively adds the vertex with the largest
marginal gain to the working subset, stopping when the budget is
exhausted or the marginal gain becomes non-positive. For monotone
submodular utilities the algorithm achieves the
Nemhauser-Wolsey-Fisher 1978 ``(1 − 1/e) ≈ 0.632`` approximation
guarantee relative to the optimal subset of the same size.

Complexity is ``O(budget × n × utility_eval_cost)`` for ``n``
vertices and an arbitrary ``utility``. The implementation uses a
straightforward linear scan rather than a lazy priority queue; the
public API would not change when that optimisation is added later.

The retrieval step is deterministic for a fixed graph, observation,
budget, and utility instance; the test suite relies on this
property.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph, WorkingGraph
from pjepa.retrieval.utility import FacilityLocationUtility, RetrievalUtility

__all__ = ["GreedyRetrieval", "RetrievalResult"]


@dataclass(frozen=True)
class RetrievalResult:
    """The output of a greedy retrieval run.

    Attributes:
        working: The working graph that was selected. Its vertex
            count never exceeds the configured budget.
        utility: The cumulative utility achieved by the greedy
            selection. ``0.0`` for empty input graphs.
        iterations: The number of greedy iterations actually performed
            (may be less than the budget when the graph is too small
            or when further additions no longer improve the utility).

    Example:
        >>> result = GreedyRetrieval(budget=8).select(graph, observation)
        >>> result.working.num_vertices() <= 8
        True
    """

    working: WorkingGraph
    utility: float
    iterations: int


class GreedyRetrieval:
    """Greedy retrieval over the persistent graph vertices.

    At each step, the algorithm selects the vertex whose addition
    yields the largest marginal utility gain. For monotone submodular
    utilities the resulting subset achieves at least ``(1 - 1/e)``
    times the optimal utility of any subset of the same size.

    Attributes:
        budget: The maximum number of vertices to include in the
            working graph.

    Raises:
        GraphError: At construction time if ``budget`` is negative.

    Example:
        >>> retriever = GreedyRetrieval(budget=32)
        >>> result = retriever.select(persistent, observation)
    """

    def __init__(self, budget: int) -> None:
        if budget < 0:
            raise GraphError(f"GreedyRetrieval: budget must be non-negative; got {budget}")
        self.budget = budget

    def select(
        self,
        graph: TypedAttributedGraph,
        observation: torch.Tensor,
        utility: RetrievalUtility | None = None,
    ) -> RetrievalResult:
        """Run the greedy algorithm and return a :class:`RetrievalResult`.

        Args:
            graph: The persistent graph state.
            observation: A tensor whose leading dimension matches the
                observation batch (``[m, d]`` or ``[d]``).
            utility: A :class:`RetrievalUtility` instance. ``None``
                constructs a default :class:`FacilityLocationUtility`
                from the graph's vertex features.

        Returns:
            A populated :class:`RetrievalResult`.

        Raises:
            GraphError: If the persistent graph has zero vertices
                and a non-zero budget (in that case the empty result
                is returned as ``utility=0.0``, ``iterations=0``).
        """
        n = graph.num_vertices()
        feature_dim = graph.vertex_features.shape[1]
        if n == 0:
            empty = TypedAttributedGraph(
                vertex_features=torch.zeros((0, feature_dim)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
            return RetrievalResult(
                working=WorkingGraph(graph=empty, budget=self.budget, parent_version=graph.version),
                utility=0.0,
                iterations=0,
            )
        if utility is None:
            utility = FacilityLocationUtility(vertex_features=graph.vertex_features)

        selected: list[int] = []
        cumulative_utility = 0.0
        iterations = 0
        max_steps = min(self.budget, n)

        for _ in range(max_steps):
            best_vertex = -1
            best_marginal = -float("inf")
            best_total = cumulative_utility
            for v in range(n):
                if v in selected:
                    continue
                trial = selected + [v]
                subset = torch.tensor(trial, dtype=torch.long)
                total = float(utility(subset, observation))
                marginal = total - cumulative_utility
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_vertex = v
                    best_total = total
            # Stop only when the best marginal gain is non-positive.
            # A negative gain would imply a non-monotone utility;
            # zero gain implies coverage has been achieved and further
            # vertices do not help.
            if best_vertex < 0 or best_marginal <= 0.0:
                break
            selected.append(best_vertex)
            cumulative_utility = best_total
            iterations += 1

        if selected:
            mask = torch.zeros(n, dtype=torch.bool)
            mask[torch.tensor(selected, dtype=torch.long)] = True
            working_graph = graph.subgraph(mask)
        else:
            working_graph = TypedAttributedGraph(
                vertex_features=torch.zeros((0, feature_dim)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )

        return RetrievalResult(
            working=WorkingGraph(
                graph=working_graph,
                budget=self.budget,
                parent_version=graph.version,
            ),
            utility=cumulative_utility,
            iterations=iterations,
        )
