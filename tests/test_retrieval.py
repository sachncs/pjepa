"""Tests for pjepa.retrieval."""

from __future__ import annotations

import itertools

import pytest
import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph
from pjepa.retrieval import (
    FacilityLocationUtility,
    GreedyRetrieval,
    InformationGainUtility,
    facility_location_weights,
    uniform_weights,
)
from pjepa.retrieval.utility import RetrievalUtility

__all__ = [
    "test_happy_greedy_returns_budget",
    "test_happy_facility_location_scores_non_negative",
    "test_bad_negative_budget",
    "test_bad_facility_location_wrong_dim",
    "test_bad_information_gain_negative_mu",
    "test_ugly_empty_graph_zero_utility",
    "test_ugly_single_vertex_graph",
    "test_leaky_repeated_retrieval_no_module_state",
    "test_round_trip_submodule_property",
    "test_cross_backend_mps_utility",
    "test_distributional_utility_is_submodular",
    "test_property_uniform_weights",
    "test_property_facility_weights_non_negative",
    "test_one_minus_one_over_e_on_synthetic",
]


def _random_graph(num_vertices: int = 8, feature_dim: int = 3, seed: int = 0) -> TypedAttributedGraph:
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn((num_vertices, feature_dim), generator=g)
    edges = []
    for i, j in itertools.combinations(range(num_vertices), 2):
        if torch.rand((1,), generator=g).item() < 0.4:
            edges.append((i, j))
            edges.append((j, i))
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).T
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
    return TypedAttributedGraph(
        vertex_features=feats,
        edge_index=ei,
        edge_features=torch.zeros((ei.shape[1], 1)),
    )


def test_happy_greedy_returns_budget() -> None:
    """A typical retrieval respects the budget and yields positive utility."""
    g = _random_graph(20, 4, seed=1)
    obs = torch.randn((5, 4))
    result = GreedyRetrieval(budget=8).select(g, obs)
    assert result.working.num_vertices() <= 8
    assert result.utility > 0.0
    assert result.iterations >= 1


def test_happy_facility_location_scores_non_negative() -> None:
    """Facility-location utility is non-negative on non-empty inputs."""
    g = _random_graph(5, 3, seed=2)
    util = FacilityLocationUtility(vertex_features=g.vertex_features)
    obs = torch.randn((4, 3))
    subset = torch.tensor([0, 2, 4], dtype=torch.long)
    score = util(subset, obs)
    assert score >= 0.0


def test_bad_negative_budget() -> None:
    """A negative budget raises GraphError."""
    with pytest.raises(GraphError):
        GreedyRetrieval(budget=-1)


def test_bad_facility_location_wrong_dim() -> None:
    """A non-2-D vertex-features tensor is rejected."""
    with pytest.raises(ValueError):
        FacilityLocationUtility(vertex_features=torch.zeros((4,)))


def test_bad_information_gain_negative_mu() -> None:
    """A negative mu is rejected."""
    with pytest.raises(ValueError):
        InformationGainUtility(mu=-0.1)


def test_ugly_empty_graph_zero_utility() -> None:
    """An empty persistent graph yields a zero-utility result."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    obs = torch.randn((2, 4))
    result = GreedyRetrieval(budget=8).select(g, obs)
    assert result.working.num_vertices() == 0
    assert result.utility == 0.0


def test_ugly_single_vertex_graph() -> None:
    """A single-vertex graph yields a one-vertex working graph."""
    g = TypedAttributedGraph(
        vertex_features=torch.tensor([[1.0, 2.0, 3.0]]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    obs = torch.tensor([1.0, 2.0, 3.0])
    result = GreedyRetrieval(budget=8).select(g, obs)
    assert result.working.num_vertices() == 1


def test_leaky_repeated_retrieval_no_module_state() -> None:
    """Two back-to-back retrievals do not see each other's state."""
    g = _random_graph(6, 3, seed=3)
    obs = torch.randn((2, 3))
    r1 = GreedyRetrieval(budget=4).select(g, obs)
    r2 = GreedyRetrieval(budget=4).select(g, obs)
    assert r1.utility == pytest.approx(r2.utility)


def test_round_trip_submodule_property() -> None:
    """Retrieval is deterministic on the same seed."""
    g = _random_graph(10, 3, seed=4)
    obs = torch.randn((2, 3))
    a = GreedyRetrieval(budget=5).select(g, obs)
    b = GreedyRetrieval(budget=5).select(g, obs)
    assert a.working.num_vertices() == b.working.num_vertices()
    assert a.utility == pytest.approx(b.utility)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_utility() -> None:
    """Utility computation works on MPS tensors."""
    g = _random_graph(8, 4, seed=5)
    g = g.to(torch.device("mps"))
    util = FacilityLocationUtility(vertex_features=g.vertex_features)
    obs = torch.randn((3, 4), device="mps")
    subset = torch.tensor([0, 1, 2], dtype=torch.long, device="mps")
    score = util(subset, obs)
    # MPS may produce small negative values from fp rounding on cosines
    # that should mathematically be non-negative; allow a small tolerance.
    assert score >= -1e-4


def test_distributional_utility_is_submodular() -> None:
    """Facility location exhibits diminishing returns on random inputs.

    The property test generates 50 random inputs and asserts the
    diminishing-returns inequality
        ``f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T)``
    for random ``S ⊂ T`` with ``v ∉ T``.
    """
    g = _random_graph(8, 4, seed=6)
    util = FacilityLocationUtility(vertex_features=g.vertex_features)
    obs = torch.randn((4, 4))
    n = g.num_vertices()
    for _ in range(50):
        s_size = torch.randint(0, n - 2, (1,)).item()
        t_size = torch.randint(s_size + 1, n - 1, (1,)).item()
        # Build a small random subset S and extend it to T
        s = torch.randperm(n)[:s_size]
        s_set = set(s.tolist())
        # Add random extra vertices to make T
        remaining = [v for v in range(n) if v not in s_set]
        extra_count = t_size - s_size
        if extra_count <= 0 or len(remaining) < extra_count + 1:
            continue
        extras_for_t = remaining[:extra_count]
        t = torch.tensor(sorted(s_set | set(extras_for_t)), dtype=torch.long)
        # Pick a vertex v outside T
        v_candidates = [v for v in remaining if v not in set(extras_for_t)]
        if not v_candidates:
            continue
        v = v_candidates[torch.randint(0, len(v_candidates), (1,)).item()]
        delta_s = (
            util(torch.cat([s, torch.tensor([v])]), obs) - util(s, obs)
        )
        delta_t = (
            util(torch.cat([t, torch.tensor([v])]), obs) - util(t, obs)
        )
        assert delta_s >= delta_t - 1e-5, (
            f"submodularity violated at S={s.tolist()}, T={t.tolist()}, v={v}: "
            f"delta_s={delta_s:.4f}, delta_t={delta_t:.4f}"
        )


def test_property_uniform_weights() -> None:
    """uniform_weights returns the requested shape."""
    weights = uniform_weights(7)
    assert weights.shape == (7,)
    assert torch.all(weights == 1.0)


def test_property_facility_weights_non_negative() -> None:
    """facility_location_weights yields non-negative values."""
    feats = torch.randn((5, 4))
    obs = torch.randn((4,))
    weights = facility_location_weights(feats, obs)
    assert torch.all(weights >= 0.0)


def test_one_minus_one_over_e_on_synthetic() -> None:
    """Greedy achieves >= (1 - 1/e) * OPT on random monotone submodular
    facility-location utilities.

    OPT is computed exactly by brute-force enumeration over all
    ``C(n, k)`` subsets of the persistent graph for each budget.
    """
    from itertools import combinations

    threshold = 1 - 1 / torch.e
    threshold = float(threshold)
    for seed in range(3):
        n = 8
        g = _random_graph(n, 3, seed=seed)
        obs = torch.randn((5, 3))
        util = FacilityLocationUtility(vertex_features=g.vertex_features)
        for budget in (2, 3, 4):
            opt_value = max(
                util(torch.tensor(list(combo), dtype=torch.long), obs)
                for combo in combinations(range(n), budget)
            )
            result = GreedyRetrieval(budget=budget).select(g, obs, utility=util)
            assert result.utility >= threshold * opt_value - 1e-5, (
                f"greedy utility {result.utility:.4f} below "
                f"({threshold:.4f}) * OPT {opt_value:.4f} "
                f"for seed={seed} budget={budget}"
            )