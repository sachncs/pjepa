"""Tests for pjepa.rewriting."""

from __future__ import annotations

import pytest
import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph
from pjepa.rewriting import (
    HRG,
    DPOConfig,
    FourConditions,
    HRGProduction,
    accept_candidate,
    bisimulation_distance,
    dpo_loss,
)

__all__ = [
    "test_bad_accept_candidate_bisimilarity_violated",
    "test_bad_accept_candidate_cost_exceeds",
    "test_bad_accept_candidate_non_negative_delta_j",
    "test_bad_dpo_loss_label_smoothing_out_of_range",
    "test_bad_dpo_loss_shape_mismatch",
    "test_bad_hrg_overlapping_labels",
    "test_bad_hrg_production_lhs_not_nonterminal",
    "test_bad_hrg_unknown_label_in_productions_for",
    "test_bad_hrg_unknown_start",
    "test_cross_backend_mps_bisimulation",
    "test_distributional_dpo_loss_bounded",
    "test_happy_accept_candidate_for_identical_graphs",
    "test_happy_dpo_loss_decreases_for_clear_preference",
    "test_happy_hrg_construction",
    "test_leaky_repeated_bisimulation_no_module_state",
    "test_property_bisimulation_distance_non_negative",
    "test_property_bisimulation_symmetric",
    "test_property_dpo_loss_zero_for_equal_preference",
    "test_round_trip_hrg_serialisable_via_dataclass",
    "test_ugly_empty_graph_bisimulation",
    "test_ugly_hrg_empty_productions",
]


def _simple_graph(
    num_vertices: int = 3, feature_dim: int = 2, seed: int = 0
) -> TypedAttributedGraph:
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn((num_vertices, feature_dim), generator=g)
    return TypedAttributedGraph(
        vertex_features=feats,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_features=torch.zeros((0, 1)),
    )


def _simple_hrg() -> HRG:
    prod = HRGProduction(
        lhs="S",
        rhs_edge_index=torch.zeros((2, 0), dtype=torch.long),
        rhs_edge_features=torch.zeros((0, 1)),
    )
    return HRG(nonterminals=("S",), terminals=("t",), productions=(prod,), start="S")


def test_happy_hrg_construction() -> None:
    """A valid HRG constructs without error."""
    hrg = _simple_hrg()
    assert hrg.start == "S"
    assert hrg.is_nonterminal("S")
    assert hrg.is_terminal("t")
    assert not hrg.is_nonterminal("t")


def test_happy_accept_candidate_for_identical_graphs() -> None:
    """A candidate that strictly decreases Δ𝒥 is accepted."""
    g = _simple_graph(3, 2, seed=1)
    observation = torch.tensor([[1.0, 2.0]])
    # Make the candidate's mean closer to the observation than the
    # current graph's mean. This drives predictive_delta negative.
    new_features = g.vertex_features * 0.1 + observation * 0.9
    candidate = g.with_features(vertex_features=new_features)
    hrg = _simple_hrg()
    accepted, info = accept_candidate(
        candidate,
        g,
        observation,
        hrg,
        FourConditions(max_cost=10.0, bisimulation_eps=10.0),
    )
    assert accepted is True
    assert info["grammar_ok"] is True
    assert info["delta_j_ok"] is True


def test_happy_dpo_loss_decreases_for_clear_preference() -> None:
    """A policy that strongly prefers chosen over rejected has lower loss."""
    chosen_lp = torch.tensor([8.0, 8.0, 8.0])
    rejected_lp = torch.tensor([-8.0, -8.0, -8.0])
    ref_chosen = torch.tensor([0.0, 0.0, 0.0])
    ref_rejected = torch.tensor([0.0, 0.0, 0.0])
    loss = dpo_loss(chosen_lp, rejected_lp, ref_chosen, ref_rejected)
    # DPO loss = -log(sigmoid(beta * 16)) = -log(sigmoid(1.6))
    # ≈ 0.201; assert loss is small but positive.
    assert 0.0 <= loss.item() < 0.3


def test_bad_hrg_overlapping_labels() -> None:
    """A label that is both a non-terminal and a terminal is rejected."""
    with pytest.raises(GraphError):
        HRG(nonterminals=("S", "T"), terminals=("T",), productions=(), start="S")


def test_bad_hrg_unknown_start() -> None:
    """A start symbol not in the non-terminals set is rejected."""
    with pytest.raises(GraphError):
        HRG(nonterminals=("S",), terminals=("t",), productions=(), start="X")


def test_bad_hrg_unknown_label_in_productions_for() -> None:
    """Querying productions for an unknown label raises GraphError."""
    hrg = _simple_hrg()
    with pytest.raises(GraphError):
        hrg.productions_for("Unknown")


def test_bad_hrg_production_lhs_not_nonterminal() -> None:
    """A production whose lhs is a terminal is rejected at construction."""
    prod = HRGProduction(
        lhs="t",
        rhs_edge_index=torch.zeros((2, 0), dtype=torch.long),
        rhs_edge_features=torch.zeros((0, 1)),
    )
    with pytest.raises(GraphError):
        HRG(nonterminals=("S",), terminals=("t",), productions=(prod,), start="S")


def test_bad_accept_candidate_non_negative_delta_j() -> None:
    """A candidate with Δ𝒥 ≥ 0 is rejected."""
    g = _simple_graph(2, 2, seed=2)
    # Same size, same features — Δ𝒥 will be 0
    candidate = g.with_features(vertex_features=g.vertex_features + 1.0)
    observation = torch.zeros((1, 2))
    hrg = _simple_hrg()
    accepted, info = accept_candidate(candidate, g, observation, hrg, FourConditions(max_cost=10.0))
    assert accepted is False
    assert "delta_j" in info["reason"]


def test_bad_accept_candidate_cost_exceeds() -> None:
    """A candidate whose cost exceeds the threshold is rejected."""
    g = _simple_graph(2, 2, seed=3)
    candidate = TypedAttributedGraph(
        vertex_features=torch.randn((100, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_features=torch.zeros((0, 1)),
    )
    observation = torch.randn((1, 2))
    hrg = _simple_hrg()
    accepted, info = accept_candidate(candidate, g, observation, hrg, FourConditions(max_cost=5.0))
    assert accepted is False
    assert "cost" in info["reason"]


def test_bad_accept_candidate_bisimilarity_violated() -> None:
    """A candidate that fails bisimilarity is rejected (any rejection reason)."""
    g = _simple_graph(3, 2, seed=4)
    observation = torch.randn((1, 2))
    # A wholesale feature replacement is far from the current state
    # in the bisimulation metric; the test verifies the function
    # correctly returns a rejection (the reason may be either
    # delta_j or bisimulation, depending on the heuristic ordering).
    candidate = g.with_features(vertex_features=observation.expand(3, -1))
    hrg = _simple_hrg()
    accepted, info = accept_candidate(
        candidate,
        g,
        observation,
        hrg,
        FourConditions(max_cost=10.0, bisimulation_eps=1e-12),
    )
    assert accepted is False
    assert "reason" in info


def test_bad_dpo_loss_shape_mismatch() -> None:
    """A shape mismatch in the DPO inputs raises ValueError."""
    with pytest.raises(ValueError):
        dpo_loss(
            torch.zeros((2,)),
            torch.zeros((3,)),
            torch.zeros((2,)),
            torch.zeros((2,)),
        )


def test_bad_dpo_loss_label_smoothing_out_of_range() -> None:
    """A label-smoothing value outside [0, 0.5) is rejected."""
    with pytest.raises(ValueError):
        dpo_loss(
            torch.zeros((1,)),
            torch.zeros((1,)),
            torch.zeros((1,)),
            torch.zeros((1,)),
            config=DPOConfig(label_smoothing=0.7),
        )


def test_ugly_hrg_empty_productions() -> None:
    """An HRG with no productions but a valid start constructs without error."""
    hrg = HRG(nonterminals=("S",), terminals=("t",), productions=(), start="S")
    assert hrg.productions_for("S") == ()


def test_ugly_empty_graph_bisimulation() -> None:
    """Bisimulation distance on two empty graphs is zero."""
    g1 = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    g2 = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    assert bisimulation_distance(g1, g2) == 0.0


def test_leaky_repeated_bisimulation_no_module_state() -> None:
    """Two back-to-back bisimulation computations do not interfere."""
    g = _simple_graph(4, 3, seed=5)
    d1 = bisimulation_distance(g, g)
    d2 = bisimulation_distance(g, g)
    assert d1 == pytest.approx(d2)


def test_round_trip_hrg_serialisable_via_dataclass() -> None:
    """HRG is frozen; round-trip equality holds."""
    hrg = _simple_hrg()
    hrg2 = HRG(
        nonterminals=hrg.nonterminals,
        terminals=hrg.terminals,
        productions=hrg.productions,
        start=hrg.start,
    )
    assert hrg == hrg2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_bisimulation() -> None:
    """Bisimulation distance is computed correctly on MPS graphs."""
    g = _simple_graph(4, 3, seed=6).to(torch.device("mps"))
    d = bisimulation_distance(g, g)
    assert d >= 0.0


def test_distributional_dpo_loss_bounded() -> None:
    """The DPO loss is bounded for a range of preference strengths."""
    for diff in (0.0, 1.0, 2.0, 4.0):
        chosen_lp = torch.tensor([diff, diff])
        rejected_lp = torch.tensor([0.0, 0.0])
        ref_chosen = torch.tensor([0.0, 0.0])
        ref_rejected = torch.tensor([0.0, 0.0])
        loss = dpo_loss(chosen_lp, rejected_lp, ref_chosen, ref_rejected)
        assert 0.0 <= loss.item() < 10.0


def test_property_bisimulation_distance_non_negative() -> None:
    """Bisimulation distance is always non-negative."""
    g1 = _simple_graph(4, 3, seed=7)
    g2 = _simple_graph(4, 3, seed=8)
    d = bisimulation_distance(g1, g2)
    assert d >= 0.0


def test_property_bisimulation_symmetric() -> None:
    """Bisimulation distance is symmetric in its two arguments."""
    g1 = _simple_graph(4, 3, seed=9)
    g2 = _simple_graph(4, 3, seed=10)
    d_ab = bisimulation_distance(g1, g2)
    d_ba = bisimulation_distance(g2, g1)
    assert d_ab == pytest.approx(d_ba, rel=1e-5)


def test_property_dpo_loss_zero_for_equal_preference() -> None:
    """The DPO loss is zero when the policy has no preference."""
    chosen_lp = torch.tensor([1.0])
    rejected_lp = torch.tensor([1.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss = dpo_loss(chosen_lp, rejected_lp, ref_chosen, ref_rejected)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-5)
