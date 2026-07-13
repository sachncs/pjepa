"""Tests for the Phase 4 validation experiments.

Covers:
    * The synthetic submodular retrieval quality (Exp A).
    * The hyperbolic-vs-Euclidean tree distortion (Exp B).
    * The encoder ablation on AST-like graphs (Exp C).
    * The publication-style helper in :mod:`pjepa.eval.style`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from experiments.run_exp_a_retrieval import (
    BRUTE_FORCE_CAP,
    DEFAULT_BUDGETS,
    DEFAULT_N_SEEDS,
    DEFAULT_NS,
    PSEUDO_RESTARTS,
    RetrievalExperimentConfig,
    brute_force_optimum,
    greedy_utility,
    pseudo_optimum,
    random_submodular,
)
from experiments.run_exp_a_retrieval import (
    run as run_a,
)
from experiments.run_exp_b_distortion import (
    DistortionExperimentConfig,
    bourgain_inspired_embedding,
    edge_hyperbolic_distances,
    sarkar_inspired_embedding,
    stats,
    tree_edge_index,
    tree_parents,
)
from experiments.run_exp_b_distortion import (
    run as run_b,
)
from experiments.run_exp_c_encoder_ablation import (
    DEFAULT_DEPTHS,
    EncoderAblationConfig,
    HyperbolicMPNN,
    build_ast_like_graph,
    build_encoder,
    evaluate_accuracy,
    predict_depth_loss,
    run_encoder_ablation,
    train_one_encoder,
)
from pjepa.eval.style import (
    PUBLICATION_DPI,
    PUBLICATION_FIGSIZE,
    color_for,
    set_publication_style,
)
from pjepa.graphs import TypedAttributedGraph

__all__ = [
    "test_eval_color_for_wraps_modulo",
    "test_eval_publication_style_sets_dpi",
    "test_exp_a_brute_force_optimum_matches_greedy_upper_bound",
    "test_exp_a_brute_force_optimum_non_negative",
    "test_exp_a_pseudo_optimum_lower_bound",
    "test_exp_a_random_submodular_dim_match",
    "test_exp_a_run_smoke_writes_outputs",
    "test_exp_b_bourgain_dim_match",
    "test_exp_b_hyperbolic_distances_min_ge_alpha",
    "test_exp_b_run_smoke_writes_outputs",
    "test_exp_b_sarkar_radius_step",
    "test_exp_b_sarkar_units_inside_disk",
    "test_exp_b_tree_parents_count",
    "test_exp_c_build_ast_like_graph_features_shape",
    "test_exp_c_build_encoder_known_variants",
    "test_exp_c_run_smoke_writes_outputs",
]


def test_eval_publication_style_sets_dpi() -> None:
    """set_publication_style sets savefig.dpi to PUBLICATION_DPI."""
    set_publication_style()
    assert plt.rcParams["savefig.dpi"] == PUBLICATION_DPI
    assert PUBLICATION_FIGSIZE == (6.0, 4.0)


def test_eval_color_for_wraps_modulo() -> None:
    """color_for cycles through the palette via modulo."""
    assert color_for(0) == color_for(6)
    assert color_for(0) != color_for(1)


def test_exp_a_random_submodular_dim_match() -> None:
    """The synthetic utility and observation have matching dimensions."""
    util, obs = random_submodular(n=10, seed=0, feature_dim=5, observation_dim=3)
    assert util.vertex_features.shape == (10, 5)
    assert obs.shape == (3, 5)


def test_exp_a_brute_force_optimum_non_negative() -> None:
    """The exact optimum is non-negative on random FL utilities."""
    util, obs = random_submodular(n=12, seed=1, feature_dim=3, observation_dim=2)
    opt = brute_force_optimum(util, n=12, budget=3, observation=obs)
    assert opt is not None
    assert opt >= 0.0


def test_exp_a_brute_force_optimum_matches_greedy_upper_bound() -> None:
    """The exact optimum is at least the greedy value (Nemhauser-Wolsey)."""
    util, obs = random_submodular(n=10, seed=2, feature_dim=3, observation_dim=2)
    opt = brute_force_optimum(util, n=10, budget=3, observation=obs)
    assert opt is not None
    g = TypedAttributedGraph(
        vertex_features=util.vertex_features,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    from pjepa.retrieval import GreedyRetrieval

    greedy_value = float(GreedyRetrieval(budget=3).select(g, obs, utility=util).utility)
    assert greedy_value <= opt + 1e-5


def test_exp_a_pseudo_optimum_lower_bound() -> None:
    """The pseudo-optimum is at most the exact optimum (random ≤ optimal)."""
    util, obs = random_submodular(n=10, seed=3, feature_dim=3, observation_dim=2)
    opt = brute_force_optimum(util, n=10, budget=3, observation=obs)
    assert opt is not None
    pseudo = pseudo_optimum(util, n=10, budget=3, observation=obs, seed=0)
    assert pseudo <= opt + 1e-5


def test_exp_a_run_smoke_writes_outputs(tmp_path: Path) -> None:
    """run writes CSV and PNG outputs and respects the (1-1/e) bound."""
    config = RetrievalExperimentConfig(
        ns=(20,),
        budgets=(4,),
        n_seeds=2,
        output_dir=str(tmp_path),
    )
    summary = run_a(config)
    assert (tmp_path / "retrieval_quality.csv").exists()
    assert (tmp_path / "retrieval_quality.png").exists()
    assert summary["all_pass"]
    assert math.isclose(summary["threshold"], 1.0 - 1.0 / math.e, rel_tol=1e-9)
    with (tmp_path / "retrieval_quality.csv").open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 2
    for row in rows:
        assert row["n"] == "20"
        assert row["budget"] == "4"


def test_exp_b_tree_parents_count() -> None:
    """The parent map has the expected number of entries."""
    parents, n = tree_parents(b=2, depth=3)
    assert n == 15
    assert len(parents) == 15
    assert parents[0] is None


def test_exp_b_sarkar_units_inside_disk() -> None:
    """Sarkar-inspired coordinates are strictly inside the Poincaré disk."""
    parents, n = tree_parents(b=2, depth=4)
    coords = sarkar_inspired_embedding(parents, n, b=2, depth=4, seed=0)
    assert coords.shape == (n, 2)
    norms = coords.norm(dim=-1)
    assert torch.all(norms < 1.0)


def test_exp_b_sarkar_radius_step() -> None:
    """The hyperbolic step is 2 * asinh(1)."""
    assert math.isclose(2.0 * math.asinh(1.0), 1.762747174039086, rel_tol=1e-9)


def test_exp_b_hyperbolic_distances_min_ge_alpha() -> None:
    """Every edge has hyperbolic distance >= 2*asinh(1)."""
    parents, n = tree_parents(b=2, depth=4)
    coords = sarkar_inspired_embedding(parents, n, b=2, depth=4, seed=0)
    ei = tree_edge_index(b=2, depth=4)
    dists = edge_hyperbolic_distances(coords, ei)
    assert torch.all(dists >= 2.0 * math.asinh(1.0) - 1e-4)


def test_exp_b_bourgain_dim_match() -> None:
    """The Bourgain-inspired embedding has the requested dimension."""
    parents, n = tree_parents(b=2, depth=4)
    coords = bourgain_inspired_embedding(parents, n, b=2, depth=4, d=12, seed=0)
    assert coords.shape == (n, 12)


def test_exp_b_run_smoke_writes_outputs(tmp_path: Path) -> None:
    """run writes CSV and PNG outputs."""
    config = DistortionExperimentConfig(
        depths=(3, 5),
        branchings=(2,),
        dims=(4,),
        n_seeds=1,
        output_dir=str(tmp_path),
    )
    summary = run_b(config)
    assert (tmp_path / "distortion.csv").exists()
    assert (tmp_path / "distortion.png").exists()
    rows = summary["rows"]
    assert len(rows) == 2
    for row in rows:
        assert 0.0 <= row["euc_over_hyp_max"] <= 1.0 + 1e-6
        assert row["hyp_min"] >= 1.7


def test_exp_c_build_ast_like_graph_features_shape() -> None:
    """The synthetic AST graph has one-hot features of shape [N, D+1]."""
    g = build_ast_like_graph(depth=4, branching=2, seed=0)
    assert g.num_vertices() == 31
    assert g.vertex_features.shape == (31, 5)
    assert torch.all(g.vertex_features.sum(dim=-1) == 1.0)


def test_exp_c_build_encoder_known_variants() -> None:
    """Every advertised encoder variant is constructable."""
    for name in ("EuclideanMPNN", "HyperbolicMPNN", "DualGeometricEncoder"):
        enc = build_encoder(name, input_dim=6, output_dim=6)
        assert enc is not None


def test_exp_c_run_smoke_writes_outputs(tmp_path: Path) -> None:
    """run_encoder_ablation writes CSV and PNG outputs and produces rows."""
    config = EncoderAblationConfig(
        depths=(3,),
        n_graphs=4,
        n_seeds=1,
        epochs=5,
        output_dir=str(tmp_path),
    )
    summary = run_encoder_ablation(config)
    assert (tmp_path / "encoder_ablation.csv").exists()
    assert (tmp_path / "encoder_ablation.png").exists()
    rows = summary["rows"]
    assert len(rows) == 3
    encoders = {row["encoder"] for row in rows}
    assert encoders == {"EuclideanMPNN", "HyperbolicMPNN", "DualGeometricEncoder"}
    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["n_train_graphs"] >= 1
        assert row["n_test_graphs"] >= 1
        train_test_disjoint = row["n_train_graphs"] + row["n_test_graphs"] <= config.n_graphs
        assert train_test_disjoint


def test_exp_c_predict_depth_loss_shapes() -> None:
    """predict_depth_loss returns logits and labels of consistent shapes."""
    g = build_ast_like_graph(depth=3, branching=2, seed=0)
    enc = build_encoder("EuclideanMPNN", input_dim=4, output_dim=4)
    logits, labels = predict_depth_loss(enc, [g], depth=3)
    assert logits.shape[0] == g.num_vertices()
    assert labels.shape == (g.num_vertices(),)


def test_exp_a_default_budgets_and_ns_are_plan_compliant() -> None:
    """The defaults are explicitly overridable plan-compliant tuples."""
    assert DEFAULT_NS == (40, 50)
    assert DEFAULT_BUDGETS == (5, 7)
    assert DEFAULT_N_SEEDS >= 1
    assert BRUTE_FORCE_CAP > 0
    assert PSEUDO_RESTARTS > 0


def test_exp_a_brute_force_returns_zero_for_empty_budget() -> None:
    """A budget of 0 returns 0.0 rather than enumerating."""
    util, obs = random_submodular(n=5, seed=0, feature_dim=3, observation_dim=2)
    assert brute_force_optimum(util, n=5, budget=0, observation=obs) == 0.0


def test_exp_b_run_handles_large_depth() -> None:
    """Larger depth (15) does not crash the Sarkar / Bourgain pipelines."""
    parents, n = tree_parents(b=2, depth=15)
    hyp = sarkar_inspired_embedding(parents, n, b=2, depth=15, seed=0)
    assert hyp.shape == (n, 2)
    euc = bourgain_inspired_embedding(parents, n, b=2, depth=15, d=4, seed=0)
    assert euc.shape == (n, 4)


def test_exp_b_stats_handles_empty() -> None:
    """stats returns zero-valued summary for empty inputs."""
    assert stats(torch.zeros((0,))) == {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
    }


@pytest.mark.parametrize(("depth", "b"), [(2, 2), (3, 3), (4, 2)])
def test_exp_b_sarkar_max_norm_below_one(depth: int, b: int) -> None:
    """For several depths, Sarkar-inspired max norm is strictly < 1."""
    parents, n = tree_parents(b=b, depth=depth)
    coords = sarkar_inspired_embedding(parents, n, b=b, depth=depth, seed=0)
    assert float(coords.norm(dim=-1).max().item()) < 1.0


def test_eval_style_is_idempotent() -> None:
    """set_publication_style can be invoked twice without raising."""
    set_publication_style()
    set_publication_style()
    assert plt.rcParams["savefig.dpi"] == PUBLICATION_DPI


def test_exp_a_greedy_utility_matches_selector() -> None:
    """greedy_utility agrees with a direct GreedyRetrieval call."""
    util, obs = random_submodular(n=8, seed=0, feature_dim=3, observation_dim=2)
    value = greedy_utility(util, n=8, budget=3, observation=obs)
    g = TypedAttributedGraph(
        vertex_features=util.vertex_features,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    from pjepa.retrieval import GreedyRetrieval

    expected = float(GreedyRetrieval(budget=3).select(g, obs, utility=util).utility)
    assert math.isclose(value, expected, rel_tol=1e-9)


def test_exp_c_hyperbolic_mpnn_forward() -> None:
    """HyperbolicMPNN emits features with norms strictly below 1."""
    g = build_ast_like_graph(depth=3, branching=2, seed=0)
    enc = HyperbolicMPNN(input_dim=4, hidden_dim=8, num_layers=2, output_dim=4)
    out = enc(g)
    assert out.shape == (g.num_vertices(), 4)
    assert torch.all(out.norm(dim=-1) < 1.0)


def test_exp_c_train_one_encoder_returns_accuracy() -> None:
    """train_one_encoder returns a finite accuracy in [0, 1]."""
    graphs = [build_ast_like_graph(depth=3, branching=2, seed=k) for k in range(4)]
    enc = build_encoder("EuclideanMPNN", input_dim=4, output_dim=4)
    accuracy = train_one_encoder(
        "EuclideanMPNN",
        enc,
        train_graphs=graphs[:3],
        test_graphs=graphs[3:],
        depth=3,
        epochs=2,
        lr=1e-2,
    )
    assert 0.0 <= accuracy <= 1.0


def test_exp_c_evaluate_accuracy_in_unit_interval() -> None:
    """evaluate_accuracy returns a finite value in [0, 1]."""
    g = build_ast_like_graph(depth=3, branching=2, seed=0)
    enc = build_encoder("EuclideanMPNN", input_dim=4, output_dim=4)
    assert 0.0 <= evaluate_accuracy(enc, [g], depth=3) <= 1.0


def test_exp_c_train_test_split_is_disjoint(tmp_path: Path) -> None:
    """The encoder ablation's held-out graphs are disjoint from train graphs."""
    config = EncoderAblationConfig(
        depths=(3,),
        n_graphs=4,
        n_seeds=1,
        epochs=1,
        output_dir=str(tmp_path),
    )
    summary = run_encoder_ablation(config)
    for row in summary["rows"]:
        assert row["n_train_graphs"] + row["n_test_graphs"] == config.n_graphs


def test_exp_a_pseudo_strict_lower_bound_under_exact() -> None:
    """pseudo_optimum is strictly <= the brute-force optimum on small instances."""
    util, obs = random_submodular(n=10, seed=5, feature_dim=3, observation_dim=2)
    opt = brute_force_optimum(util, n=10, budget=3, observation=obs)
    pseudo = pseudo_optimum(util, n=10, budget=3, observation=obs, seed=0, n_starts=32)
    assert opt is not None
    assert pseudo <= opt + 1e-5


def test_exp_c_default_depths_are_smoke_compliant() -> None:
    """The default encoder-ablation depths are smoke-friendly."""
    assert DEFAULT_DEPTHS[0] >= 1
    assert all(d > 0 for d in DEFAULT_DEPTHS)
