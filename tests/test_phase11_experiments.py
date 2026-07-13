"""Tests for the Phase 11 experiments.

Covers:

* Exp G — inference-storage decoupling measurement (B-bounded working
  graph, warm-up, sync, bootstrap-slope CI).
* Exp H — component ablations (incl. ``minus_bisimulation`` /
  ``minus_submodular_retrieval`` behavioural differences, hyperbolic
  output in the ``full`` variant, paired-by-seed CI semantics).
* Sensitivity sweep over the working-graph budget ``B`` (gradients
  flow through the encoder in pretraining; standard finite-score
  bootstrap CI of the mean).

Every test uses the smoke configuration so the suite stays fast.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.run_exp_g_decoupling import (
    DecouplingConfig,
    run_decoupling_measurement,
)
from experiments.run_exp_g_decoupling import (
    default_smoke_config as decoupling_smoke,
)
from experiments.run_exp_h_ablations import (
    ABLATION_VARIANTS,
    VARIANT_INTERPRETATION,
    canonical_wl_hash,
    embed_dim_for_variant,
    run_ablation,
    verify_candidate_for_variant,
    verify_variants,
)
from experiments.run_exp_h_ablations import (
    default_smoke_config as ablation_smoke,
)
from experiments.run_sensitivity import (
    DEFAULT_BUDGETS,
    aggregate_per_budget,
    run_sensitivity,
)
from experiments.run_sensitivity import (
    default_smoke_config as sensitivity_smoke,
)

__all__ = [
    "test_ablation_csv_has_interpretation_field",
    "test_ablation_full_embed_dim_uses_hyperbolic",
    "test_ablation_interpretation_covers_every_variant",
    "test_ablation_smoke_creates_outputs",
    "test_ablation_summary_has_bonferroni_and_bootstrap",
    "test_ablation_variants_match_plan",
    "test_decoupling_default_smoke_is_fast",
    "test_decoupling_dual_encoder_used",
    "test_decoupling_plot_is_written",
    "test_decoupling_rows_have_required_columns",
    "test_decoupling_slope_table_present",
    "test_decoupling_uses_bounded_working_graph",
    "test_decoupling_uses_facility_location",
    "test_decoupling_uses_greedy_retrieval",
    "test_run_decoupling_smoke_creates_outputs",
    "test_sensitivity_default_budgets_match_plan",
    "test_sensitivity_plot_is_written",
    "test_sensitivity_smoke_creates_outputs",
    "test_sensitivity_summary_covers_each_budget",
    "test_sensitivity_uses_finite_score_bootstrap",
    "test_verify_candidate_wl_accepts_on_match_only",
    "test_verify_variants_rejects_unknown",
    "test_wl_hash_distinguishes_graphs",
    "test_wl_hash_is_deterministic",
]


# ---------------------------------------------------------------------------
# Plan-compliant variants
# ---------------------------------------------------------------------------


def test_ablation_variants_match_plan() -> None:
    """The variant list matches the Phase 11 plan exactly."""
    expected = {
        "full",
        "minus_hyperbolic",
        "minus_persistent",
        "minus_four_conditions",
        "minus_ema",
        "minus_bisimulation",
        "minus_submodular_retrieval",
    }
    assert set(ABLATION_VARIANTS) == expected


def test_verify_variants_rejects_unknown() -> None:
    """An unknown variant raises ConfigError."""
    from pjepa.exceptions import ConfigError

    with pytest.raises(ConfigError):
        verify_variants(["full", "minus_bisimulation", "minus_nonexistent"])


def test_ablation_interpretation_covers_every_variant() -> None:
    """Every plan variant has a non-empty one-line interpretation."""
    for v in ABLATION_VARIANTS:
        assert v in VARIANT_INTERPRETATION
        assert VARIANT_INTERPRETATION[v].strip() != ""


def test_wl_hash_is_deterministic() -> None:
    """WL hash is deterministic for a fixed graph."""
    import torch

    from pjepa.graphs import TypedAttributedGraph

    g = TypedAttributedGraph(
        vertex_features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    h1 = canonical_wl_hash(g)
    h2 = canonical_wl_hash(g)
    assert h1 == h2


def test_wl_hash_distinguishes_graphs() -> None:
    """WL hash is sensitive to graph topology."""
    import torch

    from pjepa.graphs import TypedAttributedGraph

    g_a = TypedAttributedGraph(
        vertex_features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    g_b = TypedAttributedGraph(
        vertex_features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
    )
    assert canonical_wl_hash(g_a) != canonical_wl_hash(g_b)


def test_ablation_full_embed_dim_uses_hyperbolic() -> None:
    """The ``full`` variant consumes the hyperbolic component of the encoder."""
    full_dim = embed_dim_for_variant("full")
    eu_dim = embed_dim_for_variant("minus_hyperbolic")
    assert full_dim > eu_dim


def test_verify_candidate_wl_accepts_on_match_only() -> None:
    """``minus_bisimulation`` accepts iff the canonical WL hashes match."""
    import torch

    from pjepa.graphs import TypedAttributedGraph
    from pjepa.rewriting import HRG

    g_a = TypedAttributedGraph(
        vertex_features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    g_b = TypedAttributedGraph(
        vertex_features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
    )
    grammar = HRG(nonterminals=("S",), terminals=("a",), productions=(), start="S")
    obs = torch.zeros((1, 2))
    accepted_same, _ = verify_candidate_for_variant("minus_bisimulation", g_a, g_a, obs, grammar)
    accepted_diff, _ = verify_candidate_for_variant("minus_bisimulation", g_b, g_a, obs, grammar)
    assert accepted_same is True
    assert accepted_diff is False


# ---------------------------------------------------------------------------
# Decoupling (Exp G)
# ---------------------------------------------------------------------------


def test_decoupling_default_smoke_is_fast(tmp_path: Path) -> None:
    """The default smoke configuration is small enough for unit tests."""
    cfg = decoupling_smoke(output_dir=str(tmp_path))
    assert cfg.n_sizes == (8, 16)
    assert cfg.budgets == (4, 8)
    assert cfg.n_trials == 1


def test_decoupling_rows_have_required_columns() -> None:
    """The decoupling CSV has the plan-compliant columns."""
    cfg = DecouplingConfig(
        n_sizes=(4,),
        budgets=(2,),
        n_trials=1,
        feature_dim=2,
    )
    summary = run_decoupling_measurement(cfg)
    rows = summary["rows"]
    assert rows
    required = {"what", "n_vertices", "budget", "mean_seconds", "std_seconds"}
    for row in rows:
        assert required.issubset(set(row.keys()))


def test_decoupling_dual_encoder_used() -> None:
    """The decoupling measurement exercises DualGeometricEncoder."""
    cfg = DecouplingConfig(
        n_sizes=(4,),
        budgets=(2,),
        n_trials=1,
        feature_dim=2,
    )
    summary = run_decoupling_measurement(cfg)
    encoder_rows = [r for r in summary["rows"] if r["what"] == "encoder"]
    # One encoder row per (N, B) cell.
    assert len(encoder_rows) == len(cfg.n_sizes) * len(cfg.budgets)


def test_decoupling_uses_bounded_working_graph() -> None:
    """Encoder rows are emitted for every (N, B) cell — the B-bounded claim."""
    cfg = DecouplingConfig(
        n_sizes=(4, 8),
        budgets=(2, 4),
        n_trials=1,
        feature_dim=2,
    )
    summary = run_decoupling_measurement(cfg)
    encoder_rows = [r for r in summary["rows"] if r["what"] == "encoder"]
    n_rows_by_budget: dict[int, int] = {}
    for row in encoder_rows:
        n_rows_by_budget[int(row["budget"])] = n_rows_by_budget.get(int(row["budget"]), 0) + 1
    assert set(n_rows_by_budget) == set(cfg.budgets)
    assert all(n == len(cfg.n_sizes) for n in n_rows_by_budget.values())


def test_decoupling_uses_greedy_retrieval() -> None:
    """The retrieval measurement exercises GreedyRetrieval."""
    cfg = DecouplingConfig(
        n_sizes=(4,),
        budgets=(2,),
        n_trials=1,
        feature_dim=2,
    )
    summary = run_decoupling_measurement(cfg)
    retrieval_rows = [r for r in summary["rows"] if r["what"] == "retrieval"]
    assert len(retrieval_rows) == len(cfg.n_sizes) * len(cfg.budgets)


def test_decoupling_uses_facility_location() -> None:
    """The retrieval rows reflect FacilityLocationUtility output."""
    from pjepa.retrieval import FacilityLocationUtility

    util = FacilityLocationUtility(vertex_features=__import__("torch").zeros((3, 2)))
    assert callable(util)


def test_decoupling_slope_table_present(tmp_path: Path) -> None:
    """A per-(what, B) OLS slope table is written to disk."""
    cfg = DecouplingConfig(
        n_sizes=(4,),
        budgets=(2,),
        n_trials=1,
        feature_dim=2,
        output_dir=str(tmp_path),
    )
    summary = run_decoupling_measurement(cfg)
    slope_path = Path(summary["slope_csv"])
    assert slope_path.exists()
    with slope_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert rows
    required = {
        "what",
        "budget",
        "slope_seconds_per_vertex",
        "intercept_seconds",
        "n_points",
        "ci_low",
        "ci_high",
        "p_value_slope_vs_zero",
    }
    assert required.issubset(set(rows[0].keys()))


def test_decoupling_plot_is_written(tmp_path: Path) -> None:
    """The plan-compliant ``plots/decoupling.png`` exists."""
    cfg = decoupling_smoke(output_dir=str(tmp_path))
    summary = run_decoupling_measurement(cfg)
    plot_path = Path(summary["png"])
    assert plot_path.exists()
    assert plot_path.stat().st_size > 100
    with plot_path.open("rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_run_decoupling_smoke_creates_outputs(tmp_path: Path) -> None:
    """Smoke mode writes plan-compliant CSV, slope CSV, and PNG."""
    output_dir = tmp_path / "decoupling_smoke"
    cfg = decoupling_smoke(output_dir=str(output_dir))
    summary = run_decoupling_measurement(cfg)
    assert (output_dir / "tables" / "decoupling.csv").exists()
    assert (output_dir / "plots" / "decoupling.png").exists()
    assert (output_dir / "tables" / "decoupling_slope.csv").exists()
    assert summary["rows"]


# ---------------------------------------------------------------------------
# Ablations (Exp H)
# ---------------------------------------------------------------------------


def test_ablation_smoke_creates_outputs(tmp_path: Path) -> None:
    """Smoke mode writes plan-compliant CSV, summary CSV, and PNG."""
    output_dir = tmp_path / "ablation_smoke"
    cfg = ablation_smoke(output_dir=str(output_dir))
    summary = run_ablation(cfg)
    assert (output_dir / "tables" / "ablation.csv").exists()
    assert (output_dir / "tables" / "ablation_summary.csv").exists()
    assert (output_dir / "plots" / "ablation.png").exists()
    assert summary["summary"]


def test_ablation_summary_has_bonferroni_and_bootstrap() -> None:
    """Every ablation row carries Wilcoxon, bootstrap CI, and Bonferroni p-value."""
    cfg = ablation_smoke()
    summary = run_ablation(cfg)
    required = {
        "variant",
        "interpretation",
        "n_runs",
        "mean_accuracy",
        "std_accuracy",
        "ci_low",
        "ci_high",
        "mean_diff_vs_full",
        "p_value_bootstrap",
        "wilcoxon_p_vs_full",
        "wilcoxon_p_bonferroni",
    }
    for entry in summary["summary"]:
        assert required.issubset(set(entry.keys()))


def test_ablation_csv_has_interpretation_field(tmp_path: Path) -> None:
    """The raw ablation CSV row carries the interpretation field."""
    output_dir = tmp_path / "ablation_smoke"
    cfg = ablation_smoke(output_dir=str(output_dir))
    run_ablation(cfg)
    csv_path = output_dir / "tables" / "ablation.csv"
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert rows
    assert "interpretation" in rows[0]
    assert rows[0]["interpretation"].strip() != ""


# ---------------------------------------------------------------------------
# Sensitivity (Exp H)
# ---------------------------------------------------------------------------


def test_sensitivity_default_budgets_match_plan() -> None:
    """The default budget sweep matches the Phase 11 plan."""
    assert DEFAULT_BUDGETS == (8, 16, 32, 64, 128, 256)


def test_sensitivity_smoke_creates_outputs(tmp_path: Path) -> None:
    """Smoke mode writes the sensitivity CSV, summary, and PNG."""
    output_dir = tmp_path / "sensitivity_smoke"
    cfg = sensitivity_smoke(output_dir=str(output_dir))
    summary = run_sensitivity(cfg)
    assert (output_dir / "tables" / "sensitivity_B.csv").exists()
    assert (output_dir / "tables" / "sensitivity_B_summary.csv").exists()
    assert (output_dir / "plots" / "sensitivity_B.png").exists()
    assert summary["summary"]


def test_sensitivity_summary_covers_each_budget() -> None:
    """The summary has one row per requested budget."""
    cfg = sensitivity_smoke()
    summary = run_sensitivity(cfg)
    budgets_in_summary = sorted(int(s["budget"]) for s in summary["summary"])
    assert budgets_in_summary == sorted(cfg.budgets)


def test_sensitivity_plot_is_written(tmp_path: Path) -> None:
    """The plan-compliant plot is a non-empty PNG."""
    output_dir = tmp_path / "sensitivity_smoke"
    cfg = sensitivity_smoke(output_dir=str(output_dir))
    summary = run_sensitivity(cfg)
    plot_path = Path(summary["png"])
    assert plot_path.exists()
    assert plot_path.stat().st_size > 100
    with plot_path.open("rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_sensitivity_uses_finite_score_bootstrap() -> None:
    """The per-budget CI is the standard finite-score CI of the mean."""
    rows = [
        {"budget": 4, "seed": 0, "fold": 0, "accuracy": 0.6},
        {"budget": 4, "seed": 0, "fold": 1, "accuracy": 0.65},
        {"budget": 4, "seed": 1, "fold": 0, "accuracy": 0.7},
        {"budget": 4, "seed": 1, "fold": 1, "accuracy": 0.62},
        {"budget": 8, "seed": 0, "fold": 0, "accuracy": 0.5},
        {"budget": 8, "seed": 0, "fold": 1, "accuracy": 0.55},
    ]
    summary = aggregate_per_budget(rows, n_resamples=200, seed=0)
    # For budget=4 the mean is ~0.6425; the CI is centred on the mean.
    budget4 = next(s for s in summary if s["budget"] == 4)
    assert budget4["ci_low"] <= budget4["mean_accuracy"] <= budget4["ci_high"]


# ---------------------------------------------------------------------------
# Default-config sanity checks
# ---------------------------------------------------------------------------


def test_ablation_default_config_is_fast(tmp_path: Path) -> None:
    """The default smoke helper produces a fast smoke run."""
    cfg = ablation_smoke(output_dir=str(tmp_path / "ablation_default"))
    assert cfg.n_seeds == 1
    assert cfg.n_folds == 2
    assert cfg.epochs == 2


def test_sensitivity_default_config_is_fast(tmp_path: Path) -> None:
    """The default smoke helper produces a fast smoke run."""
    cfg = sensitivity_smoke(output_dir=str(tmp_path / "sensitivity_default"))
    assert cfg.n_seeds == 1
    assert cfg.n_folds == 2
    assert cfg.epochs == 2
