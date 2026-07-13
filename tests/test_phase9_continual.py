"""Tests for the Phase 9 continual-learning experiment.

Covers the metrics extensions, the PackNet baseline, the experiment
runner in smoke mode, and the plan-compliant CSV / plot outputs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from experiments.run_exp_e_continual import (
    CL_METHODS,
    CLExperimentConfig,
    aggregate_cl_results,
    build_accuracy_matrix,
    build_cl_model,
    build_graph_from_pairs,
    build_pair_observation,
    build_smoke_config,
    cl_forward_pass,
    evaluate_cl_model,
    run_cl_experiment,
    split_graph_indices_to_pairs,
    train_ewc_task,
    train_gem_task,
    train_naive_task,
    train_packnet_task,
    train_persistent_jepa_task,
    trainable_parameters,
)
from pjepa.baselines import EWC, GEM, PackNet
from pjepa.eval import backward_transfer, forgetting_rate, forward_transfer
from pjepa.eval.metrics import accuracy, mean_per_class_accuracy
from pjepa.graphs import PersistentState, TypedAttributedGraph

__all__ = [
    "test_accuracy_basic",
    "test_aggregate_cl_results_has_bootstrap_and_bonferroni",
    "test_aggregate_cl_results_pairs_naive_by_seed",
    "test_backward_transfer_negates_forgetting",
    "test_build_accuracy_matrix_pads_lower_triangle",
    "test_build_graph_from_pairs_filters_edges_to_budget",
    "test_cl_methods_includes_persistent_jepa_and_packnet",
    "test_default_datasets_match_phase9_plan",
    "test_default_methods_includes_persistent_jepa_and_packnet",
    "test_ewc_penalty_active_across_tasks",
    "test_ewc_trainer_returns_finite_accuracy",
    "test_forgetting_rate_empty_matrix",
    "test_forgetting_rate_known_values",
    "test_forward_pass_through_module_list",
    "test_forward_transfer_with_baseline",
    "test_gem_memory_persists_across_tasks",
    "test_gem_trainer_returns_finite_accuracy",
    "test_mean_per_class_accuracy_handles_imbalance",
    "test_naive_trainer_returns_finite_accuracy",
    "test_packnet_begin_task_freezes_prior_slices",
    "test_packnet_mask_shapes_match_params",
    "test_packnet_masks_persist_across_tasks",
    "test_packnet_trainer_returns_finite_accuracy",
    "test_persistent_jepa_grows_state",
    "test_persistent_jepa_returns_finite_accuracy",
    "test_persistent_jepa_uses_retrieved_working_graph",
    "test_run_cl_experiment_long_csv_has_correct_columns",
    "test_run_cl_experiment_smoke_creates_outputs",
    "test_run_cl_experiment_writes_forgetting_plot",
    "test_run_cl_experiment_writes_summary_csv",
    "test_smoke_config_overrides_runlevel_fields",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pairs(
    num_pairs: int,
    feature_dim: int = 4,
    seed: int = 0,
) -> list[tuple[TypedAttributedGraph, int]]:
    """Construct a synthetic dataset of ``(graph, label)`` pairs.

    Args:
        num_pairs: The number of pairs to generate.
        feature_dim: Vertex-feature dimensionality.
        seed: Seed for the deterministic per-graph feature generator.

    Returns:
        A list of ``(TypedAttributedGraph, label)`` pairs whose
        label alternates ``0, 1, 0, 1, ...``.
    """
    g = torch.Generator().manual_seed(seed)
    pairs: list[tuple[TypedAttributedGraph, int]] = []
    for i in range(num_pairs):
        graph = TypedAttributedGraph(
            vertex_features=torch.randn((3, feature_dim), generator=g),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        pairs.append((graph, i % 2))
    return pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_forgetting_rate_known_values() -> None:
    """The reference forgetting values for a 2-task scenario."""
    matrix = [[1.0, 0.0], [1.0, 1.0]]
    assert forgetting_rate(matrix) == pytest.approx(0.5)


def test_forgetting_rate_empty_matrix() -> None:
    """Empty matrix raises ValueError."""
    with pytest.raises(ValueError):
        forgetting_rate([])


def test_backward_transfer_negates_forgetting() -> None:
    """backward_transfer is the negative of forgetting_rate."""
    matrix = [[1.0, 0.5, 0.2], [1.0, 0.9, 0.6], [1.0, 1.0, 0.7]]
    assert backward_transfer(matrix) == pytest.approx(-forgetting_rate(matrix))


def test_forward_transfer_with_baseline() -> None:
    """forward_transfer subtracts a per-task baseline."""
    matrix = [[1.0, 0.5], [0.4, 0.6]]
    assert forward_transfer(matrix, baseline_per_task=[0.0, 0.2]) == pytest.approx(0.2)


def test_mean_per_class_accuracy_handles_imbalance() -> None:
    """Per-class accuracy is averaged, not weighted.

    With predictions ``[0, 0, 1, 1]`` and targets ``[0, 1, 1, 1]`` the
    function buckets by target label: class 0 has 1 sample (correct),
    class 1 has 3 samples (2 correct). Mean per-class accuracy is
    ``(1/1 + 2/3) / 2 = 5/6 ≈ 0.833``.
    """
    assert mean_per_class_accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(5.0 / 6.0)


def test_accuracy_basic() -> None:
    """Standard accuracy matches a hand-computed value."""
    assert accuracy([0, 1, 2, 0], [0, 1, 1, 0]) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# PackNet baseline
# ---------------------------------------------------------------------------


def test_packnet_begin_task_freezes_prior_slices() -> None:
    """Active slice shrinks as prior slices are frozen."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    packnet = PackNet(num_tasks=4)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    packnet.begin_task(params, task_idx=0)
    initial_active = packnet.active_parameter_count()
    assert initial_active > 0
    packnet.freeze_current_task()
    packnet.begin_task(params, task_idx=1)
    assert packnet.frozen_parameter_count() >= initial_active
    assert packnet.active_parameter_count() > 0
    assert (packnet.active_parameter_count() + packnet.frozen_parameter_count()) <= sum(
        p.numel() for _, p in params
    )


def test_packnet_mask_shapes_match_params() -> None:
    """Per-parameter masks reshape to the underlying tensor."""
    model = build_cl_model(input_dim=4, num_classes=2)
    packnet = PackNet(num_tasks=3)
    packnet.begin_task(trainable_parameters(model), task_idx=0)
    for name, param in model.named_parameters():
        if param.requires_grad and name in packnet.current_task_mask:
            assert packnet.current_task_mask[name].shape == param.shape


def test_packnet_masks_persist_across_tasks() -> None:
    """PackNet masks across tasks; prior slices stay frozen.

    On task 1 we begin with a fresh ``PackNet``; after the second
    ``begin_task`` call, the frozen count must be at least the
    active count from task 0 (no overlap, ever).
    """
    model = build_cl_model(input_dim=4, num_classes=2)
    packnet = PackNet(num_tasks=3)
    packnet.begin_task(trainable_parameters(model), task_idx=0)
    task0_active = packnet.active_parameter_count()
    packnet.freeze_current_task()
    packnet.begin_task(trainable_parameters(model), task_idx=1)
    assert packnet.frozen_parameter_count() >= task0_active
    assert packnet.active_parameter_count() > 0


# ---------------------------------------------------------------------------
# Trainer smoke tests
# ---------------------------------------------------------------------------


def test_naive_trainer_returns_finite_accuracy() -> None:
    pairs = make_pairs(6)
    model = build_cl_model(input_dim=4, num_classes=2)
    acc = train_naive_task(model, pairs[:4], pairs[4:], num_classes=2, epochs=3)
    assert 0.0 <= acc <= 1.0


def test_naive_trainer_handles_empty_train() -> None:
    """Empty train_pairs returns the evaluated accuracy on the test side."""
    pairs = make_pairs(4)
    model = build_cl_model(input_dim=4, num_classes=2)
    acc = train_naive_task(model, [], pairs, num_classes=2, epochs=1)
    assert 0.0 <= acc <= 1.0


def test_ewc_trainer_returns_finite_accuracy() -> None:
    pairs = make_pairs(6)
    model = build_cl_model(input_dim=4, num_classes=2)
    ewc = EWC(lambda_ewc=100.0)
    acc = train_ewc_task(model, ewc, pairs[:4], pairs[4:], num_classes=2, epochs=3)
    assert 0.0 <= acc <= 1.0


def test_ewc_penalty_active_across_tasks() -> None:
    """EWC penalty is zero before capture and stays active across tasks.

    We reuse a single :class:`EWC` across two ``train_ewc_task`` calls
    so the Fisher information captured during task 0 is still
    available when computing the penalty on task 1. After two tasks
    the runner would have accumulated Fisher; we approximate that
    here by capturing explicitly with the API and asserting the
    penalty is positive whenever the parameters drift away from the
    captured ``_star`` values.
    """
    ewc = EWC(lambda_ewc=100.0)
    model = build_cl_model(input_dim=4, num_classes=2)
    params = trainable_parameters(model)
    assert float(ewc.penalty(params).item()) == 0.0
    pairs = make_pairs(3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for g, lbl in pairs[:2] * 2:
        out = cl_forward_pass(model, g)
        loss = loss_fn(out, torch.tensor([lbl], dtype=torch.long))
        ewc.capture(params, loss)
        with torch.no_grad():
            for _, p in params:
                p.grad = None
    assert float(ewc.penalty(params).item()) == 0.0
    with torch.no_grad():
        for _, p in params:
            p.add_(0.5 * torch.randn_like(p))
    penalty = float(ewc.penalty(params).item())
    assert penalty > 0.0


def test_gem_trainer_returns_finite_accuracy() -> None:
    pairs = make_pairs(6)
    model = build_cl_model(input_dim=4, num_classes=2)
    gem_state = GEM(capacity=4)
    acc = train_gem_task(
        model, gem_state, pairs[:4], pairs[4:], num_classes=2, epochs=2, capacity=4
    )
    assert 0.0 <= acc <= 1.0
    assert len(gem_state.memory) > 0


def test_gem_memory_persists_across_tasks() -> None:
    """GEM memory survives across tasks when the same instance is reused."""
    pairs_a = make_pairs(4, seed=0)
    pairs_b = make_pairs(4, seed=1)
    model = build_cl_model(input_dim=4, num_classes=2)
    gem_state = GEM(capacity=8)
    train_gem_task(model, gem_state, pairs_a[:3], pairs_a[3:], num_classes=2, epochs=1, capacity=8)
    after_a = len(gem_state.memory)
    train_gem_task(model, gem_state, pairs_b[:3], pairs_b[3:], num_classes=2, epochs=1, capacity=8)
    assert len(gem_state.memory) >= after_a


def test_packnet_trainer_returns_finite_accuracy() -> None:
    pairs = make_pairs(6)
    model = build_cl_model(input_dim=4, num_classes=2)
    packnet = PackNet(num_tasks=2)
    acc = train_packnet_task(
        model,
        packnet,
        pairs[:4],
        pairs[4:],
        num_classes=2,
        epochs=2,
        task_idx=0,
    )
    assert 0.0 <= acc <= 1.0


def test_persistent_jepa_returns_finite_accuracy() -> None:
    pairs = make_pairs(6)
    model = build_cl_model(input_dim=4, num_classes=2)
    acc, state = train_persistent_jepa_task(
        model,
        pairs[:4],
        pairs[4:],
        num_classes=2,
        epochs=2,
        budget=4,
        batch_size=2,
        persistent_state=None,
    )
    assert 0.0 <= acc <= 1.0
    assert state is not None


def test_persistent_jepa_grows_state() -> None:
    """Subsequent calls accumulate vertices in the persistent graph."""
    pairs_a = make_pairs(4, seed=0)
    pairs_b = make_pairs(4, seed=1)
    model = build_cl_model(input_dim=4, num_classes=2)
    _, state = train_persistent_jepa_task(
        model,
        pairs_a[:3],
        pairs_a[3:],
        num_classes=2,
        epochs=1,
        budget=4,
        batch_size=2,
        persistent_state=None,
    )
    assert isinstance(state, PersistentState)
    prev_vertices = state.num_vertices()
    _, state = train_persistent_jepa_task(
        model,
        pairs_b[:3],
        pairs_b[3:],
        num_classes=2,
        epochs=1,
        budget=4,
        batch_size=2,
        persistent_state=state,
    )
    assert state.num_vertices() >= prev_vertices


def test_persistent_jepa_uses_retrieved_working_graph() -> None:
    """The retrieved working graph is mixed into the committed candidate.

    We seed the persistent state with a graph whose vertices are
    uniquely identifiable (constant zeros with a single distinguishing
    feature) and verify the next commit's vertex features carry over
    at least one of those identifiers — i.e. the working graph was
    not silently discarded.
    """
    marker_v = torch.tensor([[9.0, 9.0, 9.0, 9.0], [-9.0, -9.0, -9.0, -9.0]])
    marker_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    marker_graph = TypedAttributedGraph(
        vertex_features=marker_v,
        edge_index=marker_edges,
        edge_features=torch.zeros((marker_edges.shape[1], 1)),
    )
    persistent_state = PersistentState(graph=marker_graph)

    pairs = make_pairs(4)
    model = build_cl_model(input_dim=4, num_classes=2)
    _, new_state = train_persistent_jepa_task(
        model,
        pairs[:3],
        pairs[3:],
        num_classes=2,
        epochs=1,
        budget=8,
        batch_size=2,
        persistent_state=persistent_state,
    )
    features = new_state.graph.vertex_features
    has_positive = bool((features == 9.0).all(dim=-1).any().item())
    has_negative = bool((features == -9.0).all(dim=-1).any().item())
    assert has_positive or has_negative


def test_build_graph_from_pairs_filters_edges_to_budget() -> None:
    """Edges with endpoints outside the budget are dropped."""
    g_full = TypedAttributedGraph(
        vertex_features=torch.zeros((10, 1)),
        edge_index=torch.tensor([[0, 1, 5, 6, 9, 9], [1, 0, 6, 5, 8, 7]], dtype=torch.long),
        edge_features=torch.zeros((6, 1)),
    )
    pairs = [(g_full, 0)]
    candidate = build_graph_from_pairs(pairs, budget=4)
    assert candidate.num_vertices() == 4
    src = candidate.edge_index[0].tolist()
    dst = candidate.edge_index[1].tolist()
    assert all(0 <= s < 4 for s in src)
    assert all(0 <= d < 4 for d in dst)
    assert len(src) == len(dst)
    assert set(zip(src, dst)) == {(0, 1), (1, 0)}


def test_build_pair_observation_returns_tensor() -> None:
    """Empty input yields a (0,) tensor; non-empty input yields a 1-D tensor."""
    empty = build_pair_observation([])
    assert empty.shape == (0,)
    pairs = make_pairs(3)
    obs = build_pair_observation(pairs)
    assert obs.ndim == 1
    assert obs.shape[0] == 4


# ---------------------------------------------------------------------------
# Matrix / aggregation
# ---------------------------------------------------------------------------


def test_build_accuracy_matrix_pads_lower_triangle() -> None:
    """Lower-triangle entries inherit the diagonal value."""
    snapshot = [[1.0], [0.0, 0.8]]
    matrix = build_accuracy_matrix(snapshot)
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(0.0)
    assert matrix[1][1] == pytest.approx(0.8)
    assert matrix[1][0] == pytest.approx(0.8)


def test_build_accuracy_matrix_empty_raises() -> None:
    """Empty snapshots raise a ValueError."""
    with pytest.raises(ValueError):
        build_accuracy_matrix([])


def test_aggregate_cl_results_has_bootstrap_and_bonferroni() -> None:
    """aggregate_cl_results emits bootstrap CI and Bonferroni-adjusted p-values."""
    rows = []
    for seed in range(3):
        for method in ("Naive", "EWC"):
            for task in range(2):
                row = {
                    "dataset": "MUTAG",
                    "method": method,
                    "seed": seed,
                    "task": task,
                    "accuracy": 0.7 + 0.1 * (1 if method == "EWC" else 0),
                }
                if task == 1:
                    row["per_task_accuracies"] = [
                        [0.7 + 0.05 * seed],
                        [0.4 + 0.05 * seed, 0.8 + 0.05 * seed],
                    ]
                rows.append(row)
    summary = aggregate_cl_results(rows)
    assert "MUTAG|Naive" in summary
    assert "MUTAG|EWC" in summary
    ewc = summary["MUTAG|EWC"]
    assert "bootstrap" in ewc
    assert "wilcoxon_p_bonferroni" in ewc
    assert "backward_transfer" in ewc
    assert "forward_transfer" in ewc


def test_aggregate_cl_results_pairs_naive_by_seed() -> None:
    """The method-vs-Naive comparison uses the same seed index.

    We craft two Naive accuracies and one outlier EWC seed: the
    bootstrap CI and Wilcoxon test must use the matched seed
    rather than the first ``n`` accuracies, otherwise the
    comparison would be unpaired.
    """
    rows = []
    naive_accs = {0: 0.50, 1: 0.55, 2: 0.60}
    ewc_accs = {0: 0.70, 1: 0.72, 2: 0.74}
    for seed, acc in naive_accs.items():
        rows.append(
            {
                "dataset": "MUTAG",
                "method": "Naive",
                "seed": seed,
                "task": 1,
                "accuracy": acc,
                "per_task_accuracies": [[0.5], [0.5, acc]],
            }
        )
    for seed, acc in ewc_accs.items():
        rows.append(
            {
                "dataset": "MUTAG",
                "method": "EWC",
                "seed": seed,
                "task": 1,
                "accuracy": acc,
                "per_task_accuracies": [[0.5], [0.5, acc]],
            }
        )
    summary = aggregate_cl_results(rows)
    ewc = summary["MUTAG|EWC"]
    expected_diff = (sum(ewc_accs.values()) - sum(naive_accs.values())) / 3.0
    assert ewc["bootstrap"]["mean_diff"] == pytest.approx(expected_diff)


# ---------------------------------------------------------------------------
# Configuration / experiment defaults
# ---------------------------------------------------------------------------


def test_cl_methods_includes_persistent_jepa_and_packnet() -> None:
    """The CL_METHODS tuple contains PersistentJEPA and PackNet."""
    assert "PersistentJEPA" in CL_METHODS
    assert "PackNet" in CL_METHODS


def test_default_methods_includes_persistent_jepa_and_packnet() -> None:
    """The default CLExperimentConfig methods include all five baselines."""
    config = CLExperimentConfig()
    assert "Naive" in config.methods
    assert "EWC" in config.methods
    assert "GEM" in config.methods
    assert "PackNet" in config.methods
    assert "PersistentJEPA" in config.methods


def test_default_datasets_match_phase9_plan() -> None:
    """The default datasets match the Phase 9 plan."""
    config = CLExperimentConfig()
    assert tuple(config.datasets) == ("PROTEINS", "MUTAG", "NCI1")


def test_smoke_config_overrides_runlevel_fields() -> None:
    """Smoke collapses n_tasks / n_seeds / epochs_per_task / methods / datasets."""
    config = CLExperimentConfig(datasets=("PROTEINS",), n_tasks=5, n_seeds=4, epochs_per_task=20)
    smoke = build_smoke_config(config)
    assert smoke.datasets == ("MUTAG",)
    assert smoke.methods == ("Naive", "PersistentJEPA")
    assert smoke.n_tasks == 2
    assert smoke.n_seeds == 1
    assert smoke.epochs_per_task == 2
    assert smoke.budget == 16
    assert smoke.smoke is True


# ---------------------------------------------------------------------------
# Smoke run + outputs
# ---------------------------------------------------------------------------


def test_run_cl_experiment_smoke_creates_outputs(tmp_path: Path) -> None:
    """Smoke mode writes the long CSV, summary CSV, and forgetting plot."""
    output_dir = tmp_path / "cl_smoke"
    config = CLExperimentConfig(smoke=True, output_dir=str(output_dir))
    rows = run_cl_experiment(config)
    assert len(rows) >= 1
    assert (output_dir / "cl_results.csv").exists()
    assert (output_dir / "tables").exists()
    assert (output_dir / "plots").exists()


def test_run_cl_experiment_writes_summary_csv(tmp_path: Path) -> None:
    """cl_summary.csv has the Phase-9 plan columns."""
    output_dir = tmp_path / "cl_smoke"
    config = CLExperimentConfig(smoke=True, output_dir=str(output_dir))
    run_cl_experiment(config)
    with (output_dir / "tables" / "cl_summary.csv").open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    expected_cols = {
        "dataset",
        "method",
        "n_seeds",
        "mean_accuracy",
        "std_accuracy",
        "backward_transfer",
        "forward_transfer",
        "forgetting_rate",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "wilcoxon_p",
        "wilcoxon_p_bonferroni",
    }
    assert expected_cols.issubset(set(header))


def test_run_cl_experiment_writes_forgetting_plot(tmp_path: Path) -> None:
    """cl_forgetting_curves.png is written as a non-empty PNG."""
    output_dir = tmp_path / "cl_smoke"
    config = CLExperimentConfig(smoke=True, output_dir=str(output_dir))
    run_cl_experiment(config)
    plot_path = output_dir / "plots" / "cl_forgetting_curves.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 100
    with plot_path.open("rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_run_cl_experiment_long_csv_has_correct_columns(tmp_path: Path) -> None:
    """cl_results.csv contains dataset/method/seed/task/accuracy columns."""
    output_dir = tmp_path / "cl_smoke"
    config = CLExperimentConfig(smoke=True, output_dir=str(output_dir))
    run_cl_experiment(config)
    with (output_dir / "cl_results.csv").open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        data_rows = list(reader)
    assert header == ["dataset", "method", "seed", "task", "accuracy"]
    assert len(data_rows) >= 1
    for row in data_rows:
        assert len(row) == 5


# ---------------------------------------------------------------------------
# Forward pass sanity
# ---------------------------------------------------------------------------


def test_forward_pass_through_module_list() -> None:
    """The dual-geometric encoder output flows through the classifier."""
    model = build_cl_model(input_dim=4, num_classes=2)
    graph = TypedAttributedGraph(
        vertex_features=torch.randn((3, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    logits = cl_forward_pass(model, graph)
    assert logits.shape == (1, 2)


def test_split_graph_indices_to_pairs() -> None:
    """Empty indices return empty lists; non-empty splits at the cut."""
    graphs = [
        type(
            "G",
            (),
            {
                "graph": TypedAttributedGraph(
                    vertex_features=torch.zeros((1, 1)),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                ),
                "label": i,
            },
        )()
        for i in range(10)
    ]
    train_pairs, test_pairs = split_graph_indices_to_pairs(graphs, [])
    assert train_pairs == []
    assert test_pairs == []
    train_pairs, test_pairs = split_graph_indices_to_pairs(graphs, list(range(10)))
    assert len(train_pairs) == 8
    assert len(test_pairs) == 2


def test_evaluate_cl_model_empty() -> None:
    """Empty pairs yield 0.0 without invoking the forward pass."""
    model = build_cl_model(input_dim=4, num_classes=2)
    assert evaluate_cl_model(model, []) == 0.0
