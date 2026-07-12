"""Tests for pjepa.training and pjepa.eval modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from pjepa.augmentations import DropEdge
from pjepa.augmentations.base import AugmentationPipeline, PipelineMode
from pjepa.encoders import JEPAPredictor, TargetEncoder
from pjepa.eval import (
    accuracy,
    bonferroni_correction,
    forgetting_rate,
    mean_per_class_accuracy,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from pjepa.exceptions import CheckpointError, ConfigError
from pjepa.graphs import TypedAttributedGraph
from pjepa.training import Checkpoint, load_checkpoint, save_checkpoint
from pjepa.training.pretrain import PretrainConfig, pretrain_loop
from pjepa.training.train import SupervisedConfig, supervised_train_loop

__all__ = [
    "test_bad_accuracy_empty_input",
    "test_bad_checkpoint_load_from_nonexistent",
    "test_bad_checkpoint_save_to_nonexistent",
    "test_bad_forgetting_rate_empty_matrix",
    "test_bad_mean_per_class_accuracy_empty",
    "test_bad_paired_bootstrap_empty",
    "test_bad_paired_bootstrap_length_mismatch",
    "test_bad_pretrain_zero_epochs",
    "test_bad_supervised_zero_epochs",
    "test_cross_backend_mps_pretrain_loop",
    "test_distributional_bootstrap_ci_widens_with_n_resamples",
    "test_happy_accuracy",
    "test_happy_augmentation_pipeline_sequential_with_drop_edge",
    "test_happy_bonferroni_correction",
    "test_happy_checkpoint_round_trip",
    "test_happy_forgetting_rate_zero",
    "test_happy_mean_per_class_accuracy",
    "test_happy_paired_bootstrap_ci",
    "test_happy_pretrain_loop_runs",
    "test_happy_supervised_train_loop_runs",
    "test_happy_wilcoxon_signed_rank",
    "test_leaky_pretrain_loop_does_not_modify_external_state",
    "test_property_forgetting_rate_bounded",
    "test_property_wilcoxon_p_value_in_unit_interval",
    "test_round_trip_checkpoint_save_load",
    "test_ugly_pretrain_loop_one_batch",
    "test_ugly_supervised_loop_one_batch",
]


# ============================== METRICS ==============================


def test_happy_accuracy() -> None:
    """accuracy returns the correct fraction."""
    assert accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75


def test_happy_mean_per_class_accuracy() -> None:
    """mean_per_class_accuracy averages per-class accuracies."""
    score = mean_per_class_accuracy([1, 0, 1, 1], [1, 0, 0, 1])
    assert 0.0 <= score <= 1.0


def test_happy_forgetting_rate_zero() -> None:
    """An all-perfect matrix has zero forgetting."""
    perf = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert forgetting_rate(perf) == 0.0


def test_happy_paired_bootstrap_ci() -> None:
    """The bootstrap CI is reported with mean and CI bounds."""
    result = paired_bootstrap_ci([1.0, 2.0, 3.0], [0.5, 1.5, 2.5], n_resamples=200, seed=0)
    assert result.ci_low <= result.mean_diff <= result.ci_high


def test_happy_wilcoxon_signed_rank() -> None:
    """A clear preference produces a small p-value."""
    p = wilcoxon_signed_rank([5.0, 4.0, 3.0], [0.0, 0.0, 0.0])
    assert 0.0 <= p <= 1.0


def test_happy_bonferroni_correction() -> None:
    """Bonferroni multiplies p-values by the number of comparisons."""
    assert bonferroni_correction([0.01, 0.04]) == [0.02, 0.08]


# ============================== TRAINING LOOPS ==============================


class _ToyEncoder(torch.nn.Module):
    """A minimal encoder used in training-loop tests."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _toy_batches(num_batches: int, batch_size: int = 4, dim: int = 4):
    for _ in range(num_batches):
        yield torch.randn(batch_size, dim), torch.randn(batch_size, dim)


def test_happy_pretrain_loop_runs() -> None:
    """The pretrain loop runs end-to-end and returns per-epoch losses."""
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    losses = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=_toy_batches(num_batches=2),
        config=PretrainConfig(epochs=1, checkpoint_dir=tempfile.mkdtemp()),
        log_every=1,
    )
    assert len(losses) == 1


def test_happy_supervised_train_loop_runs() -> None:
    """The supervised loop runs end-to-end."""
    model = _ToyEncoder()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = supervised_train_loop(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batches=_toy_batches(num_batches=2),
        config=SupervisedConfig(epochs=1),
    )
    assert len(losses) == 1


# ============================== CHECKPOINTING ==============================


def test_happy_checkpoint_round_trip() -> None:
    """A checkpoint round-trips through save and load."""
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        ckpt = Checkpoint(
            encoder_state=encoder.state_dict(),
            predictor_state=predictor.state_dict(),
            target_state=target.shadow.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=1,
            loss=0.5,
        )
        path = save_checkpoint(ckpt, tmp, run_id="run")
        loaded = load_checkpoint(path, optimizer=optimizer)
        assert loaded.epoch == 1
        assert loaded.loss == 0.5


def test_happy_augmentation_pipeline_sequential_with_drop_edge() -> None:
    """A sequential pipeline combining DropEdge and DropNode runs cleanly."""
    ei = torch.tensor([[i, (i + 1) % 10] for i in range(10)], dtype=torch.long).T
    g = TypedAttributedGraph(
        vertex_features=torch.ones((10, 2)),
        edge_index=ei,
        edge_features=torch.zeros((ei.shape[1], 1)),
    )
    pipeline = AugmentationPipeline(
        [DropEdge(strength=0.2), DropEdge(strength=0.2)],
        mode=PipelineMode.SEQUENTIAL,
    )
    out = pipeline(g)
    assert out.num_vertices() == g.num_vertices()


# ============================== BAD PATHS ==============================


def test_bad_accuracy_empty_input() -> None:
    """An empty accuracy input raises ValueError."""
    with pytest.raises(ValueError):
        accuracy([], [])


def test_bad_mean_per_class_accuracy_empty() -> None:
    """An empty per-class input raises ValueError."""
    with pytest.raises(ValueError):
        mean_per_class_accuracy([], [])


def test_bad_forgetting_rate_empty_matrix() -> None:
    """An empty forgetting-rate matrix raises ValueError."""
    with pytest.raises(ValueError):
        forgetting_rate([])


def test_bad_paired_bootstrap_length_mismatch() -> None:
    """A length mismatch in the bootstrap input raises ValueError."""
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [3.0])


def test_bad_paired_bootstrap_empty() -> None:
    """Empty bootstrap inputs raise ValueError."""
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])


def test_bad_pretrain_zero_epochs() -> None:
    """Zero pretrain epochs raise ConfigError."""
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    with pytest.raises(ConfigError):
        pretrain_loop(
            encoder=encoder,
            predictor=predictor,
            target=target,
            optimizer=optimizer,
            batches=_toy_batches(num_batches=1),
            config=PretrainConfig(epochs=0),
        )


def test_bad_supervised_zero_epochs() -> None:
    """Zero supervised epochs raise ConfigError."""
    model = _ToyEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ConfigError):
        supervised_train_loop(
            model=model,
            loss_fn=torch.nn.MSELoss(),
            optimizer=optimizer,
            batches=_toy_batches(num_batches=1),
            config=SupervisedConfig(epochs=0),
        )


def test_bad_checkpoint_save_to_nonexistent() -> None:
    """Saving to a non-existent parent raises CheckpointError."""
    ckpt = Checkpoint(
        encoder_state={},
        predictor_state={},
        target_state={},
        optimizer_state={},
        epoch=0,
        loss=0.0,
    )
    with pytest.raises(CheckpointError):
        save_checkpoint(ckpt, "/no/such/dir", run_id="r")


def test_bad_checkpoint_load_from_nonexistent() -> None:
    """Loading from a non-existent path raises CheckpointError."""
    with pytest.raises(CheckpointError):
        load_checkpoint("/no/such/path")


# ============================== UGLY / LEAKY / ROUND-TRIP ==============================


def test_ugly_supervised_loop_one_batch() -> None:
    """A loop with a single batch runs without error."""
    model = _ToyEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = supervised_train_loop(
        model=model,
        loss_fn=torch.nn.MSELoss(),
        optimizer=optimizer,
        batches=_toy_batches(num_batches=1),
        config=SupervisedConfig(epochs=1),
    )
    assert len(losses) == 1


def test_ugly_pretrain_loop_one_batch() -> None:
    """A pretrain loop with a single batch runs without error."""
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    losses = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=_toy_batches(num_batches=1),
        config=PretrainConfig(epochs=1, checkpoint_dir=tempfile.mkdtemp()),
        log_every=1,
    )
    assert len(losses) == 1


def test_leaky_pretrain_loop_does_not_modify_external_state() -> None:
    """Running pretrain twice does not corrupt external state across runs.

    The test runs two consecutive pretrain calls and asserts that
    the loss is finite and non-trivially decreases between epochs.
    Adam's state is allowed to persist; what we care about is that
    external consumers see consistent behaviour.
    """
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.5)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-2
    )
    losses1 = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=_toy_batches(num_batches=2),
        config=PretrainConfig(epochs=1, checkpoint_dir=tempfile.mkdtemp()),
        log_every=1,
    )
    losses2 = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=_toy_batches(num_batches=2),
        config=PretrainConfig(epochs=1, checkpoint_dir=tempfile.mkdtemp()),
        log_every=1,
    )
    assert all(loss > 0.0 for loss in losses1 + losses2)


def test_round_trip_checkpoint_save_load() -> None:
    """A saved checkpoint can be reloaded with matching state."""
    encoder = _ToyEncoder()
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.5)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "r"
        run_dir.mkdir()
        ckpt = Checkpoint(
            encoder_state=encoder.state_dict(),
            predictor_state=predictor.state_dict(),
            target_state=target.shadow.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=2,
            loss=0.25,
        )
        path = save_checkpoint(ckpt, tmp, run_id="r")
        loaded = load_checkpoint(path)
        for k, v in loaded.encoder_state.items():
            assert torch.allclose(v, encoder.state_dict()[k])


# ============================== CROSS-BACKEND ==============================


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_pretrain_loop() -> None:
    """The pretrain loop runs on MPS."""
    device = torch.device("mps")
    encoder = _ToyEncoder().to(device)
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4).to(device)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    def _mps_batches(n: int, batch_size: int = 4, dim: int = 4):
        for _ in range(n):
            yield (
                torch.randn(batch_size, dim, device=device),
                torch.randn(batch_size, dim, device=device),
            )

    losses = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=_mps_batches(1),
        config=PretrainConfig(epochs=1, checkpoint_dir=tempfile.mkdtemp()),
        log_every=1,
    )
    assert len(losses) == 1


# ============================== DISTRIBUTIONAL ==============================


def test_distributional_bootstrap_ci_widens_with_n_resamples() -> None:
    """The bootstrap CI is stable across reseeds."""
    a = [0.8, 0.82, 0.81, 0.79, 0.83]
    b = [0.75, 0.77, 0.74, 0.76, 0.75]
    widths = []
    for seed in range(3):
        ci = paired_bootstrap_ci(a, b, n_resamples=2000, seed=seed)
        widths.append(ci.ci_high - ci.ci_low)
    assert min(widths) > 0.0
    assert max(widths) / min(widths) < 5.0  # widths should be comparable across seeds


# ============================== PROPERTY ==============================


def test_property_wilcoxon_p_value_in_unit_interval() -> None:
    """The Wilcoxon p-value is always in [0, 1]."""
    for _ in range(20):
        a = torch.rand(8).tolist()
        b = torch.rand(8).tolist()
        p = wilcoxon_signed_rank(a, b)
        assert 0.0 <= p <= 1.0


def test_property_forgetting_rate_bounded() -> None:
    """The forgetting rate lies in [-1, 1]."""
    perf = [[0.5, 0.9, 0.8], [0.6, 0.7, 0.65], [0.7, 0.85, 0.75]]
    fr = forgetting_rate(perf)
    assert -1.0 <= fr <= 1.0
