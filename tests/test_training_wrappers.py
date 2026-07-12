"""Tests for pjepa.training wrappers (SWA, TTA, ensemble, distillation)."""

from __future__ import annotations

import pytest
import torch

from pjepa.augmentations import TensorDropFeature
from pjepa.exceptions import ConfigError
from pjepa.training import (
    DistillationConfig,
    DistillationLoss,
    Ensemble,
    SWAConfig,
    SWAWrapper,
    TTAConfig,
    TTAWrapper,
    distill_kl,
)

__all__ = [
    "test_bad_distill_kl_shape_mismatch",
    "test_bad_distill_kl_zero_temperature",
    "test_bad_ensemble_empty_models",
    "test_bad_ensemble_unknown_aggregator",
    "test_bad_swa_negative_start_epoch",
    "test_bad_tta_zero_n_aug",
    "test_cross_backend_mps_ensemble",
    "test_happy_distill_kl_zero_for_identical_logits",
    "test_happy_distillation_loss_runs",
    "test_happy_ensemble_hard_vote",
    "test_happy_ensemble_rank_avg",
    "test_happy_ensemble_soft_vote",
    "test_happy_swa_records_snapshots",
    "test_happy_tta_runs_multiple_augs",
    "test_property_swa_averaged_is_average",
    "test_round_trip_swa_apply_to",
    "test_ugly_swa_no_snapshots_below_start",
    "test_ugly_tta_only_includes_original",
]


def _toy_model() -> torch.nn.Module:
    """Minimal model that returns a fixed tensor for a given input."""
    return torch.nn.Linear(4, 3)


def test_happy_swa_records_snapshots() -> None:
    """SWA records snapshots after the start epoch."""
    model = _toy_model()
    swa = SWAWrapper(model, SWAConfig(start_epoch=5))
    for epoch in range(10):
        swa.update(epoch)
    assert swa.snapshot_count == 5


def test_happy_tta_runs_multiple_augs() -> None:
    """TTA runs ``n_aug`` augmented passes plus the original."""
    model = _toy_model()
    aug = TensorDropFeature(strength=0.5, generator=torch.Generator().manual_seed(0))
    tta = TTAWrapper(model, aug, TTAConfig(n_aug=3, include_original=True))
    x = torch.randn((2, 4))
    out = tta(x)
    assert out.shape == (2, 3)


def test_happy_ensemble_soft_vote() -> None:
    """Soft-vote ensemble averages per-model logits."""
    torch.manual_seed(0)
    m1 = _toy_model()
    m2 = _toy_model()
    ensemble = Ensemble([m1, m2], "soft_vote")
    x = torch.randn((2, 4))
    out = ensemble(x)
    expected = (m1(x) + m2(x)) / 2.0
    assert torch.allclose(out, expected)


def test_happy_ensemble_hard_vote() -> None:
    """Hard-vote ensemble returns per-sample mode of argmax."""
    m1 = torch.nn.Linear(4, 3)
    m2 = torch.nn.Linear(4, 3)
    m3 = torch.nn.Linear(4, 3)
    ensemble = Ensemble([m1, m2, m3], "hard_vote")
    x = torch.randn((1, 4))
    out = ensemble(x)
    assert out.shape == (1,)


def test_happy_ensemble_rank_avg() -> None:
    """Rank-avg ensemble averages per-model ranks."""
    m1 = _toy_model()
    m2 = _toy_model()
    ensemble = Ensemble([m1, m2], "rank_avg")
    x = torch.randn((1, 4))
    out = ensemble(x)
    assert out.shape == (1, 3)


def test_happy_distill_kl_zero_for_identical_logits() -> None:
    """The KL divergence is zero when student matches teacher."""
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    loss = distill_kl(logits, logits.clone(), temperature=2.0)
    assert float(loss.item()) < 1e-5


def test_happy_distillation_loss_runs() -> None:
    """The combined loss returns a finite scalar."""
    student = torch.randn((4, 3), requires_grad=True)
    teacher = torch.randn((4, 3))
    targets = torch.tensor([0, 1, 2, 1])
    loss_fn = DistillationLoss(DistillationConfig(temperature=2.0, alpha=0.5))
    loss = loss_fn(student, teacher, targets)
    assert torch.isfinite(loss)


def test_bad_swa_negative_start_epoch() -> None:
    """A negative start_epoch is rejected."""
    with pytest.raises(ConfigError):
        SWAWrapper(_toy_model(), SWAConfig(start_epoch=-1))


def test_bad_tta_zero_n_aug() -> None:
    """A zero ``n_aug`` is rejected."""
    with pytest.raises(ConfigError):
        TTAConfig(n_aug=0)


def test_bad_ensemble_empty_models() -> None:
    """An empty ensemble is rejected."""
    with pytest.raises(ConfigError):
        Ensemble([])


def test_bad_ensemble_unknown_aggregator() -> None:
    """An unknown aggregator is rejected."""
    with pytest.raises(ConfigError):
        Ensemble([_toy_model()], aggregator="bogus")


def test_bad_distill_kl_zero_temperature() -> None:
    """A zero temperature is rejected."""
    with pytest.raises(ConfigError):
        distill_kl(torch.zeros((1, 3)), torch.zeros((1, 3)), temperature=0.0)


def test_bad_distill_kl_shape_mismatch() -> None:
    """A shape mismatch is rejected."""
    with pytest.raises(ConfigError):
        distill_kl(torch.zeros((1, 3)), torch.zeros((1, 5)))


def test_ugly_swa_no_snapshots_below_start() -> None:
    """No snapshots are taken before the start epoch."""
    swa = SWAWrapper(_toy_model(), SWAConfig(start_epoch=10))
    for epoch in range(5):
        swa.update(epoch)
    assert swa.snapshot_count == 0


def test_ugly_tta_only_includes_original() -> None:
    """TTA with n_aug=1 and include_original=False runs exactly one pass."""
    model = _toy_model()
    aug = TensorDropFeature(strength=0.5)
    tta = TTAWrapper(model, aug, TTAConfig(n_aug=1, include_original=False))
    x = torch.randn((2, 4))
    out = tta(x)
    assert out.shape == (2, 3)


def test_round_trip_swa_apply_to() -> None:
    """SWA averaged weights can be applied to the live model."""
    model = _toy_model()
    swa = SWAWrapper(model, SWAConfig(start_epoch=0))
    with torch.no_grad():
        for v in (1.0, 2.0, 3.0):
            model.weight.fill_(v)
            model.bias.fill_(0.0)
            swa.update(epoch=0)
    swa.apply_to()
    # After applying the average, the model parameters equal the
    # snapshot average (mean of 1.0, 2.0, 3.0 = 2.0).
    assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_ensemble() -> None:
    """Ensemble forward runs on MPS."""
    device = torch.device("mps")
    m1 = _toy_model().to(device)
    m2 = _toy_model().to(device)
    ensemble = Ensemble([m1, m2])
    out = ensemble(torch.randn((1, 4), device=device))
    assert out.device.type == "mps"


def test_property_swa_averaged_is_average() -> None:
    """The averaged parameter equals the mean of the snapshots."""
    model = _toy_model()
    swa = SWAWrapper(model, SWAConfig(start_epoch=0))
    # Snapshot three distinct states.
    with torch.no_grad():
        for v in (1.0, 2.0, 3.0):
            model.weight.fill_(v)
            swa.update(epoch=0)
    avg_weight = swa.averaged_parameters()["weight"]
    assert torch.allclose(avg_weight, torch.full_like(avg_weight, 2.0))
