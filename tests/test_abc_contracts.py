"""Tests for the abstract-base-class contracts.

These tests verify that the polymorphic base classes correctly
enforce their abstract methods. Any subclass that fails to
override an abstract method cannot be instantiated; the
exception raised is :class:`TypeError` (Python's standard
behaviour for ABC subclasses with unimplemented abstract
methods).
"""

from __future__ import annotations

import pytest
import torch

from pjepa.augmentations.base import Transform
from pjepa.encoders import DualGeometric, Predictor, Target
from pjepa.encoders.base import Encoder
from pjepa.encoders.jepa_predictor import Head
from pjepa.retrieval.utility import Facility, InfoGain, Utility
from pjepa.rewriting.four_conditions import Criterion, FourConditions
from pjepa.scheduler.buffer import Buffer, Step, Storage
from pjepa.scheduler.cadence import Cadence, Sleep

__all__ = [
    "test_augmentation_base_cannot_be_instantiated",
    "test_cadence_base_cannot_be_instantiated",
    "test_criterion_base_cannot_be_instantiated",
    "test_encoder_base_cannot_be_instantiated",
    "test_head_base_cannot_be_instantiated",
    "test_storage_base_cannot_be_instantiated",
    "test_target_update_moves_shadow_toward_online",
    "test_utility_base_cannot_be_instantiated",
]


def test_encoder_base_cannot_be_instantiated() -> None:
    """The Encoder base class is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Encoder()  # type: ignore[abstract]


def test_head_base_cannot_be_instantiated() -> None:
    """The Head base class is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Head()  # type: ignore[abstract]


def test_utility_base_cannot_be_instantiated() -> None:
    """The Utility base class is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Utility()  # type: ignore[abstract]


def test_criterion_base_cannot_be_instantiated() -> None:
    """The Criterion base class is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Criterion()  # type: ignore[abstract]


def test_storage_base_cannot_be_instantiated() -> None:
    """The Storage base class is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Storage()  # type: ignore[abstract]


def test_cadence_base_cannot_be_instantiated() -> None:
    """The Cadence base class is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Cadence()  # type: ignore[abstract]


def test_augmentation_base_cannot_be_instantiated() -> None:
    """The Transform base class is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Transform()  # type: ignore[abstract]


def test_target_update_moves_shadow_toward_online() -> None:
    """Target.update moves the shadow parameters toward the online parameters."""
    online = torch.nn.Linear(4, 4)
    target = Target(online, momentum=0.5)
    # Manually perturb the online parameters away from the shadow.
    with torch.no_grad():
        online.weight.fill_(1.0)
        online.bias.fill_(1.0)
        target.shadow.weight.fill_(0.0)
        target.shadow.bias.fill_(0.0)
    target.update()
    # Shadow should be 0.5 * 0 + 0.5 * 1 = 0.5 after the update.
    assert torch.allclose(target.shadow.weight, torch.full_like(target.shadow.weight, 0.5))
    assert torch.allclose(target.shadow.bias, torch.full_like(target.shadow.bias, 0.5))


def test_predictor_inherits_head() -> None:
    """Predictor is a Head subclass and exposes the output_dim property."""
    pred = Predictor(input_dim=4, hidden_dim=8, output_dim=4)
    assert isinstance(pred, Head)
    assert isinstance(pred, torch.nn.Module)
    assert pred.output_dim == 4
    out = pred(torch.randn(2, 4))
    assert tuple(out.shape) == (2, 4)


def test_facility_is_utility() -> None:
    """Facility is a Utility subclass and produces a non-negative score."""
    features = torch.randn((5, 4))
    facility = Facility(features)
    assert isinstance(facility, Utility)
    score = facility(torch.tensor([0, 1, 2]), torch.randn((3, 4)))
    assert score >= 0.0


def test_infogain_is_utility() -> None:
    """InfoGain is a Utility subclass and produces a non-negative score."""
    features = torch.randn((5, 4))
    info = InfoGain(features)
    assert isinstance(info, Utility)
    score = info(torch.tensor([0, 1, 2]), torch.randn((4,)))
    assert score >= 0.0


def test_four_conditions_is_criterion() -> None:
    """FourConditions is a Criterion subclass."""
    fc = FourConditions()
    assert isinstance(fc, Criterion)
    assert fc.beta_ib == 1e-2


def test_buffer_is_storage() -> None:
    """Buffer is a Storage subclass."""
    buf = Buffer(capacity=10)
    assert isinstance(buf, Storage)
    assert len(buf) == 0


def test_sleep_is_cadence() -> None:
    """Sleep is a Cadence subclass."""
    s = Sleep()
    assert isinstance(s, Cadence)
    assert not s.should_sleep()  # empty history returns 1.0
    s.update(accepted=True, utilisation=0.5)
    assert not s.should_sleep()
    s.update(accepted=False, utilisation=0.0)
    # After a low-stat observation, may or may not flip depending on the window.
    assert isinstance(s.should_sleep(), bool)


def test_buffer_add_and_minibatches() -> None:
    """Buffer.add and minibatches round-trip the recorded steps."""
    buf = Buffer(capacity=4)
    for _ in range(2):
        step = Step(
            state=torch.randn((3,)),
            action=0,
            logprob=torch.tensor(0.0),
            reward=1.0,
            value=0.5,
            done=False,
        )
        buf.add(step)
    assert len(buf) == 2
    batches = list(buf.minibatches(batch_size=2))
    assert len(batches) == 1
    states, actions, old_logprobs, advantages, returns = batches[0]
    assert states.shape == (2, 3)
    assert actions.tolist() == [0, 0]
    assert torch.allclose(advantages, returns)


def test_dual_geometric_is_encoder() -> None:
    """DualGeometric is an Encoder subclass and has a positive output_dim."""
    assert issubclass(DualGeometric, Encoder)
    instance = DualGeometric(input_dim=4, euclidean_dim=8, hyperbolic_dim=4, num_layers=2)
    assert instance.output_dim == 12
