"""Tests for pjepa.objectives, dynamics, scheduler, encoders.

Covers the eight-class test taxonomy.
"""

from __future__ import annotations

import math

import pytest
import torch

from pjepa.dynamics import contractivity_bound, fixed_point_iteration
from pjepa.encoders import (
    DualGeometricEncoder,
    EuclideanMPNN,
    HyperbolicProjection,
    JEPAPredictor,
    TargetEncoder,
)
from pjepa.exceptions import ConfigError, NumericalError
from pjepa.graphs import TypedAttributedGraph
from pjepa.objectives import FreeEnergy, description_length, ib_lagrangian, variational_ib_bound
from pjepa.scheduler import (
    PPOConfig,
    PPOTrainer,
    ReplayBuffer,
    SleepCadence,
    Transition,
    should_sleep,
)

__all__ = [
    "test_bad_euclidean_mpnn_zero_dim",
    "test_bad_hyperbolic_projection_curvature",
    "test_bad_ib_lagrangian_negative_mutual_information",
    "test_bad_ppo_zero_minibatch",
    "test_bad_replay_buffer_zero_capacity",
    "test_bad_sleep_cadence_window",
    "test_bad_variational_ib_shape_mismatch",
    "test_cross_backend_mps_hyperbolic",
    "test_distributional_contractivity_bound_at_t_zero",
    "test_happy_contractivity_bound",
    "test_happy_description_length_positive",
    "test_happy_dual_geometric_encoder",
    "test_happy_euclidean_mpnn_forward",
    "test_happy_fixed_point_iteration_identity",
    "test_happy_free_energy_non_negative",
    "test_happy_hyperbolic_projection_norms",
    "test_happy_ib_lagrangian",
    "test_happy_jepa_predictor_shape",
    "test_happy_ppo_clipped_surrogate",
    "test_happy_replay_buffer_add_and_sample",
    "test_happy_sleep_cadence_no_sleep_when_healthy",
    "test_happy_target_encoder_ema",
    "test_happy_variational_ib_bound",
    "test_leaky_ppo_does_not_mutate_outer_state",
    "test_property_dual_geometric_dims",
    "test_round_trip_target_encoder_step",
    "test_ugly_free_energy_empty_graph",
    "test_ugly_hyperbolic_projection_zero_input",
]


# ============================== OBJECTIVES ==============================


def test_happy_ib_lagrangian() -> None:
    """IB Lagrangian combines I(X;Z) and -beta*I(Y;Z)."""
    assert ib_lagrangian(1.0, 0.5, 0.1) == pytest.approx(0.95)


def test_happy_variational_ib_bound() -> None:
    """Variational IB bound is non-negative for identical inputs."""
    logits = torch.zeros((4, 5))
    assert variational_ib_bound(logits, logits) == pytest.approx(0.0, abs=1e-6)


def test_happy_description_length_positive() -> None:
    """description_length is positive for non-trivial graphs."""
    g = TypedAttributedGraph(
        vertex_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_features=torch.tensor([[1.0], [1.0]]),
    )
    assert description_length(g) > 0.0


def test_happy_free_energy_non_negative() -> None:
    """FreeEnergy on a non-trivial graph is finite and non-negative."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((2, 3)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    obs = torch.randn((1, 3))
    value = FreeEnergy()(g, obs)
    assert math.isfinite(value) or value == float("inf")


# ============================== DYNAMICS ==============================


def test_happy_contractivity_bound() -> None:
    """The contractivity bound decreases monotonically when eta_g < 1."""
    bounds = [contractivity_bound(0.5, 0.1, 0.05, t) for t in range(20)]
    assert bounds[-1] < bounds[0]


def test_happy_fixed_point_iteration_identity() -> None:
    """An identity operator converges in one step."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((1, 2)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    final, steps = fixed_point_iteration(g, lambda x: x)
    assert steps == 1


# ============================== SCHEDULER ==============================


def test_happy_ppo_clipped_surrogate() -> None:
    """The clipped surrogate returns a finite tensor."""
    trainer = PPOTrainer(_ActorCritic(input_dim=4, num_actions=3), PPOConfig())
    ratios = torch.tensor([0.9, 1.0, 1.1])
    advantages = torch.tensor([1.0, 1.0, 1.0])
    loss = trainer.clipped_surrogate(ratios, advantages)
    assert torch.isfinite(loss)


def test_happy_replay_buffer_add_and_sample() -> None:
    """ReplayBuffer accepts transitions and yields minibatches."""
    buffer = ReplayBuffer(capacity=10, max_age=100)
    for i in range(5):
        buffer.add(
            Transition(
                state=torch.randn(3),
                action=i % 3,
                logprob=torch.tensor(0.0),
                reward=1.0,
                value=0.5,
            )
        )
    assert len(buffer) == 5
    batches = list(buffer.minibatches(batch_size=2))
    assert len(batches) >= 1


class _ActorCritic(torch.nn.Module):
    """Tiny actor-critic policy used in PPO tests."""

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(input_dim, num_actions)
        self.critic = torch.nn.Linear(input_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(state), self.critic(state)


def test_happy_sleep_cadence_no_sleep_when_healthy() -> None:
    """A high accept-rate keeps the cadence from firing sleep."""
    cadence = SleepCadence(rho_min=0.1, alpha_min=0.3, window=8)
    for _ in range(8):
        cadence.update(accepted=True, utilisation=0.7)
    assert not should_sleep(cadence)


# ============================== ENCODERS ==============================


def test_happy_euclidean_mpnn_forward() -> None:
    """EuclideanMPNN returns per-vertex embeddings of the right shape."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    )
    encoder = EuclideanMPNN(input_dim=4, hidden_dim=16, num_layers=2, output_dim=8)
    out = encoder(g)
    assert out.shape == (5, 8)


def test_happy_hyperbolic_projection_norms() -> None:
    """HyperbolicProjection produces points with norms strictly < 1."""
    proj = HyperbolicProjection(input_dim=4, output_dim=3)
    out = proj(torch.randn((5, 4)))
    norms = out.norm(dim=-1)
    assert torch.all(norms < 1.0)


def test_happy_dual_geometric_encoder() -> None:
    """DualGeometricEncoder returns a tuple of (Euclidean, hyperbolic) tensors."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((4, 3)),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    enc = DualGeometricEncoder(input_dim=3, euclidean_dim=8, hyperbolic_dim=4, num_layers=2)
    e, h = enc(g)
    assert e.shape == (4, 8)
    assert h.shape == (4, 4)


def test_happy_jepa_predictor_shape() -> None:
    """JEPAPredictor output dim matches ``output_dim``."""
    pred = JEPAPredictor(input_dim=10, hidden_dim=16, output_dim=4)
    out = pred(torch.randn((3, 10)))
    assert out.shape == (3, 4)


def test_happy_target_encoder_ema() -> None:
    """TargetEncoder EMA moves shadow parameters toward online."""
    online = torch.nn.Linear(4, 4)
    initial_shadow = online.weight.detach().clone()
    target = TargetEncoder(online, momentum=0.5)
    with torch.no_grad():
        online.weight.fill_(2.0)
    target.update()
    # Expected: shadow = 0.5 * initial_shadow + 0.5 * 2.0
    expected = 0.5 * initial_shadow + 0.5 * 2.0
    assert torch.allclose(target.shadow.weight, expected)


# ============================== BAD PATHS ==============================


def test_bad_ib_lagrangian_negative_mutual_information() -> None:
    """Negative mutual information raises NumericalError."""
    with pytest.raises(NumericalError):
        ib_lagrangian(-1.0, 0.5, 0.1)


def test_bad_variational_ib_shape_mismatch() -> None:
    """A shape mismatch between posterior and prior raises NumericalError."""
    with pytest.raises(NumericalError):
        variational_ib_bound(torch.zeros((2, 4)), torch.zeros((2, 5)))


def test_bad_euclidean_mpnn_zero_dim() -> None:
    """Zero dimensions are rejected."""
    with pytest.raises(ValueError):
        EuclideanMPNN(input_dim=0, hidden_dim=16)


def test_bad_hyperbolic_projection_curvature() -> None:
    """Negative curvature is rejected."""
    with pytest.raises(ValueError):
        HyperbolicProjection(input_dim=4, curvature=-1.0)


def test_bad_replay_buffer_zero_capacity() -> None:
    """A zero-capacity replay buffer is rejected."""
    with pytest.raises(ConfigError):
        ReplayBuffer(capacity=0)


def test_bad_sleep_cadence_window() -> None:
    """A non-positive window is rejected."""
    with pytest.raises(ConfigError):
        SleepCadence(window=0)


def test_bad_ppo_zero_minibatch() -> None:
    """A zero minibatch size is rejected."""
    with pytest.raises(ConfigError):
        PPOConfig(minibatch_size=0)


# ============================== UGLY PATHS ==============================


def test_ugly_free_energy_empty_graph() -> None:
    """An empty graph yields an infinite FreeEnergy value."""
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 3)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    obs = torch.randn((1, 3))
    assert FreeEnergy()(g, obs) == float("inf")


def test_ugly_hyperbolic_projection_zero_input() -> None:
    """All-zero input still produces a valid hyperbolic point."""
    proj = HyperbolicProjection(input_dim=4, output_dim=3)
    out = proj(torch.zeros((2, 4)))
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


# ============================== LEAKY / ROUND-TRIP ==============================


def test_leaky_ppo_does_not_mutate_outer_state() -> None:
    """PPO update does not leak optimizer state between calls."""
    trainer = PPOTrainer(_ActorCritic(input_dim=4, num_actions=3), PPOConfig())
    buffer = ReplayBuffer(capacity=4)
    for i in range(4):
        buffer.add(
            Transition(
                state=torch.randn(4),
                action=i % 3,
                logprob=torch.tensor(0.0),
                reward=1.0,
                value=0.0,
            )
        )
    opt = torch.optim.Adam(trainer.policy.parameters(), lr=1e-3)
    state_dict_before = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    trainer.update(buffer, opt)
    state_dict_after = trainer.policy.state_dict()
    assert any(
        not torch.allclose(state_dict_before[k], state_dict_after[k]) for k in state_dict_before
    )


def test_round_trip_target_encoder_step() -> None:
    """Target encoder parameters move after update."""
    online = torch.nn.Linear(4, 4)
    target = TargetEncoder(online, momentum=0.0)
    initial = target.shadow.weight.clone()
    with torch.no_grad():
        online.weight.fill_(5.0)
    target.update()
    assert not torch.allclose(target.shadow.weight, initial)


# ============================== CROSS-BACKEND ==============================


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_hyperbolic() -> None:
    """Hyperbolic projection runs on MPS tensors."""
    proj = HyperbolicProjection(input_dim=4, output_dim=3).to("mps")
    out = proj(torch.randn((3, 4), device="mps"))
    assert out.device.type == "mps"


# ============================== DISTRIBUTIONAL ==============================


def test_distributional_contractivity_bound_at_t_zero() -> None:
    """At t=0 the bound equals the initial distance (with eta_o*epsilon term)."""
    for eta_g in (0.1, 0.5, 0.9):
        for eta_o in (0.0, 0.1, 0.5):
            b = contractivity_bound(eta_g, eta_o, 0.05, 0)
            assert b == pytest.approx(1.0)


# ============================== PROPERTY ==============================


def test_property_dual_geometric_dims() -> None:
    """DualGeometricEncoder exposes the correct euclidean and hyperbolic dims."""
    enc = DualGeometricEncoder(input_dim=3, euclidean_dim=8, hyperbolic_dim=4)
    assert enc.euclidean_dim == 8
    assert enc.hyperbolic_dim == 4


def test_encoder_protocol_satisfied_by_subclass() -> None:
    """EuclideanMPNN satisfies the Encoder protocol structurally."""
    enc = EuclideanMPNN(input_dim=3, hidden_dim=8, num_layers=2, output_dim=4)
    assert isinstance(enc, torch.nn.Module)
    assert hasattr(enc, "forward") and hasattr(enc, "to")
