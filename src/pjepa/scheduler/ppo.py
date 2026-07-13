"""PPO trainer with clipped surrogate.

Implements the clipped-surrogate PPO update of Schulman et al. 2017
(arXiv:1707.06347). The reward is unclipped (the surrogate is
clipped instead, which is the standard PPO recipe). The trainer also
exposes :meth:`compute_gae` for callers that need a true generalised
advantage estimator — :class:`ReplayBuffer`'s default ``minibatches``
does not run GAE and uses the reward directly as both the advantage
and the return-to-go.

The trainer is *stateless* across updates: every call to
:meth:`update` consumes the buffer once and returns aggregated
statistics; the live policy is the only piece of state that
persists.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pjepa.exceptions import ConfigError
from pjepa.scheduler.buffer import ReplayBuffer

__all__ = ["PPOConfig", "PPOTrainer"]


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for PPO.

    Attributes:
        clip_eps: Clipping epsilon for the surrogate objective.
        gae_lambda: GAE lambda parameter; used by
            :meth:`PPOTrainer.compute_gae` rather than by the
            :meth:`update` loop (which uses the trivial reward-as-advantage
            fallback when no GAE is supplied).
        value_coef: Coefficient for the value loss.
        entropy_coef: Coefficient for the entropy bonus.
        gamma: Discount factor.
        inner_epochs: Number of PPO epochs per update.
        minibatch_size: Minibatch size for each inner epoch.

    Raises:
        ConfigError: At construction time if ``clip_eps`` is not in
            ``(0, 1)``, ``gae_lambda`` is outside ``[0, 1]``,
            ``gamma`` is outside ``(0, 1]``, or ``inner_epochs`` /
            ``minibatch_size`` is non-positive.
    """

    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    inner_epochs: int = 4
    minibatch_size: int = 64

    def __post_init__(self) -> None:
        if self.inner_epochs <= 0:
            raise ConfigError(f"PPOConfig.inner_epochs must be positive; got {self.inner_epochs}")
        if self.minibatch_size <= 0:
            raise ConfigError(
                f"PPOConfig.minibatch_size must be positive; got {self.minibatch_size}"
            )
        if not 0.0 < self.clip_eps < 1.0:
            raise ConfigError(f"PPOConfig.clip_eps must be in (0, 1); got {self.clip_eps}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ConfigError(f"PPOConfig.gae_lambda must be in [0, 1]; got {self.gae_lambda}")
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigError(f"PPOConfig.gamma must be in (0, 1]; got {self.gamma}")


class PPOTrainer:
    """Clipped-surrogate PPO trainer.

    Attributes:
        config: The PPO configuration.
        policy: A module that produces logits and a value estimate
            given a state. The interface is
            ``policy(state) -> (logits, value)``.
    """

    def __init__(self, policy: torch.nn.Module, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()
        # PPOConfig.__post_init__ has already validated the configuration
        # at construction time, so PPOTrainer accepts any PPOConfig.
        self.policy = policy

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute generalised advantage estimation.

        Args:
            rewards: ``[T]`` reward sequence.
            values: ``[T+1]`` value sequence (last entry is the
                bootstrap value).
            dones: ``[T]`` done flags.

        Returns:
            ``[T]`` advantages tensor.

        Raises:
            ConfigError: If ``values`` has the wrong length.
        """
        if values.shape[0] != rewards.shape[0] + 1:
            raise ConfigError(
                f"compute_gae: values {values.shape[0]} must be rewards+1 = {rewards.shape[0] + 1}"
            )
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(1)
        for t in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * values[t + 1] * mask - values[t]
            last_advantage = (
                delta + self.config.gamma * self.config.gae_lambda * mask * last_advantage
            )
            advantages[t] = last_advantage
        return advantages

    def clipped_surrogate(
        self,
        ratios: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the clipped surrogate objective.

        Args:
            ratios: ``[N]`` importance ratios ``π(a|s) / π_old(a|s)``.
            advantages: ``[N]`` advantage estimates.

        Returns:
            The mean clipped surrogate loss, defined as the negative
            of the PPO objective so that minimising this loss
            maximises the surrogate.
        """
        unclipped = ratios * advantages
        clipped = (
            torch.clamp(ratios, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages
        )
        return -torch.min(unclipped, clipped).mean()

    def update(
        self,
        buffer: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Run one PPO update on a replay buffer.

        Iterates ``config.inner_epochs`` passes, each consuming
        minibatches of size ``config.minibatch_size`` from the
        buffer via :meth:`ReplayBuffer.minibatches`. Aggregates
        ``policy_loss``, ``value_loss`` and ``entropy`` statistics
        across all minibatches and returns their means.

        Args:
            buffer: The replay buffer to sample from.
            optimizer: The optimiser; Adam is the standard choice.

        Returns:
            A dict with keys ``policy_loss``, ``value_loss`` and
            ``entropy``, each averaged over the update.

        Raises:
            ConfigError: If the buffer is empty.
        """
        if len(buffer) == 0:
            raise ConfigError("PPOTrainer.update: replay buffer is empty")
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_batches = 0
        for _ in range(self.config.inner_epochs):
            for batch in buffer.minibatches(self.config.minibatch_size):
                states, actions, old_logprobs, advantages, returns = batch
                logits, values = self.policy(states)
                logprobs = torch.log_softmax(logits, dim=-1)
                new_logprobs = logprobs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                ratios = (new_logprobs - old_logprobs).exp()
                policy_loss = self.clipped_surrogate(ratios, advantages)
                value_loss = ((values.squeeze(-1) - returns) ** 2).mean()
                entropy = -(logprobs * logprobs.exp()).sum(dim=-1).mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                n_batches += 1
        if n_batches > 0:
            for k in stats:
                stats[k] /= n_batches
        return stats
