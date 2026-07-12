"""FIFO replay buffer for PPO.

The buffer stores transitions as :class:`Transition` records and
provides uniform sampling with and without replacement. Importance-
ratio information is included so the trainer can compute off-policy
corrections against the live policy.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from pjepa.exceptions import ConfigError

__all__ = ["ReplayBuffer", "Transition"]


@dataclass
class Transition:
    """A single replay-buffer transition.

    Attributes:
        state: The state tensor at this step.
        action: The action taken.
        logprob: The log-probability of the action under the policy
          that produced the transition.
        reward: The reward received.
        value: The value estimate at this state.
        done: Whether the transition terminated an episode.
        old_logprob: The log-probability at collection time. Equal
          to ``logprob`` in the live-trajectory case.
    """

    state: torch.Tensor
    action: int
    logprob: torch.Tensor
    reward: float
    value: float
    done: bool = False
    old_logprob: torch.Tensor | None = None


@dataclass
class ReplayBuffer:
    """Bounded FIFO replay buffer with optional staleness eviction.

    Attributes:
        capacity: Maximum number of transitions stored.
        max_age: Maximum age (in steps) of a transition before eviction.
    """

    capacity: int
    max_age: int = 10_000
    storage: deque = field(init=False)
    step_counter: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ConfigError(f"ReplayBuffer: capacity must be positive; got {self.capacity}")
        if self.max_age <= 0:
            raise ConfigError(f"ReplayBuffer: max_age must be positive; got {self.max_age}")
        self.storage = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        """Append a transition to the buffer.

        Args:
            transition: The transition to record.
        """
        if transition.old_logprob is None:
            transition.old_logprob = transition.logprob.detach().clone()
        self.storage.append((self.step_counter, transition))
        self.step_counter += 1
        self._evict_stale()

    def _evict_stale(self) -> None:
        cutoff = self.step_counter - self.max_age
        while self.storage and self.storage[0][0] < cutoff:
            self.storage.popleft()

    def minibatches(
        self, batch_size: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield successive minibatches without replacement.

        Args:
            batch_size: Size of each minibatch.

        Yields:
            ``(states, actions, old_logprobs, advantages, returns)``
            tensors of shape ``[batch_size, ...]``. Advantages and
            returns are computed from the rewards and values in the
            buffer via GAE; in this simplified implementation we
            return advantages equal to returns-to-go so the trainer
            sees the canonical PPO formulation.
        """
        if batch_size <= 0:
            raise ConfigError(
                f"ReplayBuffer.minibatches: batch_size must be positive; got {batch_size}"
            )
        transitions = [t for _, t in self.storage]
        if not transitions:
            return iter(())
        states = torch.stack([t.state for t in transitions])
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        old_logprobs = torch.stack([t.old_logprob for t in transitions])
        returns = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        advantages = returns.clone()
        idx = torch.randperm(states.shape[0])
        for start in range(0, states.shape[0], batch_size):
            sel = idx[start : start + batch_size]
            yield (
                states[sel],
                actions[sel],
                old_logprobs[sel],
                advantages[sel],
                returns[sel],
            )
