"""FIFO replay buffer for PPO.

The buffer stores transitions as :class:`Step` records and provides
uniform sampling both with and without replacement. Importance-ratio
information is included so the trainer can compute off-policy
corrections against the live policy.

Capacity is enforced via ``collections.deque(maxlen=capacity)`` so
the oldest entry is evicted automatically when the buffer is full.
An additional ``max_age`` guard drops entries that are too old,
even if the buffer has not yet reached capacity.

The :class:`Buffer` class is the concrete implementation; the
:class:`Storage` abstract base class is the polymorphic root of
the storage hierarchy. Other subclasses (e.g. a priority replay
buffer) can override :meth:`add`, :meth:`minibatches`, and
:meth:`evict_stale` to swap in alternative policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from pjepa.exceptions import ConfigError

__all__ = ["Buffer", "Step", "Storage"]


@dataclass
class Step:
    """A single replay-buffer step.

    The dataclass is mutable so the buffer can lazily fill in
    ``old_logprob`` from ``logprob`` when the caller has no
    recorded value to supply.

    Attributes:
        state: The state tensor at this step.
        action: The action taken (an integer index into the
            action space).
        logprob: The log-probability of the action under the
            policy that produced the transition.
        reward: The reward received.
        value: The value estimate at this state.
        done: Whether the transition terminated an episode.
        old_logprob: The log-probability at collection time.
            Equal to ``logprob`` in the live-trajectory case.
            Lazily back-filled by :meth:`Buffer.add` when
            ``None``.

    Args:
        state: The state tensor.
        action: The action taken.
        logprob: The log-probability of the action under the
            live policy.
        reward: The reward received.
        value: The value estimate at this state.
        done: Whether the transition terminated an episode.
        old_logprob: Optional collection-time log-probability.
    """

    state: torch.Tensor
    action: int
    logprob: torch.Tensor
    reward: float
    value: float
    done: bool = False
    old_logprob: torch.Tensor | None = None


#: Backward-compatible alias for the step dataclass.
Transition = Step


class Storage(ABC):
    """Abstract base class for replay-storage backends.

    The :class:`Storage` class is the polymorphic root of the
    replay-storage hierarchy. The concrete subclass :class:`Buffer`
    is the framework's FIFO implementation; subclasses with
    priority-based or reservoir sampling can override
    :meth:`add`, :meth:`minibatches`, and :meth:`evict_stale`
    without changing the trainer.

    All methods raise :class:`ConfigError` for invalid arguments
    (negative capacities, zero batch sizes, etc.).
    """

    @abstractmethod
    def add(self, step: Step) -> None:
        """Append a step to the storage.

        Args:
            step: The step to record.
        """
        ...

    @abstractmethod
    def minibatches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yield successive minibatches without replacement.

        Args:
            batch_size: Size of each minibatch.

        Yields:
            ``(states, actions, old_logprobs, advantages, returns)``
            tensors each shaped ``[batch_size, ...]``.
        """
        ...

    @abstractmethod
    def evict_stale(self) -> None:
        """Drop every entry whose age exceeds the configured bound."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored transitions."""
        ...


@dataclass
class Buffer(Storage):
    """Bounded FIFO replay buffer with staleness eviction.

    Adding a transition appends to the deque and increments the
    internal step counter; when the entry is older than
    ``max_age`` it is dropped immediately. Reading minibatches via
    :meth:`minibatches` draws without replacement.

    Attributes:
        capacity: Maximum number of transitions stored.
        max_age: Maximum age (in steps) of a transition before
            eviction. Defaults to a generous ``10_000`` so eviction
            only fires in long-running sessions.

    Raises:
        ConfigError: At construction time if ``capacity`` or
            ``max_age`` is non-positive.
    """

    capacity: int
    max_age: int = 10_000
    storage: deque = field(init=False)
    step_counter: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Validate the capacity and max_age arguments.

        Raises:
            ConfigError: If ``capacity`` or ``max_age`` is
                non-positive.
        """
        if self.capacity <= 0:
            raise ConfigError(f"Buffer: capacity must be positive; got {self.capacity}")
        if self.max_age <= 0:
            raise ConfigError(f"Buffer: max_age must be positive; got {self.max_age}")
        self.storage = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        """Return the number of stored transitions.

        Returns:
            ``len(self.storage)``.
        """
        return len(self.storage)

    def add(self, step: Step) -> None:
        """Append a step to the buffer.

        If ``step.old_logprob`` is ``None`` the buffer copies
        ``step.logprob.detach()`` so the trainer has an immutable
        snapshot of the collection-time log-probability. The
        buffer performs a stale-entry sweep at the end of each
        call.

        Args:
            step: The step to record.
        """
        if step.old_logprob is None:
            step.old_logprob = step.logprob.detach().clone()
        self.storage.append((self.step_counter, step))
        self.step_counter += 1
        self.evict_stale()

    def evict_stale(self) -> None:
        """Drop every entry whose age exceeds ``max_age``.

        The function walks the deque from the head and pops
        entries until the remaining entries are within the
        staleness window. ``O(N)`` in the worst case (when every
        entry is stale).
        """
        cutoff = self.step_counter - self.max_age
        while self.storage and self.storage[0][0] < cutoff:
            self.storage.popleft()

    def minibatches(
        self, batch_size: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield successive minibatches without replacement.

        The yielded ``advantages`` and ``returns`` tensors share
        the same values: this implementation does not run a
        separate generalised advantage estimation; the trainer
        sees the reward both as the return-to-go and as the
        advantage so the standard PPO surrogate
        ``log π(a|s) * advantage`` recovers the regular
        policy-gradient update. Callers that need a true GAE pass
        should compute it inside the trainer via
        :meth:`Trainer.compute_gae` on the collected rewards.

        Complexity: stacks all transitions once (``O(n)``) and
        then yields random ``batch_size`` slices (``O(n)`` total
        across all minibatches). Memory is ``O(n)`` over and above
        the stored tensors.

        Args:
            batch_size: Size of each minibatch.

        Yields:
            ``(states, actions, old_logprobs, advantages, returns)``
            tensors each shaped ``[batch_size, ...]``.

        Raises:
            ConfigError: If ``batch_size`` is non-positive.
        """
        if batch_size <= 0:
            raise ConfigError(f"Buffer.minibatches: batch_size must be positive; got {batch_size}")
        transitions = [t for _, t in self.storage]
        if not transitions:
            return iter(())
        states = torch.stack([t.state for t in transitions])
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        old_logprobs = torch.stack([t.old_logprob for t in transitions])
        returns = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        # Advantages equal returns here: PPO without a separate
        # GAE baseline reduces to a reward-weighted policy
        # gradient. A future revision may compute the proper GAE
        # in this method.
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


#: Backward-compatible alias for the concrete buffer.
ReplayBuffer = Buffer
