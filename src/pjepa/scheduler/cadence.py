"""Sleep-cycle cadence for the scheduler.

A sleep cycle is triggered when either:

* the rolling mean accepted-rewrite rate over the last ``window``
  observations falls below ``rho_min``, or
* the rolling mean working-graph utilisation
  ``mean(|W_t|) / B`` falls below ``alpha_min``.

The cadence is deterministic and reproduces across runs given
the same observation history. The two rolling histories are
independent ``deque(maxlen=window)`` instances so updating one
does not affect the other.

The :class:`Cadence` abstract base class is the polymorphic
root of the sleep-cadence hierarchy. The concrete subclass
:class:`Sleep` is the framework's default implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from pjepa.exceptions import ConfigError

__all__ = ["Cadence", "Sleep", "should_sleep"]


class Cadence(ABC):
    """Abstract base class for sleep cadence.

    The :class:`Cadence` class is the polymorphic root of the
    sleep-cadence hierarchy. The concrete subclass :class:`Sleep`
    is the framework's default implementation, but alternative
    strategies (e.g. time-based, loss-based) can subclass
    :class:`Cadence` and override :meth:`should_sleep`.

    The trainer consults :meth:`should_sleep` once per epoch;
    subclasses can maintain arbitrary state via ``self`` as long
    as :meth:`update` is called by the trainer.
    """

    @abstractmethod
    def should_sleep(self) -> bool:
        """Return whether a sleep cycle should fire.

        Returns:
            ``True`` if a sleep cycle should begin.
        """
        ...

    @abstractmethod
    def update(self, accepted: bool, utilisation: float) -> None:
        """Record one observation.

        Args:
            accepted: Whether the most recent rewrite was
                accepted.
            utilisation: The working-graph utilisation at this
                step, expected in ``[0, 1]``.
        """
        ...

    def reset(self) -> None:
        """Reset the cadence state.

        Default implementation is a no-op. Overridden by
        :class:`Sleep` to clear the rolling histories.
        """
        return


@dataclass
class Sleep(Cadence):
    """Configuration for the sleep-cycle trigger.

    Attributes:
        rho_min: Minimum rolling accepted-rewrite rate.
        alpha_min: Minimum rolling working-graph utilisation.
        window: Size of the rolling window (in observations).

    Raises:
        ConfigError: At construction time if any threshold is
            out of range or ``window`` is non-positive.
    """

    rho_min: float = 0.05
    alpha_min: float = 0.4
    window: int = 32
    accepted_history: deque = field(init=False)
    utilisation_history: deque = field(init=False)

    def __post_init__(self) -> None:
        """Validate the thresholds and window.

        Raises:
            ConfigError: If any threshold is out of range or
                ``window`` is non-positive.
        """
        if not 0.0 <= self.rho_min <= 1.0:
            raise ConfigError(f"Sleep: rho_min must be in [0, 1]; got {self.rho_min}")
        if not 0.0 <= self.alpha_min <= 1.0:
            raise ConfigError(f"Sleep: alpha_min must be in [0, 1]; got {self.alpha_min}")
        if self.window <= 0:
            raise ConfigError(f"Sleep: window must be positive; got {self.window}")
        self.accepted_history = deque(maxlen=self.window)
        self.utilisation_history = deque(maxlen=self.window)

    def update(self, accepted: bool, utilisation: float) -> None:
        """Record one observation.

        Args:
            accepted: Whether the most recent rewrite was
                accepted.
            utilisation: The working-graph utilisation at this
                step, expected in ``[0, 1]``. Values outside that
                range are accepted but may distort future
                ``should_sleep`` decisions.
        """
        self.accepted_history.append(1 if accepted else 0)
        self.utilisation_history.append(utilisation)

    def reset(self) -> None:
        """Clear the rolling histories.

        After a reset the cadence returns ``1.0`` for both
        :attr:`mean_accepted_rate` and :attr:`mean_utilisation`
        until a new observation is recorded.
        """
        self.accepted_history.clear()
        self.utilisation_history.clear()

    @property
    def mean_accepted_rate(self) -> float:
        """Return the rolling accepted-rewrite rate.

        When the history is empty (e.g. immediately after
        :meth:`reset`) returns ``1.0`` so the cadence does not
        fire from a cold start.
        """
        if not self.accepted_history:
            return 1.0
        return sum(self.accepted_history) / len(self.accepted_history)

    @property
    def mean_utilisation(self) -> float:
        """Return the rolling mean utilisation.

        Empty history returns ``1.0``; see
        :attr:`mean_accepted_rate`.
        """
        if not self.utilisation_history:
            return 1.0
        return sum(self.utilisation_history) / len(self.utilisation_history)

    def should_sleep(self) -> bool:
        """Return whether a sleep cycle should fire.

        Sleep fires when **either** rolling statistic drops below
        its threshold. Once a sleep cycle begins, the trainer
        typically calls :meth:`reset` so the next observations
        start fresh.
        """
        return self.mean_accepted_rate < self.rho_min or self.mean_utilisation < self.alpha_min


def should_sleep(cadence: Cadence) -> bool:
    """Functional alias for ``cadence.should_sleep()``.

    Args:
        cadence: The cadence to consult.

    Returns:
        ``True`` if a sleep cycle should begin.
    """
    return cadence.should_sleep()


#: Backward-compatible alias for the concrete cadence.
SleepCadence = Sleep
