"""Sleep-cycle cadence for the scheduler.

A sleep cycle is triggered when either (i) the rolling mean
accepted-rewrite rate falls below ``rho_min`` over ``T_w``
observations, or (ii) the working-graph utilisation
``mean(|W_t|) / B`` falls below ``alpha_min``. The cadence is
deterministic and reproduces across runs given the same observation
history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from pjepa.exceptions import ConfigError

__all__ = ["SleepCadence", "should_sleep"]


@dataclass
class SleepCadence:
    """Configuration for the sleep-cycle trigger.

    Attributes:
        rho_min: Minimum rolling accepted-rewrite rate.
        alpha_min: Minimum rolling working-graph utilisation.
        window: Size of the rolling window (in observations).
    """

    rho_min: float = 0.05
    alpha_min: float = 0.4
    window: int = 32
    accepted_history: deque = field(init=False)
    utilisation_history: deque = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho_min <= 1.0:
            raise ConfigError(f"SleepCadence: rho_min must be in [0, 1]; got {self.rho_min}")
        if not 0.0 <= self.alpha_min <= 1.0:
            raise ConfigError(f"SleepCadence: alpha_min must be in [0, 1]; got {self.alpha_min}")
        if self.window <= 0:
            raise ConfigError(f"SleepCadence: window must be positive; got {self.window}")
        self.accepted_history = deque(maxlen=self.window)
        self.utilisation_history = deque(maxlen=self.window)

    def update(self, accepted: bool, utilisation: float) -> None:
        """Record one observation.

        Args:
            accepted: Whether the most recent rewrite was accepted.
            utilisation: The working-graph utilisation at this step,
              in [0, 1].
        """
        self.accepted_history.append(1 if accepted else 0)
        self.utilisation_history.append(utilisation)

    def reset(self) -> None:
        """Clear the rolling histories."""
        self.accepted_history.clear()
        self.utilisation_history.clear()

    @property
    def mean_accepted_rate(self) -> float:
        """Return the rolling accepted-rewrite rate."""
        if not self.accepted_history:
            return 1.0
        return sum(self.accepted_history) / len(self.accepted_history)

    @property
    def mean_utilisation(self) -> float:
        """Return the rolling mean utilisation."""
        if not self.utilisation_history:
            return 1.0
        return sum(self.utilisation_history) / len(self.utilisation_history)

    def should_sleep(self) -> bool:
        """Return whether a sleep cycle should fire."""
        return self.mean_accepted_rate < self.rho_min or self.mean_utilisation < self.alpha_min


def should_sleep(cadence: SleepCadence) -> bool:
    """Functional alias for ``cadence.should_sleep()``."""
    return cadence.should_sleep()
