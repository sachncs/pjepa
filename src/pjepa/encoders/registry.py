"""Registry for encoder implementations.

The registry maps string names to :class:`Encoder` subclasses so that
new encoder implementations can be plugged in without modifying the
core library. Concrete encoders (:class:`EuclideanMPNN`,
:class:`HyperbolicProjection`, :class:`DualGeometricEncoder`,
:class:`JEPAPredictor`) are registered automatically on import.

The public API:

* :func:`register` — class decorator adding an encoder under a name.
* :func:`get_encoder` — look up an encoder class by name.
* :func:`available_encoders` — list every registered name in
  alphabetical order.
* :func:`evict_encoder` — remove a registered name (testing utility).

The registry stores class *objects*, not instances, so callers can
customise construction arguments at the point of instantiation.
"""

from __future__ import annotations

from collections.abc import Callable

from pjepa.encoders.base import Encoder
from pjepa.encoders.dual_geometric import DualGeometricEncoder
from pjepa.encoders.euclidean_mpnn import EuclideanMPNN
from pjepa.encoders.hyperbolic import HyperbolicProjection
from pjepa.encoders.jepa_predictor import JEPAPredictor
from pjepa.exceptions import ContractError

__all__ = [
    "available_encoders",
    "evict_encoder",
    "get_encoder",
    "register",
]


encoder_registry: dict[str, type[Encoder]] = {}
"""Mutable mapping from registered name to encoder subclass.

The mapping is module-level so registrations performed on import are
visible to every subsequent lookup. The dict is exposed (rather than
hidden behind a leading underscore) because the test suite uses
:func:`evict_encoder` to clean up user-registered entries.
"""


def register(name: str) -> Callable[[type[Encoder]], type[Encoder]]:
    """Class decorator that registers an :class:`Encoder` subclass.

    The decorator returns the original class unchanged. Double
    registration under the same name raises
    :class:`ContractError`.

    Args:
        name: The name under which the encoder is registered.

    Returns:
        The original class, unmodified.

    Raises:
        ContractError: If ``name`` is empty or already registered.
    """

    def decorator(cls: type[Encoder]) -> type[Encoder]:
        if not name:
            raise ContractError("register: name must be a non-empty string")
        if name in encoder_registry:
            existing = encoder_registry[name].__name__
            raise ContractError(f"register: encoder {name!r} already registered as {existing}")
        encoder_registry[name] = cls
        return cls

    return decorator


def get_encoder(name: str) -> type[Encoder]:
    """Look up a registered encoder class by name.

    Args:
        name: The registered name.

    Returns:
        The registered :class:`Encoder` subclass.

    Raises:
        ContractError: If ``name`` is not in the registry.
    """
    try:
        return encoder_registry[name]
    except KeyError as exc:
        raise ContractError(
            f"get_encoder: unknown encoder {name!r}; available: {sorted(encoder_registry)}"
        ) from exc


def available_encoders() -> tuple[str, ...]:
    """Return the names of all registered encoders in alphabetical order."""
    return tuple(sorted(encoder_registry))


def evict_encoder(name: str) -> bool:
    """Remove a registered encoder by name.

    This is a *test-only* convenience; modifying the global registry
    at runtime can affect other parts of the library. Production code
    should not depend on this function.

    Args:
        name: The registered name to remove.

    Returns:
        ``True`` if the name was registered, ``False`` otherwise.
    """
    return encoder_registry.pop(name, None) is not None


# Built-in encoders registered on import.
register("euclidean_mpnn")(EuclideanMPNN)
register("hyperbolic")(HyperbolicProjection)
register("dual_geometric")(DualGeometricEncoder)
register("jepa_predictor")(JEPAPredictor)
