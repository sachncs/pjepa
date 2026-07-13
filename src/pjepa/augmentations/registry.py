"""Registry for augmentations.

The registry allows new :class:`Augmentation` implementations to be
added at runtime without modifying the core library. Each entry is
keyed by a string name; users can :func:`register` their own classes
and :func:`get_augmentation` them later by name.
"""

from __future__ import annotations

from collections.abc import Callable

from pjepa.augmentations.base import Augmentation
from pjepa.augmentations.feature import DropFeature, FeatureMask
from pjepa.augmentations.identity import Identity
from pjepa.augmentations.structural import (
    ConnectedSubgraph,
    DropEdge,
    DropNode,
    Subgraph,
)
from pjepa.exceptions import GraphError

__all__ = [
    "augmentation_registry",
    "available_augmentations",
    "evict_augmentation",
    "get_augmentation",
    "register",
]


augmentation_registry: dict[str, type[Augmentation]] = {}
"""Mutable mapping from registered name to augmentation subclass.

Exposed at module level so tests can clean up user-registered entries
with :func:`evict_augmentation`. The dict stores class *objects*,
not instances.
"""


def register(name: str) -> Callable[[type[Augmentation]], type[Augmentation]]:
    """Class decorator that registers an :class:`Augmentation` subclass.

    Args:
        name: The name under which the augmentation is registered.

    Returns:
        The original class, unmodified.

    Raises:
        GraphError: If ``name`` is empty or already registered.
    """

    def decorator(cls: type[Augmentation]) -> type[Augmentation]:
        if not name:
            raise GraphError("register: name must be a non-empty string")
        if name in augmentation_registry:
            existing = augmentation_registry[name].__name__
            raise GraphError(f"register: augmentation {name!r} already registered as {existing}")
        augmentation_registry[name] = cls
        return cls

    return decorator


def get_augmentation(name: str) -> type[Augmentation]:
    """Look up a registered augmentation class by name.

    Args:
        name: The registered name.

    Returns:
        The registered :class:`Augmentation` subclass.

    Raises:
        GraphError: If ``name`` is not in the registry.
    """
    try:
        return augmentation_registry[name]
    except KeyError as exc:
        known = sorted(augmentation_registry)
        msg = f"get_augmentation: unknown augmentation {name!r}; available: {known}"
        raise GraphError(msg) from exc


def available_augmentations() -> tuple[str, ...]:
    """Return the names of all registered augmentations in alphabetical order."""
    return tuple(sorted(augmentation_registry))


def evict_augmentation(name: str) -> bool:
    """Remove a registered augmentation by name.

    Test-only convenience; modifying the global registry at runtime
    can affect other parts of the library.

    Args:
        name: The registered name to remove.

    Returns:
        ``True`` if the name was registered, ``False`` otherwise.
    """
    return augmentation_registry.pop(name, None) is not None


# Built-in augmentations registered on import.
register("drop_edge")(DropEdge)
register("drop_node")(DropNode)
register("drop_feature")(DropFeature)
register("feature_mask")(FeatureMask)
register("connected_subgraph")(ConnectedSubgraph)
register("subgraph")(Subgraph)
register("identity")(Identity)
