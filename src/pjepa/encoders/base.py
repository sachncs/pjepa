"""Encoder base class and contract.

The :class:`Encoder` class is the polymorphic root of the encoder
hierarchy. Every encoder in ``pjepa`` is a :class:`Encoder` subclass
and inherits the shared contract:

* ``forward`` maps a :class:`Graph` to a tensor.
* ``output_dim`` is the trailing dimension of that tensor.
* ``to`` returns the encoder itself (typed for chaining).

The base class is intentionally abstract: subclasses must implement
:meth:`forward` and expose ``output_dim`` as either a class attribute
or an instance property. The wrapper methods :meth:`encode` and
:meth:`summary` are convenience methods that subclasses can override;
the default :meth:`encode` simply forwards to :meth:`forward`, and the
default :meth:`summary` returns a dict describing the encoder's
configuration.

The class is a ``torch.nn.Module`` subclass so all encoder parameters
move with the standard ``.to(device)`` call and integrate with
``torch.compile`` and the rest of the PyTorch ecosystem.

Example:
    >>> import torch
    >>> from pjepa.encoders.base import Encoder
    >>> from pjepa.graphs import Graph
    >>> class Constant(Encoder):
    ...     def __init__(self, dim: int) -> None:
    ...         super().__init__()
    ...         self.dim = dim
    ...     @property
    ...     def output_dim(self) -> int:
    ...         return self.dim
    ...     def forward(self, graph: Graph) -> torch.Tensor:
    ...         return torch.zeros((graph.num_vertices(), self.dim))
    >>> enc = Constant(4)
    >>> enc.output_dim
    4
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from pjepa.graphs import Graph

__all__ = ["Encoder", "EncoderProtocol"]


class Encoder(ABC, torch.nn.Module):
    """Abstract base class for every encoder in the package.

    The :class:`Encoder` class is the polymorphic root of the
    encoder hierarchy. Implementations must override
    :meth:`forward` and expose ``output_dim`` as either a class
    attribute or an instance property. The base class also
    implements :meth:`encode` and :meth:`summary` as convenience
    wrappers, both of which subclasses may override.

    Attributes:
        output_dim: The trailing dimension of the tensors produced
            by :meth:`forward`. The same value is reported as the
            last dimension of the output tensor of :meth:`forward`.

    Args:
        (none — the base class is a no-op ``torch.nn.Module``).

    Raises:
        TypeError: If a subclass declares ``output_dim`` as a
            class attribute that is not a positive integer.

    Example:
        >>> import torch
        >>> from pjepa.encoders.base import Encoder
        >>> from pjepa.graphs import Graph
        >>> class Constant(Encoder):
        ...     def __init__(self, dim: int) -> None:
        ...         super().__init__()
        ...         self.dim = dim
        ...     @property
        ...     def output_dim(self) -> int:
        ...         return self.dim
        ...     def forward(self, graph: Graph) -> torch.Tensor:
        ...         return torch.zeros((graph.num_vertices(), self.dim))
        >>> enc = Constant(4)
        >>> enc.output_dim
        4
    """

    def __init__(self) -> None:
        """Initialize the underlying :class:`torch.nn.Module`."""
        super().__init__()

    @property
    def output_dim(self) -> int:
        """Return the trailing dimension of the encoder's output tensor.

        The default implementation reads the instance attribute
        ``self.output_width``. Subclasses can either set that
        attribute in ``__init__`` or override the property.

        Returns:
            A positive integer set by the subclass.

        Raises:
            AttributeError: If neither the instance attribute nor
                a subclass override is provided.
        """
        try:
            return int(self.output_width)
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__}: output_dim is not set; "
                "subclasses must set self.output_width in __init__ "
                "or override the output_dim property"
            ) from exc

    @abstractmethod
    def forward(self, graph: Graph) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Encode the graph into a tensor (or tuple of tensors).

        Args:
            graph: The input graph.

        Returns:
            A tensor whose trailing dimension is
            ``self.output_dim``. The leading dimension is the
            number of vertices ``N`` for per-vertex encoders, or
            omitted for graph-level encoders. Multi-component
            encoders (e.g. :class:`DualGeometric`) may return a
            tuple of tensors instead.
        """
        ...

    def encode(self, graph: Graph) -> torch.Tensor:
        """Encode the graph into a tensor.

        Thin wrapper around :meth:`forward` for callers that prefer
        the verb ``encode`` at the call site.

        Args:
            graph: The input graph.

        Returns:
            The same tensor :meth:`forward` would return.
        """
        return self.forward(graph)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the encoder.

        The default implementation returns the class name and the
        output dimension. Subclasses may override to add more
        fields (e.g. parameter counts, layer widths).

        Returns:
            A dictionary with keys ``class`` and ``output_dim``.
        """
        return {"class": type(self).__name__, "output_dim": int(self.output_dim)}


#: Convenience alias for the polymorphic root.
EncoderProtocol = Encoder
