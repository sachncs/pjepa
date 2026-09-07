"""Euclidean message-passing encoder.

A :class:`Euclidean` is a GIN-style message-passing encoder (Xu
et al. 2019) that operates on the Euclidean component of the
persistent graph. The update rule is::

    h_i^{(l+1)} = MLP^{(l)}([h_i^{(l)}, sum_{j->i} h_j^{(l)}])

which is the standard GIN-style concatenation of the previous-layer
hidden state with the sum of incoming messages. The aggregation is
done with :meth:`Tensor.index_add_` so the encoder runs on every
supported backend without an extra dependency.

The class is a thin specialised wrapper around :class:`Encoder`. It
inherits the polymorphic contract and stores its output width in
``self.output_width`` so the inherited :attr:`output_dim` property
returns it without further work.

Complexity per layer is ``O(E * d)`` dominated by the scatter-add
into a tensor of width ``hidden_dim``. The encoder has no
intermediate Python loops and is friendly to ``torch.compile`` on
CUDA and CPU.

Example:
    >>> import torch
    >>> from pjepa.encoders.euclidean_mpnn import Euclidean
    >>> from pjepa.graphs import Graph
    >>> v = torch.randn((4, 8))
    >>> ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    >>> g = Graph(v, ei, torch.zeros((3, 2)))
    >>> enc = Euclidean(input_dim=8, hidden_dim=16, num_layers=2, output_dim=4)
    >>> out = enc(g)
    >>> tuple(out.shape)
    (4, 4)
"""

from __future__ import annotations

import torch
from torch import nn

from pjepa.encoders.base import Encoder
from pjepa.graphs import Graph

__all__ = ["Euclidean", "UpdateMLP"]


class UpdateMLP(nn.Module):
    """Two-layer MLP used as the inner update of :class:`Euclidean`.

    The architecture is ``Linear -> ReLU -> Linear`` with no
    normalisation, mirroring the original GIN update. Both linear
    layers are initialised with PyTorch's default scheme.

    Attributes:
        lin1: First ``nn.Linear`` mapping ``in_dim`` to ``hidden_dim``.
        lin2: Second ``nn.Linear`` mapping ``hidden_dim`` to ``out_dim``.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden layer width.
        out_dim: Output feature dimension.

    Raises:
        ValueError: If any of ``in_dim``, ``hidden_dim``, ``out_dim``
            is non-positive.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialise the two linear layers.

        Args:
            in_dim: Input feature dimension.
            hidden_dim: Hidden layer width.
            out_dim: Output feature dimension.

        Raises:
            ValueError: If any dimension is non-positive.
        """
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError(
                f"UpdateMLP: all dimensions must be positive; "
                f"got in_dim={in_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}"
            )
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to ``x``.

        Args:
            x: A ``[..., in_dim]`` tensor.

        Returns:
            A ``[..., out_dim]`` tensor.
        """
        return self.lin2(torch.relu(self.lin1(x)))


class Euclidean(Encoder):
    """GIN-style message-passing encoder.

    The encoder is trainable: the trainer can update its parameters
    via the standard ``torch.optim`` machinery. Numerical sanity
    checks belong to the trainer; this class does not raise
    :class:`NumericalError`.

    The class is a polymorphic :class:`Encoder` subclass. The
    inherited :attr:`output_dim` property returns ``self.output_width``,
    which is set in ``__init__``.

    Attributes:
        hidden_dim: Width of the message-passing layers.
        num_layers: Number of message-passing layers.
        input_proj: Project input features to ``hidden_dim``.
        update: The :class:`UpdateMLP` instance used in the update.
        out_proj: Project the final hidden state to ``output_dim``.

    Args:
        input_dim: Vertex feature dimension.
        hidden_dim: Width of the message-passing layers.
        num_layers: Number of message-passing layers.
        output_dim: Dimensionality of the per-vertex embedding.

    Raises:
        ValueError: At construction time if any of ``input_dim``,
            ``hidden_dim``, ``num_layers``, ``output_dim`` is
            non-positive.

    Example:
        >>> from pjepa.encoders.euclidean_mpnn import Euclidean
        >>> enc = Euclidean(input_dim=4, hidden_dim=8, num_layers=2, output_dim=4)
        >>> enc.output_dim
        4
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 128,
    ) -> None:
        """Initialise the encoder.

        Args:
            input_dim: Vertex feature dimension.
            hidden_dim: Width of the message-passing layers.
            num_layers: Number of message-passing layers.
            output_dim: Dimensionality of the per-vertex embedding.

        Raises:
            ValueError: If any of ``input_dim``, ``hidden_dim``,
                ``num_layers``, ``output_dim`` is non-positive.
        """
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0 or output_dim <= 0:
            raise ValueError(
                f"Euclidean: all dimensions must be positive; "
                f"got input_dim={input_dim}, hidden_dim={hidden_dim}, "
                f"num_layers={num_layers}, output_dim={output_dim}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.output_width = int(output_dim)
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        # Update MLP takes concatenated [h || agg] so its input is 2 * hidden_dim.
        self.update = UpdateMLP(2 * int(hidden_dim), int(hidden_dim), int(hidden_dim))
        self.out_proj = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, graph: Graph) -> torch.Tensor:
        """Encode the graph and return a per-vertex embedding.

        Args:
            graph: The input graph.

        Returns:
            A ``[N, output_dim]`` tensor of per-vertex embeddings.
        """
        x = graph.vertex_features
        h = self.input_proj(x)
        edge_index = graph.edge_index
        for _ in range(self.num_layers):
            if edge_index.numel() == 0:
                # Edgeless graph: messages are zero, so the concat reduces to [h || 0].
                agg = torch.zeros_like(h)
            else:
                src = edge_index[0]
                dst = edge_index[1]
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, h[src])
            h = self.update(torch.cat([h, agg], dim=-1))
        return self.out_proj(h)
