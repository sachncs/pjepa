"""Euclidean message-passing neural network encoder.

A simple GIN-style MPNN (Xu et al. 2019) operating on the Euclidean
component of the persistent graph. Returns per-vertex embeddings.

The update rule is::

    h_i^{(l+1)} = MLP^{(l)}([h_i^{(l)}, sum_{j->i} h_j^{(l)}])

the standard GIN-style concatenation of the previous-layer hidden
state with the sum of incoming messages. The aggregation is done
with :meth:`Tensor.index_add_` rather than ``scatter_add`` so the
encoder runs on every supported backend without an extra dependency.

Complexity per layer is ``O(E * d)`` dominated by the scatter-add
into a tensor of width ``hidden_dim``. The encoder has no
intermediate Python loops and is friendly to ``torch.compile`` on
CUDA and CPU.
"""

from __future__ import annotations

import torch
from torch import nn

from pjepa.graphs import TypedAttributedGraph

__all__ = ["EuclideanMPNN", "UpdateMLP"]


class UpdateMLP(nn.Module):
    """Two-layer MLP used inside :class:`EuclideanMPNN`.

    The architecture is ``Linear -> ReLU -> Linear`` with no
    normalisation, mirroring the original GIN update. Both linear
    layers are initialised with PyTorch's default scheme.

    Attributes:
        lin1: First ``nn.Linear`` mapping ``in_dim`` to ``hidden_dim``.
        lin2: Second ``nn.Linear`` mapping ``hidden_dim`` to ``out_dim``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
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


class EuclideanMPNN(nn.Module):
    """GIN-style message-passing encoder.

    The encoder is ``frozen=False`` so the trainer can update its
    parameters. Numerical sanity checks belong to the trainer; this
    class does not raise :class:`NumericalError`.

    Attributes:
        hidden_dim: Width of the message-passing layers.
        num_layers: Number of message-passing layers.
        output_dim: Dimensionality of the per-vertex embedding.
        input_proj: Project input features to ``hidden_dim``.
        update: The :class:`UpdateMLP` instance used in the update.
        out_proj: Project the final hidden state to ``output_dim``.

    Raises:
        ValueError: At construction time if any of ``input_dim``,
            ``hidden_dim``, ``num_layers``, ``output_dim`` is
            non-positive.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0 or output_dim <= 0:
            raise ValueError(
                f"EuclideanMPNN: all dimensions must be positive; "
                f"got input_dim={input_dim}, hidden_dim={hidden_dim}, "
                f"num_layers={num_layers}, output_dim={output_dim}"
            )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Update MLP takes concatenated [h || agg] so its input is 2 * hidden_dim.
        self.update = UpdateMLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
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

    def to(self, device: torch.device) -> EuclideanMPNN:
        """Move parameters to ``device`` and return self.

        The override exists only to tighten the return type and
        make chained device moves read naturally.
        """
        return super().to(device)
