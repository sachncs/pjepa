"""BGRL — Bootstrap Graph Latents (Thakoor et al., 2022).

BGRL learns node-level representations without negative samples: an
online encoder + predictor is matched against a target encoder that
is updated as an exponential moving average of the online encoder.
The loss is a ``(1 - cosine similarity)`` penalty between the
online predictor output and the (frozen) target encoder output on
two augmented views of the same graph.

## Architecture

```
   view_a ──► online_encoder ──► predictor ──► z_a
   view_b ──► target_encoder ──► z_b (no grad)

   loss = (1 - cos(z_a, z_b)).mean()
   target ← momentum * target + (1 - momentum) * online
```

The class exposes :meth:`pretrain_step` for self-supervised
training and :meth:`node_logits` for downstream linear-probe
evaluation.

## Complexity

* :meth:`pretrain_step` — one forward pass through each
  encoder + one predictor pass. ``O(L * B * H)`` per ``H``
  hidden dim and ``L`` GraphSAGE layers.
* :meth:`encode` / :meth:`embed` — single online-encoder
  forward pass, ``O(|V| * H)`` for ``|V|`` vertices.

## Exceptions

:meth:`loss` raises :class:`GraphError` when the two views
have different vertex counts or empty view.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pjepa.baselines.graphsage import GraphSAGE
from pjepa.exceptions import GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = ["BGRL"]


class BGRL(nn.Module):
    """Bootstrap Graph Latents with an online/target encoder pair.

    Attributes:
        hidden_dim: Width of the online / target encoders.
        num_layers: Depth of the GraphSAGE backbones.
        momentum: Initial EMA momentum for the target encoder.
        num_classes: Classifier width (``0`` disables the head).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        momentum: float = 0.99,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("BGRL: dims must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"BGRL: momentum must be in [0, 1); got {momentum}")
        self.online_encoder = GraphSAGE(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=0
        )
        self.target_encoder = GraphSAGE(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=0
        )
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier: nn.Module | None = None
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.momentum = float(momentum)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

    def update_target(self) -> None:
        r"""Update the target encoder as an EMA of the online encoder.

        For every named parameter the new target weight is

        .. math::

            \theta_T \gets m \cdot \theta_T + (1 - m) \cdot \theta_O

        where ``m`` is :attr:`momentum` and ``\theta_O`` is the
        online parameter.
        """
        with torch.no_grad():
            for online_p, target_p in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
            ):
                target_p.data.mul_(self.momentum).add_(online_p.data, alpha=1.0 - self.momentum)

    def online_embedding(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return online predictions for ``graph``.

        Args:
            graph: The input graph.

        Returns:
            The predictor output for the online-encoder embeddings,
            shape ``[N, hidden_dim]``.
        """
        h = self.online_encoder.encode(graph)
        return self.predictor(h)

    def target_embedding(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return target embeddings for ``graph`` (no gradient).

        Args:
            graph: The input graph.

        Returns:
            The frozen target-encoder embeddings, shape
            ``[N, hidden_dim]``.
        """
        with torch.no_grad():
            return self.target_encoder.encode(graph)

    def loss(self, view_a: TypedAttributedGraph, view_b: TypedAttributedGraph) -> torch.Tensor:
        """Compute the BGRL cosine-similarity loss for two augmented views.

        The loss is the mean of ``(1 - cos(z_a, z_b))`` over the
        per-vertex features, with both ``z_a`` and ``z_b`` L2
        normalised before the dot product so the cosine similarity is
        numerically equivalent to a centred dot product.

        Args:
            view_a: The first augmented view.
            view_b: The second augmented view (must share the
              vertex count of ``view_a``).

        Returns:
            A scalar loss tensor.

        Raises:
            GraphError: If either view is empty or the two views
              have different vertex counts.
        """
        if view_a.num_vertices() == 0 or view_b.num_vertices() == 0:
            raise GraphError("BGRL.loss: cannot compute loss on an empty view")
        if view_a.num_vertices() != view_b.num_vertices():
            raise GraphError(
                f"BGRL.loss: views must have the same vertex count; "
                f"got {view_a.num_vertices()} vs {view_b.num_vertices()}"
            )
        z_a = self.online_embedding(view_a)
        with torch.no_grad():
            z_b = self.target_embedding(view_b)
        z_a = F.normalize(z_a, p=2, dim=-1)
        z_b = F.normalize(z_b, p=2, dim=-1)
        cos = (z_a * z_b).sum(dim=-1)
        return (1.0 - cos).mean()

    def pretrain_step(
        self,
        view_a: TypedAttributedGraph,
        view_b: TypedAttributedGraph,
    ) -> torch.Tensor:
        """Compute the BGRL loss and update the target encoder in-place.

        Args:
            view_a: The first augmented view.
            view_b: The second augmented view.

        Returns:
            The scalar loss tensor (the same value :meth:`loss` returns).
        """
        loss = self.loss(view_a, view_b)
        self.update_target()
        return loss

    def encode(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return (online) per-vertex embeddings for downstream probing.

        Args:
            graph: The input graph.

        Returns:
            The online-encoder per-vertex embeddings, shape
            ``[N, hidden_dim]``.
        """
        return self.online_encoder.encode(graph)

    def embed(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return mean-pooled (online) graph embedding.

        Args:
            graph: The input graph.

        Returns:
            ``[1, hidden_dim]`` pooled embedding.
        """
        return self.online_encoder.embed(graph)

    def node_logits(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return per-vertex logits using the (frozen) classifier head.

        Args:
            graph: The input graph.

        Returns:
            ``[N, num_classes]`` per-vertex logits.

        Raises:
            RuntimeError: When the classifier is disabled
              (``num_classes=0``).
        """
        if self.classifier is None:
            raise RuntimeError("BGRL: classifier is disabled (num_classes=0)")
        return self.classifier(self.online_encoder.encode(graph))

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Return per-graph logits via mean-pooled online embeddings.

        Args:
            graph: The input graph.

        Returns:
            For ``num_classes > 0``: ``[1, num_classes]`` per-graph
            logits. For ``num_classes == 0``: the mean-pooled online
            embedding, ``[1, hidden_dim]``.
        """
        if self.classifier is None:
            return self.embed(graph)
        return self.classifier(self.embed(graph))
