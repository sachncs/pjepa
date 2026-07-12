"""Typed attributed graphs.

A :class:`TypedAttributedGraph` is an immutable, fully-typed container
for the graph data structures used throughout the framework. The class
is intentionally minimal: it stores vertex and edge features plus
optional labels and a global feature vector, but does not interpret
them. Downstream modules (encoders, retrieval, rewriting) attach
meaning via well-defined protocols.

The dataclass is ``frozen=True``: every "mutation" returns a new
instance, which eliminates an entire class of bugs (in-place graph
edits) and makes the graphs hashable for caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from pjepa.exceptions import GraphError

__all__ = ["TypedAttributedGraph", "graph_from_edge_index"]


@dataclass(frozen=True)
class TypedAttributedGraph:
    """Immutable typed attributed graph.

    Attributes:
        vertex_features: A ``[N, d_v]`` tensor of vertex features.
        edge_index: A ``[2, E]`` ``long`` tensor in COO format.
        edge_features: A ``[E, d_e]`` tensor of edge features; may be
          empty when the graph has no edges.
        vertex_labels: Optional ``[N]`` ``long`` tensor of categorical
          vertex labels.
        edge_labels: Optional ``[E]`` ``long`` tensor of categorical
          edge labels.
        global_features: Optional ``[d_g]`` tensor of graph-level
          features.
        version: A monotonically increasing version counter, bumped on
          every functional update.

    Example:
        >>> v = torch.randn((3, 4))
        >>> ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        >>> e = torch.randn((4, 2))
        >>> g = TypedAttributedGraph(v, ei, e)
        >>> g.num_vertices()
        3
    """

    vertex_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor = field(default_factory=lambda: torch.empty((0, 0)))
    vertex_labels: torch.Tensor | None = None
    edge_labels: torch.Tensor | None = None
    global_features: torch.Tensor | None = None
    version: int = 0

    def __post_init__(self) -> None:
        """Validate shape consistency on construction."""
        if self.vertex_features.ndim != 2:
            raise GraphError(
                f"TypedAttributedGraph: vertex_features must be 2-D; "
                f"got shape {tuple(self.vertex_features.shape)}"
            )
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise GraphError(
                f"TypedAttributedGraph: edge_index must be [2, E]; "
                f"got shape {tuple(self.edge_index.shape)}"
            )
        if self.edge_index.dtype != torch.long:
            raise GraphError(
                f"TypedAttributedGraph: edge_index dtype must be long; got {self.edge_index.dtype}"
            )
        n_vertices = self.vertex_features.shape[0]
        if n_vertices > 0 and self.edge_index.numel() > 0:
            max_idx = int(self.edge_index.max().item())
            if max_idx >= n_vertices:
                raise GraphError(
                    f"TypedAttributedGraph: edge index {max_idx} exceeds vertex count {n_vertices}"
                )
        if self.edge_features.ndim != 2:
            raise GraphError(
                f"TypedAttributedGraph: edge_features must be 2-D; "
                f"got shape {tuple(self.edge_features.shape)}"
            )
        if (
            self.edge_features.numel() > 0
            and self.edge_features.shape[0] != self.edge_index.shape[1]
        ):
            raise GraphError(
                f"TypedAttributedGraph: edge_features first dim "
                f"{self.edge_features.shape[0]} does not match edge count "
                f"{self.edge_index.shape[1]}"
            )
        if self.vertex_labels is not None and self.vertex_labels.shape[0] != n_vertices:
            raise GraphError(
                f"TypedAttributedGraph: vertex_labels length "
                f"{self.vertex_labels.shape[0]} does not match vertex count "
                f"{n_vertices}"
            )
        if self.edge_labels is not None and self.edge_labels.numel() > 0:
            if self.edge_labels.shape[0] != self.edge_index.shape[1]:
                raise GraphError(
                    f"TypedAttributedGraph: edge_labels length "
                    f"{self.edge_labels.shape[0]} does not match edge count "
                    f"{self.edge_index.shape[1]}"
                )

    def num_vertices(self) -> int:
        """Return the number of vertices in the graph."""
        return int(self.vertex_features.shape[0])

    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return int(self.edge_index.shape[1])

    def with_features(self, **kwargs: object) -> TypedAttributedGraph:
        """Return a copy with selected fields replaced; bumps the version.

        Args:
            **kwargs: Field names to update on the new instance.

        Returns:
            A new :class:`TypedAttributedGraph` with the requested
            fields replaced and ``version + 1``.

        Raises:
            GraphError: If a provided field is not a valid attribute.

        Example:
            >>> g2 = g.with_features(global_features=torch.zeros(8))
        """
        kwargs["version"] = self.version + 1
        try:
            return replace(self, **kwargs)
        except TypeError as exc:
            raise GraphError(f"TypedAttributedGraph.with_features: {exc}") from exc

    def subgraph(self, vertex_mask: torch.Tensor) -> TypedAttributedGraph:
        """Return the vertex-induced subgraph on the given boolean mask.

        Args:
            vertex_mask: A ``[N]`` boolean or 0/1 tensor selecting the
              vertices to keep.

        Returns:
            A new :class:`TypedAttributedGraph` containing only the
            selected vertices and edges between them.

        Raises:
            GraphError: If ``vertex_mask`` has the wrong shape or
              contains non-boolean values.

        Example:
            >>> mask = torch.tensor([True, False, True])
            >>> sub = g.subgraph(mask)
            >>> sub.num_vertices()
            2
        """
        if vertex_mask.shape != (self.num_vertices(),):
            raise GraphError(
                f"TypedAttributedGraph.subgraph: mask shape "
                f"{tuple(vertex_mask.shape)} does not match vertex count "
                f"{self.num_vertices()}"
            )
        if vertex_mask.dtype != torch.bool:
            vertex_mask = vertex_mask.bool()

        new_vertices = self.vertex_features[vertex_mask]
        if new_vertices.shape[0] == 0:
            return TypedAttributedGraph(
                vertex_features=torch.zeros((0, self.vertex_features.shape[1])),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_features=torch.zeros((0, self.edge_features.shape[1])),
                vertex_labels=(
                    self.vertex_labels[vertex_mask] if self.vertex_labels is not None else None
                ),
                edge_labels=torch.zeros((0,), dtype=torch.long)
                if self.edge_labels is not None and self.edge_labels.numel() > 0
                else None,
                global_features=self.global_features,
                version=self.version + 1,
            )

        device = self.vertex_features.device
        old_to_new = -torch.ones(self.num_vertices(), dtype=torch.long, device=device)
        old_to_new[vertex_mask] = torch.arange(int(vertex_mask.sum().item()), device=device)
        edge_mask = vertex_mask[self.edge_index[0]] & vertex_mask[self.edge_index[1]]
        new_edges = self.edge_index[:, edge_mask]
        new_edges = old_to_new[new_edges]
        new_edge_features = self.edge_features[edge_mask]

        return TypedAttributedGraph(
            vertex_features=new_vertices,
            edge_index=new_edges,
            edge_features=new_edge_features,
            vertex_labels=(
                self.vertex_labels[vertex_mask] if self.vertex_labels is not None else None
            ),
            edge_labels=(self.edge_labels[edge_mask] if self.edge_labels is not None else None),
            global_features=self.global_features,
            version=self.version + 1,
        )

    def to(self, device: torch.device) -> TypedAttributedGraph:
        """Move every tensor to the given device."""
        return TypedAttributedGraph(
            vertex_features=self.vertex_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            vertex_labels=(
                self.vertex_labels.to(device) if self.vertex_labels is not None else None
            ),
            edge_labels=(self.edge_labels.to(device) if self.edge_labels is not None else None),
            global_features=(
                self.global_features.to(device) if self.global_features is not None else None
            ),
            version=self.version,
        )


def graph_from_edge_index(
    edge_index: torch.Tensor,
    num_vertices: int,
    vertex_dim: int = 0,
    edge_dim: int = 0,
) -> TypedAttributedGraph:
    """Construct a graph from an edge index, optionally synthesising features.

    Useful for tests and for ingesting raw adjacency data without
    having to fabricate features manually.

    Args:
        edge_index: A ``[2, E]`` ``long`` tensor in COO format.
        num_vertices: The number of vertices in the graph.
        vertex_dim: Feature dimension for synthesised vertex features.
          Zero yields zero-row tensors.
        edge_dim: Feature dimension for synthesised edge features.
          Zero yields zero-row tensors.

    Returns:
        A new :class:`TypedAttributedGraph` with the requested
        topology and (optional) zero-initialised features.

    Raises:
        GraphError: If ``num_vertices`` is negative.

    Example:
        >>> ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        >>> g = graph_from_edge_index(ei, num_vertices=2, vertex_dim=4)
    """
    if num_vertices < 0:
        raise GraphError(
            f"graph_from_edge_index: num_vertices must be non-negative; got {num_vertices}"
        )
    v = torch.zeros((num_vertices, vertex_dim))
    e = torch.zeros((edge_index.shape[1], edge_dim))
    return TypedAttributedGraph(vertex_features=v, edge_index=edge_index, edge_features=e)
