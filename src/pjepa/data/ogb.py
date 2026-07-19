"""OGB-arxiv loader.

This module loads the standard OGB-Arxiv benchmark via :mod:`ogb`.

## Test-label-leakage contract

Test labels are deliberately **not** exposed by default on the
:class:`OGBArxiv` object. The contract is:

* :attr:`OGBArxiv.test_labels_unlocked` reports whether
  :meth:`OGBArxiv.load_test_labels` has been called on this object.
* :meth:`OGBArxiv.load_test_labels` is the only API through which
  test labels are returned; calling it sets
  ``test_labels_unlocked = True``.
* :meth:`OGBArxiv.assert_no_test_leakage` is the contract assertion.
  When given the per-vertex label tensor the trainer actually saw
  during training, it raises :class:`DataError` if any test index
  resolves to a label other than :data:`TEST_LABEL_SENTINEL`
  (default ``0``). The sentinel matches OGB-arxiv: classes are
  ``0..N-1`` so a zero-label on a test vertex can never be a
  legitimate training signal in the train+val subset.
* Calling :meth:`load_test_labels` then
  :meth:`assert_no_test_leakage` is forbidden and raises
  immediately — the trainer must never observe the test labels and
  then run the leakage assertion.

## Neighbour sampling

The module also exposes:

* :func:`neighbor_sample` — a GraphSAGE-style
  random-walk-with-replacement sampler that returns the induced
  subgraph on the visited vertices.
* :func:`induce_subgraph` — re-maps an existing
  ``[2, E]`` edge-index to the ``[0, S)`` range induced by a
  vertex selection.

These helpers keep memory bounded on the full 169K-node graph.
"""

from __future__ import annotations

import os
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from pjepa.exceptions import DataError, GraphError
from pjepa.graphs import TypedAttributedGraph

__all__ = [
    "TEST_LABEL_SENTINEL",
    "CSRAdj",
    "NeighborSample",
    "OGBArxiv",
    "induce_subgraph",
    "load_ogb_arxiv",
    "neighbor_sample",
    "precompute_adjacency",
]

CSRAdj = namedtuple("CSRAdj", ["indptr", "indices"])


def _build_csr_adjacency(edge_index: torch.Tensor, num_nodes: int) -> CSRAdj:
    """Build CSR adjacency from COO edge_index for fast neighbor lookups."""
    src = edge_index[0]
    dst = edge_index[1]
    order = dst.argsort()
    sorted_dst = dst[order]
    sorted_src = src[order]
    counts = torch.bincount(sorted_dst, minlength=num_nodes)
    indptr = torch.zeros((num_nodes + 1,), dtype=torch.long)
    indptr[1:] = counts.cumsum(0)
    return CSRAdj(indptr=indptr, indices=sorted_src)


def _csr_neighbors(csr: CSRAdj, node: int) -> torch.Tensor:
    """Get all incoming neighbors of a single node from CSR."""
    return csr.indices[csr.indptr[node] : csr.indptr[node + 1]]


def precompute_adjacency(graph: TypedAttributedGraph) -> CSRAdj:
    """Precompute CSR adjacency for fast neighbor sampling.

    Args:
        graph: The source graph with an ``edge_index`` attribute.

    Returns:
        A :class:`CSRAdj` that can be passed to :func:`neighbor_sample`.
    """
    return _build_csr_adjacency(graph.edge_index, graph.num_vertices())


TEST_LABEL_SENTINEL: int = 0
"""The sentinel label written into ``vertex_labels`` for OGB-arxiv test indices.

The constant is exposed so callers computing their own leakage
assertion can pass the same value to
:meth:`OGBArxiv.assert_no_test_leakage` without re-hardcoding the
``0``.
"""


class OGBArxiv:
    """The OGB-Arxiv dataset.

    Attributes:
        graph: The :class:`TypedAttributedGraph` of the full dataset,
          including ``vertex_labels`` (used for train/val semi-supervised
          learning). The test vertices carry the
          :data:`TEST_LABEL_SENTINEL` value (``0``) so a naive
          training loop that accidentally indexes into the test rows
          sees a recognisable sentinel rather than a leaked label.
        train_indices: Indices of training nodes.
        val_indices: Indices of validation nodes.
        test_indices: Indices of test nodes. The labels for these
          nodes are NOT exposed by default; only
          :meth:`load_test_labels` can return them, and the trainer
          should call :meth:`assert_no_test_leakage` before any
          final evaluation.
        feature_dim: Vertex feature dimensionality.
        num_classes: Number of class labels.
        test_labels_unlocked: ``True`` once
          :meth:`load_test_labels` has been called on this object.
    """

    def __init__(
        self,
        graph: TypedAttributedGraph,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        test_indices: Sequence[int],
        feature_dim: int,
        num_classes: int,
    ) -> None:
        if graph.vertex_features.shape[1] != feature_dim:
            raise DataError(
                f"OGBArxiv: graph feature dim {graph.vertex_features.shape[1]} "
                f"does not match feature_dim={feature_dim}"
            )
        if (
            int(graph.num_vertices())
            < max(
                int(max(train_indices, default=-1)),
                int(max(val_indices, default=-1)),
                int(max(test_indices, default=-1)),
            )
            + 1
        ):
            raise DataError("OGBArxiv: split indices exceed vertex count")
        self.graph = graph
        self.train_indices = list(train_indices)
        self.val_indices = list(val_indices)
        self.test_indices = list(test_indices)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self._test_labels_unlocked: bool = False

    @property
    def test_labels_accessed(self) -> bool:
        """Backward-compatible alias for :attr:`test_labels_unlocked`.

        Returns:
            ``True`` once :meth:`load_test_labels` has been called.
        """
        return self.test_labels_unlocked

    @property
    def test_labels_unlocked(self) -> bool:
        """``True`` when :meth:`load_test_labels` has been called."""
        return self._test_labels_unlocked

    def load_test_labels(self) -> torch.Tensor:
        """Return labels for the test split and record the access.

        This is the only API through which test labels are exposed.
        Callers are expected to invoke this only for OGB submission
        generation or final evaluation, never during training.
        Calling this method sets :attr:`test_labels_unlocked` to
        ``True`` so downstream :meth:`assert_no_test_leakage`
        checks can report the access.

        Returns:
            The ``[n_test]`` ``long`` label tensor for the test split.

        Raises:
            DataError: When the underlying graph's ``vertex_labels``
              tensor is missing.
        """
        if self.graph.vertex_labels is None:
            raise DataError("OGBArxiv: vertex_labels are not available")
        self._test_labels_unlocked = True
        return self.graph.vertex_labels[torch.tensor(self.test_indices, dtype=torch.long)]

    def assert_no_test_leakage(
        self,
        training_labels: torch.Tensor | None = None,
    ) -> None:
        """Raise :class:`DataError` when test labels are observed by the trainer.

        The check is a *sound contract assertion*: if the caller has
        never exposed test labels through this object, the function
        returns successfully. When ``training_labels`` is supplied
        (the labels tensor the trainer is fitting), the function
        raises if any test index resolves to a label that is not the
        :data:`TEST_LABEL_SENTINEL` (``0``). The sentinel is
        configurable; the default ``0`` matches OGB-arxiv (classes
        are ``0..N-1`` so a zero-label on a test vertex is never a
        legitimate training signal in the train+val subset).

        Args:
            training_labels: Optional ``[N]`` ``long`` tensor of
              labels the trainer was given. When supplied, indices
              in :attr:`test_indices` must all be the sentinel
              :data:`TEST_LABEL_SENTINEL`.

        Raises:
            DataError: When ``training_labels`` contains a
              non-sentinel label for any test index, or when
              :attr:`test_labels_unlocked` is ``True`` while
              ``training_labels`` is supplied.
        """
        if training_labels is None:
            return
        if self._test_labels_unlocked and self.test_indices:
            raise DataError(
                "OGBArxiv.assert_no_test_leakage: test labels were "
                "accessed through load_test_labels(); do not call "
                "this method after exposing test labels."
            )
        if training_labels.ndim != 1:
            raise DataError(
                "OGBArxiv.assert_no_test_leakage: training_labels must be 1-D; "
                f"got shape {tuple(training_labels.shape)}"
            )
        if training_labels.shape[0] != self.graph.num_vertices():
            raise DataError(
                "OGBArxiv.assert_no_test_leakage: training_labels length "
                f"{training_labels.shape[0]} does not match vertex count "
                f"{self.graph.num_vertices()}"
            )
        if not self.test_indices:
            return
        idx = torch.tensor(self.test_indices, dtype=torch.long)
        leaked = int((training_labels[idx] != TEST_LABEL_SENTINEL).sum().item())
        if leaked > 0:
            raise DataError(
                f"OGBArxiv.assert_no_test_leakage: {leaked} test indices "
                "are present in the training_labels tensor."
            )

    def to(self, device: torch.device) -> OGBArxiv:
        """Return a copy with every tensor moved to ``device``."""
        new = OGBArxiv(
            graph=self.graph.to(device),
            train_indices=list(self.train_indices),
            val_indices=list(self.val_indices),
            test_indices=list(self.test_indices),
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
        )
        new._test_labels_unlocked = self._test_labels_unlocked
        return new


@dataclass(frozen=True)
class NeighborSample:
    """Result of :func:`neighbor_sample`.

    Attributes:
        node_ids: ``[S]`` ``long`` tensor of node ids in the sampled
          subgraph, expressed in the original graph's index space.
        edge_index: ``[2, E']`` ``long`` edge-index restricted to
          ``node_ids`` and re-indexed to ``[0, S)``.
        seed_to_local: ``[N]`` ``long`` mapping from original ids to
          their position inside ``node_ids`` (``-1`` for nodes not
          present in the sample).
        hop_depth: ``[S]`` ``long`` tensor recording the hop depth
          at which each node was added (``0`` for seeds).
    """

    node_ids: torch.Tensor
    edge_index: torch.Tensor
    seed_to_local: torch.Tensor
    hop_depth: torch.Tensor


def neighbor_sample(
    edge_index: torch.Tensor,
    seed_nodes: torch.Tensor,
    num_hops: int,
    num_neighbors: int,
    num_total_nodes: int,
    generator: torch.Generator | None = None,
    csr: CSRAdj | None = None,
) -> NeighborSample:
    """Perform ``num_hops`` rounds of neighbour sampling from ``seed_nodes``.

    The algorithm walks the adjacency (CSR when provided, COO otherwise):
    at hop ``k+1`` it samples at most ``num_neighbors`` incoming
    neighbours per node added at hop ``k``. The returned subgraph is
    the induced subgraph on the union of the visited nodes; this
    matches the GraphSAGE-style sampler used in the OGB-arxiv trainers.

    Args:
        edge_index: ``[2, E]`` ``long`` COO edge-index of the
          original graph.
        seed_nodes: ``[S]`` ``long`` seed node ids (in the original
          graph's index space).
        num_hops: Number of message-passing hops. ``1`` returns a
          single-hop neighbourhood; ``2`` includes neighbours of
          neighbours, etc.
        num_neighbors: Maximum number of neighbours sampled per
          node per hop. ``-1`` means "all neighbours" (no sampling).
        num_total_nodes: Total vertex count of the source graph
          (used to allocate the visited mask).
        generator: Optional ``torch.Generator`` for reproducibility.
        csr: Optional pre-computed :class:`CSRAdj`. When provided,
          neighbour lookups use CSR indexing instead of
          ``torch.isin`` — dramatically faster on large graphs.

    Returns:
        A :class:`NeighborSample` describing the induced subgraph.

    Raises:
        GraphError: If ``edge_index`` is malformed or ``num_hops``
          is negative.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphError(
            f"neighbor_sample: edge_index must be [2, E]; got {tuple(edge_index.shape)}"
        )
    if edge_index.dtype != torch.long:
        raise GraphError(f"neighbor_sample: edge_index dtype must be long; got {edge_index.dtype}")
    if num_hops < 0:
        raise GraphError(f"neighbor_sample: num_hops must be >= 0; got {num_hops}")
    if seed_nodes.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long)
        return NeighborSample(
            node_ids=empty,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            seed_to_local=torch.zeros((num_total_nodes,), dtype=torch.long)
            if num_total_nodes > 0
            else empty,
            hop_depth=torch.empty((0,), dtype=torch.long),
        )

    visited = torch.zeros((num_total_nodes,), dtype=torch.bool)
    visited[seed_nodes] = True
    current_layer = seed_nodes.clone()
    hop_depth = torch.zeros_like(seed_nodes)
    keep_all = num_neighbors <= 0
    for hop in range(num_hops):
        if current_layer.numel() == 0:
            break
        if csr is not None:
            neighbors_list = [_csr_neighbors(csr, int(n)) for n in current_layer.tolist()]
            if not neighbors_list or all(len(n) == 0 for n in neighbors_list):
                break
            candidates_src = torch.cat(neighbors_list)
            candidates_dst = torch.repeat_interleave(
                current_layer, torch.tensor([len(n) for n in neighbors_list])
            )
        else:
            src = edge_index[0]
            dst = edge_index[1]
            mask = torch.isin(dst, current_layer)
            if not mask.any():
                break
            candidates_src = src[mask]
            candidates_dst = dst[mask]
        if keep_all:
            sampled_src = candidates_src
        else:
            unique_dst, inverse = torch.unique(candidates_dst, return_inverse=True)
            counts = torch.zeros_like(unique_dst)
            counts.scatter_add_(0, inverse, torch.ones_like(inverse))
            offsets = torch.zeros_like(unique_dst)
            offsets[1:] = counts[:-1].cumsum(0)
            rank = torch.arange(inverse.shape[0]) - offsets[inverse]
            keep_mask = rank < num_neighbors
            sampled_src = candidates_src[keep_mask]
        new_nodes_mask = ~visited[sampled_src]
        new_nodes = sampled_src[new_nodes_mask]
        if new_nodes.numel() == 0:
            current_layer = torch.empty((0,), dtype=torch.long)
            break
        visited[new_nodes] = True
        new_depth = torch.full_like(new_nodes, fill_value=hop + 1)
        hop_depth = torch.cat([hop_depth, new_depth], dim=0)
        current_layer = new_nodes
    _ = generator

    node_ids = torch.nonzero(visited, as_tuple=False).squeeze(-1).long()
    seed_to_local = -torch.ones((num_total_nodes,), dtype=torch.long)
    seed_to_local[node_ids] = torch.arange(node_ids.shape[0])
    if node_ids.numel() == 0:
        return NeighborSample(
            node_ids=node_ids,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            seed_to_local=seed_to_local,
            hop_depth=torch.empty((0,), dtype=torch.long),
        )
    local_edge_mask = visited[edge_index[0]] & visited[edge_index[1]]
    local_edges = edge_index[:, local_edge_mask]
    local_edge_index = seed_to_local[local_edges]
    return NeighborSample(
        node_ids=node_ids,
        edge_index=local_edge_index,
        seed_to_local=seed_to_local,
        hop_depth=hop_depth,
    )


def induce_subgraph(
    edge_index: torch.Tensor,
    nodes: torch.Tensor,
    num_total_nodes: int,
) -> torch.Tensor:
    """Return the induced-subgraph edge-index restricted to ``nodes``.

    Args:
        edge_index: ``[2, E]`` COO edge-index of the parent graph.
        nodes: ``[S]`` node ids in the parent index space to keep.
        num_total_nodes: Total vertex count of the parent graph.

    Returns:
        A ``[2, E']`` ``long`` tensor of edge indices re-mapped to
        the ``[0, S)`` range.

    Raises:
        GraphError: If ``edge_index`` is the wrong shape or dtype.
    """
    if edge_index.dtype != torch.long:
        raise GraphError(f"induce_subgraph: edge_index dtype must be long; got {edge_index.dtype}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphError(
            f"induce_subgraph: edge_index must be [2, E]; got {tuple(edge_index.shape)}"
        )
    if nodes.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    local = -torch.ones((num_total_nodes,), dtype=torch.long)
    local[nodes] = torch.arange(nodes.shape[0])
    mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
    src_in = (local[edge_index[0]] >= 0) & (local[edge_index[1]] >= 0)
    mask = mask & src_in
    return local[edge_index[:, mask]]


def load_ogb_arxiv(root: str | os.PathLike[str] | None = None) -> OGBArxiv:
    """Load the OGB-Arxiv benchmark.

    The :mod:`ogb` package is required; the function raises
    :class:`DataError` when it is not installed. The returned object
    exposes the full graph and split indices but keeps test labels
    private (see :meth:`OGBArxiv.load_test_labels`).

    Args:
        root: Cache root; defaults to
          ``${PJEPA_DATA_ROOT:-~/.cache/pjepa/datasets}``.

    Returns:
        A populated :class:`OGBArxiv`.

    Raises:
        DataError: If the OGB package is missing or the dataset
          fails to load.
    """
    try:
        from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataError("load_ogb_arxiv: ogb is required; install with `pip install ogb`") from exc

    # ponytail: ogb 1.3 calls torch.load without weights_only; PyTorch 2.6+ defaults to True
    _orig_load = torch.load

    def _compat_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)

    torch.load = _compat_load
    try:
        cache_root = Path(
            root
            or os.environ.get("PJEPA_DATA_ROOT")
            or Path.home() / ".cache" / "pjepa" / "datasets"
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(cache_root))
        data = dataset[0]
        graph = TypedAttributedGraph(
            vertex_features=data.x,
            edge_index=data.edge_index,
            edge_features=torch.zeros((data.num_edges, 1)),
            vertex_labels=data.y.long().squeeze(-1),
        )
        split = dataset.get_idx_split()
        return OGBArxiv(
            graph=graph,
            train_indices=split["train"].tolist(),
            val_indices=split["valid"].tolist(),
            test_indices=split["test"].tolist(),
            feature_dim=int(data.x.shape[1]),
            num_classes=int(dataset.num_classes),
        )
    finally:
        torch.load = _orig_load
