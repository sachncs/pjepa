"""Data loaders for TUDataset, OGB, and continual-learning splits.

## Layout

* :mod:`pjepa.data.tu` — :class:`TUGraph` and :func:`load_tu_dataset`
  for the standard TUDataset family (PROTEINS, MUTAG, NCI1, ...).
* :mod:`pjepa.data.ogb` — :class:`OGBArxiv` with the
  test-label-safe-by-default contract, neighbour sampling, and
  induced-subgraph helpers used by the Phase-10 trainers.
* :mod:`pjepa.data.cl_splits` — :class:`ClassIncrementalSplit` for
  class-incremental continual-learning evaluation.
"""

from __future__ import annotations

from pjepa.data.cl_splits import ClassIncrementalSplit, make_class_incremental_split
from pjepa.data.ogb import (
    TEST_LABEL_SENTINEL,
    NeighborSample,
    OGBArxiv,
    induce_subgraph,
    load_ogb_arxiv,
    neighbor_sample,
)
from pjepa.data.tu import TUGraph, expected_checksum, load_tu_dataset

__all__ = [
    "TEST_LABEL_SENTINEL",
    "ClassIncrementalSplit",
    "NeighborSample",
    "OGBArxiv",
    "TUGraph",
    "expected_checksum",
    "induce_subgraph",
    "load_ogb_arxiv",
    "load_tu_dataset",
    "make_class_incremental_split",
    "neighbor_sample",
]
