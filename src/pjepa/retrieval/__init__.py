"""Working-graph retrieval via submodular maximisation.

The retriever selects a fixed-budget vertex-induced subgraph of the
persistent graph that maximises a monotone submodular utility. The
greedy algorithm (Algorithm 1 in the paper) achieves the
Nemhauser-Wolsey-Fisher 1978 ``(1 - 1/e)`` approximation guarantee
relative to the optimal subset of the same cardinality.

Two utilities are provided:

* :class:`FacilityLocationUtility` — a provably-submodular
  coverage utility; useful as a fallback and as a sanity baseline.
* :class:`InformationGainUtility` — the framework's headline
  utility, defined as conditional mutual information minus a
  per-vertex cost (see :class:`InformationGainUtility`).
"""

from __future__ import annotations

from pjepa.retrieval.greedy import GreedyRetrieval, RetrievalResult
from pjepa.retrieval.utility import (
    FacilityLocationUtility,
    InformationGainUtility,
    RetrievalUtility,
    facility_location_weights,
    uniform_weights,
)

__all__ = [
    "FacilityLocationUtility",
    "GreedyRetrieval",
    "InformationGainUtility",
    "RetrievalResult",
    "RetrievalUtility",
    "facility_location_weights",
    "uniform_weights",
]
