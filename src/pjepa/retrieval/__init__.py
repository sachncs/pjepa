"""Working-graph retrieval via submodular maximisation.

The retriever selects a fixed-budget vertex-induced subgraph of the
persistent graph that maximises a monotone submodular utility. The
greedy algorithm (Algorithm 1 in the paper) achieves the
Nemhauser-Wolsey-Fisher 1978 ``(1 - 1/e)`` approximation guarantee.
"""

from __future__ import annotations

from pjepa.retrieval.greedy import GreedyRetrieval
from pjepa.retrieval.utility import (
    FacilityLocationUtility,
    InformationGainUtility,
    RetrievalUtility,
    facility_location_weights,
    uniform_weights,
)

__all__ = [
    "GreedyRetrieval",
    "RetrievalUtility",
    "InformationGainUtility",
    "FacilityLocationUtility",
    "uniform_weights",
    "facility_location_weights",
]