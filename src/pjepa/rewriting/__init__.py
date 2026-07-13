"""Verified rewriting engine.

This subpackage implements the hyperedge-replacement grammar (HRG),
the bisimulation metric, the four-conditions acceptance criterion,
and the DPO rewriting loss.

Workflow::

    candidate = grammar.expand(some_nonterminal)
    accepted, info = accept_candidate(
        candidate,
        current_graph,
        observation,
        grammar,
        thresholds=FourConditions(),
    )
    if accepted:
        loss = dpo_loss(c_lp, r_lp, c_ref, r_ref)

The ``accepted`` / ``rejected`` decision is purely a function of the
acceptance criterion; the engine itself never modifies state.
"""

from __future__ import annotations

from pjepa.rewriting.bisimulation import BisimulationMetric, bisimulation_distance
from pjepa.rewriting.dpo import DPOConfig, dpo_loss
from pjepa.rewriting.four_conditions import (
    FourConditions,
    accept_candidate,
    compute_delta_j,
)
from pjepa.rewriting.hrg import HRG, HRGProduction

__all__ = [
    "HRG",
    "BisimulationMetric",
    "DPOConfig",
    "FourConditions",
    "HRGProduction",
    "accept_candidate",
    "bisimulation_distance",
    "compute_delta_j",
    "dpo_loss",
]
