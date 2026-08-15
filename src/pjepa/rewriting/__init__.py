"""Verified rewriting engine.

This subpackage implements the hyperedge-replacement grammar
(:class:`HRG`), the bisimulation metric (:class:`Bisimulation`),
the four-conditions acceptance criterion (:class:`FourConditions`),
and the DPO rewriting loss.

The :class:`Criterion` abstract base class is the polymorphic
root of the acceptance-criterion hierarchy. The concrete
:class:`FourConditions` is the headline implementation; ablations
can plug in a relaxed subclass by inheriting from :class:`Criterion`.

Workflow::

    candidate = grammar.expand(some_nonterminal)
    criterion = FourConditions()
    accepted, info = criterion.evaluate(
        candidate, current_graph, observation, grammar
    )
    if accepted:
        loss = dpo_loss(c_lp, r_lp, c_ref, r_ref)

The ``accepted`` / ``rejected`` decision is purely a function of
the acceptance criterion; the engine itself never modifies state.
"""

from __future__ import annotations

from pjepa.rewriting.bisimulation import Bisimulation, bisimulation_distance
from pjepa.rewriting.dpo import DPOConfig, dpo_loss
from pjepa.rewriting.four_conditions import (
    Criterion,
    FourConditions,
    accept,
    compute_delta_j,
)
from pjepa.rewriting.hrg import HRG, HRGProduction

__all__ = [
    "HRG",
    "Bisimulation",
    "Criterion",
    "DPOConfig",
    "FourConditions",
    "HRGProduction",
    "accept",
    "bisimulation_distance",
    "compute_delta_j",
    "dpo_loss",
]
