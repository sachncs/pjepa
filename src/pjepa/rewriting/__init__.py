"""Verified rewriting engine.

This subpackage implements the hyperedge-replacement grammar (HRG),
the bisimulation metric, the four-conditions acceptance criterion,
and the DPO rewriting loss.
"""

from __future__ import annotations

from pjepa.rewriting.bisimulation import BisimulationMetric, bisimulation_distance
from pjepa.rewriting.four_conditions import FourConditions, accept_candidate
from pjepa.rewriting.hrg import HRG, HRGProduction
from pjepa.rewriting.dpo import DPOConfig, dpo_loss

__all__ = [
    "HRG",
    "HRGProduction",
    "BisimulationMetric",
    "bisimulation_distance",
    "FourConditions",
    "accept_candidate",
    "DPOConfig",
    "dpo_loss",
]