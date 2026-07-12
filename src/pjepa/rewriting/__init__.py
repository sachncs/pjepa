"""Verified rewriting engine.

This subpackage implements the hyperedge-replacement grammar (HRG),
the bisimulation metric, the four-conditions acceptance criterion,
and the DPO rewriting loss.
"""

from __future__ import annotations

from pjepa.rewriting.bisimulation import BisimulationMetric, bisimulation_distance
from pjepa.rewriting.dpo import DPOConfig, dpo_loss
from pjepa.rewriting.four_conditions import FourConditions, accept_candidate
from pjepa.rewriting.hrg import HRG, HRGProduction

__all__ = [
    "HRG",
    "BisimulationMetric",
    "DPOConfig",
    "FourConditions",
    "HRGProduction",
    "accept_candidate",
    "bisimulation_distance",
    "dpo_loss",
]
