"""Unified information-theoretic objectives.

This subpackage implements the four-term free-energy functional 𝒥
that governs every acceptance decision in the framework.
"""

from __future__ import annotations

from pjepa.objectives.free_energy import FreeEnergy
from pjepa.objectives.ib_lagrangian import ib_lagrangian, variational_ib_bound
from pjepa.objectives.mdl import description_length

__all__ = [
    "FreeEnergy",
    "ib_lagrangian",
    "variational_ib_bound",
    "description_length",
]