"""Unified information-theoretic objectives.

This subpackage implements the four-term free-energy functional 𝒥
that governs every acceptance decision in the framework:

    𝒥(G) = E[ -log p(O | G) ]          (predictive fit)
         + β · D_KL( q(G) ‖ p(G) )      (IB complexity)
         + λ · DL(G)                    (MDL / description length)
         − γ · I(G; O_{>t})             (forward-information bonus)

The third term *encourages parsimony* and the fourth term *rewards
predictive value*; both can make ``𝒥`` smaller than its predictive
component alone. The implementation is intentionally explicit about
its terms so debugging and ablation are straightforward.
"""

from __future__ import annotations

from pjepa.objectives.free_energy import FreeEnergy
from pjepa.objectives.ib_lagrangian import ib_lagrangian, variational_ib_bound
from pjepa.objectives.mdl import description_length

__all__ = [
    "FreeEnergy",
    "description_length",
    "ib_lagrangian",
    "variational_ib_bound",
]
