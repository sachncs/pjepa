"""Unified free-energy functional 𝒥.

The functional of the framework (paper §2.7) is the sum of four
additive terms:

    𝒥(G) = E[ -log p(O | G) ]
         + β_ib · D_KL( q(G) ‖ p(G) )
         + λ_mdl · DL(G)
         − γ_forward · forward(G, O)

This module provides a dataclass wrapper plus a callable evaluation
that operates on :class:`pjepa.graphs.TypedAttributedGraph`. The
implementation is intentionally explicit about its terms so
debugging and ablation are straightforward.

The numerical value of 𝒥 is **not guaranteed non-negative**: the
forward-information bonus subtracts a similarity term, so 𝒥 can be
below zero when the candidate aligns strongly with the observation.
What matters for the framework is the *sign of Δ𝒥*, not the absolute
level.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pjepa.exceptions import NumericalError
from pjepa.graphs import TypedAttributedGraph
from pjepa.objectives.ib_lagrangian import variational_ib_bound
from pjepa.objectives.mdl import description_length

__all__ = ["FreeEnergy"]


@dataclass(frozen=True)
class FreeEnergy:
    """The four-term unified free-energy functional.

    Attributes:
        beta_ib: Coefficient of the KL term.
        lambda_mdl: Coefficient of the description-length term.
        gamma_forward: Coefficient of the forward-information bonus.
    """

    beta_ib: float = 1e-2
    lambda_mdl: float = 1e-3
    gamma_forward: float = 1e-4

    def __call__(
        self,
        graph: TypedAttributedGraph,
        observation: torch.Tensor,
        posterior_logits: torch.Tensor | None = None,
        prior_logits: torch.Tensor | None = None,
    ) -> float:
        """Evaluate 𝒥 on the given graph and observation.

        Args:
            graph: The persistent or candidate graph.
            observation: The current observation tensor.
            posterior_logits: Optional encoder logits; when supplied
                with ``prior_logits`` they feed the KL term.
            prior_logits: Optional prior logits for the KL term.

        Returns:
            The signed scalar value of 𝒥. May be negative when the
            forward-information bonus dominates and may be ``inf``
            when the input graph has no vertices.

        Raises:
            NumericalError: When the resulting value is NaN. A
                negative or infinite value is *not* an error and is
                returned as-is.
        """
        if graph.num_vertices() == 0:
            return float("inf")

        # Term 1: predictive fit (negative log-likelihood proxy)
        if observation.numel() > 0:
            mean_feat = graph.vertex_features.mean(dim=0)
            nll = float(((mean_feat - observation.squeeze(0)) ** 2).mean().item())
        else:
            nll = 0.0

        # Term 2: KL term (IB complexity). Only contributes when the
        # caller supplied both posterior and prior logits.
        if posterior_logits is not None and prior_logits is not None:
            kl = variational_ib_bound(posterior_logits, prior_logits)
        else:
            kl = 0.0

        # Term 3: description length (MDL).
        dl = description_length(graph)

        # Term 4: forward-information bonus (rewarding alignment with
        # future observations). Subtracts from the functional so the
        # candidate graph *reduces J* when it aligns with the
        # observation.
        forward = 0.0
        if observation.numel() > 0:
            sim = float(
                torch.nn.functional.cosine_similarity(
                    graph.vertex_features.mean(dim=0, keepdim=True),
                    observation,
                    dim=-1,
                )
                .mean()
                .item()
            )
            forward = sim

        value = nll + self.beta_ib * kl + self.lambda_mdl * dl - self.gamma_forward * forward
        if value != value:  # NaN check that avoids importing math.
            raise NumericalError(
                f"FreeEnergy: computed NaN for graph with {graph.num_vertices()} vertices"
            )
        return value
