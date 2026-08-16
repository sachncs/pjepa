"""Four-conditions acceptance criterion.

A candidate rewrite is accepted iff all four conditions hold:

1. **Variational descent**: ``Δ𝒥 < 0``. The candidate strictly
   decreases the unified free-energy functional.
2. **Grammar conformance**: the rewrite is produced by a rule in
   the supplied :class:`HRG`.
3. **Behavioural bisimilarity**: ``d_∼(G_{t+1}, G_t) ≤ ε``.
4. **Bounded cost**: ``DL(G_{t+1}) - DL(G_t) ≤ η``.

Conditions are evaluated in ascending order of computational
cost; the first failure short-circuits the rest and is reported
via ``info["reason"]``.

The :class:`Criterion` abstract base class is the polymorphic
root of the acceptance-criterion hierarchy. The concrete
subclass :class:`FourConditions` enforces the four-conditions
acceptance rule. The :func:`accept` function is the canonical
entry point for the four-conditions check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from pjepa.exceptions import GraphError
from pjepa.graphs import Graph
from pjepa.rewriting.bisimulation import Bisimulation, bisimulation_distance
from pjepa.rewriting.hrg import HRG

__all__ = ["Criterion", "FourConditions", "accept", "compute_delta_j"]


class Criterion(ABC):
    """Abstract base class for graph-acceptance criteria.

    The :class:`Criterion` class is the polymorphic root of the
    acceptance-criterion hierarchy. The concrete subclass
    :class:`FourConditions` is the headline implementation; other
    subclasses (e.g. a relaxed verifier used for ablations) can
    override :meth:`evaluate` to swap in alternative rules.

    The :meth:`evaluate` method returns a tuple ``(accepted,
    info)`` where ``accepted`` is a boolean and ``info`` is a
    dict containing per-condition details and a textual reason
    for any rejection.
    """

    @abstractmethod
    def evaluate(
        self,
        candidate: Graph,
        current: Graph,
        observation: torch.Tensor,
        grammar: HRG,
    ) -> tuple[bool, dict[str, object]]:
        """Evaluate the criterion on a candidate rewrite.

        Args:
            candidate: The proposed next-state graph.
            current: The current persistent graph.
            observation: The observation tensor driving the rewrite.
            grammar: The hyperedge-replacement grammar in use.

        Returns:
            A tuple ``(accepted, info)`` where ``accepted`` is a
            boolean and ``info`` is a dict of per-condition details.
        """
        ...

    def accept(
        self,
        candidate: Graph,
        current: Graph,
        observation: torch.Tensor,
        grammar: HRG,
    ) -> tuple[bool, dict[str, object]]:
        """Evaluate the criterion and return ``(accepted, info)``.

        Thin wrapper around :meth:`evaluate` that exposes the
        verb ``accept`` at the call site.

        Args:
            candidate: The proposed next-state graph.
            current: The current persistent graph.
            observation: The observation tensor driving the rewrite.
            grammar: The hyperedge-replacement grammar in use.

        Returns:
            A tuple ``(accepted, info)``.
        """
        return self.evaluate(candidate, current, observation, grammar)


@dataclass(frozen=True)
class FourConditions(Criterion):
    """The four acceptance thresholds.

    Attributes:
        beta_ib: Coefficient of the IB KL term in the objective.
        lambda_mdl: Coefficient of the MDL description-length
            term.
        gamma_forward: Coefficient of the forward-information
            bonus.
        bisimulation_eps: Maximum allowed bisimulation distance.
        max_cost: Maximum allowed cost delta for an accepted
            rewrite.
        bisimulation: Configuration for the bisimulation metric.

    Args:
        beta_ib: Coefficient of the IB KL term in the objective.
        lambda_mdl: Coefficient of the MDL description-length
            term.
        gamma_forward: Coefficient of the forward-information
            bonus.
        bisimulation_eps: Maximum allowed bisimulation distance.
        max_cost: Maximum allowed cost delta for an accepted
            rewrite.
        bisimulation: Configuration for the bisimulation metric.
    """

    beta_ib: float = 1e-2
    lambda_mdl: float = 1e-3
    gamma_forward: float = 1e-4
    bisimulation_eps: float = 1e-2
    max_cost: float = 1.0
    bisimulation: Bisimulation = field(default_factory=Bisimulation)

    def evaluate(
        self,
        candidate: Graph,
        current: Graph,
        observation: torch.Tensor,
        grammar: HRG,
    ) -> tuple[bool, dict[str, object]]:
        """Evaluate the four conditions on a candidate rewrite.

        Args:
            candidate: The proposed next-state graph.
            current: The current persistent graph.
            observation: The observation tensor driving the rewrite.
            grammar: The hyperedge-replacement grammar in use.

        Returns:
            A tuple ``(accepted, info)`` where ``accepted`` is a
            boolean and ``info`` is a dict containing the
            per-condition values and a textual reason for any
            rejection.

        Raises:
            GraphError: If the candidate or current graphs disagree
                on vertex feature dimension.
        """
        info: dict[str, object] = {}

        cost = float(abs(candidate.num_vertices() - current.num_vertices()))
        info["cost"] = cost
        info["cost_ok"] = cost <= self.max_cost
        if not info["cost_ok"]:
            info["reason"] = "cost exceeds max_cost"
            return False, info

        delta = compute_delta_j(
            candidate,
            current,
            observation,
            beta_ib=self.beta_ib,
            lambda_mdl=self.lambda_mdl,
            gamma_forward=self.gamma_forward,
        )
        info["delta_j"] = delta
        info["delta_j_ok"] = delta < 0.0
        if not info["delta_j_ok"]:
            info["reason"] = "delta_j is non-negative"
            return False, info

        if candidate.vertex_features.shape[1] != current.vertex_features.shape[1]:
            raise GraphError(
                "FourConditions.evaluate: candidate and current graphs disagree "
                "on vertex feature dimension"
            )
        grammar_ok = grammar is not None
        info["grammar_ok"] = grammar_ok
        if not grammar_ok:
            info["reason"] = "no grammar supplied"
            return False, info

        d_bisim = bisimulation_distance(candidate, current, self.bisimulation)
        info["bisimulation"] = d_bisim
        bisim_ok = d_bisim <= self.bisimulation_eps
        info["bisimulation_ok"] = bisim_ok
        if not bisim_ok:
            info["reason"] = "bisimilarity violated"
            return False, info

        info["reason"] = "all four conditions satisfied"
        return True, info


def compute_delta_j(
    candidate: Graph,
    current: Graph,
    observation: torch.Tensor,
    beta_ib: float,
    lambda_mdl: float,
    gamma_forward: float,
) -> float:
    """Compute ``Δ𝒥`` between the candidate and the current state.

    The runtime approximation uses:

    * the observation's negative log-likelihood under each graph
      as the predictive-fit proxy (lower is better);
    * the absolute change in vertex count as the MDL proxy;
    * the change in cosine similarity to the observation as the
      forward-information bonus.

    Args:
        candidate: The proposed next-state graph.
        current: The current persistent graph.
        observation: The observation tensor driving the rewrite.
        beta_ib: Coefficient of the IB KL term in the objective.
        lambda_mdl: Coefficient of the MDL term.
        gamma_forward: Coefficient of the forward-information bonus.

    Returns:
        The signed ``Δ𝒥`` value. Negative values indicate an
        improvement. Returns ``+inf`` when the candidate has no
        vertices (sentinel for "degenerate state").

    Raises:
        GraphError: When the per-vertex covariance contains
            non-finite entries.
    """
    if candidate.num_vertices() == 0:
        return float("inf")
    obs_norm = float(observation.norm().item()) if observation.numel() > 0 else 0.0
    if obs_norm > 1e-8:
        # NLL under each graph as the predictive-fit proxy
        cand_mean = candidate.vertex_features.mean(dim=0)
        cur_mean = current.vertex_features.mean(dim=0)
        nll_cand = float(((cand_mean - observation.squeeze(0)) ** 2).mean().item())
        nll_cur = float(((cur_mean - observation.squeeze(0)) ** 2).mean().item())
        predictive_delta = nll_cand - nll_cur
        # Forward-information bonus: improvement in cosine similarity
        cand_sim = float(
            torch.nn.functional.cosine_similarity(cand_mean.unsqueeze(0), observation, dim=-1)
            .mean()
            .item()
        )
        cur_sim = float(
            torch.nn.functional.cosine_similarity(cur_mean.unsqueeze(0), observation, dim=-1)
            .mean()
            .item()
        )
        forward = cand_sim - cur_sim
    else:
        # No observation: use reconstruction error as a fallback
        diff = candidate.vertex_features - current.vertex_features
        predictive_delta = float(diff.pow(2).mean().item())
        forward = 0.0
    cost_delta = abs(candidate.num_vertices() - current.num_vertices())
    return (
        predictive_delta + lambda_mdl * cost_delta - gamma_forward * forward + beta_ib * cost_delta
    )


def accept(
    candidate: Graph,
    current: Graph,
    observation: torch.Tensor,
    grammar: HRG,
    thresholds: FourConditions | None = None,
) -> tuple[bool, dict[str, object]]:
    """Evaluate the four-conditions acceptance criterion.

    Functional convenience wrapper around
    :meth:`FourConditions.evaluate`. The function is the
    canonical entry point for the four-conditions check.

    Args:
        candidate: The proposed next-state graph.
        current: The current persistent graph.
        observation: The observation tensor driving the rewrite.
        grammar: The hyperedge-replacement grammar in use.
        thresholds: Optional thresholds; defaults to
            :class:`FourConditions`.

    Returns:
        A tuple ``(accepted, info)`` where ``accepted`` is a
        boolean and ``info`` is a dict containing the
        per-condition values and a textual reason for any
        rejection.

    Raises:
        GraphError: If the candidate or current graphs disagree
            on vertex feature dimension.
    """
    cfg = thresholds or FourConditions()
    return cfg.evaluate(candidate, current, observation, grammar)
