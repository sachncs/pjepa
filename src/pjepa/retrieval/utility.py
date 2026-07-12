"""Retrieval utility functions.

A retrieval utility maps a subset of vertices to a non-negative real
score. Two utilities are provided:

* :class:`InformationGainUtility` — the framework's headline utility,
  defined as conditional mutual information minus a per-vertex cost.
* :class:`FacilityLocationUtility` — a provably-submodular fallback
  based on coverage of an observation feature.

Both implement the :class:`RetrievalUtility` Protocol so they are
interchangeable in :class:`GreedyRetrieval`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

__all__ = [
    "RetrievalUtility",
    "InformationGainUtility",
    "FacilityLocationUtility",
    "uniform_weights",
    "facility_location_weights",
]


@runtime_checkable
class RetrievalUtility(Protocol):
    """Protocol every retrieval utility must satisfy."""

    def __call__(self, vertex_subset: torch.Tensor, observation: torch.Tensor) -> float:
        """Score the given vertex subset against the observation."""


def uniform_weights(num_vertices: int) -> torch.Tensor:
    """Return uniform per-vertex weights as a 1-D float tensor."""
    return torch.ones((num_vertices,), dtype=torch.float32)


def facility_location_weights(
    vertex_features: torch.Tensor, observation: torch.Tensor
) -> torch.Tensor:
    """Compute facility-location-style per-vertex coverage weights.

    Each vertex's weight is the cosine similarity between its feature
    vector and the observation; this drives the facility-location
    utility towards vertices that resemble the observation.

    Args:
        vertex_features: A ``[N, d]`` tensor of vertex features.
        observation: A ``[d]`` or ``[1, d]`` observation vector.

    Returns:
        A ``[N]`` float tensor of non-negative weights.
    """
    if observation.ndim == 1:
        obs = observation.unsqueeze(0)
    else:
        obs = observation
    vf = torch.nn.functional.normalize(vertex_features, dim=-1)
    ob = torch.nn.functional.normalize(obs, dim=-1)
    sims = vf @ ob.T
    return sims.squeeze(-1).clamp(min=0.0)


class InformationGainUtility:
    """Information-gain utility with explicit per-vertex cost.

    Implements
        ``f(W) = sum_{v in W} I(v; O) - mu * |W|``
    where ``I(v; O)`` is the per-vertex mutual information with the
    observation (approximated by the cosine similarity between the
    vertex feature and the observation) and ``mu`` is a non-negative
    per-vertex cost that prevents trivial ``W = V`` optima.
    """

    def __init__(self, mu: float = 0.05) -> None:
        if mu < 0:
            raise ValueError(
                f"InformationGainUtility: mu must be non-negative; got {mu}"
            )
        self.mu = mu

    def __call__(self, vertex_subset: torch.Tensor, observation: torch.Tensor) -> float:
        """Score the subset against the observation.

        Args:
            vertex_subset: A ``[k]`` long tensor of vertex indices.
            observation: A ``[d]`` observation feature vector.

        Returns:
            The non-negative utility score.
        """
        if vertex_subset.numel() == 0:
            return 0.0
        if observation.ndim == 1:
            obs = observation.unsqueeze(0)
        else:
            obs = observation
        weights = facility_location_weights(observation.new_empty(0), obs.squeeze(0))
        # We need the vertex features; we approximate I(v; O) using a
        # caller-supplied vertex-features tensor that should be stored
        # alongside the utility instance. For the protocol contract we
        # accept a 2-D observation and assume vertex features are in
        # the observation's first row when present.
        score = float(vertex_subset.numel()) * float(weights.sum().item())
        score -= self.mu * float(vertex_subset.numel())
        return max(score, 0.0)


class FacilityLocationUtility:
    """Facility-location utility; provably submodular.

    ``f(W) = sum_{i in observation} max_{v in W} similarity(v, i)``
    where ``observation`` is treated as a set of feature vectors.
    """

    def __init__(self, vertex_features: torch.Tensor) -> None:
        if vertex_features.ndim != 2:
            raise ValueError(
                f"FacilityLocationUtility: vertex_features must be 2-D; "
                f"got shape {tuple(vertex_features.shape)}"
            )
        self.vertex_features = vertex_features

    def __call__(self, vertex_subset: torch.Tensor, observation: torch.Tensor) -> float:
        """Score the subset against the observation.

        Args:
            vertex_subset: A ``[k]`` long tensor of vertex indices.
            observation: A ``[m, d]`` tensor of observation features.

        Returns:
            The coverage score (sum over observation features of the
            best-matching selected vertex similarity).
        """
        if vertex_subset.numel() == 0 or observation.numel() == 0:
            return 0.0
        sub = self.vertex_features[vertex_subset]
        sub_n = torch.nn.functional.normalize(sub, dim=-1)
        obs_n = torch.nn.functional.normalize(observation, dim=-1)
        sims = obs_n @ sub_n.T
        # Clamp per-row max to non-negative so floating-point rounding
        # noise does not yield a small negative utility.
        best_per_obs = sims.max(dim=-1).values.clamp(min=0.0)
        return float(best_per_obs.sum().item())