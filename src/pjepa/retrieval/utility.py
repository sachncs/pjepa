"""Retrieval utility functions.

A retrieval utility maps a subset of vertices to a non-negative
real score. Two utilities are provided:

* :class:`InfoGain` — the framework's headline utility, defined
  as conditional mutual information minus a per-vertex cost.
* :class:`Facility` — a provably-submodular fallback based on
  coverage of an observation feature.

Both implement the :class:`Utility` abstract base class and are
therefore interchangeable in :class:`pjepa.retrieval.Retrieval`.

The utilities operate on per-vertex feature matrices supplied at
construction time; passing the same features for every evaluation
keeps the per-call cost down. All functions are pure PyTorch with
no trainable parameters and no autograd state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

__all__ = [
    "Facility",
    "InfoGain",
    "Utility",
    "facility_location_weights",
    "uniform_weights",
]


class Utility(ABC):
    """Abstract base class for retrieval utilities.

    A utility scores a vertex subset against an observation. The
    class is callable: ``util(subset, observation)`` returns a
    non-negative scalar. The two concrete subclasses are
    :class:`InfoGain` (the framework's headline utility) and
    :class:`Facility` (a provably-submodular fallback).

    Subclasses must implement :meth:`__call__`. The wrapper
    :meth:`score` is the canonical entry point and simply
    forwards to :meth:`__call__` so that subclasses can override
    just one of the two if they prefer.
    """

    @abstractmethod
    def __call__(self, vertex_subset: torch.Tensor, observation: torch.Tensor) -> float:
        """Score the given vertex subset against the observation.

        Args:
            vertex_subset: A ``[k]`` long tensor of vertex indices.
            observation: A feature tensor (``[d]`` or ``[m, d]``).

        Returns:
            A non-negative scalar score.
        """
        ...

    def score(self, vertex_subset: torch.Tensor, observation: torch.Tensor) -> float:
        """Score the given vertex subset against the observation.

        Thin wrapper around :meth:`__call__` for callers that
        prefer the verb ``score`` at the call site.

        Args:
            vertex_subset: A ``[k]`` long tensor of vertex indices.
            observation: A feature tensor.

        Returns:
            A non-negative scalar score.
        """
        return self.__call__(vertex_subset, observation)


def uniform_weights(num_vertices: int) -> torch.Tensor:
    """Return uniform per-vertex weights as a 1-D float tensor.

    Args:
        num_vertices: Number of weights to produce.

    Returns:
        A ``[num_vertices]`` float tensor of ones.
    """
    return torch.ones((num_vertices,), dtype=torch.float32)


def facility_location_weights(
    vertex_features: torch.Tensor, observation: torch.Tensor
) -> torch.Tensor:
    """Compute facility-location-style per-vertex coverage weights.

    Each vertex's weight is the cosine similarity between its
    feature vector and the observation; this drives the
    facility-location utility towards vertices that resemble the
    observation.

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


class InfoGain(Utility):
    """Information-gain utility with explicit per-vertex cost.

    Implements::

        f(W) = sum_{v in W} I(v; O) - mu * |W|

    where ``I(v; O)`` is the per-vertex mutual information with
    the observation, approximated here by the cosine similarity
    between the vertex feature and the observation (after ``L2``
    normalisation). ``mu`` is a non-negative per-vertex cost
    that prevents the trivial ``W = V`` optimum and biases the
    selection towards small, descriptive subsets.

    The utility is **not** provably submodular; for workloads
    where that matters, fall back to :class:`Facility`.

    Attributes:
        vertex_features: The ``[N, d]`` reference feature matrix.
        mu: The per-vertex cost.
    """

    def __init__(
        self,
        vertex_features: torch.Tensor,
        mu: float = 0.05,
    ) -> None:
        """Initialise the utility with the persistent graph's vertex features.

        Args:
            vertex_features: A ``[N, d]`` tensor of vertex features.
                The reference is stored by reference; mutating it
                after construction changes future evaluations.
            mu: Per-vertex cost; must be non-negative.

        Raises:
            ValueError: If ``vertex_features`` is not 2-D or ``mu``
                is negative.
        """
        if vertex_features.ndim != 2:
            raise ValueError(
                f"InfoGain: vertex_features must be 2-D; got shape {tuple(vertex_features.shape)}"
            )
        if mu < 0:
            raise ValueError(f"InfoGain: mu must be non-negative; got {mu}")
        self.vertex_features = vertex_features
        self.mu = float(mu)

    def __call__(self, vertex_subset: torch.Tensor, observation: torch.Tensor) -> float:
        """Score the subset against the observation.

        Args:
            vertex_subset: A ``[k]`` long tensor of vertex indices.
                Indices outside ``[0, N)`` raise a PyTorch
                indexing error.
            observation: A ``[d]`` observation feature vector.

        Returns:
            ``sum_{v in W} cos(vf_v, observation) - mu * |W|``,
            floored at ``0.0`` so negative scores are reported as
            zero. Returns ``0.0`` for an empty subset.
        """
        if vertex_subset.numel() == 0:
            return 0.0
        selected_features = self.vertex_features[vertex_subset]
        weights = facility_location_weights(selected_features, observation)
        gain = float(weights.sum().item())
        cost = self.mu * float(vertex_subset.numel())
        return max(gain - cost, 0.0)


class Facility(Utility):
    """Facility-location utility; provably submodular.

    Implements::

        f(W) = sum_{i in observation} max_{v in W} similarity(v, i)

    where ``observation`` is treated as a set of feature vectors.
    The facility-location function is monotone submodular, so the
    greedy retrieval algorithm achieves the standard ``(1 - 1/e)``
    approximation. Negative similarities are clamped to zero in
    ``best_per_obs``; without that clamp floating-point noise could
    produce a small negative score on edgeless inputs.

    Attributes:
        vertex_features: The ``[N, d]`` reference feature matrix.
    """

    def __init__(self, vertex_features: torch.Tensor) -> None:
        """Initialise the utility with the vertex feature matrix.

        Args:
            vertex_features: A ``[N, d]`` tensor of vertex
                features.

        Raises:
            ValueError: If ``vertex_features`` is not 2-D.
        """
        if vertex_features.ndim != 2:
            raise ValueError(
                f"Facility: vertex_features must be 2-D; got shape {tuple(vertex_features.shape)}"
            )
        self.vertex_features = vertex_features

    def __call__(self, vertex_subset: torch.Tensor, observation: torch.Tensor) -> float:
        """Score the subset against the observation.

        Args:
            vertex_subset: A ``[k]`` long tensor of vertex indices.
            observation: A ``[m, d]`` tensor of observation
                features.

        Returns:
            The coverage score, i.e. the sum over observation
            features of the best-matching selected vertex
            similarity. Returns ``0.0`` for empty inputs.
        """
        if vertex_subset.numel() == 0 or observation.numel() == 0:
            return 0.0
        sub = self.vertex_features[vertex_subset]
        sub_n = torch.nn.functional.normalize(sub, dim=-1)
        obs_n = torch.nn.functional.normalize(observation, dim=-1)
        sims = obs_n @ sub_n.T
        # Clamp per-row max to non-negative so floating-point
        # rounding noise does not yield a small negative utility.
        best_per_obs = sims.max(dim=-1).values.clamp(min=0.0)
        return float(best_per_obs.sum().item())
