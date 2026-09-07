"""Encoder base class, specialised implementations, and head modules.

Encoders map a :class:`pjepa.graphs.Graph` to a per-vertex or
graph-level embedding tensor. The dual-geometric encoder
(:class:`DualGeometric`) produces both Euclidean and hyperbolic
components, which Proposition 3 of the paper justifies.

Heads are the predictor and target modules used by the
self-supervised training loop. They share a polymorphic base
class (:class:`Head`).

Registry entry points:

* :func:`get_encoder` — look up an encoder class by name.
* :func:`available_encoders` — list every registered name.
* :func:`evict_encoder` — remove a registered name (testing
  utility).
* :func:`register` — class decorator for new implementations.
"""

from __future__ import annotations

from pjepa.encoders.base import Encoder, EncoderProtocol
from pjepa.encoders.dual_geometric import DualGeometric
from pjepa.encoders.euclidean_mpnn import Euclidean
from pjepa.encoders.hyperbolic import Hyperbolic
from pjepa.encoders.jepa_predictor import Head, Predictor, Target
from pjepa.encoders.registry import (
    available_encoders,
    encoder_registry,
    evict_encoder,
    get_encoder,
    register,
)

__all__ = [
    "DualGeometric",
    "Encoder",
    "EncoderProtocol",
    "Euclidean",
    "Head",
    "Hyperbolic",
    "Predictor",
    "Target",
    "available_encoders",
    "encoder_registry",
    "evict_encoder",
    "get_encoder",
    "register",
]
