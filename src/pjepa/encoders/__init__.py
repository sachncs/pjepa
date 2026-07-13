"""Encoder protocols and implementations.

Encoders map a :class:`pjepa.graphs.TypedAttributedGraph` to a
per-vertex or graph-level embedding tensor. The dual-geometric
encoder produces both Euclidean and hyperbolic components, which
Proposition 3 of the paper justifies.

Implementations are :class:`torch.nn.Module` subclasses but the
:class:`Encoder` protocol only requires ``forward`` and ``to``;
downstream code should depend on the protocol rather than on
concrete subclasses.

Registry entry points:

* :func:`get_encoder` — look up an encoder class by name.
* :func:`available_encoders` — list every registered name.
* :func:`evict_encoder` — remove a registered name (testing utility).
* :func:`register` — class decorator for new implementations.
"""

from __future__ import annotations

from pjepa.encoders.base import Encoder, EncoderProtocol
from pjepa.encoders.dual_geometric import DualGeometricEncoder
from pjepa.encoders.euclidean_mpnn import EuclideanMPNN
from pjepa.encoders.hyperbolic import HyperbolicProjection
from pjepa.encoders.jepa_predictor import JEPAPredictor, TargetEncoder
from pjepa.encoders.registry import (
    available_encoders,
    encoder_registry,
    evict_encoder,
    get_encoder,
    register,
)

__all__ = [
    "DualGeometricEncoder",
    "Encoder",
    "EncoderProtocol",
    "EuclideanMPNN",
    "HyperbolicProjection",
    "JEPAPredictor",
    "TargetEncoder",
    "available_encoders",
    "encoder_registry",
    "evict_encoder",
    "get_encoder",
    "register",
]
