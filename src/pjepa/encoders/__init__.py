"""Encoder protocols and implementations.

Encoders map :class:`TypedAttributedGraph` to per-vertex or graph-level
embedding tensors. The dual-geometric encoder produces both Euclidean
and hyperbolic components, which Proposition 3 justifies.
"""

from __future__ import annotations

from pjepa.encoders.base import Encoder, EncoderProtocol
from pjepa.encoders.euclidean_mpnn import EuclideanMPNN
from pjepa.encoders.hyperbolic import HyperbolicProjection
from pjepa.encoders.dual_geometric import DualGeometricEncoder
from pjepa.encoders.jepa_predictor import JEPAPredictor, TargetEncoder

__all__ = [
    "Encoder",
    "EncoderProtocol",
    "EuclideanMPNN",
    "HyperbolicProjection",
    "DualGeometricEncoder",
    "JEPAPredictor",
    "TargetEncoder",
]