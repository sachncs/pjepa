"""Baseline model implementations for SOTA comparison.

Each baseline re-implements a published self-supervised or supervised
graph-learning method so the comparison is apples-to-apples. The
implementations follow the original papers closely and are
deliberately minimal.
"""

from __future__ import annotations

from pjepa.baselines.gcn import GCN
from pjepa.baselines.gin import GIN
from pjepa.baselines.graphmae import GraphMAE
from pjepa.baselines.graphcl import GraphCL
from pjepa.baselines.infograph import InfoGraph
from pjepa.baselines.ewc import EWC
from pjepa.baselines.gem import GEM

__all__ = [
    "GCN",
    "GIN",
    "GraphMAE",
    "GraphCL",
    "InfoGraph",
    "EWC",
    "GEM",
]