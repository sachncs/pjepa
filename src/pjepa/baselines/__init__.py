"""Baseline model implementations for SOTA comparison.

Each baseline re-implements a published self-supervised or
supervised graph-learning method so the comparison is
apples-to-apples. The implementations follow the original papers
closely and are deliberately minimal.

## Layout

* :mod:`pjepa.baselines.gcn` — Kipf & Welling GCN.
* :mod:`pjepa.baselines.gin` — Xu et al. GIN (with optional VN).
* :mod:`pjepa.baselines.graphsage` — Hamilton et al. GraphSAGE.
* :mod:`pjepa.baselines.naive` — mean-pool baseline.
* :mod:`pjepa.baselines.graphcl` — You et al. GraphCL
  (NT-Xent contrastive loss).
* :mod:`pjepa.baselines.graphmae` — Hou et al. GraphMAE
  (GIN encoder + MSE on masked features).
* :mod:`pjepa.baselines.infograph` — Sun et al. InfoGraph
  (mutual-information between node and graph embeddings).
* :mod:`pjepa.baselines.bgrl` — Thakoor et al. BGRL
  (online / target encoder pair with EMA).
* :mod:`pjepa.baselines.ewc` — Kirkpatrick et al. Elastic
  Weight Consolidation (CL regulariser).
* :mod:`pjepa.baselines.gem` — Lopez-Paz & Ranzato GEM
  (gradient projection onto memory samples).
* :mod:`pjepa.baselines.packnet` — Mallya & Lazebnik PackNet
  (per-task parameter slicing).
"""

from __future__ import annotations

from pjepa.baselines.bgrl import BGRL
from pjepa.baselines.ewc import EWC
from pjepa.baselines.gcn import GCN
from pjepa.baselines.gem import GEM, MemorySample
from pjepa.baselines.gin import GIN
from pjepa.baselines.graphcl import GraphCL
from pjepa.baselines.graphmae import GraphMAE
from pjepa.baselines.graphsage import GraphSAGE
from pjepa.baselines.infograph import InfoGraph
from pjepa.baselines.naive import Naive
from pjepa.baselines.packnet import PackNet

__all__ = [
    "BGRL",
    "EWC",
    "GCN",
    "GEM",
    "GIN",
    "GraphCL",
    "GraphMAE",
    "GraphSAGE",
    "InfoGraph",
    "MemorySample",
    "Naive",
    "PackNet",
]
