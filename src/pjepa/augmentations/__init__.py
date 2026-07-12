"""Graph augmentations for self-supervised learning.

This subpackage implements the standard graph augmentations used in
GraphCL, GCA, GraphMAE, and similar self-supervised graph methods.
Each augmentation is a callable ``Graph -> Graph`` that supports
reproducible randomness via a ``torch.Generator``.
"""

from __future__ import annotations

from pjepa.augmentations.base import Augmentation, AugmentationPipeline
from pjepa.augmentations.structural import DropEdge, DropNode, RandomWalkSubgraph
from pjepa.augmentations.feature import DropFeature, FeatureMask

__all__ = [
    "Augmentation",
    "AugmentationPipeline",
    "DropEdge",
    "DropNode",
    "RandomWalkSubgraph",
    "DropFeature",
    "FeatureMask",
]