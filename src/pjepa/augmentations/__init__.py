"""Graph augmentations for self-supervised learning.

This subpackage implements the standard graph augmentations used in
GraphCL, GCA, GraphMAE, and similar self-supervised graph methods.
Each augmentation is a callable ``Graph -> Graph`` that supports
reproducible randomness via a :class:`torch.Generator`.

Composition is supported through :class:`AugmentationPipeline`, which
selects augmentations in one of three modes:

* ``SEQUENTIAL`` — apply every augmentation in order;
* ``RANDOM_SAMPLE_ONE`` — pick exactly one augmentation uniformly;
* ``RANDOM_SAMPLE_K`` — pick ``k`` augmentations without
  replacement and apply them in the sampled order.

Augmentations never mutate the input graph in place; each
``__call__`` returns a fresh :class:`TypedAttributedGraph`.
"""

from __future__ import annotations

from pjepa.augmentations.base import Augmentation, AugmentationPipeline, PipelineMode
from pjepa.augmentations.feature import DropFeature, FeatureMask
from pjepa.augmentations.identity import Identity
from pjepa.augmentations.registry import (
    augmentation_registry,
    available_augmentations,
    evict_augmentation,
    get_augmentation,
    register,
)
from pjepa.augmentations.structural import (
    ConnectedSubgraph,
    DropEdge,
    DropNode,
    Subgraph,
)
from pjepa.augmentations.tensor import TensorDropFeature, tensor_drop_feature

__all__ = [
    "Augmentation",
    "AugmentationPipeline",
    "ConnectedSubgraph",
    "DropEdge",
    "DropFeature",
    "DropNode",
    "FeatureMask",
    "Identity",
    "PipelineMode",
    "Subgraph",
    "TensorDropFeature",
    "augmentation_registry",
    "available_augmentations",
    "evict_augmentation",
    "get_augmentation",
    "register",
    "tensor_drop_feature",
]
