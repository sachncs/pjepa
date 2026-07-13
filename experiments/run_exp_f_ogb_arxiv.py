"""OGB-arxiv benchmark runner (Phase 10, ``experiments/run_exp_f_ogb_arxiv.py``).

This module implements the Phase 10 experiment matrix for the OGB-arxiv
node-classification benchmark. Five methods are compared:

* ``GCN`` — fully-supervised two-layer GCN (Kipf & Welling, 2017).
* ``GraphSAGE`` — fully-supervised neighbour-sampling GraphSAGE
  (Hamilton et al., 2017).
* ``BGRL`` — self-supervised Bootstrap Graph Latents (Thakoor et al.,
  2022) followed by a linear probe trained on the labelled split.
* ``GraphMAE`` — self-supervised masked autoencoder (Hou et al., 2022)
  followed by a linear probe.
* ``PersistentJEPA`` — the framework's headline JEPA variant with a
  dual-geometric encoder, EMA target encoder, greedy retrieval, and a
  linear probe on top of the target-encoder embeddings.

The runner uses mini-batch neighbour sampling to keep the activation
memory bounded on the 169K-node / 1.1M-edge graph. The trainer calls
:meth:`OGBArxiv.assert_no_test_leakage` before any training step (the
contract that test labels never enter the trainer's label tensor) and
honours the user-supplied RSS cap via :func:`pjepa.perf.assert_rss_cap`.

Outputs (under ``output_dir``):

* ``ogb_results.csv`` — long-format per-(method, seed) rows.
* ``tables/ogb_summary.csv`` — per-method mean / std / n_seeds plus a
  paired bootstrap CI against the GCN reference.
* ``predictions/<method>_seed<n>.csv`` — optional per-test-node
  prediction artefacts for OGB submission. The ``node_id`` column
  carries the official OGB test indices rather than positional indices.

When ``OGBArxiv`` is built from a small synthetic graph (as the test
suite does) the end-to-end flow runs without network access, so the
smoke configuration can be exercised locally.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pjepa.baselines import BGRL, GCN, GraphMAE, GraphSAGE
from pjepa.data.ogb import (
    NeighborSample,
    OGBArxiv,
    load_ogb_arxiv,
    neighbor_sample,
)
from pjepa.encoders import DualGeometricEncoder, JEPAPredictor, TargetEncoder
from pjepa.eval import bonferroni_correction, paired_bootstrap_ci, wilcoxon_signed_rank
from pjepa.exceptions import ConfigError, DataError, GraphError
from pjepa.graphs import PersistentState, TypedAttributedGraph
from pjepa.logging_setup import LogFormat, configure_logging, get_logger
from pjepa.perf import assert_rss_cap, load_sharded_state_dict, shard_state_dict
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "OGB_METHODS",
    "OGBConfig",
    "OGBRunResult",
    "PersistentJEPAClassifier",
    "aggregate_ogb_results",
    "apply_rss_cap",
    "bootstrap_persistent_graph",
    "build_labeled_subgraph",
    "build_predictions_artifact",
    "default_smoke_config",
    "map_global_to_local",
    "mask_test_labels",
    "per_class_accuracy",
    "predict_graphmae_probe",
    "predict_method_dispatch",
    "predict_node_classifier",
    "predict_persistent_jepa",
    "predictions_artifact_path",
    "run_ogb_experiment",
    "safe_supervised_labels",
    "subgraph_features",
    "subgraph_labels",
    "train_bgrl_with_probe",
    "train_gcn_supervised",
    "train_graphmae_with_probe",
    "train_graphsage_supervised",
    "train_method_dispatch",
    "train_node_classifier",
    "train_persistent_jepa_with_probe",
]


OGB_METHODS: tuple[str, ...] = (
    "GCN",
    "GraphSAGE",
    "BGRL",
    "GraphMAE",
    "PersistentJEPA",
)


def per_class_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the mean per-class accuracy for two 1-D label tensors.

    The score is ``(1/C) Σ_c correct_c / total_c`` where ``C`` is the
    number of distinct target classes observed in ``targets``. This is
    the standard balanced-accuracy metric for the OGB-arxiv protocol.

    Args:
        predictions: ``[N]`` ``long`` tensor of predicted labels.
        targets: ``[N]`` ``long`` tensor of ground-truth labels.

    Returns:
        The mean per-class accuracy in [0, 1]. Returns ``0.0`` when
        ``predictions`` is empty.
    """
    if predictions.numel() == 0:
        return 0.0
    correct: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)
    preds_list = predictions.tolist()
    targets_list = targets.tolist()
    for p, t in zip(preds_list, targets_list, strict=True):
        total[int(t)] += 1
        if int(p) == int(t):
            correct[int(t)] += 1
    if not total:
        return 0.0
    return sum(correct[c] / total[c] for c in total) / len(total)


def safe_supervised_labels(graph: TypedAttributedGraph) -> torch.Tensor:
    """Return a defensive copy of the per-vertex labels.

    Zeroing the test vertices on the returned tensor (see
    :func:`mask_test_labels`) yields the *safe* training-label tensor
    accepted by :meth:`OGBArxiv.assert_no_test_leakage`: every test
    index holds the sentinel ``0`` so the trainer can never observe
    the true test label, mirroring the OGB-arxiv submission protocol.

    Args:
        graph: The graph carrying ``vertex_labels``.

    Returns:
        A clone of ``graph.vertex_labels``.

    Raises:
        GraphError: If ``graph.vertex_labels`` is ``None``.
    """
    if graph.vertex_labels is None:
        raise GraphError("safe_supervised_labels: graph has no vertex_labels")
    return graph.vertex_labels.clone()


def mask_test_labels(labels: torch.Tensor, test_indices: Sequence[int]) -> torch.Tensor:
    """Return ``labels`` with the test vertices replaced by the sentinel ``0``.

    Args:
        labels: The full per-vertex label tensor.
        test_indices: Indices of the test vertices to mask.

    Returns:
        A new ``labels.clone()`` with the test indices zeroed.
    """
    masked = labels.clone()
    if test_indices:
        masked[torch.tensor(list(test_indices), dtype=torch.long)] = 0
    return masked


def subgraph_features(
    graph: TypedAttributedGraph,
    sample: NeighborSample,
) -> torch.Tensor:
    """Gather the per-vertex features of the sampled subgraph.

    Args:
        graph: The parent graph holding the vertex features.
        sample: The neighbour-sampling result.

    Returns:
        A ``[S, d_v]`` tensor of the sampled vertices' features.
    """
    return graph.vertex_features[sample.node_ids]


def subgraph_labels(
    labels: torch.Tensor,
    sample: NeighborSample,
) -> torch.Tensor:
    """Gather the per-vertex labels of the sampled subgraph.

    Args:
        labels: The full ``[N]`` label tensor.
        sample: The neighbour-sampling result.

    Returns:
        A ``[S]`` ``long`` tensor of sampled labels (test labels are
        expected to already be the sentinel ``0``).
    """
    return labels[sample.node_ids]


def build_labeled_subgraph(
    graph: TypedAttributedGraph,
    sample: NeighborSample,
    labels: torch.Tensor,
) -> TypedAttributedGraph:
    """Materialise a :class:`TypedAttributedGraph` for ``sample``.

    The synthetic ``edge_features`` tensor is a zero column matching
    the edge count; downstream encoders that consume edge features will
    treat this as a fully uninformative edge signal.

    Args:
        graph: The parent graph (source of vertex features).
        sample: The neighbour-sampling result.
        labels: The safe label tensor (test indices zeroed).

    Returns:
        A populated :class:`TypedAttributedGraph`.
    """
    feats = subgraph_features(graph, sample)
    sampled_labels = subgraph_labels(labels, sample)
    edge_features = torch.zeros((sample.edge_index.shape[1], 1))
    return TypedAttributedGraph(
        vertex_features=feats,
        edge_index=sample.edge_index,
        edge_features=edge_features,
        vertex_labels=sampled_labels,
    )


def map_global_to_local(sample: NeighborSample, node_ids: torch.Tensor) -> torch.Tensor:
    """Re-index global node ids into the sampled subgraph's local ids.

    Args:
        sample: The neighbour-sampling result carrying the
          ``seed_to_local`` mapping.
        node_ids: ``[k]`` ``long`` tensor of global node ids.

    Returns:
        A ``[k]`` ``long`` tensor of local ids in ``[0, S)``.
    """
    return sample.seed_to_local[node_ids]


def train_node_classifier(
    model: nn.Module,
    dataset: OGBArxiv,
    config: OGBConfig,
    training_labels: torch.Tensor,
) -> nn.Module:
    """Train a node-classification model end-to-end via neighbour sampling.

    The trainer uses AdamW with the configured learning rate and weight
    decay, shuffled mini-batches of seed nodes, and a fixed ``num_hops``
    / ``num_neighbors`` fan-in for every epoch. The returned model is
    the trained version of ``model`` (mutated in place by the optimiser).

    Args:
        model: The model to train; mutated in place.
        dataset: The OGB-arxiv dataset.
        config: The experiment configuration.
        training_labels: The safe label tensor (test indices zeroed).

    Returns:
        ``model`` after the final ``optimizer.step``.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    train_idx = torch.tensor(dataset.train_indices, dtype=torch.long)
    graph = dataset.graph
    num_nodes = graph.num_vertices()

    for _epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(train_idx.shape[0])
        for start in range(0, perm.shape[0], config.batch_size):
            batch_idx = train_idx[perm[start : start + config.batch_size]]
            sample = neighbor_sample(
                graph.edge_index,
                batch_idx,
                num_hops=config.num_hops,
                num_neighbors=config.num_neighbors,
                num_total_nodes=num_nodes,
            )
            sub = build_labeled_subgraph(graph, sample, training_labels)
            sub_logits = model.node_logits(sub)
            local_targets = map_global_to_local(sample, batch_idx)
            target_labels = training_labels[batch_idx]
            loss = loss_fn(sub_logits[local_targets], target_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_node_classifier(
    model: nn.Module,
    dataset: OGBArxiv,
    indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Predict labels for ``indices`` using neighbour-sampled subgraphs.

    A 2-hop, full-fan-in neighbour sampler is used at inference time so
    every reachable neighbour participates in the prediction, which
    matches the GraphSAGE-style inference protocol of OGB-arxiv
    trainers.

    Args:
        model: The trained classifier.
        dataset: The OGB-arxiv dataset.
        indices: ``[N_q]`` ``long`` tensor of query indices.
        batch_size: Mini-batch size for the inference loop.

    Returns:
        A ``[N_q]`` ``long`` prediction tensor aligned with ``indices``.
    """
    model.eval()
    graph = dataset.graph
    num_nodes = graph.num_vertices()
    all_preds = torch.empty((indices.shape[0],), dtype=torch.long)
    for start in range(0, indices.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        sample = neighbor_sample(
            graph.edge_index,
            batch_idx,
            num_hops=2,
            num_neighbors=-1,
            num_total_nodes=num_nodes,
        )
        sub = TypedAttributedGraph(
            vertex_features=graph.vertex_features[sample.node_ids],
            edge_index=sample.edge_index,
            edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
        )
        sub_logits = model.node_logits(sub)
        local_idx = map_global_to_local(sample, batch_idx)
        all_preds[start : start + batch_idx.shape[0]] = sub_logits[local_idx].argmax(dim=-1)
    return all_preds


def train_gcn_supervised(
    dataset: OGBArxiv,
    config: OGBConfig,
    training_labels: torch.Tensor,
) -> GCN:
    """Train a two-layer GCN with neighbour sampling.

    Args:
        dataset: The OGB-arxiv dataset.
        config: The experiment configuration.
        training_labels: The safe label tensor (test indices zeroed).

    Returns:
        The trained :class:`GCN`.
    """
    model = GCN(
        input_dim=dataset.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=dataset.num_classes,
    )
    return train_node_classifier(model, dataset, config, training_labels)


def train_graphsage_supervised(
    dataset: OGBArxiv,
    config: OGBConfig,
    training_labels: torch.Tensor,
) -> GraphSAGE:
    """Train a GraphSAGE encoder with neighbour sampling.

    Args:
        dataset: The OGB-arxiv dataset.
        config: The experiment configuration.
        training_labels: The safe label tensor (test indices zeroed).

    Returns:
        The trained :class:`GraphSAGE`.
    """
    model = GraphSAGE(
        input_dim=dataset.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=dataset.num_classes,
    )
    return train_node_classifier(model, dataset, config, training_labels)


def train_bgrl_with_probe(
    dataset: OGBArxiv,
    config: OGBConfig,
    training_labels: torch.Tensor,
    seed: int = 0,
) -> BGRL:
    """Train BGRL via self-supervised pre-training then fit a linear probe.

    The pretraining loop uses two augmented views of every sampled
    subgraph and the BGRL cosine-similarity loss; the target encoder
    is updated as an EMA of the online encoder after every mini-batch.
    After pretraining, a linear probe (the model's classifier head) is
    fit on top of the frozen online encoder for the labelled split —
    fitting the probe is what the previous version of this function
    silently skipped, leaving the head at its random initialisation.

    Args:
        dataset: The OGB-arxiv dataset.
        config: The experiment configuration.
        training_labels: The safe label tensor (test indices zeroed).
        seed: Per-run seed; seeded into a ``torch.Generator`` for the
          pretraining view sampling.

    Returns:
        The trained :class:`BGRL` with its ``classifier`` head fit to
        the labelled split.
    """
    model = BGRL(
        input_dim=dataset.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=dataset.num_classes,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    graph = dataset.graph
    num_nodes = graph.num_vertices()
    train_idx = torch.tensor(dataset.train_indices, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed) * 7919)
    for _ in range(config.epochs):
        model.train()
        perm = torch.randperm(train_idx.shape[0], generator=generator)
        for start in range(0, perm.shape[0], config.batch_size):
            batch_idx = train_idx[perm[start : start + config.batch_size]]
            sample = neighbor_sample(
                graph.edge_index,
                batch_idx,
                num_hops=config.num_hops,
                num_neighbors=config.num_neighbors,
                num_total_nodes=num_nodes,
                generator=generator,
            )
            feats = subgraph_features(graph, sample)
            sub_edges = sample.edge_index
            view_a = TypedAttributedGraph(
                vertex_features=feats,
                edge_index=sub_edges,
                edge_features=torch.zeros((sub_edges.shape[1], 1)),
            )
            view_b = TypedAttributedGraph(
                vertex_features=feats + 0.01 * torch.randn_like(feats),
                edge_index=sub_edges,
                edge_features=torch.zeros((sub_edges.shape[1], 1)),
            )
            optimizer.zero_grad()
            loss = model.loss(view_a, view_b)
            loss.backward()
            optimizer.step()
            model.update_target()

    # Probe-fitting stage: encode every training vertex (with full-fan-in
    # 2-hop sampling) using the now-trained online encoder, then fit the
    # classifier head end-to-end on the labelled split. The encoder is
    # frozen so the probe fitting does not destabilise the BGRL features.
    if model.classifier is None:
        return model
    model.eval()
    sample = neighbor_sample(
        graph.edge_index,
        train_idx,
        num_hops=2,
        num_neighbors=-1,
        num_total_nodes=num_nodes,
    )
    sub_full = TypedAttributedGraph(
        vertex_features=graph.vertex_features[sample.node_ids],
        edge_index=sample.edge_index,
        edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
    )
    with torch.no_grad():
        embeddings = model.online_encoder.encode(sub_full)
    local_train = sample.seed_to_local[train_idx]
    h_train = embeddings[local_train].detach()
    y_train = training_labels[train_idx]
    probe_optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    probe_epochs = max(1, config.epochs // 2)
    for _ in range(probe_epochs):
        probe_optimizer.zero_grad()
        loss = loss_fn(model.classifier(h_train), y_train)
        loss.backward()
        probe_optimizer.step()
    return model


def train_graphmae_with_probe(
    dataset: OGBArxiv,
    config: OGBConfig,
    training_labels: torch.Tensor,
) -> tuple[GraphMAE, nn.Linear]:
    """Pretrain a GraphMAE encoder and fit a linear probe on top.

    The encoder is pretrained with the masked-MSE objective for
    ``config.epochs`` epochs. The encoder is then frozen and a linear
    classifier is fit on the labelled split for ``config.epochs // 2``
    epochs. The encoder and the probe are returned separately so the
    dispatcher can route them to the inference path.

    Args:
        dataset: The OGB-arxiv dataset.
        config: The experiment configuration.
        training_labels: The safe label tensor (test indices zeroed).

    Returns:
        A tuple ``(encoder, probe)`` where ``encoder`` is the
        pretrained, frozen :class:`GraphMAE` and ``probe`` is the
        freshly-fit ``nn.Linear`` classifier head.
    """
    encoder = GraphMAE(
        input_dim=dataset.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        mask_ratio=0.5,
    )
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    graph = dataset.graph
    num_nodes = graph.num_vertices()
    train_idx = torch.tensor(dataset.train_indices, dtype=torch.long)
    for _ in range(config.epochs):
        encoder.train()
        perm = torch.randperm(train_idx.shape[0])
        for start in range(0, perm.shape[0], config.batch_size):
            batch_idx = train_idx[perm[start : start + config.batch_size]]
            sample = neighbor_sample(
                graph.edge_index,
                batch_idx,
                num_hops=config.num_hops,
                num_neighbors=config.num_neighbors,
                num_total_nodes=num_nodes,
            )
            feats = subgraph_features(graph, sample)
            sub = TypedAttributedGraph(
                vertex_features=feats,
                edge_index=sample.edge_index,
                edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
            )
            out = encoder(sub)
            target = feats
            mask = out["mask"]
            if mask.any():
                loss = ((out["reconstruction"][mask] - target[mask]) ** 2).mean()
            else:
                loss = (out["reconstruction"] - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe = nn.Linear(config.hidden_dim, dataset.num_classes)
    encoder.eval()
    sample = neighbor_sample(
        graph.edge_index,
        train_idx,
        num_hops=2,
        num_neighbors=-1,
        num_total_nodes=num_nodes,
    )
    sub = TypedAttributedGraph(
        vertex_features=graph.vertex_features[sample.node_ids],
        edge_index=sample.edge_index,
        edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
    )
    with torch.no_grad():
        embeddings = encoder.encode(sub)
    local_train = sample.seed_to_local[train_idx]
    h_train = embeddings[local_train].detach()
    y_train = training_labels[train_idx]
    probe_optimizer = torch.optim.AdamW(
        probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(max(1, config.epochs // 2)):
        probe_optimizer.zero_grad()
        loss = loss_fn(probe(h_train), y_train)
        loss.backward()
        probe_optimizer.step()
    return encoder, probe


class PersistentJEPAClassifier(nn.Module):
    """Linear probe head sitting on top of the Persistent-JEPA target encoder.

    Attributes:
        classifier: The linear ``[hidden_dim, num_classes]`` head.
    """

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        """Initialise the linear probe head.

        Args:
            hidden_dim: Input feature dimension.
            num_classes: Output dimension (number of classes).
        """
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the linear head to ``h``.

        Args:
            h: ``[..., hidden_dim]`` feature tensor.

        Returns:
            ``[..., num_classes]`` logits tensor.
        """
        return self.classifier(h)


def bootstrap_persistent_graph(graph: TypedAttributedGraph) -> TypedAttributedGraph:
    """Build the initial persistent state from a few seed vertices.

    A fixed set of the first ``min(8, N)`` vertices is selected as
    the seed set; their induced subgraph is returned with no edges
    (the persistent graph is grown via :meth:`PersistentState.commit`
    as the trainer observes new vertices).

    Args:
        graph: The source graph to draw seed features from.

    Returns:
        A :class:`TypedAttributedGraph` with at most 8 vertices and
        no edges, sharing feature dimensionality with ``graph``.
    """
    n = min(8, graph.num_vertices())
    seed_ids = torch.arange(n, dtype=torch.long)
    sample = NeighborSample(
        node_ids=seed_ids,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        seed_to_local=-torch.ones((graph.num_vertices(),), dtype=torch.long),
        hop_depth=torch.zeros_like(seed_ids),
    )
    sample.seed_to_local[seed_ids] = torch.arange(n)
    feats = subgraph_features(graph, sample)
    return TypedAttributedGraph(
        vertex_features=feats,
        edge_index=sample.edge_index,
        edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
    )


def train_persistent_jepa_with_probe(
    dataset: OGBArxiv,
    config: OGBConfig,
    training_labels: torch.Tensor,
) -> tuple[TargetEncoder, PersistentJEPAClassifier]:
    """Train Persistent-JEPA on OGB-arxiv.

    The training loop performs BYOL-style EMA pretraining of a
    :class:`DualGeometricEncoder` against a target encoder, then fits
    a linear probe that consumes the target encoder's outputs on the
    neighbour-sampled subgraphs. The persistent graph is grown via
    :meth:`PersistentState.commit` as the trainer observes each
    mini-batch.

    Args:
        dataset: The OGB-arxiv dataset.
        config: The experiment configuration.
        training_labels: The safe label tensor (test indices zeroed).

    Returns:
        A tuple ``(target, probe)`` where ``target`` is the trained
        :class:`TargetEncoder` and ``probe`` is the linear probe.
    """
    encoder = DualGeometricEncoder(
        input_dim=dataset.feature_dim,
        euclidean_dim=config.hidden_dim,
        hyperbolic_dim=32,
        num_layers=config.num_layers,
    )
    predictor = JEPAPredictor(
        input_dim=config.hidden_dim,
        hidden_dim=max(64, config.hidden_dim * 2),
        output_dim=config.hidden_dim,
    )
    target = TargetEncoder(encoder, momentum=0.99)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    probe = PersistentJEPAClassifier(config.hidden_dim, dataset.num_classes)
    probe_optimizer = torch.optim.AdamW(
        probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    graph = dataset.graph
    num_nodes = graph.num_vertices()
    train_idx = torch.tensor(dataset.train_indices, dtype=torch.long)

    persistent = PersistentState(graph=bootstrap_persistent_graph(graph))

    for _epoch in range(config.epochs):
        encoder.train()
        predictor.train()
        probe.train()
        perm = torch.randperm(train_idx.shape[0])
        for start in range(0, perm.shape[0], config.batch_size):
            batch_idx = train_idx[perm[start : start + config.batch_size]]
            sample = neighbor_sample(
                graph.edge_index,
                batch_idx,
                num_hops=config.num_hops,
                num_neighbors=config.num_neighbors,
                num_total_nodes=num_nodes,
            )
            feats = subgraph_features(graph, sample)
            sub = TypedAttributedGraph(
                vertex_features=feats,
                edge_index=sample.edge_index,
                edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
            )
            # JEPA pretraining step: encoder receives gradients via the
            # predictor loss; the target encoder is updated as an EMA
            # *after* the optimiser step.
            e, _ = encoder(sub)
            e_pooled = e.mean(dim=0, keepdim=True)
            predicted = predictor(e_pooled)
            with torch.no_grad():
                target_emb, _ = target.shadow(sub)
                target_pooled = target_emb.mean(dim=0, keepdim=True)
            jepa_loss = nn.functional.smooth_l1_loss(predicted, target_pooled.detach())
            optimizer.zero_grad()
            jepa_loss.backward()
            optimizer.step()
            target.update()

            # Probe step on the (now-updated) target encoder.
            local_targets = map_global_to_local(sample, batch_idx)
            with torch.no_grad():
                target_emb, _ = target.shadow(sub)
            logits = probe(target_emb[local_targets])
            target_labels = training_labels[batch_idx]
            probe_loss = loss_fn(logits, target_labels)
            probe_optimizer.zero_grad()
            probe_loss.backward()
            probe_optimizer.step()

            # Commit the candidate working graph to the persistent state.
            candidate = TypedAttributedGraph(
                vertex_features=feats.detach(),
                edge_index=sample.edge_index,
                edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
            )
            if candidate.num_vertices() > 0:
                persistent = persistent.commit(
                    candidate=candidate,
                    cost=0.0,
                    timestamp=float(time.time()),
                    delta_j=-1e-3,
                )

    encoder.eval()
    target.shadow.eval()
    probe.eval()
    return target, probe


@torch.no_grad()
def predict_persistent_jepa(
    target: TargetEncoder,
    probe: PersistentJEPAClassifier,
    dataset: OGBArxiv,
    indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Run a prediction through the Persistent-JEPA target + probe.

    A 2-hop, full-fan-in neighbour sampler is used at inference time so
    every reachable neighbour participates in the prediction.

    Args:
        target: The trained :class:`TargetEncoder`.
        probe: The trained :class:`PersistentJEPAClassifier`.
        dataset: The OGB-arxiv dataset.
        indices: ``[N_q]`` ``long`` tensor of query indices.
        batch_size: Mini-batch size for the inference loop.

    Returns:
        A ``[N_q]`` ``long`` prediction tensor aligned with ``indices``.
    """
    target.shadow.eval()
    probe.eval()
    graph = dataset.graph
    num_nodes = graph.num_vertices()
    all_preds = torch.empty((indices.shape[0],), dtype=torch.long)
    for start in range(0, indices.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        sample = neighbor_sample(
            graph.edge_index,
            batch_idx,
            num_hops=2,
            num_neighbors=-1,
            num_total_nodes=num_nodes,
        )
        sub = TypedAttributedGraph(
            vertex_features=graph.vertex_features[sample.node_ids],
            edge_index=sample.edge_index,
            edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
        )
        emb, _ = target.shadow(sub)
        local_idx = map_global_to_local(sample, batch_idx)
        logits = probe(emb[local_idx])
        all_preds[start : start + batch_idx.shape[0]] = logits.argmax(dim=-1)
    return all_preds


@torch.no_grad()
def predict_graphmae_probe(
    encoder: GraphMAE,
    probe: nn.Linear,
    dataset: OGBArxiv,
    indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Predict labels via the GraphMAE encoder + linear probe.

    Args:
        encoder: The trained (frozen) :class:`GraphMAE`.
        probe: The trained linear probe.
        dataset: The OGB-arxiv dataset.
        indices: ``[N_q]`` ``long`` tensor of query indices.
        batch_size: Mini-batch size for the inference loop.

    Returns:
        A ``[N_q]`` ``long`` prediction tensor aligned with ``indices``.
    """
    encoder.eval()
    probe.eval()
    graph = dataset.graph
    num_nodes = graph.num_vertices()
    all_preds = torch.empty((indices.shape[0],), dtype=torch.long)
    for start in range(0, indices.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        sample = neighbor_sample(
            graph.edge_index,
            batch_idx,
            num_hops=2,
            num_neighbors=-1,
            num_total_nodes=num_nodes,
        )
        sub = TypedAttributedGraph(
            vertex_features=graph.vertex_features[sample.node_ids],
            edge_index=sample.edge_index,
            edge_features=torch.zeros((sample.edge_index.shape[1], 1)),
        )
        emb = encoder.encode(sub)
        local_idx = map_global_to_local(sample, batch_idx)
        logits = probe(emb[local_idx])
        all_preds[start : start + batch_idx.shape[0]] = logits.argmax(dim=-1)
    return all_preds


def train_method_dispatch(
    method: str,
    dataset: OGBArxiv,
    config: OGBConfig,
    training_labels: torch.Tensor,
    seed: int = 0,
) -> tuple[Any, Any]:
    """Train ``method`` on ``dataset`` and return ``(encoder, head)``.

    ``encoder`` is the trained backbone; ``head`` is either the
    classifier attached to ``encoder`` (GCN/GraphSAGE/BGRL) or the
    linear probe (GraphMAE/Persistent-JEPA).

    Args:
        method: One of :data:`OGB_METHODS`.
        dataset: The OGB-arxiv dataset.
        config: The experiment configuration.
        training_labels: The safe label tensor (test indices zeroed).
        seed: Per-run seed for the BGRL pretraining generator.

    Returns:
        A tuple ``(encoder, head)``.

    Raises:
        ConfigError: When ``method`` is not in :data:`OGB_METHODS`.
    """
    if method == "GCN":
        return train_gcn_supervised(dataset, config, training_labels), None
    if method == "GraphSAGE":
        return train_graphsage_supervised(dataset, config, training_labels), None
    if method == "BGRL":
        return train_bgrl_with_probe(dataset, config, training_labels, seed=seed), None
    if method == "GraphMAE":
        encoder, probe = train_graphmae_with_probe(dataset, config, training_labels)
        return encoder, probe
    if method == "PersistentJEPA":
        target, probe = train_persistent_jepa_with_probe(dataset, config, training_labels)
        return target, probe
    raise ConfigError(f"train_method_dispatch: unknown method {method!r}")


def predict_method_dispatch(
    method: str,
    encoder: Any,
    head: Any,
    dataset: OGBArxiv,
    indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Run inference for ``method`` on ``indices``.

    Args:
        method: One of :data:`OGB_METHODS`.
        encoder: The trained encoder (or target encoder) from
          :func:`train_method_dispatch`.
        head: The trained head (or probe).
        dataset: The OGB-arxiv dataset.
        indices: ``[N_q]`` ``long`` tensor of query indices.
        batch_size: Mini-batch size for the inference loop.

    Returns:
        A ``[N_q]`` ``long`` prediction tensor aligned with ``indices``.

    Raises:
        ConfigError: When ``method`` is not in :data:`OGB_METHODS`.
    """
    if method in ("GCN", "GraphSAGE", "BGRL"):
        return predict_node_classifier(encoder, dataset, indices, batch_size)
    if method == "GraphMAE":
        return predict_graphmae_probe(encoder, head, dataset, indices, batch_size)
    if method == "PersistentJEPA":
        return predict_persistent_jepa(encoder, head, dataset, indices, batch_size)
    raise ConfigError(f"predict_method_dispatch: unknown method {method!r}")


def apply_rss_cap(config: OGBConfig) -> None:
    """Apply the RSS cap from ``config`` if configured.

    When ``config.rss_cap_mb`` is positive, :func:`assert_rss_cap`
    raises :class:`BackendError` if the current RSS exceeds the cap;
    when the cap is non-positive the check is disabled.

    Args:
        config: The experiment configuration.

    Raises:
        pjepa.exceptions.BackendError: When the RSS exceeds the cap.
    """
    if config.rss_cap_mb <= 0:
        return
    assert_rss_cap(config.rss_cap_mb)


def predictions_artifact_path(output_dir: Path, method: str, seed: int) -> Path:
    """Return the conventional path for a per-method prediction CSV.

    Args:
        output_dir: Root output directory.
        method: Method name used for the file name.
        seed: Seed index used for the file name.

    Returns:
        The path ``<output_dir>/predictions/<safe-method>_seed<n>.csv``.
    """
    safe = method.replace("/", "_").replace(" ", "_")
    return output_dir / "predictions" / f"{safe}_seed{seed}.csv"


def default_smoke_config(config: OGBConfig) -> OGBConfig:
    """Return a fast variant of ``config`` for smoke testing.

    The smoke configuration uses a single seed, single epoch, single
    hidden layer, and a small batch size so the entire experiment
    matrix completes in well under a second on a synthetic dataset.

    Args:
        config: The base configuration; ``rss_cap_mb``, ``shard_bytes``,
          ``emit_predictions``, ``output_dir`` are inherited.

    Returns:
        A smoke-tuned :class:`OGBConfig`.
    """
    return OGBConfig(
        methods=OGB_METHODS,
        n_seeds=1,
        epochs=1,
        hidden_dim=8,
        num_layers=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=4,
        num_neighbors=2,
        num_hops=1,
        rss_cap_mb=config.rss_cap_mb,
        shard_bytes=config.shard_bytes,
        emit_predictions=config.emit_predictions,
        output_dir=config.output_dir,
        smoke=True,
    )


def build_predictions_artifact(
    predictions: torch.Tensor,
    output_dir: str | Path,
    method: str,
    seed: int,
    node_ids: Sequence[int] | None = None,
) -> Path:
    """Write a per-test-node prediction CSV in the OGB submission format.

    The CSV has two columns:

    * ``node_id`` — the official OGB test index when ``node_ids`` is
      supplied, otherwise the positional index in ``[0, len(predictions))``.
      Supplying ``node_ids`` is the path used by :func:`run_ogb_experiment`
      so the produced artefact is directly uploadable to the OGB
      leaderboard.
    * ``prediction`` — the predicted class id.

    Args:
        predictions: ``[N_test]`` ``long`` prediction tensor.
        output_dir: Root output directory.
        method: Method name used for the file name.
        seed: Seed index used for the file name.
        node_ids: Optional ``Sequence`` of length ``len(predictions)``
          carrying the official OGB test indices.

    Returns:
        The path of the written CSV.

    Raises:
        OSError: If the file cannot be written.
        ValueError: When ``node_ids`` is supplied but its length does
          not match ``len(predictions)``.
    """
    path = predictions_artifact_path(Path(output_dir), method, seed)
    if node_ids is not None and len(node_ids) != int(predictions.numel()):
        raise ValueError(
            f"build_predictions_artifact: node_ids length {len(node_ids)} does not match "
            f"prediction count {int(predictions.numel())}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node_id", "prediction"])
        for i, pred in enumerate(predictions.tolist()):
            writer.writerow([int(node_ids[i]) if node_ids is not None else int(i), int(pred)])
    return path


@dataclass(frozen=True)
class OGBConfig:
    """Configuration for the OGB-arxiv experiment.

    Attributes:
        methods: Methods to evaluate.
        n_seeds: Number of seeds per method.
        epochs: Number of supervised training epochs.
        hidden_dim: Encoder width.
        num_layers: Number of message-passing layers.
        learning_rate: Optimiser learning rate.
        weight_decay: Optimiser weight decay.
        batch_size: Per-epoch batch size (mini-batches of seed nodes).
        num_neighbors: Neighbour fan-in per hop.
        num_hops: Number of message-passing hops.
        rss_cap_mb: Practical RSS ceiling for the run. ``0`` (the
          default) disables the check; the Phase 10 plan recommends
          ``6144`` MiB for the real OGB-arxiv run on the target M3 Pro.
        shard_bytes: Maximum shard size for :func:`shard_state_dict`.
        emit_predictions: When ``True``, write per-test-node
          prediction CSVs under ``<output_dir>/predictions``. The
          ``node_id`` column carries the official OGB test indices.
        output_dir: Root directory for CSV / table outputs.
        smoke: When ``True``, the experiment runs a fast smoke
          configuration (single seed, single epoch, all five methods).
    """

    methods: tuple[str, ...] = OGB_METHODS
    n_seeds: int = 3
    epochs: int = 100
    hidden_dim: int = 256
    num_layers: int = 3
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    batch_size: int = 1024
    num_neighbors: int = 15
    num_hops: int = 2
    rss_cap_mb: float = 0.0
    shard_bytes: int = 100 * 1024 * 1024
    emit_predictions: bool = False
    output_dir: str = "results/ogb"
    smoke: bool = False


@dataclass(frozen=True)
class OGBRunResult:
    """Per-(method, seed) result row.

    Attributes:
        method: The method name.
        seed: The seed index.
        val_acc: Per-class accuracy on the validation split.
        test_acc: Per-class accuracy on the test split.
        elapsed_seconds: Wall-clock seconds for the training step.
        n_train_nodes: Number of training nodes visited.
        n_val_nodes: Number of validation nodes evaluated.
        n_test_nodes: Number of test nodes evaluated.
        test_predictions: Optional ``[N_test]`` ``long`` prediction
          tensor, only populated when
          :attr:`OGBConfig.emit_predictions` is ``True``.
    """

    method: str
    seed: int
    val_acc: float
    test_acc: float
    elapsed_seconds: float
    n_train_nodes: int
    n_val_nodes: int
    n_test_nodes: int
    test_predictions: torch.Tensor | None = None

    def as_row(self) -> dict[str, object]:
        """Return a CSV-friendly representation of this result."""
        return {
            "method": self.method,
            "seed": self.seed,
            "val_acc": self.val_acc,
            "test_acc": self.test_acc,
            "elapsed_seconds": self.elapsed_seconds,
            "n_train_nodes": self.n_train_nodes,
            "n_val_nodes": self.n_val_nodes,
            "n_test_nodes": self.n_test_nodes,
        }


def write_long_results_csv(results: list[OGBRunResult], path: Path) -> None:
    """Write the long-format ``ogb_results.csv``.

    Args:
        results: The per-(method, seed) result rows.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "seed",
                "val_acc",
                "test_acc",
                "elapsed_seconds",
                "n_train_nodes",
                "n_val_nodes",
                "n_test_nodes",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_row())


def aggregate_ogb_results(
    results: list[OGBRunResult],
    reference: str = "GCN",
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, object]]:
    """Aggregate per-(method, seed) results into a per-method summary.

    The bootstrap CI and the Wilcoxon signed-rank test are paired *by
    seed*: for every method ``M ≠ reference`` the test compares the
    ``M`` accuracy on seed ``s`` against the ``reference`` accuracy on
    the same seed ``s``. Seeds present in only one of the two
    methods are dropped, so the comparison is always seed-matched and
    the CI / p-values are not inflated by unpaired replications.

    The reference method is included in the summary with a degenerate
    CI (``ci_low = ci_high = 0``) and a Wilcoxon p-value of ``1.0``,
    matching the convention that "self vs self" is the null hypothesis.
    Bonferroni-adjusted p-values are computed across every method.

    Args:
        results: Per-(method, seed) result rows.
        reference: The method used as the comparison baseline. Defaults
          to ``"GCN"``.
        n_resamples: Bootstrap resample count.
        seed: Random seed for the bootstrap resampler.

    Returns:
        A dict keyed by method name with the per-method summary.
    """
    grouped: dict[str, list[OGBRunResult]] = defaultdict(list)
    for r in results:
        grouped[r.method].append(r)
    summary: dict[str, dict[str, object]] = {}
    reference_accs = [r.test_acc for r in grouped.get(reference, [])]
    reference_by_seed = {r.seed: r.test_acc for r in grouped.get(reference, [])}
    for method, rows in grouped.items():
        test_accs = [r.test_acc for r in rows]
        n = len(test_accs)
        if n == 0:
            continue
        mean = sum(test_accs) / n
        std = (sum((a - mean) ** 2 for a in test_accs) / max(n - 1, 1)) ** 0.5
        if method == reference or not reference_accs:
            ci = paired_bootstrap_ci(test_accs, test_accs, n_resamples=n_resamples, seed=seed)
            wilcoxon_p = 1.0
        else:
            # Pair by seed: only seeds present in both methods contribute
            # to the bootstrap CI / Wilcoxon test.
            rows_by_seed = {r.seed: r.test_acc for r in rows}
            common_seeds = sorted(set(reference_by_seed) & set(rows_by_seed))
            test_subset = [rows_by_seed[s] for s in common_seeds]
            ref_subset = [reference_by_seed[s] for s in common_seeds]
            if len(common_seeds) >= 1:
                ci = paired_bootstrap_ci(
                    test_subset, ref_subset, n_resamples=n_resamples, seed=seed
                )
                wilcoxon_p = wilcoxon_signed_rank(test_subset, ref_subset)
            else:
                ci = paired_bootstrap_ci(test_accs, test_accs, n_resamples=n_resamples, seed=seed)
                wilcoxon_p = 1.0
        summary[method] = {
            "method": method,
            "n_seeds": n,
            "mean_test_acc": mean,
            "std_test_acc": std,
            "mean_val_acc": sum(r.val_acc for r in rows) / n,
            "mean_elapsed_seconds": sum(r.elapsed_seconds for r in rows) / n,
            "bootstrap_ci_low": ci.ci_low,
            "bootstrap_ci_high": ci.ci_high,
            "bootstrap_p_value": ci.p_value,
            "wilcoxon_p": wilcoxon_p,
        }
    raw_ps = [float(summary[m]["wilcoxon_p"]) for m in summary]
    corrected = bonferroni_correction(raw_ps)
    for m, p_corr in zip(summary, corrected, strict=True):
        summary[m]["wilcoxon_p_bonferroni"] = p_corr
    return summary


def write_summary_table_csv(summary: dict[str, dict[str, object]], path: Path) -> None:
    """Write the headline ``tables/ogb_summary.csv``.

    Args:
        summary: The per-method summary from :func:`aggregate_ogb_results`.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "method",
                "n_seeds",
                "mean_test_acc",
                "std_test_acc",
                "mean_val_acc",
                "mean_elapsed_seconds",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "bootstrap_p_value",
                "wilcoxon_p",
                "wilcoxon_p_bonferroni",
            ]
        )
        for entry in summary.values():
            writer.writerow(
                [
                    entry["method"],
                    int(entry["n_seeds"]),
                    f"{float(entry['mean_test_acc']):.4f}",
                    f"{float(entry['std_test_acc']):.4f}",
                    f"{float(entry['mean_val_acc']):.4f}",
                    f"{float(entry['mean_elapsed_seconds']):.4f}",
                    f"{float(entry['bootstrap_ci_low']):.4f}",
                    f"{float(entry['bootstrap_ci_high']):.4f}",
                    f"{float(entry['bootstrap_p_value']):.4g}",
                    f"{float(entry['wilcoxon_p']):.4g}",
                    f"{float(entry['wilcoxon_p_bonferroni']):.4g}",
                ]
            )


def write_per_method_csv(results: list[OGBRunResult], tables_dir: Path) -> None:
    """Write one CSV per method under ``tables/ogb_<method>.csv``.

    Args:
        results: Per-(method, seed) result rows.
        tables_dir: Directory where the per-method CSVs are written.
    """
    grouped: dict[str, list[OGBRunResult]] = defaultdict(list)
    for r in results:
        grouped[r.method].append(r)
    for method, rows in grouped.items():
        safe = method.replace("/", "_").replace(" ", "_")
        path = tables_dir / f"ogb_{safe}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["seed", "val_acc", "test_acc", "elapsed_seconds"])
            for r in sorted(rows, key=lambda r: r.seed):
                writer.writerow(
                    [
                        int(r.seed),
                        f"{r.val_acc:.4f}",
                        f"{r.test_acc:.4f}",
                        f"{r.elapsed_seconds:.4f}",
                    ]
                )


def run_ogb_experiment(
    config: OGBConfig,
    output_dir: str | None = None,
    dataset: OGBArxiv | None = None,
) -> list[OGBRunResult]:
    """Run the OGB-arxiv experiment across the configured methods.

    The function:

    1. Loads (or accepts a pre-loaded) :class:`OGBArxiv`.
    2. Calls :meth:`OGBArxiv.assert_no_test_leakage` with the safe
       label tensor (test indices zeroed).
    3. Iterates over ``(seed, method)``, training + evaluating each.
    4. Optionally writes the per-test-node prediction artefact using
       the official OGB test indices.
    5. Writes the long CSV, the summary CSV, and the per-method CSVs.

    Args:
        config: The experiment configuration.
        output_dir: Optional override for the output directory.
        dataset: Optional pre-loaded :class:`OGBArxiv`. When ``None``,
          :func:`load_ogb_arxiv` is invoked and may raise
          :class:`DataError` if the ``ogb`` package is unavailable.

    Returns:
        A list of :class:`OGBRunResult` rows, one per ``(method, seed)``.
    """
    log = get_logger(__name__)
    effective = default_smoke_config(config) if config.smoke else config
    root = Path(output_dir if output_dir is not None else effective.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    apply_rss_cap(effective)
    if dataset is None:
        try:
            dataset = load_ogb_arxiv()
        except DataError as exc:
            raise ConfigError(f"run_ogb_experiment: failed to load OGB-arxiv: {exc}") from exc

    dataset.assert_no_test_leakage(
        training_labels=mask_test_labels(
            safe_supervised_labels(dataset.graph), dataset.test_indices
        ),
    )

    val_idx = torch.tensor(dataset.val_indices, dtype=torch.long)
    test_idx = torch.tensor(dataset.test_indices, dtype=torch.long)
    training_labels = mask_test_labels(safe_supervised_labels(dataset.graph), dataset.test_indices)
    val_targets = (
        dataset.graph.vertex_labels[val_idx] if dataset.graph.vertex_labels is not None else None
    )
    test_targets = (
        dataset.graph.vertex_labels[test_idx] if dataset.graph.vertex_labels is not None else None
    )
    if val_targets is None or test_targets is None:
        raise ConfigError("run_ogb_experiment: dataset is missing vertex_labels")

    results: list[OGBRunResult] = []
    for seed in range(effective.n_seeds):
        set_global_seed(seed * 1009 + 7)
        for method in effective.methods:
            log.info(
                "training method",
                extra={
                    "event": "ogb.train.start",
                    "method": method,
                    "seed": seed,
                },
            )
            start = time.time()
            encoder, head = train_method_dispatch(
                method, dataset, effective, training_labels, seed=seed
            )
            elapsed = time.time() - start

            val_preds = predict_method_dispatch(
                method, encoder, head, dataset, val_idx, effective.batch_size
            )
            test_preds = predict_method_dispatch(
                method, encoder, head, dataset, test_idx, effective.batch_size
            )
            val_acc = per_class_accuracy(val_preds, val_targets)
            test_acc = per_class_accuracy(test_preds, test_targets)
            predictions_tensor = test_preds.detach().clone() if effective.emit_predictions else None
            if effective.emit_predictions and predictions_tensor is not None:
                build_predictions_artifact(
                    predictions=predictions_tensor,
                    output_dir=root,
                    method=method,
                    seed=seed,
                    node_ids=dataset.test_indices,
                )

            if effective.shard_bytes > 0:
                shardable: dict[str, torch.Tensor] = {}
                if isinstance(encoder, nn.Module):
                    shardable.update({f"encoder.{k}": v for k, v in encoder.state_dict().items()})
                elif isinstance(encoder, TargetEncoder):
                    shardable.update(
                        {f"target.{k}": v for k, v in encoder.shadow.state_dict().items()}
                    )
                if isinstance(head, nn.Module):
                    shardable.update({f"head.{k}": v for k, v in head.state_dict().items()})
                if shardable:
                    shard_path = root / "shards" / method / f"seed{seed}"
                    state = {k: v.detach().cpu() for k, v in shardable.items()}
                    shard_state_dict(state, shard_path, max_shard_bytes=effective.shard_bytes)
                    load_sharded_state_dict(shard_path, map_location="cpu")

            results.append(
                OGBRunResult(
                    method=method,
                    seed=seed,
                    val_acc=val_acc,
                    test_acc=test_acc,
                    elapsed_seconds=elapsed,
                    n_train_nodes=len(dataset.train_indices),
                    n_val_nodes=val_idx.shape[0],
                    n_test_nodes=test_idx.shape[0],
                    test_predictions=predictions_tensor,
                )
            )
            log.info(
                "method trained",
                extra={
                    "event": "ogb.train.complete",
                    "method": method,
                    "seed": seed,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "elapsed_seconds": elapsed,
                },
            )

    write_long_results_csv(results, root / "ogb_results.csv")
    summary = aggregate_ogb_results(results)
    write_summary_table_csv(summary, tables_dir / "ogb_summary.csv")
    write_per_method_csv(results, tables_dir)
    return results


def main() -> int:
    """CLI entry point for the OGB-arxiv experiment.

    Returns:
        ``0`` on a successful run.
    """
    parser = argparse.ArgumentParser(description="Run the OGB-arxiv experiment.")
    parser.add_argument("--methods", nargs="*", default=list(OGBConfig.methods))
    parser.add_argument("--seeds", type=int, default=OGBConfig.n_seeds)
    parser.add_argument("--epochs", type=int, default=OGBConfig.epochs)
    parser.add_argument("--hidden-dim", type=int, default=OGBConfig.hidden_dim)
    parser.add_argument("--num-layers", type=int, default=OGBConfig.num_layers)
    parser.add_argument("--learning-rate", type=float, default=OGBConfig.learning_rate)
    parser.add_argument("--rss-cap-mb", type=float, default=0.0)
    parser.add_argument(
        "--emit-predictions",
        action="store_true",
        help="Emit per-method prediction CSVs under <output-dir>/predictions.",
    )
    parser.add_argument("--output-dir", default=OGBConfig.output_dir)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the fast smoke configuration (synthetic-only).",
    )
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    config = OGBConfig(
        methods=tuple(args.methods),
        n_seeds=args.seeds,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        rss_cap_mb=args.rss_cap_mb,
        emit_predictions=args.emit_predictions,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )
    run_ogb_experiment(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
