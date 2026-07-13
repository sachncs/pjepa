"""Ablation study: which component of Persistent-JEPA contributes most?

Phase 11 of the implementation plan. The seven ablation variants are
exactly the ones specified in ``plans/phase_11/plan.md``:

* ``full`` — full Persistent-JEPA. The encoder is the dual-geometric
  encoder and the classification head is fed the concatenation of the
  Euclidean + hyperbolic components (so "full" really uses the
  hyperbolic output, not the Euclidean projection alone). The
  persistent graph is committed to via the four-conditions check
  (which acts as the bisimulation proxy in this smoke configuration
  since the HRG is degenerate); the working graph is selected by
  greedy submodular retrieval; the predictor is matched against an
  EMA target encoder.
* ``minus_hyperbolic`` — Euclidean-only MPNN (no hyperbolic branch).
* ``minus_persistent`` — replace the persistent graph with a fixed-size
  FIFO replay buffer. No commits are attempted: every working graph
  is drawn from the replay buffer (or, when the buffer is empty,
  from the observation directly).
* ``minus_four_conditions`` — skip the four-conditions check; always
  commit the candidate.
* ``minus_ema`` — drop the EMA target encoder; use the online encoder
  directly. The pretraining step is skipped and the classifier head
  is fed embeddings from the (non-EMA) online encoder.
* ``minus_bisimulation`` — replace the bisimulation metric with the
  canonical 1-Weisfeiler-Lehman hash. The acceptance criterion is
  the canonical WL condition: accept iff the candidate and current
  hashes match (i.e. they are 1-WL equivalent).
* ``minus_submodular_retrieval`` — replace greedy retrieval with
  uniform random sampling (the retriever is bypassed entirely).

Each variant is trained for ``n_seeds x n_folds`` runs and the
per-run mean per-class accuracy is recorded. The reference variant
(``full``) is used for the paired bootstrap CI and Wilcoxon
signed-rank test; both statistics pair seeds across methods so the
comparison is always seed-matched.

Outputs (plan-compliant):

* ``<output_dir>/tables/ablation.csv`` — per-(variant, seed, fold)
  raw accuracy plus per-variant bootstrap CI, Wilcoxon
  signed-rank p-value, Bonferroni-corrected p-value, and an
  interpretation field.
* ``<output_dir>/tables/ablation_summary.csv`` — per-variant summary.
* ``<output_dir>/plots/ablation.png`` — publication-quality bar
  chart with bootstrap-CI error bars.
* Legacy ``<output_dir>/ablation.csv`` retained for compatibility.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from pjepa.augmentations.base import AugmentationPipeline, PipelineMode
from pjepa.augmentations.feature import DropFeature
from pjepa.augmentations.structural import DropEdge, DropNode
from pjepa.data.tu import load_tu_dataset
from pjepa.encoders import DualGeometricEncoder, EuclideanMPNN, JEPAPredictor, TargetEncoder
from pjepa.eval import (
    bonferroni_correction,
    color_for,
    mean_per_class_accuracy,
    paired_bootstrap_ci,
    set_publication_style,
    wilcoxon_signed_rank,
)
from pjepa.exceptions import ConfigError
from pjepa.graphs import PersistentState, TypedAttributedGraph, WorkingGraph
from pjepa.logging_setup import LogFormat, configure_logging, get_logger
from pjepa.retrieval import FacilityLocationUtility, GreedyRetrieval
from pjepa.rewriting import HRG, accept_candidate
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "ABLATION_VARIANTS",
    "VARIANT_INTERPRETATION",
    "AblationConfig",
    "aggregate_variant_summary",
    "build_classifier_head",
    "build_encoder_for_variant",
    "build_jepa_augmentation_pipeline",
    "build_variant_model",
    "canonical_wl_hash",
    "default_smoke_config",
    "embed_dim_for_variant",
    "encode_and_mean_pool",
    "is_recognised_variant",
    "kfold_indices",
    "pretrain_jepa_one_epoch",
    "random_subset_indices",
    "render_ablation_figure",
    "run_ablation",
    "select_working_graph_for_variant",
    "train_one_variant",
    "verify_candidate_for_variant",
    "verify_variants",
    "write_table_csv",
]


ABLATION_VARIANTS: tuple[str, ...] = (
    "full",
    "minus_hyperbolic",
    "minus_persistent",
    "minus_four_conditions",
    "minus_ema",
    "minus_bisimulation",
    "minus_submodular_retrieval",
)


VARIANT_INTERPRETATION: dict[str, str] = {
    "full": "Baseline Persistent-JEPA with all components.",
    "minus_hyperbolic": "Removes the hyperbolic branch of the encoder; "
    "tests the importance of hierarchical geometry.",
    "minus_persistent": "Replaces the persistent graph with a FIFO "
    "replay buffer; tests the persistent sufficient statistic.",
    "minus_four_conditions": "Skips the four-conditions acceptance "
    "check and always commits; tests the verification step.",
    "minus_ema": "Drops the EMA target encoder and uses the online "
    "encoder directly; tests the BYOL-style target update.",
    "minus_bisimulation": "Replaces the bisimulation metric with the "
    "canonical 1-WL hash; tests the behavioural-equivalence "
    "verification.",
    "minus_submodular_retrieval": "Replaces greedy submodular "
    "retrieval with uniform random sampling; tests the working-graph "
    "selection rule.",
}


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for the ablation study.

    Attributes:
        dataset: The TU dataset to run on. Default per Phase 11 plan: ``"PROTEINS"``.
        variants: The ablation variants to evaluate. Default is the
          plan-compliant :data:`ABLATION_VARIANTS` tuple.
        n_seeds: The number of seeds per variant.
        n_folds: The number of cross-validation folds.
        epochs: The number of training epochs per run.
        budget: Working-graph budget ``B`` for the full pipeline.
        learning_rate: Optimiser learning rate.
        batch_size: Mini-batch size.
        output_dir: Output directory; ``tables/`` and ``plots/``
          sub-directories are created underneath.
        smoke: When ``True``, the experiment runs the fast smoke
          configuration used by the unit tests.
        bootstrap_resamples: Number of bootstrap resamples.
    """

    dataset: str = "PROTEINS"
    variants: tuple[str, ...] = ABLATION_VARIANTS
    n_seeds: int = 3
    n_folds: int = 5
    epochs: int = 200
    budget: int = 32
    learning_rate: float = 1e-2
    batch_size: int = 32
    output_dir: str = "results"
    smoke: bool = False
    bootstrap_resamples: int = 1000


def default_smoke_config(output_dir: str = "results/ablation_smoke") -> AblationConfig:
    """A fast smoke configuration used by the unit tests.

    Args:
        output_dir: Output directory; defaults to the standard
          ``results/ablation_smoke`` location.

    Returns:
        A smoke-tuned :class:`AblationConfig`.
    """
    return AblationConfig(
        dataset="MUTAG",
        variants=ABLATION_VARIANTS,
        n_seeds=1,
        n_folds=2,
        epochs=2,
        budget=4,
        learning_rate=1e-2,
        batch_size=4,
        output_dir=output_dir,
        smoke=True,
        bootstrap_resamples=200,
    )


def build_encoder_for_variant(variant: str, input_dim: int) -> torch.nn.Module:
    """Construct the encoder for ``variant``.

    The ``minus_hyperbolic`` variant uses the Euclidean-only MPNN;
    every other variant uses the dual-geometric encoder. Both
    constructors expose a forward pass that returns either a
    per-vertex tensor (Euclidean MPNN) or a ``(euclidean,
    hyperbolic)`` tuple (dual-geometric encoder).

    Args:
        variant: One of :data:`ABLATION_VARIANTS`.
        input_dim: Per-vertex feature dimension.

    Returns:
        A ``torch.nn.Module`` encoder.
    """
    if variant == "minus_hyperbolic":
        return EuclideanMPNN(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=4,
            output_dim=128,
        )
    return DualGeometricEncoder(
        input_dim=input_dim,
        euclidean_dim=128,
        hyperbolic_dim=32,
        num_layers=4,
    )


def build_classifier_head(embed_dim: int, num_classes: int) -> torch.nn.Sequential:
    """Build the standard MLP classification head used by every variant.

    Args:
        embed_dim: Input feature dimension.
        num_classes: Number of output classes.

    Returns:
        A ``torch.nn.Sequential`` ``[Linear, ReLU, Linear]`` head.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(embed_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    )


def embed_dim_for_variant(variant: str) -> int:
    """Return the embedding dimensionality produced by ``variant``'s encoder.

    The dual-geometric encoder returns the concatenation of its
    Euclidean and hyperbolic components in :func:`encode_and_mean_pool`;
    both have width 128 in the plan-compliant configuration, so the
    classifier receives a 256-dim input. The Euclidean-only variant
    (``minus_hyperbolic``) returns just the Euclidean component, so
    the classifier receives a 128-dim input.

    Args:
        variant: One of :data:`ABLATION_VARIANTS`.

    Returns:
        The per-graph embedding dimension consumed by the classifier head.
    """
    if variant == "minus_hyperbolic":
        return 128
    return 128 + 32


def encode_and_mean_pool(encoder: torch.nn.Module, graph: TypedAttributedGraph) -> torch.Tensor:
    """Mean-pool the per-vertex encoder output into a 1-D tensor.

    The dual-geometric encoder returns a ``(e, h)`` tuple; this
    function concatenates the two components before pooling so the
    hyperbolic branch actually contributes to the classifier input
    (which is the bug that the previous version of this experiment
    silently introduced — it dropped ``h`` and only consumed ``e``).

    Args:
        encoder: The encoder module.
        graph: The input graph.

    Returns:
        A 1-D ``float`` tensor of shape ``[embed_dim]``.
    """
    out = encoder(graph)
    if isinstance(out, tuple):
        eu, hyp = out
        out = torch.cat([eu, hyp], dim=-1)
    if out.ndim == 2:
        out = out.mean(dim=0)
    return out


def canonical_wl_hash(graph: TypedAttributedGraph, iterations: int = 2) -> str:
    """Return a canonical 1-Weisfeiler-Lehman hash of ``graph``.

    The algorithm matches the standard 1-WL relabelling procedure:

    1. Initialise every vertex label from its feature vector (rounded
       to 3 decimal places for determinism).
    2. For ``iterations`` rounds, replace each vertex label with
       ``(current_label, sorted(neighbour_labels))``.
    3. Hash the resulting multiset of labels to a Python string.

    Two graphs are 1-WL equivalent iff their hashes match. The hash
    is sensitive to topology (it changes when edges are added /
    removed) and deterministic (the same graph always produces the
    same hash).

    Args:
        graph: The input graph.
        iterations: Number of WL relabelling rounds. Defaults to
          ``2`` to keep the hash compact.

    Returns:
        A deterministic string hash.
    """
    feats = graph.vertex_features.detach().clone()
    if feats.numel() == 0:
        return "empty"
    n = feats.shape[0]
    labels = [tuple(round(float(v), 3) for v in feats[i].tolist()) for i in range(n)]
    for _ in range(iterations):
        adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
        if graph.edge_index.numel() > 0:
            for s, d in zip(
                graph.edge_index[0].tolist(),
                graph.edge_index[1].tolist(),
                strict=False,
            ):
                adjacency[s].append(d)
        new_labels: list[tuple] = []
        for v in range(n):
            neigh_labels = sorted(labels[u] for u in adjacency[v])
            new_labels.append((labels[v], tuple(neigh_labels)))
        labels = new_labels
    return str(hash(tuple(labels)))


def verify_candidate_for_variant(
    variant: str,
    candidate: TypedAttributedGraph,
    current: TypedAttributedGraph,
    observation: torch.Tensor,
    grammar: HRG,
) -> tuple[bool, str]:
    """Verify the candidate; ``variant`` selects the proxy used.

    Behaviour:

    * ``minus_four_conditions`` accepts every candidate (the
      verification step is removed entirely).
    * ``minus_bisimulation`` accepts iff the candidate's canonical
      WL hash equals the current's WL hash (the canonical 1-WL
      equivalence test).
    * Every other variant defers to
      :func:`pjepa.rewriting.accept_candidate`, which evaluates
      the four conditions in order: cost, ``Δ𝒥``, grammar
      conformance, bisimulation.

    Args:
        variant: The ablation variant.
        candidate: The candidate next-state graph.
        current: The current persistent graph.
        observation: The observation driving the rewrite.
        grammar: The hyperedge-replacement grammar in use.

    Returns:
        A tuple ``(accepted, info_str)``.
    """
    if variant == "minus_four_conditions":
        return True, "always commit"
    if variant == "minus_bisimulation":
        current_hash = canonical_wl_hash(current)
        candidate_hash = canonical_wl_hash(candidate)
        if current_hash == candidate_hash:
            return True, "wl_hash match"
        return False, "wl_hash differs"
    accepted, info = accept_candidate(
        candidate=candidate,
        current=current,
        observation=observation,
        grammar=grammar,
    )
    if accepted:
        return True, "accepted"
    return False, str(info.get("reason", "rejected"))


def random_subset_indices(num_vertices: int, budget: int, seed: int) -> torch.Tensor:
    """Pick ``budget`` random indices from ``[0, num_vertices)`` deterministically.

    Args:
        num_vertices: The size of the index space.
        budget: The number of indices to return (capped at
          ``num_vertices``).
        seed: Seed for the ``torch.Generator``.

    Returns:
        A 1-D ``long`` tensor of indices.
    """
    g = torch.Generator().manual_seed(int(seed))
    if num_vertices <= 0:
        return torch.zeros((0,), dtype=torch.long)
    target = min(budget, num_vertices)
    return torch.randperm(num_vertices, generator=g)[:target]


def select_working_graph_for_variant(
    variant: str,
    persistent: PersistentState | None,
    graph: TypedAttributedGraph,
    observation: torch.Tensor,
    budget: int,
    seed: int,
) -> WorkingGraph:
    """Pick the working graph for this ablation variant.

    Behaviour:

    * ``minus_submodular_retrieval`` returns a uniformly-random
      ``budget``-vertex subset (the retriever is bypassed).
    * ``minus_persistent`` returns a greedy retrieval over the
      observation graph (no persistent graph is consulted).
    * All other variants greedily retrieve from the persistent
      graph; when the persistent graph is empty they fall back to
      a greedy retrieval on the observation graph.

    Args:
        variant: The ablation variant.
        persistent: The current persistent state (may be ``None``).
        graph: The observation graph (or replay-buffer graph).
        observation: The observation feature vector.
        budget: The working-graph budget ``B``.
        seed: The seed for the random selection.

    Returns:
        A populated :class:`WorkingGraph`.
    """
    if variant == "minus_submodular_retrieval":
        idx = random_subset_indices(graph.num_vertices(), budget, seed=seed)
        mask = torch.zeros(graph.num_vertices(), dtype=torch.bool)
        if idx.numel() > 0:
            mask[idx] = True
        sub = graph.subgraph(mask)
        return WorkingGraph(graph=sub, budget=budget, parent_version=graph.version)
    if variant == "minus_persistent":
        utility = FacilityLocationUtility(vertex_features=graph.vertex_features)
        retriever = GreedyRetrieval(budget=budget)
        result = retriever.select(graph, observation, utility=utility)
        return result.working
    if persistent is None or persistent.num_vertices() == 0:
        utility = FacilityLocationUtility(vertex_features=graph.vertex_features)
        retriever = GreedyRetrieval(budget=budget)
        result = retriever.select(graph, observation, utility=utility)
        return result.working
    utility = FacilityLocationUtility(vertex_features=persistent.graph.vertex_features)
    retriever = GreedyRetrieval(budget=budget)
    result = retriever.select(persistent.graph, observation, utility=utility)
    return result.working


def build_variant_model(
    variant: str, input_dim: int, num_classes: int
) -> tuple[torch.nn.Module, torch.nn.Module, JEPAPredictor | None, TargetEncoder | None]:
    """Return ``(encoder, classifier, predictor, target)`` for ``variant``.

    The ``minus_ema`` variant drops both the JEPA predictor and the
    EMA target encoder — its forward path uses the online encoder
    directly, so the pretraining step is skipped. Every other variant
    instantiates the predictor and the EMA target encoder.

    Args:
        variant: The ablation variant.
        input_dim: Per-vertex feature dimension.
        num_classes: Number of output classes.

    Returns:
        A tuple ``(encoder, classifier, predictor, target)``.
    """
    encoder = build_encoder_for_variant(variant, input_dim)
    embed_dim = embed_dim_for_variant(variant)
    classifier = build_classifier_head(embed_dim, num_classes)
    if variant == "minus_ema":
        return encoder, classifier, None, None
    predictor = JEPAPredictor(input_dim=embed_dim, hidden_dim=embed_dim * 2, output_dim=embed_dim)
    target = TargetEncoder(encoder, momentum=0.99)
    return encoder, classifier, predictor, target


def build_jepa_augmentation_pipeline() -> AugmentationPipeline:
    """Composite graph augmentation pipeline for the JEPA pretraining step.

    Each of the three augmentations (DropEdge, DropNode, DropFeature)
    is selected with equal probability by ``RANDOM_SAMPLE_ONE``.

    Returns:
        A configured :class:`AugmentationPipeline`.
    """
    return AugmentationPipeline(
        [
            DropEdge(strength=0.2),
            DropNode(strength=0.2),
            DropFeature(strength=0.2),
        ],
        mode=PipelineMode.RANDOM_SAMPLE_ONE,
    )


def pretrain_jepa_one_epoch(
    encoder: torch.nn.Module,
    predictor: JEPAPredictor,
    target: TargetEncoder,
    train_pairs: list[tuple[TypedAttributedGraph, int]],
    config: AblationConfig,
) -> None:
    """Run a single JEPA pretraining epoch across ``train_pairs``.

    The encoder is invoked **outside** any ``torch.no_grad`` block so
    it receives gradients through the predictor; the target encoder
    is invoked inside a ``torch.no_grad`` block (it is supposed to
    provide a stable target). After every mini-batch the target
    encoder is updated as an EMA of the online encoder.

    Args:
        encoder: The online encoder.
        predictor: The JEPA predictor head.
        target: The EMA target encoder wrapper.
        train_pairs: The training (graph, label) pairs.
        config: The experiment configuration (learning rate, weight decay).
    """
    aug = build_jepa_augmentation_pipeline()
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=1e-4)
    for g, _ in train_pairs:
        aug_g = aug(g)
        # The online encoder is *not* inside no_grad: it must
        # receive gradients through the predictor.
        ctx_feat = encode_and_mean_pool(encoder, aug_g)
        with torch.no_grad():
            tgt_feat = encode_and_mean_pool(target.shadow, g)
        predicted = predictor(ctx_feat.unsqueeze(0))
        target_t = tgt_feat.unsqueeze(0)
        loss = torch.nn.functional.smooth_l1_loss(predicted, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        target.update()


def train_one_variant(
    variant: str,
    train_pairs: list[tuple[TypedAttributedGraph, int]],
    test_pairs: list[tuple[TypedAttributedGraph, int]],
    num_classes: int,
    config: AblationConfig,
    seed: int,
) -> float:
    """Train one ablation variant and return test mean per-class accuracy.

    Behaviour:

    * ``minus_persistent`` maintains a fixed-capacity FIFO replay
      buffer (``config.budget * 4`` pairs) and draws every working
      graph from it. No persistent graph is committed.
    * ``minus_ema`` skips the JEPA pretraining step entirely; the
      classifier is fed embeddings from the (non-EMA) online
      encoder.
    * Every other variant pretrains via
      :func:`pretrain_jepa_one_epoch`, then commits working graphs
      via :func:`verify_candidate_for_variant` and the persistent
      graph is grown via :meth:`PersistentState.commit`.

    Args:
        variant: The ablation variant.
        train_pairs: The training (graph, label) pairs.
        test_pairs: The test (graph, label) pairs.
        num_classes: Number of output classes.
        config: The experiment configuration.
        seed: Per-run seed.

    Returns:
        Mean per-class test accuracy in [0, 1].
    """
    if not train_pairs or not test_pairs:
        return 0.0
    input_dim = train_pairs[0][0].vertex_features.shape[1]
    encoder, classifier, predictor, target = build_variant_model(variant, input_dim, num_classes)
    persistent: PersistentState | None = None
    grammar = HRG(nonterminals=("S",), terminals=("a",), productions=(), start="S")

    if variant != "minus_ema" and predictor is not None and target is not None:
        pretrain_jepa_one_epoch(encoder, predictor, target, train_pairs, config)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    loss_fn = torch.nn.CrossEntropyLoss()

    replay_buffer: list[tuple[TypedAttributedGraph, torch.Tensor]] = []
    replay_capacity = max(1, config.budget * 4)
    bs = max(1, min(config.batch_size, len(train_pairs)))
    for _epoch in range(config.epochs):
        perm = torch.randperm(len(train_pairs))
        for start in range(0, len(train_pairs), bs):
            idx = perm[start : start + bs].tolist()
            batch = [train_pairs[i] for i in idx]
            optimizer.zero_grad()
            logits_list: list[torch.Tensor] = []
            targets_list: list[int] = []
            for g, lbl in batch:
                obs = (
                    g.vertex_features.mean(dim=0, keepdim=True)
                    if g.num_vertices() > 0
                    else torch.zeros((1, input_dim))
                )
                # Variant-specific working-graph selection.
                if variant == "minus_persistent":
                    # FIFO replay buffer: draw from the buffer when
                    # possible, fall back to the observation.
                    if replay_buffer:
                        _replay_graph, replay_feats = replay_buffer[0]
                        feats = replay_feats.detach()
                    else:
                        working = select_working_graph_for_variant(
                            variant, persistent, g, obs, config.budget, seed=seed
                        )
                        target_graph = working.graph if working.num_vertices() > 0 else g
                        feats = encode_and_mean_pool(encoder, target_graph)
                else:
                    working = select_working_graph_for_variant(
                        variant, persistent, g, obs, config.budget, seed=seed
                    )
                    if working.num_vertices() > 0:
                        candidate = working.graph
                        if persistent is not None and persistent.num_vertices() > 0:
                            accepted, _ = verify_candidate_for_variant(
                                variant, candidate, persistent.graph, obs, grammar
                            )
                            if accepted:
                                persistent = persistent.commit(
                                    candidate=candidate,
                                    cost=float(
                                        abs(candidate.num_vertices() - persistent.num_vertices())
                                    ),
                                    timestamp=float(_epoch),
                                    delta_j=-1e-3,
                                )
                        else:
                            persistent = PersistentState(graph=candidate)
                        feats = encode_and_mean_pool(encoder, candidate)
                    else:
                        feats = encode_and_mean_pool(encoder, g)
                logits_list.append(classifier(feats.unsqueeze(0)))
                targets_list.append(lbl)
                if variant == "minus_persistent":
                    replay_buffer.append((g, feats.detach()))
                    if len(replay_buffer) > replay_capacity:
                        replay_buffer.pop(0)
            logits = torch.cat(logits_list, dim=0)
            tgt = torch.tensor(targets_list, dtype=torch.long)
            loss = loss_fn(logits, tgt)
            loss.backward()
            optimizer.step()
            if target is not None:
                target.update()
        scheduler.step()

    encoder.eval()
    classifier.eval()
    preds: list[int] = []
    labels_eval: list[int] = []
    with torch.no_grad():
        for g, lbl in test_pairs:
            obs = (
                g.vertex_features.mean(dim=0, keepdim=True)
                if g.num_vertices() > 0
                else torch.zeros((1, input_dim))
            )
            if variant == "minus_persistent":
                seed_for_pick = seed * 7919
                working = select_working_graph_for_variant(
                    variant, persistent, g, obs, config.budget, seed=seed_for_pick
                )
                target_graph = working.graph if working.num_vertices() > 0 else g
            else:
                working = select_working_graph_for_variant(
                    variant, persistent, g, obs, config.budget, seed=seed
                )
                target_graph = working.graph if working.num_vertices() > 0 else g
            feats = encode_and_mean_pool(encoder, target_graph)
            preds.append(int(classifier(feats.unsqueeze(0)).argmax(dim=-1).item()))
            labels_eval.append(lbl)
    return mean_per_class_accuracy(preds, labels_eval)


def kfold_indices(n: int, k: int, seed_split: int) -> list[tuple[list[int], list[int]]]:
    """Yield ``(train_indices, test_indices)`` for ``k`` cross-validation folds.

    The split is deterministic for a fixed ``seed_split``; the k-th
    fold's test set is the k-th contiguous chunk of a seed-permuted
    ordering of the indices.

    Args:
        n: Total number of items.
        k: Number of folds (must be positive).
        seed_split: Seed for the index-shuffling generator.

    Returns:
        A list of ``(train_indices, test_indices)`` tuples.

    Raises:
        ConfigError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ConfigError(f"kfold_indices: k must be positive; got {k}")
    g = torch.Generator().manual_seed(int(seed_split))
    indices = torch.randperm(n, generator=g).tolist()
    fold_size = (n + k - 1) // k
    folds: list[tuple[list[int], list[int]]] = []
    for fold_idx in range(k):
        start = fold_idx * fold_size
        end = min(start + fold_size, n)
        train = indices[:start] + indices[end:]
        test = indices[start:end]
        folds.append((train, test))
    return folds


def aggregate_variant_summary(
    raw_rows: list[dict[str, object]],
    reference_variant: str,
    n_resamples: int,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Aggregate per-variant rows into the plan-compliant summary.

    The bootstrap CI and the Wilcoxon signed-rank test are paired
    *by seed*: for every variant ``V ≠ reference``, the test compares
    the ``V`` accuracy on seed ``s`` against the ``reference``
    accuracy on the same seed ``s``. Seeds present in only one of
    the two variants are dropped, so the comparison is always
    seed-matched and the CI / p-values are not inflated by unpaired
    replications. Bonferroni-adjusted p-values are computed across
    every variant (excluding the reference, which is degenerate).

    Args:
        raw_rows: The per-(variant, seed, fold) raw rows.
        reference_variant: The reference variant (typically ``"full"``).
        n_resamples: Bootstrap resample count.
        seed: Random seed for the bootstrap resampler.

    Returns:
        A list of per-variant summary dicts.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    grouped_by_seed: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for row in raw_rows:
        scores = grouped_by_seed.setdefault(str(row["variant"]), {})
        scores.setdefault(int(row["seed"]), []).append(float(row["accuracy"]))
        grouped[str(row["variant"])].append(float(row["accuracy"]))
    # Aggregate per-seed means (so each seed contributes one
    # matched-sample observation, matching the plan's "paired by
    # seed" requirement).
    per_seed_means: dict[str, dict[int, float]] = {}
    for variant, by_seed in grouped_by_seed.items():
        per_seed_means[variant] = {s: sum(v) / len(v) for s, v in by_seed.items()}
    reference_scores = list(per_seed_means.get(reference_variant, {}).values())
    raw_pvalues: list[float] = []
    raw_meta: list[tuple[str, float]] = []
    summary: list[dict[str, object]] = []
    for variant in ABLATION_VARIANTS:
        scores = list(grouped.get(variant, []))
        n = len(scores)
        if n == 0:
            continue
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
        std = var**0.5
        if variant == reference_variant or not reference_scores:
            ci_low, ci_high, mean_diff, p_value = 0.0, 0.0, 0.0, 1.0
            wilcoxon_p = 1.0
        else:
            ref_means = per_seed_means[reference_variant]
            this_means = per_seed_means.get(variant, {})
            common_seeds = sorted(set(ref_means) & set(this_means))
            this_subset = [this_means[s] for s in common_seeds]
            ref_subset = [ref_means[s] for s in common_seeds]
            if len(common_seeds) >= 1:
                ci = paired_bootstrap_ci(
                    this_subset, ref_subset, n_resamples=n_resamples, seed=seed
                )
                mean_diff = ci.mean_diff
                ci_low = ci.ci_low
                ci_high = ci.ci_high
                p_value = ci.p_value
                wilcoxon_p = wilcoxon_signed_rank(this_subset, ref_subset)
            else:
                ci_low, ci_high, mean_diff, p_value = 0.0, 0.0, 0.0, 1.0
                wilcoxon_p = 1.0
        raw_pvalues.append(wilcoxon_p)
        raw_meta.append((variant, wilcoxon_p))
        summary.append(
            {
                "variant": variant,
                "interpretation": VARIANT_INTERPRETATION.get(variant, ""),
                "n_runs": n,
                "mean_accuracy": mean,
                "std_accuracy": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_diff_vs_full": mean_diff,
                "p_value_bootstrap": p_value,
                "wilcoxon_p_vs_full": wilcoxon_p,
            }
        )
    corrected = bonferroni_correction(raw_pvalues)
    for (variant, _), p_corr in zip(raw_meta, corrected, strict=True):
        for entry in summary:
            if entry["variant"] == variant:
                entry["wilcoxon_p_bonferroni"] = p_corr
                break
    return summary


def render_ablation_figure(summary: list[dict[str, object]], png_path: Path) -> None:
    """Render the plan-compliant ablation bar chart with bootstrap-CI error bars.

    Args:
        summary: The per-variant summary produced by
          :func:`aggregate_variant_summary`.
        png_path: Destination file path for the PNG figure.
    """
    set_publication_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    variants = [str(s["variant"]) for s in summary]
    means = [float(s["mean_accuracy"]) for s in summary]
    ci_lows = [float(s["ci_low"]) for s in summary]
    ci_highs = [float(s["ci_high"]) for s in summary]
    n = len(variants)
    x_positions = list(range(n))
    ax.bar(
        x_positions,
        means,
        yerr=[
            [max(0.0, means[i] - ci_lows[i]) for i in range(n)],
            [max(0.0, ci_highs[i] - means[i]) for i in range(n)],
        ],
        color=[color_for(i) for i in range(n)],
        capsize=3.0,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean accuracy (bootstrap 95% CI)")
    ax.set_title("Ablation: per-variant accuracy on PROTEINS")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def write_table_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write a list of row-dicts to ``path`` using the union of keys as fieldnames.

    Args:
        rows: The row dicts to write.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_recognised_variant(variant: str) -> bool:
    """Return whether ``variant`` is a recognised ablation name.

    Args:
        variant: The variant name to test.

    Returns:
        ``True`` iff ``variant`` is in :data:`ABLATION_VARIANTS`.
    """
    return variant in ABLATION_VARIANTS


def verify_variants(variants: Sequence[str]) -> None:
    """Raise :class:`ConfigError` for unknown variant names.

    Args:
        variants: The candidate variant list.

    Raises:
        ConfigError: If any element of ``variants`` is not in
          :data:`ABLATION_VARIANTS`.
    """
    for v in variants:
        if not is_recognised_variant(v):
            raise ConfigError(
                f"verify_variants: unknown ablation variant {v!r}; "
                f"expected one of {list(ABLATION_VARIANTS)}"
            )


def run_ablation(config: AblationConfig) -> dict[str, object]:
    """Run the ablation study.

    The function:

    1. Verifies the requested variant list.
    2. Loads the TU dataset.
    3. Iterates over ``(seed, fold, variant)``, training + evaluating
       each combination.
    4. Aggregates the per-variant rows into a plan-compliant summary.
    5. Writes the long CSV, the summary CSV, the legacy CSV, and the
       publication-quality bar chart.

    Args:
        config: Experiment configuration. Use
          :func:`default_smoke_config` for fast tests.

    Returns:
        A dictionary with ``raw_rows``, ``summary``, and the output
        paths ``csv``, ``summary_csv``, ``legacy_csv``, and ``png``.
    """
    log = get_logger(__name__)
    verify_variants(config.variants)
    log.info(
        "ablation starting",
        extra={
            "event": "ablation.start",
            "dataset": config.dataset,
            "variants": list(config.variants),
            "n_seeds": config.n_seeds,
            "n_folds": config.n_folds,
        },
    )
    graphs, num_classes = load_tu_dataset(config.dataset)
    pairs = [(g.graph, g.label) for g in graphs]
    raw_rows: list[dict[str, object]] = []
    for seed in range(config.n_seeds):
        fold_splits = kfold_indices(len(pairs), config.n_folds, seed * 1000)
        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            train_pairs = [pairs[i] for i in train_idx]
            test_pairs = [pairs[i] for i in test_idx]
            if not train_pairs or not test_pairs:
                continue
            set_global_seed(seed * 1000 + fold_idx)
            for variant in config.variants:
                start = time.time()
                accuracy = train_one_variant(
                    variant,
                    train_pairs,
                    test_pairs,
                    num_classes,
                    config,
                    seed=seed * 1000 + fold_idx,
                )
                elapsed = time.time() - start
                raw_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "fold": fold_idx,
                        "accuracy": accuracy,
                        "elapsed_seconds": elapsed,
                        "interpretation": VARIANT_INTERPRETATION.get(variant, ""),
                    }
                )
                log.info(
                    "ablation variant complete",
                    extra={
                        "event": "ablation.variant_complete",
                        "variant": variant,
                        "seed": seed,
                        "fold": fold_idx,
                        "accuracy": accuracy,
                    },
                )

    summary = aggregate_variant_summary(
        raw_rows,
        reference_variant="full",
        n_resamples=config.bootstrap_resamples,
    )

    out_root = Path(config.output_dir)
    tables_dir = out_root / "tables"
    plots_dir = out_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_table_path = tables_dir / "ablation.csv"
    summary_table_path = tables_dir / "ablation_summary.csv"
    png_path = plots_dir / "ablation.png"
    legacy_csv = out_root / "ablation.csv"

    raw_with_summary_cols: list[dict[str, object]] = []
    summary_index = {str(s["variant"]): s for s in summary}
    for row in raw_rows:
        s = summary_index.get(str(row["variant"]), {})
        merged = dict(row)
        merged["mean_accuracy"] = s.get("mean_accuracy", float("nan"))
        merged["std_accuracy"] = s.get("std_accuracy", float("nan"))
        merged["ci_low"] = s.get("ci_low", float("nan"))
        merged["ci_high"] = s.get("ci_high", float("nan"))
        merged["mean_diff_vs_full"] = s.get("mean_diff_vs_full", float("nan"))
        merged["p_value_bootstrap"] = s.get("p_value_bootstrap", float("nan"))
        merged["wilcoxon_p_vs_full"] = s.get("wilcoxon_p_vs_full", float("nan"))
        merged["wilcoxon_p_bonferroni"] = s.get("wilcoxon_p_bonferroni", float("nan"))
        raw_with_summary_cols.append(merged)

    write_table_csv(raw_with_summary_cols, raw_table_path)
    write_table_csv(summary, summary_table_path)
    write_table_csv(raw_rows, legacy_csv)

    render_ablation_figure(summary, png_path)

    log.info(
        "ablation complete",
        extra={
            "event": "ablation.complete",
            "n_rows": len(raw_rows),
            "n_summary_rows": len(summary),
            "csv": str(raw_table_path),
            "summary_csv": str(summary_table_path),
            "png": str(png_path),
        },
    )
    return {
        "raw_rows": raw_rows,
        "summary": summary,
        "csv": str(raw_table_path),
        "summary_csv": str(summary_table_path),
        "legacy_csv": str(legacy_csv),
        "png": str(png_path),
    }


def main() -> int:
    """CLI entry point for the ablation study.

    Returns:
        ``0`` on a successful run.
    """
    parser = argparse.ArgumentParser(description="Run the Persistent-JEPA ablation study.")
    parser.add_argument("--dataset", default=AblationConfig.dataset)
    parser.add_argument("--seeds", type=int, default=AblationConfig.n_seeds)
    parser.add_argument("--folds", type=int, default=AblationConfig.n_folds)
    parser.add_argument("--epochs", type=int, default=AblationConfig.epochs)
    parser.add_argument("--budget", type=int, default=AblationConfig.budget)
    parser.add_argument("--output-dir", default=AblationConfig.output_dir)
    parser.add_argument("--smoke", action="store_true", help="Run the fast smoke configuration.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(ABLATION_VARIANTS),
        help="Subset of ablation variants to run.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=AblationConfig.bootstrap_resamples,
    )
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    if args.smoke:
        cfg = default_smoke_config(output_dir=args.output_dir)
    else:
        cfg = AblationConfig(
            dataset=args.dataset,
            variants=tuple(args.variants),
            n_seeds=args.seeds,
            n_folds=args.folds,
            epochs=args.epochs,
            budget=args.budget,
            output_dir=args.output_dir,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    run_ablation(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
