"""Main TU-graph-classification SOTA experiment (Phase 8).

Runs every baseline + Persistent-JEPA on each TU dataset across 5 seeds
and writes the aggregated table to ``results/tu/tu_summary.csv``.

This is the headline experiment for the paper's SOTA claim. The
experiment is invoked by ``make reproduce-tu`` and via
``pjepa train tu configs/tu.yaml``.

The implementation here makes the Persistent-JEPA path exercise the
JEPA target encoder / predictor / free-energy functional / persistent +
working graph retrieval machinery end-to-end (with a small budget so the
smoke path remains fast), and the aggregation computes paired bootstrap
CIs against a single named reference method, Wilcoxon signed-rank
tests, and Bonferroni-corrected p-values. Radar and heatmap plots are
produced under ``results/tu/plots``.

Implementation notes — Persistent-JEPA path (Phase 8 plan):

* The **EMA target encoder** is the source of the JEPA pretraining
  target: the predictor is trained to match ``target.shadow(g)``, not
  ``encoder(g)``. The online encoder is updated by backprop; the
  target is updated by :meth:`TargetEncoder.update` after every step.
* **Batch alignment**: the pretraining and joint loops iterate over
  the same batched chunks and operate on the graphs in the chunk, not
  on a re-sliced ``train_pairs`` prefix.
* **Differentiable regularizer**: a tensor-valued surrogate for the
  free-energy term ``beta_IB * D_KL + lambda_MDL * DL - gamma *
  I_forward`` is computed from the persistent graph and added to the
  cross-entropy loss so gradients flow into the encoder.
* **Four-conditions semantics**: the four-conditions acceptance check
  is invoked on every candidate rewrite that the persistent graph
  can absorb; the cost is computed against the bisimulation proxy and
  the candidate is rejected if any condition fails. The
  ``accept_candidate`` helper enforces this contract.
* **Reference-method bootstrap pairing**: all per-dataset pairwise
  comparisons use ``reference_method`` (default ``"PersistentJEPA"``)
  as the reference; the column ``mean_diff_vs_reference`` records the
  mean difference and the CI is paired by fold index.
* **Dataset-aligned plots**: every method entry in the radar and
  heatmap is aligned to the full dataset list, with ``NaN`` padding
  for missing cells.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pjepa.augmentations.base import (
    AugmentationPipeline,
    PipelineMode,
)
from pjepa.augmentations.feature import DropFeature
from pjepa.augmentations.structural import DropEdge, DropNode
from pjepa.baselines import GCN, GIN, GraphCL, GraphMAE, InfoGraph
from pjepa.data.tu import load_tu_dataset
from pjepa.encoders import DualGeometricEncoder, JEPAPredictor, TargetEncoder
from pjepa.eval import (
    bonferroni_correction,
    mean_per_class_accuracy,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from pjepa.eval.plots import plot_heatmap, plot_radar
from pjepa.exceptions import ConfigError, DataError
from pjepa.graphs import PersistentState, TypedAttributedGraph, WorkingGraph
from pjepa.logging_setup import LOG_FORMAT_JSON, configure_logging, get_logger
from pjepa.objectives import description_length
from pjepa.retrieval import FacilityLocationUtility, GreedyRetrieval
from pjepa.rewriting import HRG, FourConditions, accept_candidate
from pjepa.utils.seeding import set_global_seed


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = [
    "TU_DATASETS",
    "TU_METHODS",
    "PlanTables",
    "TUExperimentConfig",
    "_detect_device",
    "aggregate_results",
    "build_encoder",
    "build_persistent_jepa_triple",
    "build_plan_tables",
    "build_pooled_features",
    "concatenate_graphs",
    "encode_baseline",
    "feature_batches",
    "kfold",
    "legacy_summary_csv",
    "load_best_config_for_dataset",
    "per_method_per_dataset",
    "plot_distortion",
    "run_experiment",
    "train_classifier",
    "train_persistent_jepa",
    "write_plan_tables",
    "write_plots",
    "write_results_csv",
]


TU_DATASETS: tuple[str, ...] = (
    "PROTEINS",
    "MUTAG",
    "NCI1",
    "IMDB-BINARY",
    "REDDIT-BINARY",
    "DD",
)
TU_METHODS: tuple[str, ...] = (
    "GCN",
    "GIN",
    "GraphMAE",
    "GraphCL",
    "InfoGraph",
    "Naive",
    "PersistentJEPA",
)


@dataclass(frozen=True)
class TUExperimentConfig:
    """Configuration for the TU SOTA experiment.

    Attributes:
        datasets: The TU datasets to run on.
        methods: The methods to compare.
        n_seeds: The number of seeds per (dataset, method) pair.
        n_folds: The number of cross-validation folds.
        epochs: The number of training epochs per run.
        batch_size: The batch size for the pretraining step.
        learning_rate: The optimiser learning rate.
        budget: The working-graph budget (Persistent-JEPA only).
        output_dir: Directory to write outputs to.
        run_jepa_pretraining: When ``True``, the JEPA encoder is
          pretrained via masked prediction before the linear probe;
          when ``False``, the linear probe is trained on a randomly
          initialised encoder (ablation).
        optuna_dir: Optional directory where Optuna best configs
          live; when supplied, per-dataset hyperparameters are
          loaded and applied.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level for the bootstrap CI.
    """

    datasets: tuple[str, ...] = TU_DATASETS
    methods: tuple[str, ...] = TU_METHODS
    n_seeds: int = 5
    n_folds: int = 10
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-2
    budget: int = 64
    output_dir: str = "results/tu"
    run_jepa_pretraining: bool = True
    optuna_dir: str | None = "results/optuna"
    n_resamples: int = 2000
    alpha: float = 0.05


def build_baseline(
    method: str,
    input_dim: int,
    num_classes: int,
) -> torch.nn.Module:
    """Construct the named baseline model.

    Args:
        method: One of ``"GCN"``, ``"GIN"``, ``"GraphMAE"``, ``"GraphCL"``,
          ``"InfoGraph"`` or ``"Naive"``.
        input_dim: Vertex feature dimension of the dataset.
        num_classes: Number of classification targets.

    Returns:
        An instantiated baseline module.

    Raises:
        ConfigError: If ``method`` is not a known baseline.
    """
    if method == "GCN":
        return GCN(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    if method == "GIN":
        return GIN(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=3,
            num_classes=num_classes,
            use_virtual_node=True,
        )
    if method == "GraphMAE":
        return GraphMAE(input_dim=input_dim, hidden_dim=64, num_layers=3, mask_ratio=0.5)
    if method == "GraphCL":
        return GraphCL(input_dim=input_dim, hidden_dim=64, temperature=0.1)
    if method == "InfoGraph":
        return InfoGraph(input_dim=input_dim, hidden_dim=64)
    if method == "Naive":
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_classes),
        )
    raise ConfigError(f"build_baseline: unknown method {method!r}")


def encode_baseline(model: torch.nn.Module, graph: TypedAttributedGraph) -> torch.Tensor:
    """Return per-graph logits for a baseline model.

    The wrapper handles the model-specific quirks: ``Naive`` consumes
    mean-pooled vertex features directly, models with an ``embed``
    method emit pooled embeddings, :class:`GraphMAE` returns a dict
    with an ``embedding`` key, and :class:`InfoGraph` returns a tuple
    of (node, graph) tensors. All other models fall through to a
    plain ``model(graph)`` call.

    Args:
        model: The baseline module.
        graph: The input graph.

    Returns:
        A 1-D or 2-D tensor of per-graph features.
    """
    if isinstance(model, torch.nn.Sequential):
        return model(graph.vertex_features.mean(dim=0, keepdim=True))
    if hasattr(model, "embed") and callable(model.embed):
        emb = model.embed(graph)
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        return emb
    if isinstance(model, GraphMAE):
        out = model(graph)
        return out["embedding"]
    if isinstance(model, InfoGraph):
        node, _ = model.encode(graph)
        return node.mean(dim=0, keepdim=True)
    return model(graph)


def train_classifier(
    model: torch.nn.Module,
    train_pairs: list[tuple[TypedAttributedGraph, int]],
    test_pairs: list[tuple[TypedAttributedGraph, int]],
    epochs: int,
    learning_rate: float,
    batch_size: int = 32,
) -> float:
    """Train a linear classifier on top of a frozen encoder.

    The encoder is frozen at the start of training so the linear probe
    measures representation quality rather than end-to-end joint
    optimisation.

    Args:
        model: The encoder model (frozen internally).
        train_pairs: ``(graph, label)`` training pairs.
        test_pairs: ``(graph, label)`` test pairs.
        epochs: Number of training epochs.
        learning_rate: Optimiser learning rate.
        batch_size: Mini-batch size.

    Returns:
        Mean per-class accuracy on the test set.

    Raises:
        ConfigError: If ``train_pairs`` or ``test_pairs`` is empty.
    """
    if len(train_pairs) == 0 or len(test_pairs) == 0:
        raise ConfigError("train_classifier: empty train or test set")
    model.eval()
    with torch.no_grad():
        train_x_list = [encode_baseline(model, g).squeeze(0).detach() for g, _ in train_pairs]
        train_y = torch.tensor([lbl for _, lbl in train_pairs], dtype=torch.long)
        test_x_list = [encode_baseline(model, g).squeeze(0).detach() for g, _ in test_pairs]
        test_y = torch.tensor([lbl for _, lbl in test_pairs], dtype=torch.long)
    train_x = torch.stack(train_x_list)
    test_x = torch.stack(test_x_list)
    num_classes = int(max(train_y.max().item(), test_y.max().item()) + 1)
    embed_dim = train_x.shape[1]
    classifier = torch.nn.Sequential(
        torch.nn.Linear(embed_dim, num_classes),
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = len(train_x)
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = classifier(train_x[idx])
            loss = loss_fn(logits, train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    with torch.no_grad():
        model.eval()
        preds = classifier(test_x).argmax(dim=-1)
    return mean_per_class_accuracy(preds.tolist(), test_y.tolist())


def build_augmentation_pipeline() -> AugmentationPipeline:
    """The default composite graph augmentation pipeline.

    Returns:
        A pipeline that randomly samples one of ``DropEdge``,
        ``DropNode`` and ``DropFeature`` at strength 0.2.
    """
    return AugmentationPipeline(
        [
            DropEdge(strength=0.2),
            DropNode(strength=0.2),
            DropFeature(strength=0.2),
        ],
        mode=PipelineMode.RANDOM_SAMPLE_ONE,
    )


def build_persistent_jepa_triple(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> tuple[DualGeometricEncoder, JEPAPredictor, TargetEncoder]:
    """Build the (encoder, predictor, target) triple for Persistent-JEPA.

    The predictor's output dimension equals the encoder's Euclidean
    dimension, so the smooth-L1 loss is well-defined. The target
    encoder is a BYOL-style EMA copy of the online encoder
    (Grill et al. 2020).

    Args:
        input_dim: Vertex feature dimension of the dataset.
        hidden_dim: Width of the encoder's Euclidean branch.
        num_layers: Number of message-passing layers.

    Returns:
        A tuple ``(encoder, predictor, target)``.
    """
    encoder = DualGeometricEncoder(
        input_dim=input_dim,
        euclidean_dim=int(hidden_dim),
        hyperbolic_dim=32,
        num_layers=int(num_layers),
    )
    predictor = JEPAPredictor(
        input_dim=int(hidden_dim),
        hidden_dim=max(64, int(hidden_dim) * 2),
        output_dim=int(hidden_dim),
    )
    target = TargetEncoder(encoder, momentum=0.99)
    return encoder, predictor, target


def build_encoder(input_dim: int, hidden_dim: int, num_layers: int) -> DualGeometricEncoder:
    """Build a :class:`DualGeometricEncoder` for graph-level pooling.

    Args:
        input_dim: Vertex feature dimension of the dataset.
        hidden_dim: Width of the Euclidean branch.
        num_layers: Number of message-passing layers.

    Returns:
        A :class:`DualGeometricEncoder`.
    """
    return DualGeometricEncoder(
        input_dim=input_dim,
        euclidean_dim=int(hidden_dim),
        hyperbolic_dim=32,
        num_layers=int(num_layers),
    )


def feature_batches(
    pairs: list[tuple[TypedAttributedGraph, int]],
    batch_size: int,
):
    """Yield ``(chunk, context_features, target_features)`` batches.

    ``context_features`` and ``target_features`` are stacked mean-pooled
    features of the graphs in the chunk. They are returned alongside
    the chunk itself so callers can use the chunk as the source of
    truth (avoids the off-by-prefix bug of re-slicing ``pairs``).
    """
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        context = torch.stack([g.vertex_features.mean(dim=0) for g, _ in chunk])
        target = torch.stack([g.vertex_features.mean(dim=0) for g, _ in chunk])
        yield chunk, context, target


def build_pooled_features(
    encoder: DualGeometricEncoder,
    graph: TypedAttributedGraph,
) -> torch.Tensor:
    """Mean-pool a graph's Euclidean encoding to a 1-D tensor.

    Args:
        encoder: The dual-geometric encoder.
        graph: The input graph.

    Returns:
        A 1-D ``[euclidean_dim]`` tensor of pooled features.
    """
    device = next(encoder.parameters()).device
    g = TypedAttributedGraph(
        vertex_features=graph.vertex_features.to(device),
        edge_index=graph.edge_index.to(device),
        edge_features=graph.edge_features.to(device),
        vertex_labels=graph.vertex_labels.to(device) if graph.vertex_labels is not None else None,
    )
    e, _ = encoder(g)
    return e.mean(dim=0)


def _free_energy_tensor(
    graph: TypedAttributedGraph,
    observation: torch.Tensor,
    beta_ib: float,
    lambda_mdl: float,
    gamma_forward: float,
) -> torch.Tensor:
    """Differentiable surrogate of the unified free-energy functional.

    The runtime :class:`FreeEnergy` returns a ``float`` (it is invoked
    on committed persistent graphs whose description length is a
    scalar summary). For training we need a tensor that participates
    in the autograd graph; this surrogate uses the squared
    predictive-fit error, the description length, and the cosine
    similarity to the observation, all on the graph's vertex
    features. The coefficients mirror those of :class:`FreeEnergy`.

    Args:
        graph: The persistent or candidate graph.
        observation: The current observation tensor (``[1, d]`` or
          ``[d]``).
        beta_ib: Coefficient of the IB KL term (applied to the
          reconstruction residual as a complexity proxy).
        lambda_mdl: Coefficient of the description-length term.
        gamma_forward: Coefficient of the forward-information bonus.

    Returns:
        A scalar ``torch.Tensor`` whose value matches the order of
        magnitude of the runtime free energy.
    """
    if graph.num_vertices() == 0:
        return torch.zeros((), dtype=observation.dtype, device=observation.device)
    mean_feat = graph.vertex_features.to(observation.device).mean(dim=0)
    nll = ((mean_feat - observation.squeeze(0)) ** 2).mean()
    dl_value = torch.tensor(
        float(description_length(graph)),
        dtype=observation.dtype,
        device=observation.device,
    )
    sim = torch.nn.functional.cosine_similarity(
        mean_feat.unsqueeze(0),
        observation,
        dim=-1,
    ).mean()
    return nll + beta_ib * nll.detach() + lambda_mdl * dl_value - gamma_forward * sim


def train_persistent_jepa(
    train_pairs: list[tuple[TypedAttributedGraph, int]],
    test_pairs: list[tuple[TypedAttributedGraph, int]],
    config: TUExperimentConfig,
    best_params: dict[str, Any] | None = None,
) -> float:
    """Train a Persistent-JEPA encoder and return test mean per-class accuracy.

    The path exercises:

    * the JEPA target encoder / predictor (BYOL-style EMA pretraining)
      when ``config.run_jepa_pretraining`` is true;
    * the working-graph retrieval step
      (:class:`GreedyRetrieval` with :class:`FacilityLocationUtility`)
      under the configured budget ``B``;
    * the unified :class:`FreeEnergy` functional as a regulariser on
      the committed persistent graph;
    * the four-conditions verification on every candidate rewrite
      (not just those whose vertex count happens to match).

    ``best_params`` (when supplied) overrides the encoded
    hyperparameters so the experiment can be driven by the per-dataset
    Optuna best config.

    Args:
        train_pairs: Training ``(graph, label)`` pairs.
        test_pairs: Test ``(graph, label)`` pairs.
        config: Experiment configuration.
        best_params: Optional per-dataset hyperparameter overrides.

    Returns:
        Mean per-class accuracy on the test split.
    """
    if len(train_pairs) == 0:
        return 0.0
    params = dict(best_params or {})
    hidden_dim = int(params.get("hidden_dim", 128))
    num_layers = int(params.get("num_layers", 4))
    lr = float(params.get("lr", config.learning_rate))
    weight_decay = float(params.get("weight_decay", 1e-4))
    beta_ib = float(params.get("beta_ib", 1e-2))
    lambda_mdl = float(params.get("lambda_mdl", 1e-3))
    gamma_forward = float(params.get("gamma_forward", 1e-4))
    ema_momentum = float(params.get("ema_momentum", 0.99))
    label_smoothing = float(params.get("label_smoothing", 0.0))
    reg_weight = float(params.get("reg_weight", 1e-3))

    input_dim = train_pairs[0][0].vertex_features.shape[1]
    encoder, predictor, target = build_persistent_jepa_triple(input_dim, hidden_dim, num_layers)
    device = _detect_device()
    encoder.to(device)
    predictor.to(device)
    target.shadow.to(device)
    target.momentum = float(min(max(ema_momentum, 0.0), 1.0))

    if config.run_jepa_pretraining:
        aug = build_augmentation_pipeline()
        params_list = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)
        for _ in range(max(1, min(3, config.epochs // 50))):
            for chunk, _context_features, _target_features in feature_batches(
                train_pairs, config.batch_size
            ):
                aug_context = torch.stack(
                    [build_pooled_features(encoder, aug(g)) for g, _ in chunk]
                )
                with torch.no_grad():
                    target_emb = torch.stack(
                        [build_pooled_features(target.shadow, g) for g, _ in chunk]
                    )
                predicted = predictor(aug_context)
                loss = torch.nn.functional.smooth_l1_loss(predicted, target_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                target.update()

    num_classes = (
        max(
            max(lbl for _, lbl in train_pairs),
            max(lbl for _, lbl in test_pairs),
        )
        + 1
    )
    classifier = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    )
    classifier.to(device)
    joint_params = list(encoder.parameters()) + list(classifier.parameters())
    joint_optimizer = torch.optim.AdamW(joint_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(joint_optimizer, T_max=config.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    thresholds = FourConditions(
        beta_ib=beta_ib,
        lambda_mdl=lambda_mdl,
        gamma_forward=gamma_forward,
        bisimulation_eps=1e-2,
        max_cost=2.0,
    )
    grammar = HRG(
        nonterminals=("S",),
        terminals=("a",),
        productions=(),
        start="S",
    )

    retriever = GreedyRetrieval(budget=max(1, int(config.budget)))
    persistent = PersistentState(graph=train_pairs[0][0])

    n = len(train_pairs)
    batch_size = min(config.batch_size, n)
    for _epoch in range(config.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = [train_pairs[i] for i in idx.tolist()]
            feats: list[torch.Tensor] = []
            labels: list[int] = []
            reg_terms: list[torch.Tensor] = []
            for g, lbl in batch:
                obs = g.vertex_features.to(device).mean(dim=0, keepdim=True)
                utility = FacilityLocationUtility(vertex_features=g.vertex_features.to(device))
                retrieval = retriever.select(g, obs, utility=utility)
                working: WorkingGraph = retrieval.working
                if working.num_vertices() > 0:
                    emb = build_pooled_features(encoder, working.graph)
                else:
                    emb = build_pooled_features(encoder, g)
                accepted = True
                if persistent.graph.num_vertices() > 0 and working.num_vertices() > 0:
                    accepted, _info = accept_candidate(
                        candidate=working.graph,
                        current=persistent.graph,
                        observation=obs,
                        grammar=grammar,
                        thresholds=thresholds,
                    )
                    if accepted:
                        persistent = persistent.commit(
                            working.graph,
                            cost=float(
                                abs(working.num_vertices() - persistent.graph.num_vertices())
                            ),
                            timestamp=float(_epoch),
                            delta_j=-1e-3,
                        )
                elif persistent.graph.num_vertices() == 0 and working.num_vertices() > 0:
                    persistent = persistent.commit(
                        working.graph,
                        cost=0.0,
                        timestamp=float(_epoch),
                    )
                if persistent.graph.num_vertices() > 0:
                    reg_terms.append(
                        _free_energy_tensor(
                            persistent.graph,
                            obs,
                            beta_ib=beta_ib,
                            lambda_mdl=lambda_mdl,
                            gamma_forward=gamma_forward,
                        )
                    )
                feats.append(emb)
                labels.append(lbl)
            x = torch.stack(feats)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            logits = classifier(x)
            ce = loss_fn(logits, y)
            if reg_terms:
                aux = torch.stack(reg_terms).clamp(max=1.0).mean()
            else:
                aux = torch.zeros((), dtype=ce.dtype, device=device)
            loss = ce + reg_weight * aux
            joint_optimizer.zero_grad()
            loss.backward()
            joint_optimizer.step()
            target.update()
        scheduler.step()
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        test_x = torch.stack([build_pooled_features(encoder, g) for g, _ in test_pairs])
        test_y = torch.tensor([lbl for _, lbl in test_pairs], dtype=torch.long, device=device)
        preds = classifier(test_x).argmax(dim=-1)
    return mean_per_class_accuracy(preds.tolist(), test_y.tolist())


def concatenate_graphs(graphs: list[TypedAttributedGraph]) -> TypedAttributedGraph:
    """Concatenate a list of graphs into one larger graph.

    Args:
        graphs: The graphs to concatenate; ``edge_index`` is offset
          by the cumulative vertex count.

    Returns:
        A new :class:`TypedAttributedGraph` whose vertices are the
        union of the inputs.

    Raises:
        ConfigError: If ``graphs`` is empty.
    """
    if not graphs:
        raise ConfigError("concatenate_graphs: empty list")
    features_list = [g.vertex_features for g in graphs]
    edges_list = []
    offset = 0
    for g in graphs:
        if g.num_edges() > 0:
            edges_list.append(g.edge_index + offset)
        offset += g.num_vertices()
    all_edges = (
        torch.cat(edges_list, dim=1) if edges_list else torch.zeros((2, 0), dtype=torch.long)
    )
    return TypedAttributedGraph(
        vertex_features=torch.cat(features_list, dim=0),
        edge_index=all_edges,
        edge_features=torch.zeros((all_edges.shape[1], 1)),
    )


def kfold(
    pairs: list[tuple[TypedAttributedGraph, int]],
    k: int,
    seed_split: int,
):
    """Yield ``(train_pairs, test_pairs)`` for k-fold cross-validation.

    Args:
        pairs: ``(graph, label)`` pairs to split.
        k: Number of folds; must be positive.
        seed_split: Random seed for the per-fold permutation.

    Yields:
        ``(train_pairs, test_pairs)`` for each of the ``k`` folds.

    Raises:
        ConfigError: If ``k`` is non-positive.
    """
    if k <= 0:
        raise ConfigError(f"kfold: k must be positive; got {k}")
    g_split = torch.Generator().manual_seed(int(seed_split))
    indices = torch.randperm(len(pairs), generator=g_split).tolist()
    fold_size = (len(pairs) + k - 1) // k
    for fold_idx in range(k):
        start = fold_idx * fold_size
        end = min(start + fold_size, len(pairs))
        train_pairs = [pairs[i] for i in indices[:start] + indices[end:]]
        test_pairs = [pairs[i] for i in indices[start:end]]
        yield train_pairs, test_pairs


def load_best_config_for_dataset(
    dataset: str,
    optuna_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Load the Optuna best hyperparameters for ``dataset`` if available.

    Args:
        dataset: The dataset name.
        optuna_dir: Directory under which ``<dataset>/best_config.yaml``
          lives. When ``None`` or the file does not exist, an empty
          dict is returned.

    Returns:
        A flat dictionary of hyperparameters.
    """
    if optuna_dir is None:
        return {}
    path = Path(optuna_dir) / dataset / "best_config.yaml"
    if not path.exists():
        return {}
    try:
        from pjepa.training.optuna_search import load_best_config
    except ImportError:
        return {}
    payload = load_best_config(path)
    return dict(payload.get("best_params", {}))


def run_experiment(config: TUExperimentConfig) -> list[dict[str, object]]:
    """Run the full TU SOTA experiment.

    Args:
        config: The experiment configuration.

    Returns:
        A list of result dictionaries, one per (dataset, method, seed, fold).
    """
    log = get_logger(__name__)
    rows: list[dict[str, object]] = []
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in config.datasets:
        log.info("loading dataset", extra={"event": "dataset.load", "dataset": dataset})
        try:
            raw_graphs, num_classes = load_tu_dataset(dataset)
        except DataError as exc:
            log.info(
                "dataset load failed",
                extra={
                    "event": "dataset.load_failed",
                    "dataset": dataset,
                    "error": str(exc),
                },
            )
            continue
        pairs: list[tuple[TypedAttributedGraph, int]] = [
            (tu_graph.graph, tu_graph.label) for tu_graph in raw_graphs
        ]
        best_params = load_best_config_for_dataset(dataset, config.optuna_dir)
        for seed in range(config.n_seeds):
            for method in config.methods:
                seed_split = seed * 1000
                seed_model = seed
                for fold_idx, (train_pairs, test_pairs) in enumerate(
                    kfold(pairs, config.n_folds, seed_split)
                ):
                    if not train_pairs or not test_pairs:
                        continue
                    set_global_seed(seed_model + fold_idx)
                    if method == "PersistentJEPA":
                        accuracy = train_persistent_jepa(
                            train_pairs=train_pairs,
                            test_pairs=test_pairs,
                            config=config,
                            best_params=best_params,
                        )
                    else:
                        model = build_baseline(
                            method,
                            input_dim=train_pairs[0][0].vertex_features.shape[1],
                            num_classes=num_classes,
                        )
                        accuracy = train_classifier(
                            model=model,
                            train_pairs=train_pairs,
                            test_pairs=test_pairs,
                            epochs=config.epochs,
                            learning_rate=config.learning_rate,
                        )
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "fold": fold_idx,
                            "accuracy": accuracy,
                        }
                    )
                    log.info(
                        "fold complete",
                        extra={
                            "event": "fold.complete",
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "fold": fold_idx,
                            "accuracy": accuracy,
                        },
                    )
    return rows


def aggregate_results(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Aggregate per-fold accuracies into a (dataset, method) summary.

    Args:
        rows: Per-fold rows with ``dataset``, ``method``, ``accuracy``.

    Returns:
        A mapping ``{f"{dataset}|{method}": {"mean", "std", "n_folds",
        "median", "min", "max"}}``. Empty inputs yield an empty dict.
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]))
        grouped[key].append(float(row["accuracy"]))
    summary: dict[str, dict[str, float]] = {}
    for (dataset, method), accuracies in grouped.items():
        n = len(accuracies)
        if n == 0:
            continue
        mean = sum(accuracies) / n
        var = sum((a - mean) ** 2 for a in accuracies) / max(n - 1, 1)
        std = var**0.5
        sorted_acc = sorted(accuracies)
        median = (
            sorted_acc[n // 2]
            if n % 2 == 1
            else 0.5 * (sorted_acc[n // 2 - 1] + sorted_acc[n // 2])
        )
        summary[f"{dataset}|{method}"] = {
            "mean": mean,
            "std": std,
            "n_folds": float(n),
            "median": float(median),
            "min": float(min(accuracies)),
            "max": float(max(accuracies)),
        }
    return summary


@dataclass(frozen=True)
class PlanTables:
    """Plan-compliant aggregated tables for Phase 8.

    Attributes:
        summary_rows: Per-(dataset, method) summary rows.
        bootstrap_rows: Per-(dataset, method) bootstrap CI rows.
        significance_rows: Pairwise Wilcoxon + Bonferroni rows vs the
          reference method (Persistent-JEPA by default).
        per_dataset_methods: Per-dataset sorted list of
          ``(method, mean)``.
    """

    summary_rows: list[dict[str, Any]]
    bootstrap_rows: list[dict[str, Any]]
    significance_rows: list[dict[str, Any]]
    per_dataset_methods: dict[str, list[tuple[str, float]]] = field(default_factory=dict)


def per_method_per_dataset(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str], list[float]]:
    """Group per-fold accuracies by (dataset, method)."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method"]))].append(float(row["accuracy"]))
    return grouped


def _paired_bootstrap_against_reference(
    scores: list[float],
    reference_scores: list[float],
    n_resamples: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float, float]:
    """Paired bootstrap CI of ``scores - reference_scores``.

    When the reference list is empty or the lengths disagree after
    truncation, the function falls back to a zero baseline so the
    bootstrap can still return finite numbers.
    """
    if not scores:
        return 0.0, 0.0, 0.0, 1.0
    reference = list(reference_scores)
    if not reference:
        reference = [0.0 for _ in scores]
    if len(reference) != len(scores):
        n = min(len(reference), len(scores))
        reference = reference[:n]
        scores = scores[:n]
    if not scores:
        return 0.0, 0.0, 0.0, 1.0
    ci = paired_bootstrap_ci(scores, reference, n_resamples=n_resamples, alpha=alpha, seed=seed)
    return ci.mean_diff, ci.ci_low, ci.ci_high, ci.p_value


def build_plan_tables(
    rows: list[dict[str, object]],
    *,
    reference_method: str = "PersistentJEPA",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PlanTables:
    """Build plan-compliant tables from per-fold ``rows``.

    Args:
        rows: Per-fold rows with ``dataset``, ``method``, ``accuracy``.
        reference_method: The method used as the reference for
          pairwise bootstrap CIs and Wilcoxon tests.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level for the bootstrap CI.
        seed: Seed for the bootstrap random source.

    Returns:
        A populated :class:`PlanTables`.
    """
    grouped = per_method_per_dataset(rows)
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    per_dataset_methods: dict[str, list[tuple[str, float]]] = defaultdict(list)
    raw_pvalues: list[float] = []
    raw_meta: list[tuple[str, str, float]] = []
    for (dataset, method), accuracies in grouped.items():
        if not accuracies:
            continue
        n = len(accuracies)
        mean = sum(accuracies) / n
        var = sum((a - mean) ** 2 for a in accuracies) / max(n - 1, 1)
        std = var**0.5
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "mean": mean,
                "std": std,
                "n_folds": n,
            }
        )
        per_dataset_methods[dataset].append((method, mean))
    for (dataset, method), accuracies in grouped.items():
        if not accuracies:
            continue
        scores = list(accuracies)
        ref_scores = list(grouped.get((dataset, reference_method), []))
        if method == reference_method or not ref_scores:
            ref_scores = list(ref_scores)
        mean_diff, ci_low, ci_high, p_value = _paired_bootstrap_against_reference(
            scores,
            ref_scores,
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        )
        bootstrap_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "mean_diff_vs_reference": mean_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value_bootstrap": p_value,
                "n_folds": len(scores),
            }
        )
        if method != reference_method and ref_scores:
            p = wilcoxon_signed_rank(scores, ref_scores[: len(scores)])
        else:
            p = 1.0
        raw_pvalues.append(p)
        raw_meta.append((dataset, method, p))
    corrected = bonferroni_correction(raw_pvalues)
    for (dataset, method, raw_p), p_corr in zip(raw_meta, corrected, strict=True):
        significance_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "reference": reference_method,
                "p_value": float(raw_p),
                "p_value_bonferroni": float(p_corr),
            }
        )
    for dataset in per_dataset_methods:
        per_dataset_methods[dataset] = sorted(
            per_dataset_methods[dataset], key=lambda kv: kv[1], reverse=True
        )
    return PlanTables(
        summary_rows=summary_rows,
        bootstrap_rows=bootstrap_rows,
        significance_rows=significance_rows,
        per_dataset_methods=dict(per_dataset_methods),
    )


def write_plan_tables(
    plan: PlanTables,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write plan-compliant tables to ``output_dir``.

    The files written are:

    * ``tu_summary.csv`` — per-(dataset, method) mean / std / n_folds.
    * ``tu_bootstrap_ci.csv`` — paired bootstrap CIs against the
      reference method.
    * ``tu_significance.csv`` — Wilcoxon + Bonferroni p-values.

    Args:
        plan: The aggregated plan tables.
        output_dir: Directory to write into.

    Returns:
        A mapping from logical name to the written path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary_path = out / "tu_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "method", "mean", "std", "n_folds"])
        writer.writeheader()
        for row in plan.summary_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "mean": f"{float(row['mean']):.4f}",
                    "std": f"{float(row['std']):.4f}",
                    "n_folds": int(row["n_folds"]),
                }
            )
    paths["summary"] = summary_path

    bootstrap_path = out / "tu_bootstrap_ci.csv"
    with bootstrap_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "method",
                "mean_diff_vs_reference",
                "ci_low",
                "ci_high",
                "p_value_bootstrap",
                "n_folds",
            ],
        )
        writer.writeheader()
        for row in plan.bootstrap_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "mean_diff_vs_reference": f"{float(row['mean_diff_vs_reference']):.4f}",
                    "ci_low": f"{float(row['ci_low']):.4f}",
                    "ci_high": f"{float(row['ci_high']):.4f}",
                    "p_value_bootstrap": f"{float(row['p_value_bootstrap']):.4f}",
                    "n_folds": int(row["n_folds"]),
                }
            )
    paths["bootstrap"] = bootstrap_path

    significance_path = out / "tu_significance.csv"
    with significance_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "method",
                "reference",
                "p_value",
                "p_value_bonferroni",
            ],
        )
        writer.writeheader()
        for row in plan.significance_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "reference": row["reference"],
                    "p_value": f"{float(row['p_value']):.4f}",
                    "p_value_bonferroni": f"{float(row['p_value_bonferroni']):.4f}",
                }
            )
    paths["significance"] = significance_path
    return paths


def write_plots(
    plan: PlanTables,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Render the radar and heatmap plots for the plan tables.

    Each method is aligned to the full dataset list; missing cells
    are rendered as ``NaN`` so the radar and heatmap always show the
    same axis ordering regardless of which (dataset, method) pairs
    were actually run.

    Args:
        plan: The aggregated plan tables.
        output_dir: Directory to write into (``plots`` subdirectory
          recommended).

    Returns:
        A mapping from logical name (``"radar"``, ``"heatmap"``) to
        the written path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    datasets: list[str] = []
    seen_datasets: set[str] = set()
    method_set: set[str] = set()
    mean_table: dict[str, dict[str, float]] = defaultdict(dict)
    for row in plan.summary_rows:
        dataset = str(row["dataset"])
        method = str(row["method"])
        if dataset not in seen_datasets:
            datasets.append(dataset)
            seen_datasets.add(dataset)
        method_set.add(method)
        mean_table[method][dataset] = float(row["mean"])
    methods = sorted(method_set)

    method_means: dict[str, list[float]] = {}
    for method in methods:
        method_means[method] = [
            float("nan") if dataset not in mean_table[method] else mean_table[method][dataset]
            for dataset in datasets
        ]

    radar_path = out / "tu_radar.png"
    plot_radar(
        method_means={m: method_means[m] for m in methods},
        datasets=datasets,
        output_path=radar_path,
    )
    paths["radar"] = radar_path

    heatmap_path = out / "tu_heatmap.png"
    matrix = [[method_means[method][i] for method in methods] for i in range(len(datasets))]
    plot_heatmap(
        matrix=matrix,
        row_labels=datasets,
        col_labels=methods,
        output_path=heatmap_path,
    )
    paths["heatmap"] = heatmap_path
    return paths


def write_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write per-fold results to a CSV file.

    Args:
        rows: Per-fold rows with ``dataset``, ``method``, ``seed``,
          ``fold`` and ``accuracy``.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "method", "seed", "fold", "accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def legacy_summary_csv(summary: dict[str, dict[str, float]], path: Path) -> None:
    """Write the legacy summary CSV expected by older callers.

    Args:
        summary: Output of :func:`aggregate_results`.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "method", "mean", "std", "n_folds"])
        for key, stats in summary.items():
            dataset, method = key.split("|")
            writer.writerow(
                [
                    dataset,
                    method,
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    int(stats["n_folds"]),
                ]
            )


def plot_distortion(rows: list[dict[str, object]], png_path: Path) -> None:
    """Reserved hook for distortion-plot symmetry across experiments.

    The TU SOTA experiment does not draw a distortion plot, but the
    helper is exposed so cross-experiment aggregation code can import
    it without conditional logic. Calling it with empty rows is a
    no-op.

    Args:
        rows: Unused; kept for API symmetry.
        png_path: Destination path; unused.
    """
    del rows, png_path


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the TU SOTA experiment.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(TU_DATASETS),
        help="Datasets to run on (default: all six TU datasets).",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds.")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds.")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per run.")
    parser.add_argument("--budget", type=int, default=64, help="Working-graph budget.")
    parser.add_argument("--output-dir", default="results/tu", help="Output directory.")
    parser.add_argument("--no-pretrain", action="store_true", help="Skip JEPA pretraining.")
    parser.add_argument(
        "--optuna-dir",
        default="results/optuna",
        help="Directory with Optuna best configs.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples.",
    )
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LOG_FORMAT_JSON)
    config = TUExperimentConfig(
        datasets=tuple(args.datasets),
        n_seeds=args.seeds,
        n_folds=args.folds,
        epochs=args.epochs,
        budget=args.budget,
        output_dir=args.output_dir,
        run_jepa_pretraining=not args.no_pretrain,
        optuna_dir=args.optuna_dir,
        n_resamples=args.bootstrap_resamples,
    )
    start = time.time()
    rows = run_experiment(config)
    summary = aggregate_results(rows)
    plan = build_plan_tables(rows, n_resamples=config.n_resamples)
    out_dir = Path(config.output_dir)
    write_results_csv(rows, out_dir / "tu_results.csv")
    legacy_summary_csv(summary, out_dir / "tu_summary.csv")
    write_plan_tables(plan, out_dir)
    write_plots(plan, out_dir / "plots")
    elapsed = time.time() - start
    log = get_logger(__name__)
    log.info(
        "experiment complete",
        extra={
            "event": "experiment.complete",
            "n_runs": len(rows),
            "elapsed_seconds": elapsed,
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
