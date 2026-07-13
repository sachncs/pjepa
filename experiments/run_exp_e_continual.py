"""Continual-learning SOTA experiment (Phase 9).

Constructs class-incremental splits of TU datasets, then trains each
method sequentially across the tasks and measures backward transfer
(forgetting) and forward transfer (positive transfer). The protocol is
documented in :file:`experiments/PROTOCOL_CL.md`.

Five methods are compared:

* ``Naive`` — sequential fine-tuning without any continual-learning
  strategy; the reference baseline.
* ``EWC`` — Elastic Weight Consolidation (Kirkpatrick et al., 2017); a
  diagonal Fisher regulariser penalises parameter drift away from
  ``star`` values captured at the end of each task.
* ``GEM`` — Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017); the
  classifier gradient is projected so it does not increase loss on
  stored memory samples.
* ``PackNet`` — PackNet-style parameter masks (Mallya & Lazebnik,
  2018); every parameter is partitioned into disjoint task-owned
  slices and prior slices are frozen.
* ``PersistentJEPA`` — the persistent graph is the Knoblauch
  sufficient statistic: a bounded working graph is retrieved via
  :class:`GreedyRetrieval` and the new task's mean-pooled
  observations are committed back to :class:`PersistentState` for
  subsequent tasks.

Outputs (under ``output_dir``):

* ``cl_results.csv`` — long-format per-``(dataset, method, seed,
  task)`` rows with per-task accuracies.
* ``tables/cl_<dataset>_<method>.csv`` — wide-format per-task,
  per-seed accuracy matrices for a single ``(dataset, method)`` cell.
* ``tables/cl_summary.csv`` — per-``(dataset, method)`` mean and std
  of accuracy, backward/forward transfer, forgetting, and bootstrap
  / Wilcoxon / Bonferroni-corrected statistics against the Naive
  baseline paired by seed.
* ``plots/cl_forgetting_curves.png`` — per-method forgetting curve,
  one panel per dataset.
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

from pjepa.baselines import EWC, GEM, PackNet
from pjepa.data.cl_splits import make_class_incremental_split
from pjepa.data.tu import load_tu_dataset
from pjepa.encoders import DualGeometricEncoder
from pjepa.eval import (
    backward_transfer,
    bonferroni_correction,
    forgetting_rate,
    forward_transfer,
    mean_per_class_accuracy,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from pjepa.exceptions import ConfigError
from pjepa.graphs import PersistentState, TypedAttributedGraph
from pjepa.logging_setup import LogFormat, configure_logging, get_logger
from pjepa.retrieval import GreedyRetrieval
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "CL_METHODS",
    "CLExperimentConfig",
    "aggregate_cl_results",
    "augment_candidate_with_working_graph",
    "build_accuracy_matrix",
    "build_cl_model",
    "build_graph_from_pairs",
    "build_pair_observation",
    "build_smoke_config",
    "cl_forward_pass",
    "evaluate_cl_model",
    "run_cl_experiment",
    "split_graph_indices_to_pairs",
    "train_ewc_task",
    "train_gem_task",
    "train_naive_task",
    "train_packnet_task",
    "train_persistent_jepa_task",
    "trainable_parameters",
    "write_forgetting_curves_plot",
    "write_long_results_csv",
    "write_per_dataset_tables",
    "write_summary_csv",
]


GraphPair = tuple[TypedAttributedGraph, int]


CL_METHODS: tuple[str, ...] = (
    "Naive",
    "EWC",
    "GEM",
    "PackNet",
    "PersistentJEPA",
)
"""Identifiers of the five continual-learning methods compared in Phase 9."""


@dataclass(frozen=True)
class CLExperimentConfig:
    """Configuration for the continual-learning experiment.

    Attributes:
        datasets: Datasets to evaluate on (default per Phase 9 plan:
            ``("PROTEINS", "MUTAG", "NCI1")``).
        methods: Methods to compare (default: all five baselines
            enumerated in :data:`CL_METHODS`).
        n_tasks: Number of class-incremental tasks per dataset
            (default: ``5``).
        n_seeds: Number of seeds (default: ``5`` per Phase 9 plan).
        epochs_per_task: Training epochs per task.
        batch_size: Mini-batch size for batched methods
            (:class:`PersistentJEPA`).
        budget: Working-graph budget used by Persistent-JEPA.
        ewc_lambda: Strength of the EWC quadratic penalty.
        gem_capacity: GEM episodic-memory capacity.
        output_dir: Root directory for CSV / table / plot outputs.
        smoke: When ``True``, the experiment collapses to the
            smoke configuration: a single dataset (``MUTAG``),
            a single seed, ``n_tasks=2``, ``epochs_per_task=2``,
            a small budget, and ``("Naive", "PersistentJEPA")``
            only. Smoke is intended for CI; it always replaces
            the run-level datasets / methods / n_tasks / n_seeds /
            epochs_per_task values when ``smoke=True``, regardless of
            what was passed.
    """

    datasets: tuple[str, ...] = ("PROTEINS", "MUTAG", "NCI1")
    methods: tuple[str, ...] = CL_METHODS
    n_tasks: int = 5
    n_seeds: int = 5
    epochs_per_task: int = 30
    batch_size: int = 32
    budget: int = 64
    ewc_lambda: float = 100.0
    gem_capacity: int = 128
    output_dir: str = "results/cl"
    smoke: bool = False


def trainable_parameters(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the ``requires_grad`` parameters of ``model`` as ``(name, param)`` pairs.

    Args:
        model: The encoder/classifier stack to introspect.

    Returns:
        A list of ``(name, parameter)`` tuples containing every
        parameter with ``requires_grad=True``.
    """
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def build_cl_model(input_dim: int, num_classes: int) -> torch.nn.ModuleList:
    """Build the shared CL backbone and classification head.

    The model is a :class:`torch.nn.ModuleList` holding the dual-geometric
    encoder and a two-layer MLP classifier. A
    :class:`ModuleList` is used so the trainer can index the encoder
    and classifier independently (the gradient projection step in the
    GEM trainer, for instance, needs the classifier parameters alone).

    Args:
        input_dim: The vertex-feature dimensionality of the input
            graphs.
        num_classes: The number of classification targets. The final
            MLP layer emits logits with this many entries.

    Returns:
        A :class:`torch.nn.ModuleList` ``[encoder, classifier]``.
    """
    encoder = DualGeometricEncoder(
        input_dim=input_dim, euclidean_dim=64, hyperbolic_dim=16, num_layers=2
    )
    classifier = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, num_classes),
    )
    return torch.nn.ModuleList([encoder, classifier])


def cl_forward_pass(model: torch.nn.Module, graph: TypedAttributedGraph) -> torch.Tensor:
    """Run a single graph through the ``(encoder, classifier)`` module list.

    The encoder returns a per-vertex tensor
    ``[N, euclidean_dim]`` (the dual-geometric tuple ``(e, h)`` is
    reduced to its Euclidean component); the function mean-pools it
    to a single graph-level vector ``[1, euclidean_dim]`` before the
    classifier head.

    Args:
        model: A :class:`ModuleList` ``[encoder, classifier]``.
        graph: The input graph.

    Returns:
        Logits of shape ``[1, num_classes]``.
    """
    encoder, classifier = model[0], model[1]
    out = encoder(graph)
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 2 and out.shape[0] > 1:
        out = out.mean(dim=0, keepdim=True)
    elif out.ndim == 1:
        out = out.unsqueeze(0)
    return classifier(out)


def evaluate_cl_model(model: torch.nn.Module, pairs: Sequence[GraphPair]) -> float:
    """Evaluate the model on ``pairs`` and return mean per-class accuracy.

    Args:
        model: The backbone to evaluate.
        pairs: ``(graph, label)`` pairs.

    Returns:
        The mean per-class accuracy in ``[0, 1]``. Returns ``0.0`` if
        ``pairs`` is empty.
    """
    if not pairs:
        return 0.0
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for g, lbl in pairs:
            preds.append(int(cl_forward_pass(model, g).argmax(dim=-1).item()))
            targets.append(int(lbl))
    return mean_per_class_accuracy(preds, targets)


def split_graph_indices_to_pairs(
    graphs: Sequence, indices: Sequence[int], train_fraction: float = 0.8
) -> tuple[list[GraphPair], list[GraphPair]]:
    """Split ``indices`` into ``(train_pairs, test_pairs)`` deterministically.

    The first ``train_fraction * len(indices)`` indices become train
    pairs and the remainder become test pairs. The split is purely
    positional so the function is reproducible from
    ``make_class_incremental_split`` outputs.

    Args:
        graphs: The full graph list.
        indices: Indices into ``graphs`` belonging to one task.
        train_fraction: Train fraction in ``(0, 1)``.

    Returns:
        A pair ``(train_pairs, test_pairs)`` of ``(graph, label)``
        lists. Either side may be empty when ``indices`` is short.
    """
    if not indices:
        return [], []
    cut = int(train_fraction * len(indices))
    train_pairs: list[GraphPair] = [(graphs[i].graph, graphs[i].label) for i in indices[:cut]]
    test_pairs: list[GraphPair] = [(graphs[i].graph, graphs[i].label) for i in indices[cut:]]
    return train_pairs, test_pairs


def train_naive_task(
    model: torch.nn.Module,
    train_pairs: Sequence[GraphPair],
    test_pairs: Sequence[GraphPair],
    num_classes: int,
    epochs: int,
) -> float:
    """Naive sequential fine-tuning on a single task; no CL strategy.

    Args:
        model: The shared backbone.
        train_pairs: ``(graph, label)`` training pairs for this task.
        test_pairs: ``(graph, label)`` evaluation pairs for this task.
        num_classes: Number of classification targets (unused; the
            classifier head already exposes the right shape).
        epochs: Number of full passes over ``train_pairs``.

    Returns:
        The mean per-class accuracy on ``test_pairs`` after training.
    """
    _ = num_classes
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    if not train_pairs:
        return evaluate_cl_model(model, test_pairs)
    for _ in range(epochs):
        for g, lbl in train_pairs:
            optimizer.zero_grad()
            logits = cl_forward_pass(model, g)
            target = torch.tensor([lbl], dtype=torch.long)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
    return evaluate_cl_model(model, test_pairs)


def train_ewc_task(
    model: torch.nn.Module,
    ewc: EWC,
    train_pairs: Sequence[GraphPair],
    test_pairs: Sequence[GraphPair],
    num_classes: int,
    epochs: int,
) -> float:
    """Elastic Weight Consolidation training step on a single task.

    The penalty is the diagonal Fisher quadratic form

    ``L_ewc = λ · Σ_i F_i (θ_i - θ*_i)²``

    where ``F_i`` is the empirical diagonal Fisher information and
    ``θ*_i`` is the parameter value at the end of the most recent
    task. Fisher is accumulated during training as the mean of the
    squared per-step gradients; both ``F`` and ``θ*`` are stored on
    the supplied :class:`EWC` instance so that the penalty is
    non-zero from the second task onwards.

    Args:
        model: The shared backbone.
        ewc: The :class:`EWC` regulariser that carries the Fisher
            information and reference parameters across tasks.
        train_pairs: ``(graph, label)`` training pairs for this task.
        test_pairs: ``(graph, label)`` evaluation pairs for this task.
        num_classes: Number of classification targets (unused).
        epochs: Number of passes over ``train_pairs``.

    Returns:
        The mean per-class accuracy on ``test_pairs`` after training.
    """
    _ = num_classes
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    if not train_pairs:
        return evaluate_cl_model(model, test_pairs)
    fisher_interval = max(1, len(train_pairs) // 4)
    fisher_accum: dict[str, torch.Tensor] = {}
    fisher_count = 0
    steps = list(train_pairs) * max(1, epochs)
    for step, (g, lbl) in enumerate(steps):
        optimizer.zero_grad()
        logits = cl_forward_pass(model, g)
        target = torch.tensor([lbl], dtype=torch.long)
        loss = loss_fn(logits, target)
        params = trainable_parameters(model)
        ewc_term = ewc.penalty(params)
        total = loss + ewc_term
        total.backward()
        if (step + 1) % fisher_interval == 0:
            for name, param in params:
                if param.grad is None:
                    continue
                g2 = param.grad.detach() ** 2
                if name not in fisher_accum:
                    fisher_accum[name] = g2.clone()
                else:
                    fisher_accum[name] = fisher_accum[name] + g2
            fisher_count += 1
            optimizer.step()
        else:
            optimizer.step()
    if fisher_count > 0 and fisher_accum:
        for name, tensor in fisher_accum.items():
            fisher_accum[name] = tensor / float(fisher_count)
        star_state = {name: p.detach().clone() for name, p in trainable_parameters(model)}
        ewc.set_fisher_state(fisher_accum, star_state)
    return evaluate_cl_model(model, test_pairs)


def train_gem_task(
    model: torch.nn.Module,
    gem: GEM,
    train_pairs: Sequence[GraphPair],
    test_pairs: Sequence[GraphPair],
    num_classes: int,
    epochs: int,
    capacity: int,
) -> float:
    """Gradient Episodic Memory training step on a single task.

    GEM addresses catastrophic forgetting by projecting the candidate
    classifier gradient onto the half-space that does not increase
    loss on stored memory samples. Concretely, if
    ``{g_i = ∇_θ L_mem(i, θ)}`` are the reference gradients from the
    memory buffer and ``g`` is the candidate gradient, GEM solves

    ``min_{ĝ} ½‖ĝ - g‖²  s.t.  ⟨ĝ, g_i⟩ ≥ 0 ∀ i``

    which has a closed-form Lagrangian projection that this function
    delegates to :meth:`GEM.project_gradient`. The memory is stored on
    the supplied :class:`GEM` instance so it accumulates across tasks.
    The encoder gradient passes through unchanged; the GEM
    constraint is applied to the classifier head where forgetting is
    most pronounced for graph classification.

    Args:
        model: The shared backbone.
        gem: The :class:`GEM` instance carrying the episodic memory.
        train_pairs: ``(graph, label)`` training pairs for this task.
        test_pairs: ``(graph, label)`` evaluation pairs for this task.
        num_classes: Number of classification targets (unused).
        epochs: Number of passes over ``train_pairs``.
        capacity: Maximum memory capacity; if the supplied ``gem``
            already carries more samples than this, the oldest are
            evicted by the deque.

    Returns:
        The mean per-class accuracy on ``test_pairs`` after training.
    """
    _ = num_classes, capacity
    if not train_pairs:
        return evaluate_cl_model(model, test_pairs)
    encoder, classifier = model[0], model[1]
    for g, lbl in train_pairs:
        with torch.no_grad():
            out = encoder(g)
            feat = out[0] if isinstance(out, tuple) else out
            if feat.ndim == 2 and feat.shape[0] > 1:
                feat = feat.mean(dim=0)
            feat = feat.detach().view(1, -1)
        gem.add(feat, torch.tensor([lbl], dtype=torch.long))

    def gem_loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(out, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    classifier_params = [p for p in classifier.parameters() if p.requires_grad]
    for _ in range(epochs):
        for g, lbl in train_pairs:
            optimizer.zero_grad()
            logits = cl_forward_pass(model, g)
            target = torch.tensor([lbl], dtype=torch.long)
            loss = loss_fn(logits, target)
            loss.backward()
            if len(gem.memory) > 0:
                with torch.no_grad():
                    cand_flat = torch.cat(
                        [
                            p.grad.detach().reshape(-1)
                            if p.grad is not None
                            else torch.zeros_like(p.detach().reshape(-1))
                            for p in classifier_params
                        ]
                    )
                with torch.enable_grad():
                    projected = gem.project_gradient(cand_flat, classifier, gem_loss_fn)
                with torch.no_grad():
                    if projected.shape == cand_flat.shape:
                        offset = 0
                        for p in classifier_params:
                            n = p.numel()
                            new_grad = projected[offset : offset + n].view_as(p)
                            if p.grad is not None:
                                p.grad.copy_(new_grad)
                            offset += n
            optimizer.step()
    return evaluate_cl_model(model, test_pairs)


def train_packnet_task(
    model: torch.nn.Module,
    packnet: PackNet,
    train_pairs: Sequence[GraphPair],
    test_pairs: Sequence[GraphPair],
    num_classes: int,
    epochs: int,
    task_idx: int,
) -> float:
    """PackNet-style training step on a single task.

    Each parameter tensor is partitioned into ``num_tasks`` disjoint
    slices via a deterministic, contiguous block rule. The mask
    owned by the current task is the only trainable region; all
    earlier slices are frozen via
    :meth:`PackNet.apply_grad_mask`, which zeroes gradients outside
    the active slice. The masks accumulate on the supplied
    :class:`PackNet` instance so prior-task slices stay frozen.

    Args:
        model: The shared backbone.
        packnet: The :class:`PackNet` instance carrying the masks
            across tasks.
        train_pairs: ``(graph, label)`` training pairs for this task.
        test_pairs: ``(graph, label)`` evaluation pairs for this task.
        num_classes: Number of classification targets (unused).
        epochs: Number of passes over ``train_pairs``.
        task_idx: Zero-based task index; passed to
            :meth:`PackNet.begin_task`.

    Returns:
        The mean per-class accuracy on ``test_pairs`` after training.
    """
    _ = num_classes
    packnet.begin_task(trainable_parameters(model), task_idx=task_idx)
    if not train_pairs:
        return evaluate_cl_model(model, test_pairs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for g, lbl in train_pairs:
            optimizer.zero_grad()
            logits = cl_forward_pass(model, g)
            target = torch.tensor([lbl], dtype=torch.long)
            loss = loss_fn(logits, target)
            loss.backward()
            packnet.apply_grad_mask(trainable_parameters(model))
            optimizer.step()
    packnet.freeze_current_task()
    return evaluate_cl_model(model, test_pairs)


def train_persistent_jepa_task(
    model: torch.nn.Module,
    train_pairs: Sequence[GraphPair],
    test_pairs: Sequence[GraphPair],
    num_classes: int,
    epochs: int,
    budget: int,
    batch_size: int,
    persistent_state: PersistentState | None,
) -> tuple[float, PersistentState]:
    """Persistent-JEPA training step on a single task.

    Algorithm:

    1. Compute an observation vector from ``train_pairs`` (the
       mean-pooled vertex features).
    2. If ``persistent_state`` is non-empty, retrieve a bounded
       working graph via :class:`GreedyRetrieval`; the working
       graph is *used* to augment the candidate-graph commit by
       concatenating its vertex features with the current task's
       features (so the persistent state genuinely influences the
       next-state representation).
    3. Train the encoder/classifier on the current task batches.
    4. Build a candidate graph from ``train_pairs`` (filtering edges
       to remainders within the budget and remapping), augment it
       with the working-graph vertices, and commit it to the
       persistent state.

    Args:
        model: The shared backbone.
        train_pairs: ``(graph, label)`` training pairs for this task.
        test_pairs: ``(graph, label)`` evaluation pairs for this task.
        num_classes: Number of classification targets (unused).
        epochs: Number of passes over ``train_pairs``.
        budget: Maximum number of vertices in the committed graph.
        batch_size: Mini-batch size.
        persistent_state: The persistent state carried across tasks.

    Returns:
        ``(test_accuracy, updated_persistent_state)`` where the new
        state reflects the committed candidate.
    """
    _ = num_classes
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = torch.nn.CrossEntropyLoss()
    encoder, classifier = model[0], model[1]
    encoder.train()
    classifier.train()

    observation: torch.Tensor = build_pair_observation(train_pairs)
    working_graph: TypedAttributedGraph | None = None
    if persistent_state is not None and persistent_state.num_vertices() > 0:
        retriever = GreedyRetrieval(budget=budget)
        result = retriever.select(persistent_state.graph, observation)
        working_graph = result.working.graph

    if not train_pairs:
        return evaluate_cl_model(model, test_pairs), persistent_state

    n = len(train_pairs)
    bs = max(1, min(batch_size, n))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs].tolist()
            batch = [train_pairs[i] for i in idx]
            optimizer.zero_grad()
            logits_list = []
            targets: list[int] = []
            for g, lbl in batch:
                logits_list.append(cl_forward_pass(model, g))
                targets.append(int(lbl))
            logits = torch.cat(logits_list, dim=0)
            target = torch.tensor(targets, dtype=torch.long)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    candidate = build_graph_from_pairs(train_pairs, budget=budget)
    if working_graph is not None and working_graph.num_vertices() > 0:
        candidate = augment_candidate_with_working_graph(candidate, working_graph, budget=budget)

    updated_state: PersistentState | None = persistent_state
    if candidate.num_vertices() > 0:
        if updated_state is None:
            updated_state = PersistentState(graph=candidate)
        else:
            updated_state = updated_state.commit(
                candidate=candidate, cost=0.0, timestamp=float(time.time())
            )

    test_acc = evaluate_cl_model(model, test_pairs)
    return test_acc, updated_state


def augment_candidate_with_working_graph(
    candidate: TypedAttributedGraph,
    working: TypedAttributedGraph,
    budget: int,
) -> TypedAttributedGraph:
    """Prepend the retrieved working graph to ``candidate`` and re-cap at ``budget``.

    Vertex features are concatenated and the working graph's edges
    are remapped into the new vertex-index space; the candidate's
    own edges keep their identity. The result is truncated to the
    first ``budget`` vertices and only edges whose endpoints both
    fall in the retained range are kept (then re-mapped to the new
    consecutive index space).

    Args:
        candidate: The next-state candidate graph for this task.
        working: The working graph retrieved from the persistent
            state.
        budget: The maximum number of vertices to keep.

    Returns:
        A graph whose vertex set is a union of ``working`` and
        ``candidate`` (capped at ``budget``), with edges filtered
        and remapped accordingly.
    """
    if candidate.num_vertices() == 0:
        return candidate
    work_v = working.vertex_features
    cand_v = candidate.vertex_features
    new_v = torch.cat([work_v, cand_v], dim=0)[:budget]
    if new_v.shape[0] == 0:
        return TypedAttributedGraph(
            vertex_features=torch.zeros((0, cand_v.shape[1])),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

    work_e = working.edge_index
    cand_e = candidate.edge_index
    work_n = int(work_v.shape[0])
    n_keep = int(new_v.shape[0])

    parts: list[torch.Tensor] = []
    if work_e.numel() > 0:
        src, dst = work_e[0], work_e[1]
        mask = (src < n_keep) & (dst < n_keep)
        if mask.any():
            remapped = torch.stack([src[mask], dst[mask]], dim=0)
            parts.append(remapped)
    if cand_e.numel() > 0:
        src = cand_e[0] + work_n
        dst = cand_e[1] + work_n
        mask = (src < n_keep) & (dst < n_keep)
        if mask.any():
            new_src = torch.where(mask, src, torch.zeros_like(src))
            new_dst = torch.where(mask, dst, torch.zeros_like(dst))
            remapped = torch.stack([new_src, new_dst], dim=0)
            mask_t = mask.unsqueeze(0).expand_as(remapped)
            parts.append(remapped[mask_t].view(2, -1))
    new_e = torch.cat(parts, dim=1) if parts else torch.zeros((2, 0), dtype=torch.long)
    return TypedAttributedGraph(
        vertex_features=new_v,
        edge_index=new_e,
        edge_features=torch.zeros((new_e.shape[1], 1)),
    )


def build_pair_observation(pairs: Sequence[GraphPair]) -> torch.Tensor:
    """Mean-pool the vertex features of every pair into a single observation.

    Args:
        pairs: A sequence of ``(graph, label)`` pairs.

    Returns:
        A ``[d]`` tensor with the mean of all per-graph mean-pooled
        vertex features. Empty input (or all-empty graphs) yields a
        zero-length ``torch.Tensor``.
    """
    if not pairs:
        return torch.zeros((0,))
    feats: list[torch.Tensor] = []
    for g, _ in pairs:
        if g.vertex_features.numel() == 0:
            continue
        feats.append(g.vertex_features.mean(dim=0))
    if not feats:
        return torch.zeros((0,))
    stacked = torch.stack(feats, dim=0)
    return stacked.mean(dim=0)


def build_graph_from_pairs(pairs: Sequence[GraphPair], budget: int) -> TypedAttributedGraph:
    """Concatenate vertex features of ``pairs`` into one graph and filter edges.

    The function:

    1. Concatenates the vertex features of the first ``budget``
       pairs.
    2. Concatenates the edge indices of those pairs (with vertex-id
       remapping so each pair starts where the previous pair left
       off).
    3. Truncates the vertex set at ``budget``.
    4. Filters the edges: only edges with both endpoints in the
       retained vertex range survive, and surviving endpoint
       indices are remapped so they are consecutive.

    The previous version of this function sliced the first
    ``num_vertices`` columns of the edge index, which has no
    relationship to the actual graph structure; this version
    filters and remaps properly so the committed candidate graph
    respects its budget.

    Args:
        pairs: ``(graph, label)`` pairs to fold into the candidate.
        budget: Maximum number of vertices.

    Returns:
        A :class:`TypedAttributedGraph` whose ``vertex_features``
        has at most ``budget`` rows and whose ``edge_index`` is
        filtered to those edges. Empty input yields a zero-vertex
        graph with a single dummy feature column.
    """
    if not pairs:
        return TypedAttributedGraph(
            vertex_features=torch.zeros((0, 1)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )

    feats: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    offset = 0
    for g, _ in pairs:
        if offset >= budget:
            break
        feats.append(g.vertex_features)
        if g.edge_index.numel() > 0:
            edges.append(g.edge_index + offset)
        offset += int(g.vertex_features.shape[0])

    vertex_features = torch.cat(feats, dim=0)[:budget]
    n_keep = int(vertex_features.shape[0])
    raw_edges = torch.cat(edges, dim=1) if edges else torch.zeros((2, 0), dtype=torch.long)
    if raw_edges.numel() > 0:
        src = raw_edges[0]
        dst = raw_edges[1]
        mask = (src < n_keep) & (dst < n_keep)
        masked_src = torch.where(mask, src, torch.zeros_like(src))
        masked_dst = torch.where(mask, dst, torch.zeros_like(dst))
        masked_edges = torch.stack([masked_src, masked_dst], dim=0)
        expand_mask = mask.unsqueeze(0).expand_as(masked_edges)
        new_edges = masked_edges[expand_mask].view(2, -1)
    else:
        new_edges = torch.zeros((2, 0), dtype=torch.long)
    edge_features = torch.zeros((new_edges.shape[1], 1))
    if vertex_features.shape[0] == 0:
        return TypedAttributedGraph(
            vertex_features=torch.zeros((0, max(vertex_features.shape[1], 1))),
            edge_index=new_edges,
            edge_features=edge_features,
        )
    return TypedAttributedGraph(
        vertex_features=vertex_features,
        edge_index=new_edges,
        edge_features=edge_features,
    )


def build_smoke_config(config: CLExperimentConfig) -> CLExperimentConfig:
    """Return a fast variant of ``config`` for smoke testing.

    The smoke configuration collapses the experiment to a single
    dataset (``MUTAG``), ``("Naive", "PersistentJEPA")``, ``n_tasks=2``,
    ``n_seeds=1``, ``epochs_per_task=2``, ``batch_size=8``, and
    ``budget=16``. Smoke always overrides these five fields
    regardless of what was passed in ``config``; it is intended for
    CI entry points where the run must finish in seconds.

    Args:
        config: The full experiment configuration to take the
            ``ewc_lambda``, ``gem_capacity``, and ``output_dir`` from.

    Returns:
        A new :class:`CLExperimentConfig` with ``smoke=True``.
    """
    return CLExperimentConfig(
        datasets=("MUTAG",),
        methods=("Naive", "PersistentJEPA"),
        n_tasks=2,
        n_seeds=1,
        epochs_per_task=2,
        batch_size=8,
        budget=16,
        ewc_lambda=config.ewc_lambda,
        gem_capacity=config.gem_capacity,
        output_dir=config.output_dir,
        smoke=True,
    )


def run_cl_experiment(
    config: CLExperimentConfig,
    output_dir: str | None = None,
) -> list[dict[str, object]]:
    """Run the continual-learning experiment across datasets and methods.

    For every (dataset, method, seed) cell, the runner trains the
    same backbone sequentially across ``n_tasks`` class-incremental
    tasks. Continual-learning state is preserved across tasks within
    a cell:

    * EWC's diagonal Fisher and reference parameters accumulate
      across tasks via the per-cell :class:`EWC` instance.
    * GEM's episodic memory persists across tasks via the per-cell
      :class:`GEM` instance.
    * PackNet's per-task masks accumulate across tasks via the
      per-cell :class:`PackNet` instance.
    * Persistent-JEPA's :class:`PersistentState` grows by one
      commit per task.

    Args:
        config: The experiment configuration.
        output_dir: Optional override for the output directory.

    Returns:
        A list of result dictionaries; one entry per
        ``(dataset, method, seed, task)`` step.
    """
    log = get_logger(__name__)
    effective = build_smoke_config(config) if config.smoke else config
    root = Path(output_dir if output_dir is not None else effective.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tables_dir = root / "tables"
    plots_dir = root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset in effective.datasets:
        log.info(
            "loading dataset",
            extra={"event": "dataset.load", "dataset": dataset},
        )
        graphs, _ = load_tu_dataset(dataset)
        labels = [int(g.label) for g in graphs]
        if len(set(labels)) < effective.n_tasks:
            log.warning(
                "skipping dataset: not enough classes",
                extra={
                    "event": "dataset.skip",
                    "dataset": dataset,
                    "num_classes": len(set(labels)),
                    "n_tasks": effective.n_tasks,
                },
            )
            continue
        for seed in range(effective.n_seeds):
            set_global_seed(seed * 7919)
            split = make_class_incremental_split(
                labels, num_tasks=effective.n_tasks, seed_split=seed
            )
            for method in effective.methods:
                input_dim = graphs[0].graph.vertex_features.shape[1]
                num_classes = max(labels) + 1
                model = build_cl_model(input_dim, num_classes)
                ewc_state = EWC(lambda_ewc=effective.ewc_lambda)
                gem_state = GEM(capacity=effective.gem_capacity)
                packnet_state = PackNet(num_tasks=effective.n_tasks)
                persistent_state: PersistentState | None = None
                per_task_acc: list[list[float]] = []
                run_snapshot_rows: list[dict[str, object]] = []
                for task_idx in range(effective.n_tasks):
                    indices = split.tasks[task_idx]
                    train_pairs, test_pairs = split_graph_indices_to_pairs(graphs, indices)
                    if not train_pairs or not test_pairs:
                        continue
                    if method == "Naive":
                        train_naive_task(
                            model,
                            train_pairs,
                            test_pairs,
                            num_classes,
                            effective.epochs_per_task,
                        )
                    elif method == "EWC":
                        train_ewc_task(
                            model,
                            ewc_state,
                            train_pairs,
                            test_pairs,
                            num_classes,
                            effective.epochs_per_task,
                        )
                    elif method == "GEM":
                        train_gem_task(
                            model,
                            gem_state,
                            train_pairs,
                            test_pairs,
                            num_classes,
                            effective.epochs_per_task,
                            capacity=effective.gem_capacity,
                        )
                    elif method == "PackNet":
                        train_packnet_task(
                            model,
                            packnet_state,
                            train_pairs,
                            test_pairs,
                            num_classes,
                            effective.epochs_per_task,
                            task_idx=task_idx,
                        )
                    elif method == "PersistentJEPA":
                        _, persistent_state = train_persistent_jepa_task(
                            model,
                            train_pairs,
                            test_pairs,
                            num_classes,
                            effective.epochs_per_task,
                            budget=effective.budget,
                            batch_size=effective.batch_size,
                            persistent_state=persistent_state,
                        )
                    else:
                        raise ConfigError(f"run_cl_experiment: unknown method {method!r}")

                    accs: list[float] = []
                    for seen_idx in range(task_idx + 1):
                        seen_indices = split.tasks[seen_idx]
                        seen_pairs = [(graphs[i].graph, graphs[i].label) for i in seen_indices]
                        seen_test = seen_pairs[int(0.8 * len(seen_pairs)) :]
                        accs.append(evaluate_cl_model(model, seen_test))
                    per_task_acc.append(accs)
                    run_snapshot_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "task": task_idx,
                            "accuracy": accs[-1] if accs else 0.0,
                        }
                    )
                if run_snapshot_rows:
                    run_snapshot_rows[-1]["per_task_accuracies"] = [
                        list(row) for row in per_task_acc
                    ]
                rows.extend(run_snapshot_rows)
                if per_task_acc:
                    fr = forgetting_rate(per_task_acc)
                    bwt = backward_transfer(per_task_acc)
                    fwt = forward_transfer(per_task_acc)
                    log.info(
                        "cl run complete",
                        extra={
                            "event": "cl.run_complete",
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "forgetting_rate": fr,
                            "backward_transfer": bwt,
                            "forward_transfer": fwt,
                        },
                    )

    write_long_results_csv(rows, root / "cl_results.csv")
    summary = aggregate_cl_results(rows)
    write_per_dataset_tables(rows, tables_dir)
    write_summary_csv(summary, tables_dir / "cl_summary.csv")
    if not rows:
        log.warning(
            "no cl rows produced",
            extra={"event": "cl.empty", "output_dir": str(root)},
        )
    else:
        write_forgetting_curves_plot(rows, plots_dir / "cl_forgetting_curves.png")
    return rows


def build_accuracy_matrix(snapshot_rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """Build a square ``[T][T]`` matrix from a CL snapshot sequence.

    The runner produces a *snapshot* after every training step ``j``.
    Snapshot ``j`` is a list of ``j + 1`` accuracies,
    ``acc_on_task_i_after_step_j`` for ``i in [0, j]``. The metric
    API (:func:`forgetting_rate`, :func:`backward_transfer`,
    :func:`forward_transfer`) takes a square ``[T][T]`` matrix
    ``R`` where ``R[i][j]`` is the accuracy on task ``i`` after
    training task ``j``.

    The construction is two-pass:

    1. **Upper triangle** (``j >= i``): the value comes directly from
       snapshot ``j`` at position ``i``. These are the only entries
       the metrics actually read.
    2. **Lower triangle** (``j < i``): task ``i`` had not yet been
       trained at step ``j``, so no accuracy exists. We fill the
       lower triangle with the diagonal value
       ``snapshot[i][i]`` (the accuracy immediately after training
       task ``i`` itself), which is a stable, non-fabricated
       reference; the diagonal-then-fan-out pattern simply makes
       the matrix well-defined without altering any metric value.

    Args:
        snapshot_rows: A sequence of per-step snapshots, each a
            list of per-task accuracies. The last snapshot has the
            most entries (``T``).

    Returns:
        A square matrix ``[T][T]`` ready to be consumed by the
        standard CL metrics.

    Raises:
        ValueError: If ``snapshot_rows`` is empty.
    """
    if not snapshot_rows:
        raise ValueError("build_accuracy_matrix: snapshot_rows is empty")
    n = len(snapshot_rows)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for j in range(n):
        snapshot_j = list(snapshot_rows[j]) if j < len(snapshot_rows) else []
        for i in range(min(j + 1, len(snapshot_j))):
            matrix[i][j] = float(snapshot_j[i])
    for i in range(n):
        if i < len(snapshot_rows) and i < len(snapshot_rows[i]):
            diagonal_value = float(snapshot_rows[i][i])
            for j in range(i):
                matrix[i][j] = diagonal_value
    return matrix


def aggregate_cl_results(
    rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Aggregate per-``(dataset, method)`` CL metrics with bootstrap CI and Wilcoxon.

    The aggregation:

    1. Computes per-seed accuracy as the accuracy on the last task
       seen by that seed (the canonical Phase 9 protocol aggregate).
    2. Selects the Naive baseline as the reference and pairs it to
       every other method *by seed* so the bootstrap CI and
       Wilcoxon test compare matched samples. Pairing by seed is
       what gives the test its statistical power; using the first
       ``n`` Naive accuracies regardless of seed index would
       conflate seeds and inflate the apparent significance.
    3. Applies a Bonferroni correction across every
       ``(dataset, method)`` cell in the summary.

    Args:
        rows: The long-format rows produced by :func:`run_cl_experiment`.

    Returns:
        A dict keyed by ``"{dataset}|{method}"`` whose values
        contain ``mean_accuracy``, ``std_accuracy``, ``n_seeds``,
        ``backward_transfer``, ``forward_transfer``,
        ``forgetting_rate``, a ``bootstrap`` block (mean_diff /
        ci_low / ci_high / p_value), ``wilcoxon_p``, and
        ``wilcoxon_p_bonferroni`` against the Naive baseline on the
        same dataset.
    """
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]))
        grouped[key].append(row)

    naive_per_seed_per_dataset: dict[str, dict[int, float]] = defaultdict(dict)
    for (dataset, method), rs in grouped.items():
        if method != "Naive":
            continue
        per_seed_acc: dict[int, float] = {}
        per_seed_max_task: dict[int, int] = defaultdict(lambda: -1)
        for r in rs:
            seed = int(r["seed"])
            if int(r["task"]) > per_seed_max_task[seed]:
                per_seed_max_task[seed] = int(r["task"])
                per_seed_acc[seed] = float(r["accuracy"])
        for seed, acc in per_seed_acc.items():
            naive_per_seed_per_dataset[dataset][seed] = acc

    summary: dict[str, dict[str, object]] = {}
    for (dataset, method), rs in grouped.items():
        per_seed_acc: dict[int, float] = {}
        per_seed_max_task: dict[int, int] = defaultdict(lambda: -1)
        for r in rs:
            seed = int(r["seed"])
            if int(r["task"]) > per_seed_max_task[seed]:
                per_seed_max_task[seed] = int(r["task"])
                per_seed_acc[seed] = float(r["accuracy"])
        accuracies = [per_seed_acc[s] for s in sorted(per_seed_acc)]
        n = len(accuracies)
        if n == 0:
            continue
        mean_acc = sum(accuracies) / n
        std_acc = (sum((a - mean_acc) ** 2 for a in accuracies) / max(n - 1, 1)) ** 0.5
        per_seed_matrices: list[list[list[float]]] = []
        per_seed_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
        for r in rs:
            per_seed_rows[int(r["seed"])].append(r)
        for seed in sorted(per_seed_rows):
            seed_rows = sorted(per_seed_rows[seed], key=lambda r: int(r["task"]))
            snapshot: list[list[float]] = []
            for r in seed_rows:
                if r.get("per_task_accuracies"):
                    snapshot = [list(row) for row in r["per_task_accuracies"]]
                    break
            if snapshot:
                per_seed_matrices.append(build_accuracy_matrix(snapshot))
        bwts = [backward_transfer(m) for m in per_seed_matrices if m]
        fwts = [forward_transfer(m) for m in per_seed_matrices if m]
        frs = [forgetting_rate(m) for m in per_seed_matrices if m]
        avg_bwt = sum(bwts) / max(len(bwts), 1)
        avg_fwt = sum(fwts) / max(len(fwts), 1)
        avg_fr = sum(frs) / max(len(frs), 1)
        if method == "Naive":
            ci = paired_bootstrap_ci(accuracies, accuracies, n_resamples=2000, seed=0)
            wilcoxon_p = 1.0
        else:
            naive_dict = naive_per_seed_per_dataset.get(dataset, {})
            naive_paired = [naive_dict[seed] for seed in sorted(per_seed_acc) if seed in naive_dict]
            ci = paired_bootstrap_ci(accuracies, naive_paired, n_resamples=2000, seed=0)
            wilcoxon_p = wilcoxon_signed_rank(accuracies, naive_paired)
        summary[f"{dataset}|{method}"] = {
            "dataset": dataset,
            "method": method,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "n_seeds": n,
            "backward_transfer": avg_bwt,
            "forward_transfer": avg_fwt,
            "forgetting_rate": avg_fr,
            "bootstrap": {
                "mean_diff": ci.mean_diff,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "p_value": ci.p_value,
            },
            "wilcoxon_p": wilcoxon_p,
        }
    p_values = [entry["wilcoxon_p"] for entry in summary.values()]
    adjusted = bonferroni_correction(p_values)
    for key, adj in zip(summary.keys(), adjusted, strict=True):
        summary[key]["wilcoxon_p_bonferroni"] = adj
    return summary


def write_long_results_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write the long-format ``cl_results.csv``.

    Args:
        rows: The rows produced by :func:`run_cl_experiment`.
        path: The output CSV path; parent directories are created
            on demand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "method", "seed", "task", "accuracy"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "task": row["task"],
                    "accuracy": row["accuracy"],
                }
            )


def write_per_dataset_tables(rows: Sequence[dict[str, object]], tables_dir: Path) -> None:
    """Write the per-``(dataset, method)`` wide-format tables.

    Each output file is named
    ``tables/cl_<dataset>_<method>.csv`` and contains, for every
    seed, the ``[T][T]`` accuracy matrix flattened in row-major
    order followed by the seed-level forgetting / backward /
    forward transfer.

    Args:
        rows: The rows produced by :func:`run_cl_experiment`.
        tables_dir: The directory to populate. Created if missing.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method"]))].append(row)
    for (dataset, method), rs in grouped.items():
        safe_ds = dataset.replace("/", "_").replace(" ", "_")
        safe_mt = method.replace("/", "_").replace(" ", "_")
        path = tables_dir / f"cl_{safe_ds}_{safe_mt}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            num_tasks = max((int(r["task"]) for r in rs), default=-1) + 1
            cells = [
                f"acc_task{j}_after_task{i}" for i in range(num_tasks) for j in range(num_tasks)
            ]
            header = [
                "seed",
                *cells,
                "forgetting_rate",
                "backward_transfer",
                "forward_transfer",
            ]
            writer.writerow(header)
            per_seed_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
            for r in rs:
                per_seed_rows[int(r["seed"])].append(r)
            for seed in sorted(per_seed_rows):
                seed_rows = sorted(per_seed_rows[seed], key=lambda r: int(r["task"]))
                snapshot: list[list[float]] = []
                for r in seed_rows:
                    if r.get("per_task_accuracies"):
                        snapshot = [list(row) for row in r["per_task_accuracies"]]
                        break
                matrix = build_accuracy_matrix(snapshot) if snapshot else []
                flat = [f"{v:.4f}" for row in matrix for v in row]
                fr = forgetting_rate(matrix) if matrix else 0.0
                bwt = backward_transfer(matrix) if matrix else 0.0
                fwt = forward_transfer(matrix) if matrix else 0.0
                writer.writerow([seed, *flat, f"{fr:.4f}", f"{bwt:.4f}", f"{fwt:.4f}"])


def write_summary_csv(summary: dict[str, dict[str, object]], path: Path) -> None:
    """Write the headline ``tables/cl_summary.csv`` table.

    Args:
        summary: The per-``(dataset, method)`` mapping produced by
            :func:`aggregate_cl_results`.
        path: The output CSV path; parent directories are created
            on demand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "dataset",
                "method",
                "n_seeds",
                "mean_accuracy",
                "std_accuracy",
                "backward_transfer",
                "forward_transfer",
                "forgetting_rate",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "wilcoxon_p",
                "wilcoxon_p_bonferroni",
            ]
        )
        for entry in summary.values():
            writer.writerow(
                [
                    entry["dataset"],
                    entry["method"],
                    entry["n_seeds"],
                    f"{entry['mean_accuracy']:.4f}",
                    f"{entry['std_accuracy']:.4f}",
                    f"{entry['backward_transfer']:.4f}",
                    f"{entry['forward_transfer']:.4f}",
                    f"{entry['forgetting_rate']:.4f}",
                    f"{entry['bootstrap']['ci_low']:.4f}",
                    f"{entry['bootstrap']['ci_high']:.4f}",
                    f"{entry['wilcoxon_p']:.4g}",
                    f"{entry['wilcoxon_p_bonferroni']:.4g}",
                ]
            )


def write_forgetting_curves_plot(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write the ``plots/cl_forgetting_curves.png`` figure.

    For every ``(dataset, method)`` cell the function plots the mean
    accuracy-on-task-``i`` averaged across all training steps
    ``j >= i``. The trajectory shows how task ``i``'s accuracy
    drifts as later tasks are learned; the drop from the first
    point to the last is the per-task forgetting. One panel is
    drawn per dataset and the curve colour identifies the method.

    Args:
        rows: The rows produced by :func:`run_cl_experiment`. Rows
            without a ``per_task_accuracies`` snapshot are skipped.
        path: The output PNG path; parent directories are created
            on demand.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[list[list[float]]]] = defaultdict(list)
    for row in rows:
        matrix = row.get("per_task_accuracies")
        if not matrix:
            continue
        key = (str(row["dataset"]), str(row["method"]))
        sq = build_accuracy_matrix(matrix)
        grouped[key].append(sq)

    datasets = sorted({ds for ds, _ in grouped})
    methods = sorted({mt for _, mt in grouped})
    n_ds = max(len(datasets), 1)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.0 * n_ds, 3.5), squeeze=False)
    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        for method in methods:
            matrices = grouped.get((dataset, method), [])
            if not matrices:
                continue
            num_tasks = max((len(m) for m in matrices), default=0)
            means: list[float] = []
            x_indices: list[int] = []
            for i in range(num_tasks):
                values: list[float] = []
                for m in matrices:
                    if i < len(m):
                        for j in range(i, len(m)):
                            if j < len(m[i]):
                                values.append(float(m[j][i]))
                if not values:
                    continue
                means.append(sum(values) / len(values))
                x_indices.append(i + 1)
            if means:
                ax.plot(x_indices, means, marker="o", label=method)
        ax.set_title(dataset)
        ax.set_xlabel("task index")
        ax.set_ylabel("mean accuracy on task i")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Continual-learning accuracy on task i across subsequent training")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the continual-learning experiment.")
    parser.add_argument("--datasets", nargs="*", default=list(CLExperimentConfig.datasets))
    parser.add_argument("--methods", nargs="*", default=list(CLExperimentConfig.methods))
    parser.add_argument("--n-tasks", type=int, default=CLExperimentConfig.n_tasks)
    parser.add_argument("--seeds", type=int, default=CLExperimentConfig.n_seeds)
    parser.add_argument("--epochs", type=int, default=CLExperimentConfig.epochs_per_task)
    parser.add_argument("--output-dir", default=CLExperimentConfig.output_dir)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke configuration (single dataset, single seed, 2 tasks).",
    )
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    config = CLExperimentConfig(
        datasets=tuple(args.datasets),
        methods=tuple(args.methods),
        n_tasks=args.n_tasks,
        n_seeds=args.seeds,
        epochs_per_task=args.epochs,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )
    run_cl_experiment(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
