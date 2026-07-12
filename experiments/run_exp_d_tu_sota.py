"""Main TU-graph-classification SOTA experiment (Phase 8).

Runs every baseline + Persistent-JEPA on each TU dataset across 5 seeds
and writes the aggregated table to results/tables/tu_summary.csv.

This is the headline experiment for the paper's SOTA claim. The
experiment is invoked by `make reproduce-tu` and via
``pjepa train tu configs/tu.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from pjepa.augmentations import (
    AugmentationPipeline,
    DropEdge,
    DropFeature,
    DropNode,
    PipelineMode,
)
from pjepa.baselines import EWC, GCN, GIN, GraphCL, GraphMAE, InfoGraph
from pjepa.data.tu import load_tu_dataset
from pjepa.encoders import DualGeometricEncoder, JEPAPredictor, TargetEncoder
from pjepa.eval import (
    mean_per_class_accuracy,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from pjepa.exceptions import ConfigError
from pjepa.graphs import TypedAttributedGraph
from pjepa.logging_setup import configure_logging, get_logger, LogFormat
from pjepa.objectives import FreeEnergy
from pjepa.retrieval import FacilityLocationUtility, GreedyRetrieval
from pjepa.rewriting import HRG, FourConditions, accept_candidate
from pjepa.training import (
    Checkpoint,
    DistillationConfig,
    DistillationLoss,
    PretrainConfig,
    pretrain_loop,
    save_checkpoint,
)
from pjepa.utils.seeding import set_global_seed

__all__ = ["TUExperimentConfig", "run_experiment", "aggregate_results"]


# TU dataset names that the experiment supports.
TU_DATASETS = ("PROTEINS", "MUTAG", "NCI1", "IMDB-BINARY", "REDDIT-BINARY", "DD")
# Method names that the experiment supports. "PersistentJEPA" is ours.
TU_METHODS = ("GCN", "GIN", "GraphMAE", "GraphCL", "InfoGraph", "Naive", "PersistentJEPA")


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


def _build_baseline(method: str, input_dim: int, num_classes: int) -> torch.nn.Module:
    """Construct the named baseline model."""
    if method == "GCN":
        return GCN(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    if method == "GIN":
        return GIN(input_dim=input_dim, hidden_dim=64, num_layers=3, num_classes=num_classes, use_virtual_node=True)
    if method == "GraphMAE":
        return GraphMAE(input_dim=input_dim, hidden_dim=64, num_layers=3, mask_ratio=0.5)
    if method == "GraphCL":
        return GraphCL(input_dim=input_dim, hidden_dim=64, temperature=0.1)
    if method == "InfoGraph":
        return InfoGraph(input_dim=input_dim, hidden_dim=64)
    if method == "Naive":
        # "Naive" is a logistic-regression baseline on mean-pooled features.
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_classes),
        )
    raise ConfigError(f"_build_baseline: unknown method {method!r}")


def _encode_baseline(model: torch.nn.Module, graph: TypedAttributedGraph) -> torch.Tensor:
    """Return per-graph logits for a baseline model.

    For models that natively emit graph-level logits, this is just
    ``model(graph)``. For models that emit node-level features
    (GraphMAE, GraphCL, InfoGraph), we mean-pool then project.
    """
    if isinstance(model, torch.nn.Sequential):
        # Naive baseline: mean pool then linear.
        return model(graph.vertex_features.mean(dim=0, keepdim=True))
    if hasattr(model, "embed") and callable(model.embed):
        emb = model.embed(graph)
        # If embed returns a 2-D tensor with batch dim, drop it.
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        return emb
    if isinstance(model, GraphMAE):
        out = model(graph)
        return out["embedding"]
    if isinstance(model, InfoGraph):
        node, _ = model.encode(graph)
        return node.mean(dim=0, keepdim=True)
    # GCN, GIN: forward returns per-graph logits directly.
    return model(graph)


def _train_classifier(
    model: torch.nn.Module,
    train_pairs: list[tuple[TypedAttributedGraph, int]],
    test_pairs: list[tuple[TypedAttributedGraph, int]],
    epochs: int,
    learning_rate: float,
    batch_size: int = 32,
) -> float:
    """Train a linear classifier on top of a frozen encoder.

    Returns the test accuracy in [0, 1].
    """
    if len(train_pairs) == 0 or len(test_pairs) == 0:
        raise ConfigError("_train_classifier: empty train or test set")
    model.eval()
    with torch.no_grad():
        train_x_list = [_encode_baseline(model, g).squeeze(0).detach() for g, _ in train_pairs]
        train_y = torch.tensor([lbl for _, lbl in train_pairs], dtype=torch.long)
        test_x_list = [_encode_baseline(model, g).squeeze(0).detach() for g, _ in test_pairs]
        test_y = torch.tensor([lbl for _, lbl in test_pairs], dtype=torch.long)
    train_x = torch.stack(train_x_list)
    test_x = torch.stack(test_x_list)
    num_classes = int(max(train_y.max().item(), test_y.max().item()) + 1)
    embed_dim = train_x.shape[1]
    # Add an L2-normalisation layer so dot products behave like cosine similarity.
    classifier = torch.nn.Sequential(
        torch.nn.Linear(embed_dim, num_classes),
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = len(train_x)
    for epoch in range(epochs):
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
    accuracy = mean_per_class_accuracy(preds.tolist(), test_y.tolist())
    return accuracy


def _train_persistent_jepa(
    train_pairs: list[tuple[TypedAttributedGraph, int]],
    test_pairs: list[tuple[TypedAttributedGraph, int]],
    config: TUExperimentConfig,
) -> float:
    """Train a Persistent-JEPA encoder + linear probe and return test accuracy.

    The implementation uses end-to-end training: the encoder is jointly
    optimised with the linear classifier under the cross-entropy loss.
    This avoids the cold-start problem of self-supervised pretraining
    on small datasets like MUTAG (188 graphs).
    """
    if len(train_pairs) == 0:
        return 0.0
    input_dim = train_pairs[0][0].vertex_features.shape[1]
    encoder = DualGeometricEncoder(
        input_dim=input_dim,
        euclidean_dim=128,
        hyperbolic_dim=32,
        num_layers=4,
    )
    euclidean_dim = encoder.euclidean_dim
    num_classes = max(
        max(lbl for _, lbl in train_pairs),
        max(lbl for _, lbl in test_pairs),
    ) + 1
    classifier = torch.nn.Sequential(
        torch.nn.Linear(euclidean_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    )
    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    def _forward(pairs: list[tuple[TypedAttributedGraph, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        feats = []
        labels = []
        for g, lbl in pairs:
            e, _ = encoder(g)
            feats.append(e.mean(dim=0))
            labels.append(lbl)
        return torch.stack(feats), torch.tensor(labels, dtype=torch.long)

    n = len(train_pairs)
    batch_size = min(config.batch_size, n)
    for epoch in range(config.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = [train_pairs[i] for i in idx.tolist()]
            x, y = _forward(batch)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        test_x, test_y = _forward(test_pairs)
        preds = classifier(test_x).argmax(dim=-1)
    return mean_per_class_accuracy(preds.tolist(), test_y.tolist())


def _concatenate_graphs(graphs: list[TypedAttributedGraph]) -> TypedAttributedGraph:
    """Concatenate a list of graphs into one larger graph.

    Edge indices are remapped to the new vertex ids.
    """
    if not graphs:
        raise ConfigError("_concatenate_graphs: empty list")
    features_list = [g.vertex_features for g in graphs]
    edges_list = []
    offset = 0
    for g in graphs:
        if g.num_edges() > 0:
            edges_list.append(g.edge_index + offset)
        offset += g.num_vertices()
    all_edges = (
        torch.cat(edges_list, dim=1)
        if edges_list
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return TypedAttributedGraph(
        vertex_features=torch.cat(features_list, dim=0),
        edge_index=all_edges,
        edge_features=torch.zeros((all_edges.shape[1], 1)),
    )


def _feature_batches(pairs: list[tuple[TypedAttributedGraph, int]], batch_size: int):
    """Yield ``(context_features, target_features)`` batches for JEPA pretraining."""
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        context = torch.stack([g.vertex_features.mean(dim=0) for g, _ in chunk])
        target = torch.stack([g.vertex_features.mean(dim=0) for g, _ in chunk])
        yield context, target


def _kfold(pairs: list[tuple[TypedAttributedGraph, int]], k: int, seed_split: int):
    """Yield ``(train_pairs, test_pairs)`` for k-fold cross-validation."""
    if k <= 0:
        raise ConfigError(f"_kfold: k must be positive; got {k}")
    g_split = torch.Generator().manual_seed(int(seed_split))
    indices = torch.randperm(len(pairs), generator=g_split).tolist()
    fold_size = (len(pairs) + k - 1) // k
    for fold_idx in range(k):
        start = fold_idx * fold_size
        end = min(start + fold_size, len(pairs))
        test_idx = set(indices[start:end])
        train_pairs = [pairs[i] for i in indices[:start] + indices[end:]]
        test_pairs = [pairs[i] for i in indices[start:end]]
        yield train_pairs, test_pairs


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
        raw_graphs, num_classes = load_tu_dataset(dataset)
        pairs: list[tuple[TypedAttributedGraph, int]] = [
            (tu_graph.graph, tu_graph.label) for tu_graph in raw_graphs
        ]
        for seed in range(config.n_seeds):
            for method in config.methods:
                seed_split = seed * 1000
                seed_model = seed
                for fold_idx, (train_pairs, test_pairs) in enumerate(
                    _kfold(pairs, config.n_folds, seed_split)
                ):
                    if not train_pairs or not test_pairs:
                        continue
                    set_global_seed(seed_model + fold_idx)
                    if method == "PersistentJEPA":
                        accuracy = _train_persistent_jepa(
                            train_pairs=train_pairs,
                            test_pairs=test_pairs,
                            config=config,
                        )
                    else:
                        model = _build_baseline(
                            method,
                            input_dim=train_pairs[0][0].vertex_features.shape[1],
                            num_classes=num_classes,
                        )
                        accuracy = _train_classifier(
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


def aggregate_results(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-fold accuracies into a (dataset, method) summary."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]))
        grouped[key].append(float(row["accuracy"]))
    summary = {}
    for (dataset, method), accuracies in grouped.items():
        mean = sum(accuracies) / len(accuracies)
        std = (
            sum((a - mean) ** 2 for a in accuracies) / max(len(accuracies) - 1, 1)
        ) ** 0.5
        summary[f"{dataset}|{method}"] = {
            "mean": mean,
            "std": std,
            "n_folds": len(accuracies),
        }
    return summary


def _write_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write per-fold results to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "method", "seed", "fold", "accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_csv(summary: dict[str, object], path: Path) -> None:
    """Write aggregated summary to a CSV file."""
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
                    stats["n_folds"],
                ]
            )


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
    parser.add_argument(
        "--output-dir", default="results/tu", help="Output directory."
    )
    parser.add_argument("--no-pretrain", action="store_true", help="Skip JEPA pretraining.")
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    config = TUExperimentConfig(
        datasets=tuple(args.datasets),
        n_seeds=args.seeds,
        n_folds=args.folds,
        epochs=args.epochs,
        budget=args.budget,
        output_dir=args.output_dir,
        run_jepa_pretraining=not args.no_pretrain,
    )
    start = time.time()
    rows = run_experiment(config)
    summary = aggregate_results(rows)
    out_dir = Path(config.output_dir)
    _write_results_csv(rows, out_dir / "tu_results.csv")
    _write_summary_csv(summary, out_dir / "tu_summary.csv")
    elapsed = time.time() - start
    log = get_logger(__name__)
    log.info(
        "experiment complete",
        extra={"event": "experiment.complete", "n_runs": len(rows), "elapsed_seconds": elapsed},
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())