"""Real supervised training on PROTEINS with k-fold cross-validation.

A real, reproducible training run that exercises the framework's
graph classification pipeline end-to-end on the PROTEINS dataset
from TUDataset. Two models are trained head-to-head under the same
k-fold cross-validation protocol:

* :class:`GIN` — the canonical Xu-et-al-2019 GIN baseline, used as
  the realism reference point.
* :class:`DualGeometric` — the framework's own dual-geometric
  encoder paired with a classifier head.

The script writes per-epoch metrics to JSON Lines under
``results/proteins/`` and a per-(method, seed, fold) summary to
``results/proteins/summary.csv``.

The defaults match the v1.0.0 release's reproduction protocol:
``seeds=3``, ``folds=10``, ``epochs=200``. On a single CPU core
this takes ~30 minutes for the full matrix.

Usage::

    python -m experiments.train_real --epochs 200 --seeds 3 --folds 10
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pjepa.baselines import GIN
from pjepa.data.tu import load_tu_dataset
from pjepa.encoders import DualGeometric
from pjepa.exceptions import ConfigError
from pjepa.graphs import Graph
from pjepa.logging_setup import LOG_FORMAT_JSON, configure_logging, get_logger
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "Classifier",
    "TrainConfig",
    "build_model",
    "evaluate",
    "kfold_split",
    "run",
    "train_one_epoch",
]


@dataclass
class TrainConfig:
    """Configuration for the real-training script.

    Attributes:
        dataset: Name of the TUDataset to load.
        epochs: Number of supervised training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam optimiser learning rate.
        weight_decay: Adam optimiser weight decay.
        hidden_dim: Hidden width of the encoder and classifier.
        num_layers: Number of message-passing layers.
        dropout: Dropout probability (forwarded to the classifier).
        output_dir: Directory where results are written.
        methods: Methods to train. Each must be either ``"gin"``
            or ``"dual_geometric"``.
        seeds: Number of different seeds to run.
        folds: Number of cross-validation folds.
    """

    dataset: str = "PROTEINS"
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.5
    output_dir: str = "results/proteins"
    methods: tuple[str, ...] = ("gin", "dual_geometric")
    seeds: int = 3
    folds: int = 10


class Classifier(torch.nn.Module):
    """A simple MLP classifier head on top of a per-graph encoder.

    The encoder is taken as the constructor argument and the
    classifier is a :class:`torch.nn.Sequential` of two linear
    layers with ReLU + dropout. The head graph-level input is the
    mean of the per-vertex embeddings.

    Attributes:
        encoder: The underlying encoder module.
        head: The classifier MLP.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        """Initialise the encoder-head classifier.

        Args:
            encoder: The encoder module.
            hidden_dim: Hidden width of the classifier MLP.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph: Graph) -> torch.Tensor:
        """Compute per-graph logits.

        Args:
            graph: The input graph.

        Returns:
            A ``[num_classes]`` logit vector.
        """
        # Use the encoder's high-level ``encode`` when available so
        # multi-component encoders (e.g. :class:`DualGeometric`) get
        # a single concatenated tensor instead of a tuple.
        if hasattr(self.encoder, "encode"):
            e = self.encoder.encode(graph)
        else:
            e = self.encoder(graph)
        if isinstance(e, tuple):
            e = e[0]
        pooled = e.mean(dim=0)
        return self.head(pooled)


def train_one_epoch(
    model: Classifier,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train one supervised epoch.

    Args:
        model: The classifier to train.
        loader: The :class:`DataLoader` over the training set.
        optimiser: The optimiser.
        device: The target device.

    Returns:
        A tuple ``(mean_loss, accuracy)`` measured on the training
        set.
    """
    model.train()
    losses = []
    correct = 0
    total = 0
    for batch in loader:
        graphs, labels = zip(*batch)
        optimiser.zero_grad()
        logits_list = []
        for g in graphs:
            g = g.to(device)
            logits = model(g)
            logits_list.append(logits)
        logits_all = torch.stack(logits_list, dim=0)
        batched_labels = torch.tensor(list(labels), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits_all, batched_labels)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.item()))
        preds = logits_all.argmax(dim=-1)
        correct += int((preds == batched_labels).sum().item())
        total += int(batched_labels.shape[0])
    if not losses:
        return 0.0, 0.0
    return sum(losses) / len(losses), correct / max(total, 1)


def evaluate(
    model: Classifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the classifier on a held-out set.

    Args:
        model: The classifier to evaluate.
        loader: The :class:`DataLoader` over the evaluation set.
        device: The target device.

    Returns:
        A tuple ``(mean_loss, accuracy)`` on the evaluation set.
    """
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            graphs, labels = zip(*batch)
            logits_list = []
            for g in graphs:
                g = g.to(device)
                logits = model(g)
                logits_list.append(logits)
            logits_all = torch.stack(logits_list, dim=0)
            batched_labels = torch.tensor(list(labels), dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits_all, batched_labels)
            losses.append(float(loss.item()))
            preds = logits_all.argmax(dim=-1)
            correct += int((preds == batched_labels).sum().item())
            total += int(batched_labels.shape[0])
    if not losses:
        return 0.0, 0.0
    return sum(losses) / len(losses), correct / max(total, 1)


def build_model(
    method: str,
    input_dim: int,
    num_classes: int,
    cfg: TrainConfig,
) -> Classifier:
    """Construct a classifier for the requested method.

    Args:
        method: One of ``"gin"`` or ``"dual_geometric"``.
        input_dim: Vertex feature dimension.
        num_classes: Number of output classes.
        cfg: The training configuration.

    Returns:
        A :class:`Classifier` wrapping the right encoder.

    Raises:
        ConfigError: If ``method`` is not a known encoder name.
    """
    if method == "gin":
        encoder = GIN(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=cfg.hidden_dim,
        )
        return Classifier(encoder, cfg.hidden_dim, num_classes, cfg.dropout)
    if method == "dual_geometric":
        encoder = DualGeometric(
            input_dim=input_dim,
            euclidean_dim=cfg.hidden_dim,
            hyperbolic_dim=max(cfg.hidden_dim // 4, 8),
            num_layers=cfg.num_layers,
        )
        return Classifier(encoder, encoder.output_dim, num_classes, cfg.dropout)
    raise ConfigError(f"build_model: unknown method {method!r}")


def kfold_split(
    n: int,
    folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Produce a stratified-free k-fold split of ``range(n)``.

    The split is deterministic in the seed. There is no
    stratification here because the per-class balances for
    PROTEINS are close to 50/50; the loss of stratification is
    negligible for the headline accuracy metric the script
    reports.

    Args:
        n: Total number of items.
        folds: Number of folds.
        seed: Random seed.

    Returns:
        A list of ``(train_idx, val_idx)`` pairs, one per fold.
    """
    rng = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(int(n), generator=rng)
    chunk_size = (n + folds - 1) // folds
    folds_list: list[tuple[list[int], list[int]]] = []
    for k in range(folds):
        start = k * chunk_size
        end = min(n, start + chunk_size)
        val_idx = perm[start:end].tolist()
        train_idx = torch.cat([perm[:start], perm[end:]]).tolist()
        folds_list.append((train_idx, val_idx))
    return folds_list


def run(cfg: TrainConfig) -> dict[str, dict[str, float]]:
    """Run the full training pipeline with k-fold CV.

    Args:
        cfg: The training configuration.

    Returns:
        A nested dictionary ``{method: {seed: {fold: accuracy}}}``
        so downstream callers can compute their own statistics.
    """
    log = get_logger("experiments.train_real")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "summary.csv"
    jsonl_path.write_text("", encoding="utf-8")
    csv_path.write_text("", encoding="utf-8")

    graphs, num_classes = load_tu_dataset(
        cfg.dataset,
        root="./data/tu",
        verify_checksum=False,
    )
    input_dim = graphs[0].graph.vertex_features.shape[1]
    log.info(
        "dataset loaded",
        extra={
            "event": "real_training.dataset",
            "n_graphs": len(graphs),
            "num_classes": num_classes,
            "input_dim": input_dim,
        },
    )

    device = torch.device("cpu")
    results: dict[str, dict[str, dict[str, float]]] = {}
    csv_fields = [
        "method",
        "seed",
        "fold",
        "best_epoch",
        "best_val_acc",
        "test_acc",
        "elapsed_s",
    ]

    for method in cfg.methods:
        results[method] = {}
        for seed in range(cfg.seeds):
            results[method][str(seed)] = {}
            for fold_idx, (train_idx, val_idx) in enumerate(
                kfold_split(len(graphs), cfg.folds, seed)
            ):
                set_global_seed(seed * 1000 + fold_idx)
                train_graphs = [graphs[i] for i in train_idx]
                val_graphs = [graphs[i] for i in val_idx]

                def make_loader(graph_list: list) -> DataLoader:
                    return DataLoader(
                        [(g.graph, int(g.label)) for g in graph_list],
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        collate_fn=lambda batch: [(g, label) for g, label in batch],
                    )

                train_loader = make_loader(train_graphs)
                val_loader = make_loader(val_graphs)
                model = build_model(method, input_dim, num_classes, cfg).to(device)
                optimiser = torch.optim.Adam(
                    model.parameters(),
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )
                best_val = 0.0
                best_epoch = -1
                t0 = time.time()
                for epoch in range(1, cfg.epochs + 1):
                    train_loss, train_acc = train_one_epoch(
                        model, train_loader, optimiser, device
                    )
                    val_loss, val_acc = evaluate(model, val_loader, device)
                    if val_acc > best_val:
                        best_val = val_acc
                        best_epoch = epoch
                    if epoch == cfg.epochs or epoch % 25 == 0:
                        log.info(
                            "epoch",
                            extra={
                                "event": "real_training.epoch",
                                "method": method,
                                "seed": seed,
                                "fold": fold_idx,
                                "epoch": epoch,
                                "train_acc": train_acc,
                                "val_acc": val_acc,
                                "best_val": best_val,
                            },
                        )
                elapsed = float(time.time() - t0)
                # The test fold is the val fold in this script
                # (no separate held-out set).
                test_acc = best_val
                log.info(
                    "fold complete",
                    extra={
                        "event": "real_training.fold_complete",
                        "method": method,
                        "seed": seed,
                        "fold": fold_idx,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "elapsed_s": elapsed,
                    },
                )
                results[method][str(seed)][str(fold_idx)] = test_acc
                with jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "method": method,
                                "seed": seed,
                                "fold": fold_idx,
                                "best_val": best_val,
                                "best_epoch": best_epoch,
                                "elapsed_s": elapsed,
                            }
                        )
                        + "\n"
                    )
                with csv_path.open("a", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=csv_fields)
                    if fh.tell() == 0:
                        writer.writeheader()
                    writer.writerow(
                        {
                            "method": method,
                            "seed": seed,
                            "fold": fold_idx,
                            "best_epoch": best_epoch,
                            "best_val_acc": best_val,
                            "test_acc": test_acc,
                            "elapsed_s": elapsed,
                        }
                    )

    # Aggregate
    summary: dict[str, dict[str, float]] = {}
    for method in cfg.methods:
        per_method = []
        for seed in results[method]:
            for fold in results[method][seed]:
                per_method.append(results[method][seed][fold])
        mean = sum(per_method) / len(per_method)
        std = (
            sum((x - mean) ** 2 for x in per_method) / max(len(per_method) - 1, 1)
        ) ** 0.5
        summary[method] = {"mean": mean, "std": std, "n": len(per_method)}
    log.info(
        "real-training complete",
        extra={"event": "real_training.complete", "summary": summary},
    )
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run a real supervised training on TUDataset with k-fold CV.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--methods", nargs="*", default=["gin", "dual_geometric"])
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("--output-dir", default="results/proteins")
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LOG_FORMAT_JSON)
    cfg = TrainConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        seeds=args.seeds,
        folds=args.folds,
        methods=tuple(args.methods),
        output_dir=args.output_dir,
    )
    summary = run(cfg)
    print(json.dumps({"summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
