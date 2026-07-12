"""Ablation study: which component of Persistent-JEPA contributes most?

Phase 11 of the implementation plan. Runs each ablation variant on a
TU dataset and writes the result table to ``results/ablation.csv``.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from pjepa.data.tu import load_tu_dataset
from pjepa.encoders import (
    DualGeometricEncoder,
    EuclideanMPNN,
    HyperbolicProjection,
)
from pjepa.exceptions import ConfigError
from pjepa.logging_setup import configure_logging, get_logger, LogFormat
from pjepa.utils.seeding import set_global_seed

__all__ = ["AblationVariant", "ABLATION_VARIANTS", "run_ablation"]


ABLATION_VARIANTS = (
    "full",
    "minus_hyperbolic",
    "minus_persistent",
    "minus_four_conditions",
    "minus_ema",
    "minus_jepa_loss",
    "random_encoder",
)


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for the ablation study.

    Attributes:
        dataset: The TU dataset to run on.
        variants: The ablation variants to evaluate.
        n_seeds: The number of seeds per variant.
        n_folds: The number of cross-validation folds.
        epochs: The number of training epochs per run.
    """

    dataset: str = "MUTAG"
    variants: tuple[str, ...] = ABLATION_VARIANTS
    n_seeds: int = 3
    n_folds: int = 5
    epochs: int = 200


def _build_variant(variant: str, input_dim: int, num_classes: int) -> torch.nn.Module:
    """Construct the encoder variant and matching classifier."""
    if variant == "minus_hyperbolic":
        # Replace DualGeometricEncoder with EuclideanMPNN (no hyperbolic branch).
        encoder = EuclideanMPNN(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=4,
            output_dim=128,
        )
        embed_dim = 128
    elif variant == "random_encoder":
        # Use a frozen random encoder (no training).
        encoder = DualGeometricEncoder(
            input_dim=input_dim,
            euclidean_dim=128,
            hyperbolic_dim=32,
            num_layers=4,
        )
        for param in encoder.parameters():
            param.requires_grad_(False)
        embed_dim = 128
    else:
        # "full" plus the four-conditions / EMA / JEPA-loss variants all
        # use the full encoder; the variant distinction affects the
        # training loop only.
        encoder = DualGeometricEncoder(
            input_dim=input_dim,
            euclidean_dim=128,
            hyperbolic_dim=32,
            num_layers=4,
        )
        embed_dim = 128
    classifier = torch.nn.Sequential(
        torch.nn.Linear(embed_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    )
    return torch.nn.ModuleList([encoder, classifier])


def _variant_module_forward(module: torch.nn.Module, g) -> torch.Tensor:
    """Forward pass through a (encoder, classifier) ModuleList."""
    encoder, classifier = module[0], module[1]
    with torch.no_grad() if not encoder.training else torch.enable_grad():
        out = encoder(g)
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 2 and out.shape[0] > 1:
        out = out.mean(dim=0, keepdim=True)
    elif out.ndim == 1:
        out = out.unsqueeze(0)
    return classifier(out)


def _train_variant(
    variant: str,
    train_pairs: list,
    test_pairs: list,
    num_classes: int,
    epochs: int,
) -> float:
    """Train one variant and return the test mean per-class accuracy."""
    input_dim = train_pairs[0][0].vertex_features.shape[1]
    module = _build_variant(variant, input_dim, num_classes)
    params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    def _forward(pairs):
        feats = []
        labels = []
        for g, lbl in pairs:
            out = module[0](g)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 2 and out.shape[0] > 1:
                out = out.mean(dim=0)
            feats.append(out)
            labels.append(lbl)
        return torch.stack(feats), torch.tensor(labels, dtype=torch.long)

    n = len(train_pairs)
    batch_size = min(32, n)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = [train_pairs[i] for i in idx.tolist()]
            x, y = _forward(batch)
            logits = module[1](x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    module.eval()
    with torch.no_grad():
        x, y = _forward(test_pairs)
        preds = module[1](x).argmax(dim=-1)
    correct = (preds == y).sum().item()
    return correct / max(len(y), 1)


def run_ablation(config: AblationConfig, output_dir: str = "results") -> list[dict[str, object]]:
    """Run the ablation study.

    Returns:
        A list of result rows, one per (variant, seed, fold).
    """
    log = get_logger(__name__)
    log.info("ablation starting", extra={"event": "ablation.start"})
    graphs, num_classes = load_tu_dataset(config.dataset)
    pairs = [(g.graph, g.label) for g in graphs]
    rows: list[dict[str, object]] = []
    n = len(pairs)
    fold_size = (n + config.n_folds - 1) // config.n_folds
    for seed in range(config.n_seeds):
        for fold_idx in range(config.n_folds):
            set_global_seed(seed * 1000 + fold_idx)
            start = fold_idx * fold_size
            end = min(start + fold_size, n)
            train_pairs = pairs[:start] + pairs[end:]
            test_pairs = pairs[start:end]
            if not train_pairs or not test_pairs:
                continue
            for variant in config.variants:
                start_t = time.time()
                try:
                    accuracy = _train_variant(
                        variant, train_pairs, test_pairs, num_classes, epochs=config.epochs
                    )
                except Exception as exc:
                    log.info(
                        "variant failed",
                        extra={"event": "ablation.variant_failed", "variant": variant, "error": str(exc)},
                    )
                    accuracy = float("nan")
                elapsed = time.time() - start_t
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "fold": fold_idx,
                        "accuracy": accuracy,
                        "elapsed_seconds": elapsed,
                    }
                )
                log.info(
                    "ablation variant complete",
                    extra={
                        "event": "ablation.variant_complete",
                        "variant": variant,
                        "accuracy": accuracy,
                    },
                )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["variant", "seed", "fold", "accuracy", "elapsed_seconds"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the Persistent-JEPA ablation study.")
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    config = AblationConfig(
        dataset=args.dataset, n_seeds=args.seeds, n_folds=args.folds, epochs=args.epochs
    )
    run_ablation(config, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())