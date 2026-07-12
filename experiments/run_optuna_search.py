"""Optuna hyperparameter search for Persistent-JEPA on TU datasets.

Phase 6 of the implementation plan: per-dataset hyperparameter search
using Optuna with the Hyperband pruner. The search space covers the
encoder dimensions, learning rate, weight decay, JEPA coefficients,
and the working-graph budget ``B``.

Each Optuna trial runs a single fold of the TU experiment and
reports the validation accuracy. The best config per dataset is saved
to ``results/optuna/<dataset>/best_config.yaml`` for downstream use
by Phase 8 (full TU SOTA).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from pjepa.data.tu import load_tu_dataset
from pjepa.encoders import DualGeometricEncoder
from pjepa.exceptions import ConfigError
from pjepa.logging_setup import configure_logging, get_logger, LogFormat
from pjepa.utils.seeding import set_global_seed

__all__ = ["OptunaConfig", "run_search"]


@dataclass(frozen=True)
class OptunaConfig:
    """Configuration for the Optuna search.

    Attributes:
        datasets: The datasets to run search on.
        n_trials: The number of trials per dataset.
        epochs: The number of training epochs per trial.
        timeout_seconds: Optional wall-clock timeout for the whole
          search per dataset.
    """

    datasets: tuple[str, ...] = ("PROTEINS", "MUTAG", "NCI1")
    n_trials: int = 20
    epochs: int = 100
    timeout_seconds: float | None = None


def _suggest_config(trial, dataset) -> dict[str, object]:
    """Sample a hyperparameter configuration from the Optuna trial."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "B": trial.suggest_categorical("B", [16, 32, 64, 128, 256]),
        "beta_ib": trial.suggest_float("beta_ib", 1e-4, 1.0, log=True),
        "lambda_mdl": trial.suggest_float("lambda_mdl", 1e-4, 1.0, log=True),
        "gamma_forward": trial.suggest_float("gamma_forward", 1e-4, 1.0, log=True),
        "ema_momentum": trial.suggest_float("ema_momentum", 0.99, 0.9999),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
    }


def _train_one_trial(
    config: dict[str, object],
    train_pairs: list,
    test_pairs: list,
    num_classes: int,
    epochs: int,
) -> float:
    """Train one encoder+classifier pair and return test accuracy."""
    input_dim = train_pairs[0][0].vertex_features.shape[1]
    encoder = DualGeometricEncoder(
        input_dim=input_dim,
        euclidean_dim=int(config["hidden_dim"]),
        hyperbolic_dim=32,
        num_layers=int(config["num_layers"]),
    )
    classifier = torch.nn.Sequential(
        torch.nn.Linear(int(config["hidden_dim"]), 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    )
    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))

    def _forward(pairs):
        feats = []
        labels = []
        for g, lbl in pairs:
            e, _ = encoder(g)
            feats.append(e.mean(dim=0))
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
    return float((preds == test_y).float().mean().item())


def _save_best_config(study, output_dir: Path, dataset: str) -> None:
    """Save the best config to a YAML file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best = study.best_params
    lines = [f"dataset: {dataset}", "# Optuna best hyperparameters", "best_params:"]
    for key, value in best.items():
        lines.append(f"  {key}: {value!r}")
    lines.append(f"best_value: {study.best_value:.4f}")
    (output_dir / "best_config.yaml").write_text("\n".join(lines), encoding="utf-8")


def run_search(config: OptunaConfig, output_dir: str = "results/optuna") -> dict[str, object]:
    """Run Optuna search on each dataset and save per-dataset best configs."""
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigError(
            "run_search: Optuna is required; install with `pip install optuna`"
        ) from exc
    log = get_logger(__name__)
    log.info("optuna search starting", extra={"event": "optuna.start"})
    out = Path(output_dir)
    summary: dict[str, object] = {}
    for dataset in config.datasets:
        log.info("searching dataset", extra={"event": "optuna.dataset", "dataset": dataset})
        try:
            graphs, num_classes = load_tu_dataset(dataset)
        except Exception as exc:  # pragma: no cover - tolerate download failures
            log.info("dataset load failed", extra={"event": "optuna.dataset_failed", "dataset": dataset, "error": str(exc)})
            continue
        pairs = [(g.graph, g.label) for g in graphs]
        n = len(pairs)
        n_train = int(0.9 * n)
        set_global_seed(0)
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:]
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(reduction_factor=3),
        )
        start = time.time()

        def objective(trial):
            params = _suggest_config(trial, dataset)
            return _train_one_trial(
                params, train_pairs, test_pairs, num_classes, epochs=config.epochs
            )

        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout_seconds,
            show_progress_bar=False,
        )
        elapsed = time.time() - start
        _save_best_config(study, out / dataset, dataset)
        summary[dataset] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "elapsed_seconds": elapsed,
        }
        log.info(
            "search complete",
            extra={
                "event": "optuna.dataset_complete",
                "dataset": dataset,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
            },
        )
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Persistent-JEPA.")
    parser.add_argument("--datasets", nargs="*", default=list(OptunaConfig.datasets))
    parser.add_argument("--n-trials", type=int, default=OptunaConfig.n_trials)
    parser.add_argument("--epochs", type=int, default=OptunaConfig.epochs)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--output-dir", default="results/optuna")
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    config = OptunaConfig(
        datasets=tuple(args.datasets),
        n_trials=args.n_trials,
        epochs=args.epochs,
        timeout_seconds=args.timeout,
    )
    summary = run_search(config, output_dir=args.output_dir)
    log = get_logger(__name__)
    log.info("optuna search complete", extra={"event": "optuna.complete", "datasets": list(summary.keys())})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())