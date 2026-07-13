"""Optuna hyperparameter search for Persistent-JEPA on TU datasets.

Phase 6 of the implementation plan: per-dataset hyperparameter search
using Optuna with the Hyperband pruner. The search space covers the
encoder dimensions, learning rate, weight decay, JEPA coefficients,
the working-graph budget ``B``, and the augmentation family.

Implementation layout:

* The canonical implementation lives in
  :mod:`pjepa.training.optuna_search`. That module provides the
  :class:`OptunaSearch` runner, the SQLite-backed study
  construction with WAL journaling, the
  :func:`suggest_hyperparameters` search-space sampler, and the
  :func:`build_augmentation_from_name` augmentation factory.
* This module is a thin CLI on top of the package implementation so
  the experiment runner can be invoked via ``python
  experiments/run_optuna_search.py``. It re-exports the public
  names :func:`suggest_config`, :func:`train_one_trial`, and
  :func:`save_best_config` (formerly ``_suggest_config``,
  ``_train_one_trial``, and ``_save_best_config``) so existing
  callers continue to work.

The Hyperband pruning strategy and the search-space sampling are
performed inside :class:`OptunaSearch` and are not duplicated here;
see :mod:`pjepa.training.optuna_search` for the full implementation
details.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pjepa.data.tu import load_tu_dataset
from pjepa.logging_setup import LogFormat, configure_logging, get_logger
from pjepa.training.optuna_search import (
    OPTUNA_SEARCH_SPACE,
    OptunaSearch,
    OptunaSearchConfig,
    build_augmentation_from_name,
    suggest_hyperparameters,
)

__all__ = [
    "OPTUNA_SEARCH_SPACE",
    "OptunaConfig",
    "build_augmentation_from_name",
    "run_search",
    "save_best_config",
    "suggest_config",
    "train_one_trial",
]


class OptunaConfig:
    """Backwards-compatible CLI configuration for the Optuna search.

    Attributes:
        datasets: The dataset names to run search on. The default is
            the Phase 6 plan's TU trio: ``("PROTEINS", "MUTAG",
            "NCI1")``.
        n_trials: The number of trials per dataset.
        epochs: The number of training epochs per trial.
        timeout_seconds: Optional wall-clock timeout for the whole
            search per dataset; ``None`` disables the timeout.
    """

    def __init__(
        self,
        datasets: Sequence[str] = ("PROTEINS", "MUTAG", "NCI1"),
        n_trials: int = 20,
        epochs: int = 100,
        timeout_seconds: float | None = None,
    ) -> None:
        """Store the Optuna search parameters.

        Args:
            datasets: Dataset names. Stored as a tuple.
            n_trials: Integer number of trials per dataset.
            epochs: Integer number of training epochs per trial.
            timeout_seconds: Wall-clock timeout in seconds, or
                ``None`` for no timeout.
        """
        self.datasets = tuple(datasets)
        self.n_trials = int(n_trials)
        self.epochs = int(epochs)
        self.timeout_seconds = timeout_seconds


def suggest_config(trial: Any, dataset: str) -> dict[str, Any]:
    """Sample a hyperparameter configuration from ``trial``.

    Thin wrapper around :func:`suggest_hyperparameters` kept for
    backwards compatibility. ``dataset`` is accepted for signature
    symmetry but is not used; the search space is shared across
    datasets and is parameterised only by the Optuna trial.

    Args:
        trial: An Optuna trial object that exposes ``suggest_*``
            methods.
        dataset: Dataset name (currently unused; retained for API
            stability).

    Returns:
        A dictionary mapping every parameter name in
        :data:`OPTUNA_SEARCH_SPACE` to a sampled value.
    """
    _ = dataset
    return suggest_hyperparameters(trial, OPTUNA_SEARCH_SPACE)


def train_one_trial(
    config: dict[str, Any],
    train_pairs: Sequence[Any],
    test_pairs: Sequence[Any],
    num_classes: int,
    epochs: int,
) -> float:
    """Train a single (encoder, classifier) for ``epochs`` and report test accuracy.

    This is the CLI-friendly entry point that wraps the package's
    :class:`OptunaSearch`. It instantiates a one-trial Optuna search
    so the same :meth:`OptunaSearch.evaluate` path used for the full
    study is exercised (including ``optim.AdamW``, the cosine
    schedule, and label smoothing).

    Args:
        config: Hyperparameter dictionary. Must contain the keys
            referenced by :meth:`OptunaSearch.evaluate` (``lr``,
            ``weight_decay``, ``hidden_dim``, ``num_layers``, and
            ``augmentation``); ``label_smoothing`` defaults to
            ``0.0``.
        train_pairs: Training ``(graph, label)`` pairs.
        test_pairs: Test ``(graph, label)`` pairs.
        num_classes: Number of classification targets.
        epochs: Number of training epochs.

    Returns:
        The mean per-class accuracy on ``test_pairs`` after training,
        in ``[0, 1]``.
    """
    search = OptunaSearch(OptunaSearchConfig(n_trials=1, epochs=epochs))
    return float(
        search.evaluate(
            params=dict(config),
            train_pairs=list(train_pairs),
            test_pairs=list(test_pairs),
            num_classes=int(num_classes),
            augmentation=build_augmentation_from_name(str(config.get("augmentation", "none"))),
        )
    )


def save_best_config(study: Any, output_dir: Path, dataset: str) -> Path:
    """Persist the best hyperparameters from ``study`` to disk.

    The ``output_dir`` argument is forwarded as
    :attr:`OptunaSearchConfig.storage_path`; the
    :meth:`OptunaSearch.save_best_config` method then writes the
    best hyperparameters to ``<storage_path>/<dataset>/best_config.yaml``.

    Args:
        study: An Optuna study object (post-optimisation).
        output_dir: Base directory whose ``dataset`` sub-directory
            receives the best-config YAML.
        dataset: Dataset name; used both as a sub-directory name and
            inside the YAML payload.

    Returns:
        The :class:`pathlib.Path` of the YAML file written.
    """
    search = OptunaSearch(OptunaSearchConfig(storage_path=str(output_dir)))
    return search.save_best_config(study, dataset)


def run_search(
    config: OptunaConfig, output_dir: str = "results/optuna"
) -> dict[str, dict[str, Any]]:
    """Run the Optuna search on every dataset listed in ``config``.

    The ``output_dir`` parameter is forwarded as
    :attr:`OptunaSearchConfig.storage_path`. Two artefacts are
    written per dataset under this directory:

    * ``<output_dir>/<dataset>.db`` — the SQLite-backed Optuna
      study with WAL journaling enabled.
    * ``<output_dir>/<dataset>/best_config.yaml`` — the best
      hyperparameters found by TPE sampling under Hyperband
      pruning.

    Each Optuna trial is a single fold of the TU experiment;
    :class:`OptunaSearch.evaluate` reports the validation accuracy
    and the Hyperband pruner can short-circuit unpromising trials
    via :func:`trial.report` / :func:`trial.should_prune` (see
    :mod:`pjepa.training.optuna_search`).

    Args:
        config: The Optuna configuration.
        output_dir: The storage path; defaults to
            ``"results/optuna"``.

    Returns:
        A mapping from dataset name to the per-dataset summary
        returned by :meth:`OptunaSearch.run` (``best_value``,
        ``best_params``, ``n_trials``, ``n_pruned``, ``storage_path``,
        ``best_config_path``).
    """
    log = get_logger(__name__)
    log.info("optuna search starting", extra={"event": "optuna.start"})
    search = OptunaSearch(
        OptunaSearchConfig(
            storage_path=output_dir,
            n_trials=int(config.n_trials),
            epochs=int(config.epochs),
            timeout_seconds=config.timeout_seconds,
        )
    )

    def loader(name: str) -> tuple[list[tuple[Any, int]], int]:
        graphs, num_classes = load_tu_dataset(name)
        return [(g.graph, g.label) for g in graphs], int(num_classes)

    summary = search.run_many(list(config.datasets), loader)
    log.info(
        "optuna search complete",
        extra={"event": "optuna.complete", "datasets": list(summary.keys())},
    )
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for Persistent-JEPA."
    )
    parser.add_argument("--datasets", nargs="*", default=list(OptunaConfig().datasets))
    parser.add_argument("--n-trials", type=int, default=OptunaConfig().n_trials)
    parser.add_argument("--epochs", type=int, default=OptunaConfig().epochs)
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
    run_search(config, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
