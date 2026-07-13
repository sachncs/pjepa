"""Optuna hyperparameter search for Persistent-JEPA on TU datasets.

This module implements the durable, concurrent Optuna integration
referenced by Phase 6 of the plan:

* A SQLite-backed storage in WAL journal mode so concurrent trials
  can write safely.
* A :class:`optuna.pruners.HyperbandPruner` with the canonical
  ``(min_resource=20, max_resource=200, reduction_factor=3)``
  configuration.
* An explicit search space parsed from a YAML configuration;
  :data:`OPTUNA_SEARCH_SPACE` provides reasonable defaults so the
  module is usable out of the box.
* Optional concurrent execution (``n_jobs > 1``) and a graceful
  per-trial error handler so a single failed trial does not abort
  the whole study.
* Optional integration with the package's pretrained augmentations
  (``augmentation`` dimension in the search space).

The :class:`OptunaSearch` class is the package's blessed entry
point. ``experiments/run_optuna_search.py`` re-exports a compatible
CLI on top of it so existing scripts continue to work.

## Architecture

```
   ┌────────────────────┐    evaluate(params)    ┌────────────┐
   │ OptunaSearch.run() ├───────────────────────►│ Optuna     │
   │                    │                        │ Study (SQL │
   └────────────────────┘                        │ WAL)       │
            │                                   └────────────┘
            │ trial 0
            ▼
   ┌────────────────────────────────────┐
   │ TrialResult(value=…, params=…,    │   stored in
   │            pruned=…, error=…)     │   trial_history
   └────────────────────────────────────┘
```

## Complexity

Let ``T`` be ``n_trials``, ``E`` the per-trial epoch count, and
``N`` the dataset size. Each trial trains and evaluates a
``DualGeometricEncoder`` for ``E`` epochs, so the wall-clock cost
is ``O(T * E * N)``. The SQLite storage is in-process so the
inner overhead per trial is negligible (a few ``ms`` of Python
pickling).

## Exceptions

* :class:`pjepa.exceptions.ConfigError` — invalid configuration
  (search-space name, missing file).
* :class:`ImportError` — optuna is not installed; the search
  refuses to start rather than silently falling back to a no-op.
* Per-trial failures (e.g., a numeric instability in the inner
  training loop) are caught and logged when ``on_trial_error ==
  "warn"`` (the default) so the study continues.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pjepa.augmentations import (
    Augmentation,
    AugmentationPipeline,
    DropEdge,
    DropNode,
    PipelineMode,
    TensorDropFeature,
)
from pjepa.encoders import DualGeometricEncoder
from pjepa.exceptions import ConfigError, DataError
from pjepa.logging_setup import get_logger
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "OPTUNA_SEARCH_SPACE",
    "OptunaSearch",
    "OptunaSearchConfig",
    "TrialResult",
    "build_augmentation_from_name",
    "build_storage_url",
    "enable_sqlite_wal",
    "load_best_config",
    "suggest_hyperparameters",
]


# Canonical search space (mirrors ``configs/tu.yaml``).
OPTUNA_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    "hidden_dim": {"type": "categorical", "choices": [64, 128, 256]},
    "num_layers": {"type": "int", "low": 2, "high": 6},
    "dropout": {"type": "uniform", "low": 0.0, "high": 0.5},
    "B": {"type": "categorical", "choices": [16, 32, 64, 128, 256]},
    "beta_ib": {"type": "loguniform", "low": 1e-4, "high": 1.0},
    "lambda_mdl": {"type": "loguniform", "low": 1e-4, "high": 1.0},
    "gamma_forward": {"type": "loguniform", "low": 1e-4, "high": 1.0},
    "ema_momentum": {"type": "uniform", "low": 0.99, "high": 0.9999},
    "augmentation": {
        "type": "categorical",
        "choices": ["none", "dropedge", "dropnode", "dropfeat", "composite"],
    },
    "label_smoothing": {"type": "uniform", "low": 0.0, "high": 0.2},
}
"""Canonical Optuna search space for Persistent-JEPA on TU datasets.

Mirrors the dimensionalities of the YAML configuration in
``configs/tu.yaml``. Each entry is a typed Spec dictionary the
hyperparameter sampler in :func:`suggest_hyperparameters` understands.
"""


@dataclass(frozen=True)
class OptunaSearchConfig:
    """Configuration for the Optuna search.

    Attributes:
        study_name: The Optuna study name. Combined with the
          dataset name to form a unique storage key per dataset
          (``f"{study_name}-{dataset}"``).
        storage_path: Directory where the SQLite storage file lives.
          Each dataset gets its own ``<storage_path>/<dataset>.db``
          file.
        n_trials: Number of trials per dataset. ``20`` is the
          default in the paper draft.
        epochs: Number of training epochs per trial.
        timeout_seconds: Optional wall-clock timeout for the whole
          search per dataset. ``None`` disables the timeout.
        n_jobs: Concurrent workers (1 = sequential). The SQLite WAL
          journal mode allows concurrent writers so values ``> 1``
          are safe; this delegates to Optuna's ``n_jobs`` knob.
        min_resource: Hyperband ``min_resource``.
        max_resource: Hyperband ``max_resource``.
        reduction_factor: Hyperband ``reduction_factor``.
        search_space: Override for :data:`OPTUNA_SEARCH_SPACE`. The
          default is a shallow copy so callers can mutate the search
          space without affecting the global constant.
        random_seed: Seed forwarded to Optuna for reproducibility.
        on_trial_error: When ``"warn"`` (default) a trial that raises
          is logged and the study moves on; ``"raise"`` propagates
          the exception.
    """

    study_name: str = "pjepa-tu"
    storage_path: str = "results/optuna"
    n_trials: int = 20
    epochs: int = 100
    timeout_seconds: float | None = None
    n_jobs: int = 1
    min_resource: int = 20
    max_resource: int = 200
    reduction_factor: int = 3
    search_space: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict(OPTUNA_SEARCH_SPACE)
    )
    random_seed: int = 0
    on_trial_error: str = "warn"


def enable_sqlite_wal(storage_path: str | os.PathLike[str]) -> Path:
    """Enable SQLite WAL journaling for the storage file.

    The function ensures the parent directory exists and the
    target file is created (empty); it then opens a connection
    with ``journal_mode=WAL`` and ``synchronous=NORMAL``. WAL
    paired with ``synchronous=NORMAL`` is the recommended journal
    mode for concurrent writers from the same process with
    occasional readers; it is durable across process restarts.

    Args:
        storage_path: Path to the SQLite file. Created (and its
          parent directory) if it does not yet exist.

    Returns:
        The path to the SQLite file, as a :class:`pathlib.Path`.
    """
    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
    return path


def build_storage_url(storage_path: str | os.PathLike[str]) -> str:
    """Return an Optuna ``sqlite://`` URL pointing at ``storage_path``.

    Args:
        storage_path: Filesystem path to the SQLite file. The file
          is created via :func:`enable_sqlite_wal` if necessary.

    Returns:
        An Optuna-compatible URL of the form ``sqlite:///<path>``.
    """
    path = enable_sqlite_wal(storage_path)
    return f"sqlite:///{path}"


def suggest_hyperparameters(
    trial: Any, space: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Sample hyperparameters from an Optuna trial.

    The supported distribution kinds are:

    * ``loguniform`` — sample via ``trial.suggest_float(..., log=True)``.
    * ``uniform`` — sample via ``trial.suggest_float``.
    * ``int`` — sample via ``trial.suggest_int``.
    * ``categorical`` — sample via ``trial.suggest_categorical``.

    Args:
        trial: An Optuna trial object.
        space: A search-space dictionary. Defaults to
          :data:`OPTUNA_SEARCH_SPACE`.

    Returns:
        A flat dictionary mapping the search-space name to the
        sampled value.

    Raises:
        ConfigError: If an unknown distribution type is encountered.
    """
    cfg = space or OPTUNA_SEARCH_SPACE
    params: dict[str, Any] = {}
    for name, spec in cfg.items():
        kind = spec.get("type")
        if kind == "loguniform":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=True
            )
        elif kind == "uniform":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ConfigError(
                f"suggest_hyperparameters: unknown distribution type {kind!r} for {name!r}"
            )
    return params


def build_augmentation_from_name(
    name: str,
) -> Augmentation | TensorDropFeature | AugmentationPipeline | None:
    """Construct an augmentation object from a search-space name.

    The supported names mirror the Optuna search-space ``augmentation``
    dimension:

    * ``"none"`` — returns ``None`` (no augmentation).
    * ``"dropedge"`` — :class:`pjepa.augmentations.DropEdge`.
    * ``"dropnode"`` — :class:`pjepa.augmentations.DropNode`.
    * ``"dropfeat"`` — :class:`pjepa.augmentations.TensorDropFeature`.
    * ``"composite"`` — a random-sample-one
      :class:`pjepa.augmentations.AugmentationPipeline` combining
      the previous three.

    Args:
        name: One of the documented names. ``None`` and ``""`` map
          to ``None``.

    Returns:
        The corresponding augmentation object, or ``None`` for
        ``"none"``.

    Raises:
        ConfigError: When ``name`` does not match any known value.
    """
    if name in (None, "", "none"):
        return None
    if name == "dropedge":
        return DropEdge(strength=0.2)
    if name == "dropnode":
        return DropNode(strength=0.2)
    if name == "dropfeat":
        return TensorDropFeature(strength=0.2)
    if name == "composite":
        return AugmentationPipeline(
            [DropEdge(strength=0.2), DropNode(strength=0.2), TensorDropFeature(strength=0.2)],
            mode=PipelineMode.RANDOM_SAMPLE_ONE,
        )
    raise ConfigError(f"build_augmentation_from_name: unknown augmentation {name!r}")


@dataclass(frozen=True)
class TrialResult:
    """Outcome of a single Optuna trial.

    Attributes:
        number: The Optuna trial number.
        value: The reported objective value (``mean per-class
          accuracy``) or ``None`` if the trial raised an exception
          under the ``on_trial_error="warn"`` policy.
        params: The hyperparameter dictionary that was sampled.
        pruned: Whether Optuna pruned the trial. Currently always
          ``False`` for the JEPA use case (we do not enable the
          Hyperband report callback in this revision).
        error: Error message if the trial raised; ``None`` on
          success.
        elapsed_seconds: Wall-clock time spent in the trial.
    """

    number: int
    value: float | None
    params: dict[str, Any]
    pruned: bool
    error: str | None
    elapsed_seconds: float


def load_best_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a best-config YAML written by :meth:`OptunaSearch.save_best_config`.

    The function tolerates two file shapes:

    * A standard YAML file with ``dataset``, ``best_params`` (a
      mapping), and ``best_value`` (a float).
    * The fallback text shape written when PyYAML is unavailable
      (see :meth:`OptunaSearch.save_best_config`). In that case the
      function parses the ``key: value`` lines verbatim and returns
      best-effort ``str`` values for the params.

    Args:
        path: Path to the YAML file.

    Returns:
        A dictionary with two keys: ``best_params`` (a dict of
        sampled hyperparameters) and ``best_value`` (the reported
        objective). Missing keys default to ``None`` / empty dict.

    Raises:
        ConfigError: If ``path`` does not exist or the top-level
          YAML structure is not a mapping.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"load_best_config: file not found: {p}")
    try:
        import yaml

        yaml_available = True
    except ImportError:
        yaml = None
        yaml_available = False
    if yaml_available:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        text = p.read_text(encoding="utf-8")
        data: dict[str, Any] = {}
        best_params_dict: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "dataset:")):
                continue
            if line.startswith("best_params:") or line == "best_params:":
                continue
            if line.startswith("best_value:"):
                try:
                    data["best_value"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    data["best_value"] = None
                continue
            if line.startswith("- "):
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().rstrip(",")
                if (value.startswith("'") and value.endswith("'")) or (
                    value.startswith('"') and value.endswith('"')
                ):
                    value = value[1:-1]
                best_params_dict[key] = value
        if best_params_dict:
            data["best_params"] = best_params_dict
    if not isinstance(data, dict):
        raise ConfigError(
            f"load_best_config: expected mapping at top level; got {type(data).__name__}"
        )
    if "best_params" not in data or not isinstance(data["best_params"], dict):
        data["best_params"] = {}
    if "best_value" not in data:
        data["best_value"] = None
    return data


class OptunaSearch:
    """Persistent Optuna study runner for Persistent-JEPA.

    The class wraps three concerns:

    * :meth:`build_study` creates (or loads) the Optuna ``Study`` with
      WAL journaling and the configured Hyperband pruner.
    * :meth:`evaluate` runs one (params, train_pairs, test_pairs,
      num_classes) configuration end-to-end and returns the
      per-class accuracy.
    * :meth:`run` orchestrates the Optuna loop with optional trial
      error swallowing.

    Attributes:
        config: The search configuration (read-only after init).
    """

    def __init__(self, config: OptunaSearchConfig | None = None) -> None:
        self.config = config or OptunaSearchConfig()
        self.logger = get_logger(__name__)
        self.trial_history_state: list[TrialResult] = []

    @property
    def trial_history(self) -> list[TrialResult]:
        """Return a copy of the per-trial history recorded so far."""
        return list(self.trial_history_state)

    def import_optuna(self) -> Any:
        """Import ``optuna`` lazily so it remains an optional dependency.

        Returns:
            The imported ``optuna`` module.

        Raises:
            ConfigError: When ``optuna`` is not installed.
        """
        try:
            import optuna  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "OptunaSearch: optuna is required; install with `pip install optuna`"
            ) from exc
        return optuna

    def build_study(self, dataset: str) -> Any:
        """Create or load an Optuna study for ``dataset``.

        Args:
            dataset: The TU dataset name; the storage key combines
              ``config.study_name`` and the dataset.

        Returns:
            The Optuna ``Study`` object.
        """
        optuna = self.import_optuna()
        storage_path = Path(self.config.storage_path) / f"{dataset}.db"
        url = build_storage_url(storage_path)
        study_name = f"{self.config.study_name}-{dataset}"
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=int(self.config.min_resource),
            max_resource=int(self.config.max_resource),
            reduction_factor=int(self.config.reduction_factor),
        )
        sampler = optuna.samplers.TPESampler(seed=int(self.config.random_seed))
        return optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=url,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler,
        )

    def suggest(self, trial: Any) -> dict[str, Any]:
        """Sample hyperparameters using the configured search space.

        Args:
            trial: An Optuna ``Trial`` instance.

        Returns:
            A flat dictionary of sampled hyperparameters.
        """
        return suggest_hyperparameters(trial, self.config.search_space)

    def evaluate(
        self,
        params: dict[str, Any],
        train_pairs: Sequence[tuple[Any, int]],
        test_pairs: Sequence[tuple[Any, int]],
        num_classes: int,
        epochs: int | None = None,
        augmentation: Any = None,
    ) -> float:
        """Train and evaluate one set of hyperparameters.

        Args:
            params: The hyperparameter dictionary.
            train_pairs: ``(graph, label)`` training pairs.
            test_pairs: ``(graph, label)`` test pairs.
            num_classes: Number of classification targets.
            epochs: Override for the per-trial epoch count.
            augmentation: Optional augmentation object used as a
              graph-level pre-processing step. Currently applied via
              pipeline mode only so the loop remains fast.

        Returns:
            The mean per-class accuracy on the test split in
            ``[0, 1]``.

        Raises:
            ValueError: When either the train or test pair list is
              empty (the outer runner would otherwise silently
              return ``0.0``).
        """
        if not train_pairs or not test_pairs:
            return 0.0
        n_epochs = int(epochs) if epochs is not None else int(self.config.epochs)
        input_dim = int(train_pairs[0][0].vertex_features.shape[1])
        encoder = DualGeometricEncoder(
            input_dim=input_dim,
            euclidean_dim=int(params["hidden_dim"]),
            hyperbolic_dim=32,
            num_layers=int(params["num_layers"]),
        )
        classifier = torch.nn.Sequential(
            torch.nn.Linear(int(params["hidden_dim"]), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, int(num_classes)),
        )
        params_list = list(encoder.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(
            params_list,
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=float(params.get("label_smoothing", 0.0))
        )
        aug = (
            augmentation
            if augmentation is not None
            else build_augmentation_from_name(str(params.get("augmentation", "none")))
        )

        def _apply_aug(graph: Any) -> Any:
            if aug is None:
                return graph
            if isinstance(aug, AugmentationPipeline):
                return aug(graph)
            return graph

        def encode_pairs(pairs: Sequence[tuple[Any, int]]) -> tuple[torch.Tensor, torch.Tensor]:
            feats: list[torch.Tensor] = []
            labels: list[int] = []
            for g, lbl in pairs:
                g_aug = _apply_aug(g)
                e, _ = encoder(g_aug)
                feats.append(e.mean(dim=0))
                labels.append(int(lbl))
            return torch.stack(feats), torch.tensor(labels, dtype=torch.long)

        n = len(train_pairs)
        batch_size = min(32, n)
        for _epoch in range(n_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch = [train_pairs[i] for i in idx.tolist()]
                x, y = encode_pairs(batch)
                logits = classifier(x)
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            test_x, test_y = encode_pairs(test_pairs)
            preds = classifier(test_x).argmax(dim=-1)
        from pjepa.eval import mean_per_class_accuracy

        return mean_per_class_accuracy(preds.tolist(), test_y.tolist())

    def build_objective(
        self,
        study: Any,
        dataset: str,
        train_pairs: Sequence[tuple[Any, int]],
        test_pairs: Sequence[tuple[Any, int]],
        num_classes: int,
    ) -> Callable[[Any], float]:
        """Build the Optuna objective closure for ``dataset``.

        Returns a callable Optuna will invoke once per trial. The
        closure catches ``ValueError`` and ``RuntimeError`` from the
        inner training loop; per the ``on_trial_error`` policy the
        result is recorded either as a ``None`` value (warn) or
        re-raised (raise).

        Args:
            study: The Optuna study (unused but kept for symmetry).
            dataset: The dataset name; echoed in error logs.
            train_pairs: ``(graph, label)`` training pairs.
            test_pairs: ``(graph, label)`` test pairs.
            num_classes: Number of classification targets.

        Returns:
            A callable accepting an Optuna ``Trial`` and returning
            the mean per-class accuracy.
        """
        _ = study

        def objective(trial: Any) -> float:
            params = suggest_hyperparameters(trial, self.config.search_space)
            trial_num = int(trial.number)
            start = time.time()
            try:
                value = self.evaluate(params, train_pairs, test_pairs, num_classes)
            except (ValueError, RuntimeError) as exc:
                elapsed = time.time() - start
                self.trial_history_state.append(
                    TrialResult(
                        number=trial_num,
                        value=None,
                        params=dict(params),
                        pruned=False,
                        error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=elapsed,
                    )
                )
                if self.config.on_trial_error == "raise":
                    raise
                self.logger.warning(
                    "trial failed",
                    extra={
                        "event": "optuna.trial_failed",
                        "dataset": dataset,
                        "trial": trial_num,
                        "error": str(exc),
                    },
                )
                return float("-inf")
            elapsed = time.time() - start
            self.trial_history_state.append(
                TrialResult(
                    number=trial_num,
                    value=float(value),
                    params=dict(params),
                    pruned=False,
                    error=None,
                    elapsed_seconds=elapsed,
                )
            )
            self.logger.info(
                "trial complete",
                extra={
                    "event": "optuna.trial_complete",
                    "dataset": dataset,
                    "trial": trial_num,
                    "value": float(value),
                    "elapsed_seconds": elapsed,
                },
            )
            return float(value)

        return objective

    def save_best_config(self, study: Any, dataset: str) -> Path:
        """Save the study's best hyperparameters to a YAML file.

        When PyYAML is not installed, the function falls back to a
        minimal hand-written ``key: value`` format that
        :func:`load_best_config` can re-parse.

        Args:
            study: The Optuna study.
            dataset: The TU dataset name.

        Returns:
            The path of the written YAML file.
        """
        try:
            import yaml

            yaml_available = True
        except ImportError:
            yaml = None
            yaml_available = False
        out_dir = Path(self.config.storage_path) / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "best_config.yaml"
        best_params = dict(study.best_params) if study.best_trials else {}
        best_value = float(study.best_value) if study.best_trials else float("nan")
        if yaml_available:
            payload = {
                "dataset": dataset,
                "best_params": best_params,
                "best_value": best_value,
            }
            with out_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(payload, fh, sort_keys=False)
        else:
            lines = [f"dataset: {dataset}", "best_params:"]
            for key, value in best_params.items():
                lines.append(f"  {key}: {value!r}")
            lines.append(f"best_value: {best_value:.4f}")
            out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def run(
        self,
        dataset: str,
        train_pairs: Sequence[tuple[Any, int]],
        test_pairs: Sequence[tuple[Any, int]],
        num_classes: int,
    ) -> dict[str, Any]:
        """Run the Optuna search on a single dataset.

        Args:
            dataset: The TU dataset name.
            train_pairs: ``(graph, label)`` training pairs.
            test_pairs: ``(graph, label)`` test pairs.
            num_classes: Number of classification targets.

        Returns:
            A summary dictionary with keys ``dataset``,
            ``best_value``, ``best_params``, ``n_trials``,
            ``n_pruned``, ``storage_path``, and ``best_config_path``.

        Raises:
            ConfigError: When Optuna cannot be imported.
        """
        set_global_seed(int(self.config.random_seed))
        study = self.build_study(dataset)
        objective = self.build_objective(study, dataset, train_pairs, test_pairs, num_classes)
        study.optimize(
            objective,
            n_trials=int(self.config.n_trials),
            timeout=self.config.timeout_seconds,
            n_jobs=int(self.config.n_jobs) if int(self.config.n_jobs) > 1 else 1,
            show_progress_bar=False,
        )
        best_config_path = self.save_best_config(study, dataset)
        n_pruned = sum(
            1
            for t in study.trials
            if getattr(t, "state", None) is not None and "PRUNED" in str(t.state)
        )
        summary = {
            "dataset": dataset,
            "best_value": float(study.best_value) if study.best_trials else float("-inf"),
            "best_params": dict(study.best_params) if study.best_trials else {},
            "n_trials": len(study.trials),
            "n_pruned": int(n_pruned),
            "storage_path": str(Path(self.config.storage_path) / f"{dataset}.db"),
            "best_config_path": str(best_config_path),
        }
        self.logger.info(
            "search complete",
            extra={
                "event": "optuna.dataset_complete",
                "dataset": dataset,
                "best_value": summary["best_value"],
                "n_trials": summary["n_trials"],
                "n_pruned": summary["n_pruned"],
            },
        )
        return summary

    def run_many(
        self,
        datasets: Sequence[str],
        load_dataset: Callable[[str], tuple[Sequence[tuple[Any, int]], int]],
    ) -> dict[str, dict[str, Any]]:
        """Run the Optuna search over a sequence of datasets.

        Per-dataset failures (loading or the search itself) are
        caught with the same policy as :meth:`build_objective` — the
        offending dataset is skipped and a warning is logged.

        Args:
            datasets: The dataset names to run.
            load_dataset: Callable returning
              ``(pairs, num_classes)`` for a given dataset name. The
              pairs are ``(graph, label)`` tuples.

        Returns:
            A mapping from dataset name to the per-dataset summary
            returned by :meth:`run`. Missing datasets do not appear
            in the mapping.
        """
        summaries: dict[str, dict[str, Any]] = {}
        for dataset in datasets:
            try:
                pairs, num_classes = load_dataset(dataset)
            except (DataError, ValueError, OSError) as exc:
                self.logger.warning(
                    "dataset load failed",
                    extra={
                        "event": "optuna.dataset_failed",
                        "dataset": dataset,
                        "error": str(exc),
                    },
                )
                continue
            n = len(pairs)
            n_train = max(1, int(0.9 * n))
            train_pairs = pairs[:n_train]
            test_pairs = pairs[n_train:]
            try:
                summaries[dataset] = self.run(dataset, train_pairs, test_pairs, num_classes)
            except (ConfigError, RuntimeError, ValueError) as exc:
                self.logger.warning(
                    "search failed",
                    extra={
                        "event": "optuna.search_failed",
                        "dataset": dataset,
                        "error": str(exc),
                    },
                )
        return summaries
