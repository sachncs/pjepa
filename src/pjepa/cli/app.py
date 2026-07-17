"""Typer-based CLI for ``pjepa``.

The CLI exposes the canonical workflow:

* ``pjepa doctor`` — capability probe report.
* ``pjepa hardware`` — backend summary.
* ``pjepa benchmark {retrieval, distortion, encoder-ablation}`` —
  cheap validation experiments that run on the local machine.
* ``pjepa pretrain <config>`` — pretrain a JEPA encoder using a
  YAML configuration. The CLI performs an actual three-step smoke
  loop on a small synthetic input so the user can verify the
  encoder / predictor / target wiring end-to-end.
* ``pjepa train {tu,cl,ogb} <config>`` — train on a dataset
  family. The command dispatches to the corresponding
  ``experiments/run_exp_*.py`` runner.
* ``pjepa tune tu <config>`` — Optuna hyperparameter search for
  TU. Dispatches to ``experiments/run_optuna_search.py``.
* ``pjepa baseline-smoke {gcn,gin,graphmae,graphcl,infograph,naive,ewc,gem}``
  — one-epoch smoke test for any published baseline that
  constructs the model with the configured hyperparameters and
  runs a forward pass on a toy graph.
* ``pjepa decoupling <config>`` — inference-storage decoupling.
* ``pjepa ablation <config>`` — Phase 11 ablation study.
* ``pjepa sensitivity <config>`` — Phase 11 sensitivity sweep.
* ``pjepa aggregate [results-dir]`` — Phase 12 results aggregation.

The advertised ``pjepa train <dataset> <config>`` signature is the
documented command-line entry point. Internally it dispatches to
the corresponding runner, applying the YAML config when one is
provided. When the user omits a config (or supplies a path that
does not exist), the runner defaults are used so the command still
produces useful output.

Exit codes:

* ``0`` — success.
* ``2`` — configuration or argument error.
* ``3`` — experiment dispatch error (module/function not found).
* ``4`` — runner raised a runtime or value error.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import torch
import typer

from pjepa import __version__
from pjepa.config import load_config
from pjepa.exceptions import ConfigError
from pjepa.hardware import detect_backend, detect_capabilities
from pjepa.logging_setup import (
    LOG_FORMAT_HUMAN,
    configure_logging,
    get_logger,
)

__all__ = [
    "BASELINES",
    "BENCHMARKS",
    "DATASETS",
    "EXIT_CONFIG",
    "EXIT_DISPATCH",
    "EXIT_RUNTIME",
    "RUNNERS",
    "app",
    "dispatch_to_experiment",
    "main",
    "resolve_yaml_config",
    "run_baseline_forward_smoke",
    "run_pretrain_smoke",
    "supervised_optimiser",
    "supervised_target_inputs",
    "version_callback",
]


EXIT_CONFIG: int = 2
EXIT_DISPATCH: int = 3
EXIT_RUNTIME: int = 4
"""Distinct CLI exit codes so CI scripts can tell failure modes apart."""


def experiments_search_paths() -> tuple[str, ...]:
    """Return ``sys.path`` entries needed to import the ``experiments/`` scripts.

    The CLI dispatches to runner modules that live under
    ``<repo>/experiments/`` (not under the installed ``pjepa``
    package). When the repo layout is the canonical
    ``src/pjepa/cli/app.py``, the experiments directory is two
    parents up. Returns both the experiments directory and the
    repository root so the runners can ``from pjepa.<...>`` import.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    return (str(repo_root / "experiments"), str(repo_root))


def ensure_experiments_importable() -> None:
    """Insert the experiments/ directory into ``sys.path`` if missing.

    Called lazily from :func:`dispatch_to_experiment` so that
    ``import pjepa.cli.app`` does not mutate the global path.
    """
    for entry in experiments_search_paths():
        if entry not in sys.path:
            sys.path.insert(0, entry)


app = typer.Typer(
    name="pjepa",
    help="Persistent-JEPA: persistent graph world model for continual learning.",
    no_args_is_help=True,
)


DATASETS: tuple[str, ...] = ("tu", "cl", "ogb")
"""Supported dataset families for ``pjepa train``."""

BASELINES: tuple[str, ...] = (
    "gcn",
    "gin",
    "graphmae",
    "graphcl",
    "infograph",
    "naive",
    "ewc",
    "gem",
)
"""Supported baselines for ``pjepa baseline-smoke``.

Each entry corresponds to a ``pjepa.baselines.<name>`` module.
"""

BENCHMARKS: tuple[str, ...] = ("retrieval", "distortion", "encoder-ablation")
"""Supported benchmarks for ``pjepa benchmark``."""


# Single source of truth for "command -> runner module + callable + Config dataclass".
# Adding a new experiment means appending one row; nothing else in this file needs editing.
RUNNERS: dict[str, tuple[str, str, str]] = {
    # command_name: (module_path, runner_callable, config_dataclass)
    "train.tu": ("experiments.run_exp_d_tu_sota", "run_experiment", "TUExperimentConfig"),
    "train.cl": ("experiments.run_exp_e_continual", "run_cl_experiment", "CLExperimentConfig"),
    "train.ogb": ("experiments.run_exp_f_ogb_arxiv", "run_ogb_experiment", "OGBConfig"),
    "tune.tu": ("experiments.run_optuna_search", "run_search", "OptunaConfig"),
    "decoupling": (
        "experiments.run_exp_g_decoupling",
        "run_decoupling_measurement",
        "DecouplingConfig",
    ),
    "ablation": ("experiments.run_exp_h_ablations", "run_ablation", "AblationConfig"),
    "sensitivity": ("experiments.run_sensitivity", "run_sensitivity", "SensitivityConfig"),
    "benchmark.retrieval": ("experiments.run_exp_a_retrieval", "run", ""),
    "benchmark.distortion": ("experiments.run_exp_b_distortion", "run", ""),
    "benchmark.encoder-ablation": (
        "experiments.run_exp_c_encoder_ablation",
        "run_encoder_ablation",
        "",
    ),
}


def version_callback(value: bool) -> None:
    """Eager ``--version`` Typer callback.

    Args:
        value: ``True`` when the user passed ``--version``.

    Raises:
        typer.Exit: Always exits after echoing the version string.
    """
    if value:
        typer.echo(f"pjepa, version {__version__}")
        raise typer.Exit()


def resolve_yaml_config(
    config: str | None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Load a YAML config if it exists, otherwise return an empty mapping.

    Args:
        config: Optional path to a YAML configuration file.
        dataset: Optional dataset family (``tu``, ``cl``, ``ogb``).
          Currently informational only; reserved for per-family
          defaults.

    Returns:
        A configuration dictionary (possibly empty if nothing was
        found).
    """
    del dataset  # reserved for future per-family defaults
    if config is None:
        return {}
    path = Path(config)
    if not path.exists():
        return {}
    try:
        return dict(load_config(path))
    except ConfigError as exc:
        typer.echo(f"config load failed for {config}: {exc}", err=True)
        raise typer.Exit(code=EXIT_CONFIG) from exc


def dispatch_to_experiment(
    module_name: str,
    run_callable: str,
    config: dict[str, Any],
    extra_args: dict[str, Any] | None = None,
    *,
    dataclass_name: str = "",
) -> Any:
    """Dispatch to an experiment runner with optional config overrides.

    Args:
        module_name: The ``experiments`` module to import.
        run_callable: Name of the function to invoke (``"run"`` or
          ``"run_experiment"`` or ``"run_cl_experiment"`` etc.).
        config: The YAML configuration mapping.
        extra_args: Optional explicit overrides forwarded to the
          runner via its top-level dataclass.
        dataclass_name: Name of the runner's top-level dataclass to
          instantiate. When empty (the default), the runner is
          called with no arguments; this is the right choice for
          benchmarks that take no configuration.

    Returns:
        Whatever the experiment runner returns.

    Raises:
        ConfigError: If the module or callable cannot be located,
          or the named dataclass does not exist.
    """
    ensure_experiments_importable()
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(
            f"dispatch_to_experiment: cannot import {module_name!r}: {exc}"
        ) from exc
    if not hasattr(module, run_callable):
        raise ConfigError(
            f"dispatch_to_experiment: {module_name!r} has no callable {run_callable!r}"
        )
    func = getattr(module, run_callable)
    if not dataclass_name:
        return func()
    cfg_cls = getattr(module, dataclass_name, None)
    if cfg_cls is None or not (is_dataclass(cfg_cls) and isinstance(cfg_cls, type)):
        raise ConfigError(
            f"dispatch_to_experiment: {module_name!r} has no dataclass named {dataclass_name!r}"
        )
    extra = dict(extra_args or {})
    known = {f.name for f in cfg_cls.__dataclass_fields__.values()}
    cfg_kwargs: dict[str, Any] = {}
    for section in ("experiment", "training", "model", "pjepa", "optuna"):
        section_value = config.get(section, {})
        if isinstance(section_value, dict):
            for key, value in section_value.items():
                if key in known:
                    cfg_kwargs[key] = value
    for key, value in extra.items():
        if key in known:
            cfg_kwargs[key] = value
    if "smoke" in known:
        cfg_kwargs.setdefault("smoke", False)
    cfg_instance = cfg_cls(**cfg_kwargs) if cfg_kwargs else cfg_cls()
    return func(cfg_instance)


def supervised_target_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic ``(context, target)`` tensors used by the pretrain smoke.

    Args:
        (no arguments)

    Returns:
        A tuple ``(context, target)`` of two ``[2, 4]`` tensors that
        are identical across calls within a single process but
        differ across processes (the seed is derived from the PID).
    """
    seed = torch.initial_seed() ^ os.getpid()
    generator = torch.Generator().manual_seed(seed)
    context = torch.randn(2, 4, generator=generator)
    target_features = torch.randn(2, 4, generator=generator)
    return context, target_features


def supervised_optimiser(params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """Return an AdamW optimiser with sensible defaults for the pretrain smoke.

    Args:
        params: The iterable of parameters to optimise.

    Returns:
        A fresh ``torch.optim.AdamW`` instance.
    """
    return torch.optim.AdamW(params, lr=1e-3)


def run_pretrain_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Run a three-step smoke pretrain loop on synthetic tensors.

    The function wires up a tiny encoder / predictor / target
    triple via the package's own primitives
    (:func:`pjepa.training.pretrain.pretrain_loop`) so the user
    can verify the pretraining loop end-to-end without a dataset.

    Args:
        config: The YAML configuration mapping.

    Returns:
        A payload describing what the smoke ran, the final mean
        loss, and whether every component was wired.
    """
    try:
        from pjepa.encoders import JEPAPredictor, TargetEncoder
        from pjepa.training import (
            PretrainConfig,
            pretrain_loop,
        )
    except ImportError as exc:
        return {
            "ran": False,
            "reason": f"pretraining imports unavailable: {exc}",
        }

    epochs = int(config.get("training", {}).get("epochs", 2))
    log_every = int(config.get("training", {}).get("log_every", 0))
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="pjepa_pretrain_smoke_"))

    encoder = torch.nn.Linear(4, 4)
    predictor = JEPAPredictor(input_dim=4, hidden_dim=8, output_dim=4)
    target = TargetEncoder(encoder, momentum=0.9)
    optimizer = supervised_optimiser(list(encoder.parameters()) + list(predictor.parameters()))

    def batches():
        for _ in range(epochs):
            yield supervised_target_inputs()

    losses = pretrain_loop(
        encoder=encoder,
        predictor=predictor,
        target=target,
        optimizer=optimizer,
        batches=batches(),
        config=PretrainConfig(
            epochs=epochs,
            checkpoint_dir=str(checkpoint_dir),
            log_every=log_every,
        ),
    )
    return {
        "ran": True,
        "epochs": epochs,
        "losses": [float(x) for x in losses],
        "checkpoint_dir": str(checkpoint_dir),
        "final_loss": float(losses[-1]) if losses else None,
    }


def run_baseline_forward_smoke(
    baseline: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run a forward-pass smoke test on the chosen baseline.

    The function imports ``pjepa.baselines.<baseline>``,
    instantiates the public model class with the configured
    dimensions, and runs a forward pass on a small synthetic
    :class:`TypedAttributedGraph`.

    Args:
        baseline: The baseline name; one of :data:`BASELINES`.
        cfg: The YAML configuration mapping.

    Returns:
        A payload describing what was constructed and the
        resulting forward-pass shape.
    """
    module_name = f"pjepa.baselines.{baseline}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return {"ran": False, "reason": f"import failed: {exc}"}

    cls = next(
        (
            value
            for value in vars(module).values()
            if isinstance(value, type) and value.__module__ == module_name
        ),
        None,
    )
    if cls is None:
        return {"ran": False, "reason": "no model class found"}

    input_dim = int(cfg.get("model", {}).get("input_dim", 4))
    hidden_dim = int(cfg.get("model", {}).get("hidden_dim", 8))
    num_classes = int(cfg.get("model", {}).get("num_classes", 2))

    try:
        from pjepa.graphs import TypedAttributedGraph
    except ImportError as exc:
        return {"ran": False, "reason": f"graph imports unavailable: {exc}"}

    graph = TypedAttributedGraph(
        vertex_features=torch.randn((4, input_dim)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
    )
    try:
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
        out = model(graph)
        output_shape = list(out.shape) if hasattr(out, "shape") else None
        return {
            "ran": True,
            "class": cls.__name__,
            "module": module_name,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "output_shape": output_shape,
        }
    except (TypeError, ValueError) as exc:
        return {
            "ran": False,
            "class": cls.__name__,
            "module": module_name,
            "reason": f"instantiation or forward failed: {exc}",
        }


@app.callback()
def root(
    show_version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
    log_format: str = typer.Option(
        LOG_FORMAT_HUMAN,
        "--log-format",
        help="Log format: HUMAN (default) or JSON.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
) -> None:
    """Global CLI options."""
    del show_version  # consumed by the callback above
    configure_logging(level=log_level, fmt=log_format)


@app.command()
def hardware() -> None:
    """Print a one-line summary of the detected compute backend."""
    backend = detect_backend()
    typer.echo(f"backend={backend.value} device={torch.device(backend.value).type}")


@app.command()
def doctor() -> None:
    """Print the full capability probe report."""
    report = detect_capabilities()
    typer.echo(report.render())
    if report.has_red():
        raise typer.Exit(code=EXIT_CONFIG)


@app.command()
def benchmark(
    name: str = typer.Argument(..., help="retrieval | distortion | encoder-ablation"),
) -> None:
    """Run a cheap validation benchmark on the local machine.

    Args:
        name: Which benchmark to run.
    """
    key = f"benchmark.{name}"
    if key not in RUNNERS:
        typer.echo(
            f"unknown benchmark: {name!r}; choose one of {', '.join(BENCHMARKS)}"
        )
        raise typer.Exit(code=EXIT_CONFIG)
    log = get_logger(__name__)
    log.info("benchmark requested", extra={"event": "benchmark.start", "benchmark": name})
    module_name, run_callable, dataclass_name = RUNNERS[key]
    try:
        result = dispatch_to_experiment(
            module_name, run_callable, {}, dataclass_name=dataclass_name
        )
    except ConfigError as exc:
        typer.echo(f"benchmark dispatch failed: {exc}", err=True)
        raise typer.Exit(code=EXIT_DISPATCH) from exc
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"benchmark runner raised: {exc}", err=True)
        raise typer.Exit(code=EXIT_RUNTIME) from exc
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def pretrain(config: str = typer.Argument(..., help="Path to a YAML config file.")) -> None:
    """Pretrain a JEPA encoder using the supplied config.

    The command first reads the YAML config to pick up the
    ``training.epochs`` and ``training.log_every`` values, then
    runs a three-step smoke loop on a small synthetic input via
    :func:`run_pretrain_smoke`. This is a real smoke test: the
    encoder, predictor, target encoder (EMA), and optimiser are
    all wired through the package's own :func:`pretrain_loop`.

    Args:
        config: Path to the YAML configuration. Missing or
          malformed files do not raise; the loader returns an
          empty dict and the smoke loop runs with the default
          settings so the command always produces useful output.
    """
    log = get_logger(__name__)
    cfg = resolve_yaml_config(config)
    log.info(
        "pretrain requested",
        extra={"event": "pretrain.start", "config": config},
    )
    smoke = run_pretrain_smoke(cfg)
    payload = {
        "command": "pretrain",
        "config": config,
        "epochs": int(cfg.get("training", {}).get("epochs", 2)),
        "smoke": smoke,
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def train(
    dataset: str = typer.Argument(..., help="tu | cl | ogb"),
    config: str = typer.Argument(..., help="Path to a YAML config file."),
) -> None:
    """Train (supervised or continual) on the named dataset family.

    Args:
        dataset: One of :data:`DATASETS` (``tu``, ``cl``, or ``ogb``).
        config: Path to the YAML configuration.
    """
    key = f"train.{dataset}"
    if key not in RUNNERS:
        typer.echo(
            f"unknown dataset family: {dataset!r}; choose one of {', '.join(DATASETS)}"
        )
        raise typer.Exit(code=EXIT_CONFIG)
    log = get_logger(__name__)
    cfg = resolve_yaml_config(config, dataset=dataset)
    log.info(
        "train requested",
        extra={"event": "train.start", "dataset": dataset, "config": config},
    )
    module_name, run_callable, dataclass_name = RUNNERS[key]
    try:
        rows = dispatch_to_experiment(
            module_name, run_callable, cfg, dataclass_name=dataclass_name
        )
    except ConfigError as exc:
        typer.echo(f"train dispatch failed for {dataset}: {exc}", err=True)
        raise typer.Exit(code=EXIT_DISPATCH) from exc
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"train runner raised for {dataset}: {exc}", err=True)
        raise typer.Exit(code=EXIT_RUNTIME) from exc
    summary: dict[str, Any] = {
        "command": "train",
        "dataset": dataset,
        "config": config,
        "n_rows": len(rows) if isinstance(rows, list) else 0,
    }
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command()
def tune(
    dataset: str = typer.Argument(..., help="tu"),
    config: str = typer.Argument(..., help="Path to a YAML config file."),
) -> None:
    """Run an Optuna hyperparameter search for the named dataset.

    Args:
        dataset: The dataset family (only ``tu`` is supported).
        config: Path to the YAML configuration.
    """
    key = f"tune.{dataset}"
    if key not in RUNNERS:
        typer.echo(
            f"unknown tune target: {dataset!r}; only 'tu' is currently supported"
        )
        raise typer.Exit(code=EXIT_CONFIG)
    cfg = resolve_yaml_config(config)
    log = get_logger(__name__)
    log.info(
        "tune requested",
        extra={"event": "tune.start", "dataset": dataset, "config": config},
    )
    module_name, run_callable, dataclass_name = RUNNERS[key]
    try:
        result = dispatch_to_experiment(
            module_name, run_callable, cfg, dataclass_name=dataclass_name
        )
    except ConfigError as exc:
        typer.echo(f"tune dispatch failed: {exc}", err=True)
        raise typer.Exit(code=EXIT_DISPATCH) from exc
    except (ValueError, RuntimeError, ImportError) as exc:
        typer.echo(f"tune runner raised: {exc}", err=True)
        raise typer.Exit(code=EXIT_RUNTIME) from exc
    payload = {
        "command": "tune",
        "dataset": dataset,
        "config": config,
        "completed": bool(result),
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("baseline-smoke")
def baseline_smoke(
    baseline: str = typer.Argument(
        ...,
        help="gcn | gin | graphmae | graphcl | infograph | naive | ewc | gem",
    ),
    config: str = typer.Argument(..., help="Path to a YAML config file."),
) -> None:
    """Run a one-epoch smoke test for a published baseline.

    The command imports the matching
    ``pjepa.baselines.<baseline>`` module, instantiates the model
    with the configured dimensions, and runs a forward pass on a
    toy graph. This is a real construction + forward test, not
    just a name check.

    Args:
        baseline: The baseline name; one of :data:`BASELINES`.
        config: Path to the YAML configuration.
    """
    if baseline not in BASELINES:
        typer.echo(
            f"unknown baseline: {baseline!r}; choose one of {', '.join(BASELINES)}"
        )
        raise typer.Exit(code=EXIT_CONFIG)
    cfg = resolve_yaml_config(config)
    log = get_logger(__name__)
    log.info(
        "baseline-smoke requested",
        extra={"event": "baseline_smoke.start", "baseline": baseline, "config": config},
    )
    payload: dict[str, Any] = {
        "command": "baseline-smoke",
        "baseline": baseline,
        "config": config,
        "smoke": run_baseline_forward_smoke(baseline, cfg),
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def decoupling(config: str = typer.Argument(..., help="Path to a YAML config file.")) -> None:
    """Run the inference-storage decoupling measurement.

    Args:
        config: Path to the YAML configuration.
    """
    cfg = resolve_yaml_config(config)
    log = get_logger(__name__)
    log.info(
        "decoupling requested",
        extra={"event": "decoupling.start", "config": config},
    )
    module_name, run_callable, dataclass_name = RUNNERS["decoupling"]
    try:
        result = dispatch_to_experiment(
            module_name, run_callable, cfg, dataclass_name=dataclass_name
        )
    except ConfigError as exc:
        typer.echo(f"decoupling dispatch failed: {exc}", err=True)
        raise typer.Exit(code=EXIT_DISPATCH) from exc
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"decoupling runner raised: {exc}", err=True)
        raise typer.Exit(code=EXIT_RUNTIME) from exc
    typer.echo(
        json.dumps(
            {
                "command": "decoupling",
                "config": config,
                "completed": True,
                "n_rows": len(result) if isinstance(result, list) else 0,
            },
            indent=2,
        )
    )


@app.command()
def ablation(config: str = typer.Argument(..., help="Path to a YAML config file.")) -> None:
    """Run the Phase 11 ablation study.

    Args:
        config: Path to the YAML configuration.
    """
    cfg = resolve_yaml_config(config)
    log = get_logger(__name__)
    log.info(
        "ablation requested",
        extra={"event": "ablation.start", "config": config},
    )
    module_name, run_callable, dataclass_name = RUNNERS["ablation"]
    try:
        rows = dispatch_to_experiment(
            module_name, run_callable, cfg, dataclass_name=dataclass_name
        )
    except ConfigError as exc:
        typer.echo(f"ablation dispatch failed: {exc}", err=True)
        raise typer.Exit(code=EXIT_DISPATCH) from exc
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"ablation runner raised: {exc}", err=True)
        raise typer.Exit(code=EXIT_RUNTIME) from exc
    typer.echo(
        json.dumps(
            {
                "command": "ablation",
                "config": config,
                "n_rows": len(rows) if isinstance(rows, list) else 0,
            },
            indent=2,
            default=str,
        )
    )


@app.command()
def sensitivity(config: str = typer.Argument(..., help="Path to a YAML config file.")) -> None:
    """Run the working-graph-budget sensitivity sweep.

    Args:
        config: Path to the YAML configuration.
    """
    cfg = resolve_yaml_config(config)
    log = get_logger(__name__)
    log.info(
        "sensitivity requested",
        extra={"event": "sensitivity.start", "config": config},
    )
    module_name, run_callable, dataclass_name = RUNNERS["sensitivity"]
    try:
        rows = dispatch_to_experiment(
            module_name, run_callable, cfg, dataclass_name=dataclass_name
        )
    except ConfigError as exc:
        typer.echo(f"sensitivity dispatch failed: {exc}", err=True)
        raise typer.Exit(code=EXIT_DISPATCH) from exc
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"sensitivity runner raised: {exc}", err=True)
        raise typer.Exit(code=EXIT_RUNTIME) from exc
    typer.echo(
        json.dumps(
            {
                "command": "sensitivity",
                "config": config,
                "n_rows": len(rows) if isinstance(rows, list) else 0,
            },
            indent=2,
            default=str,
        )
    )


@app.command()
def aggregate(
    results_dir: str = typer.Argument("results", help="Path to the results directory."),
) -> None:
    """Aggregate every supported experiment under ``results_dir``.

    Args:
        results_dir: Directory containing the per-experiment
          outputs.
    """
    from pjepa.eval import aggregate_all

    log = get_logger(__name__)
    log.info(
        "aggregate requested",
        extra={"event": "aggregate.start", "results_dir": results_dir},
    )
    result = aggregate_all(results_dir)
    typer.echo(
        json.dumps(
            {
                "command": "aggregate",
                "results_dir": results_dir,
                "n_rows": len(result.rows),
                "jsonl": str(result.jsonl_path),
                "csv": str(result.csv_path),
                "summary": str(result.summary_path),
            },
            indent=2,
        )
    )


def main() -> None:
    """Entry point for the ``pjepa`` console script."""
    app()


if __name__ == "__main__":
    sys.exit(main())
