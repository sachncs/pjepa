"""Typer-based CLI for ``pjepa``.

The CLI exposes the canonical workflow:

* ``pjepa doctor`` — capability probe report.
* ``pjepa hardware`` — backend summary.
* ``pjepa benchmark {retrieval, distortion}`` — cheap validation
  experiments that run on the local machine.
* ``pjepa pretrain <config>`` — pretrain a JEPA encoder on a dataset.
* ``pjepa train <tu|cl> <config>`` — train (supervised or CL) on a dataset.
* ``pjepa eval  <tu|cl|ogb> <run-dir>`` — evaluate a saved checkpoint.
"""

from __future__ import annotations

import json
import sys

import typer

import torch

from pjepa import __version__
from pjepa.hardware import detect_capabilities, detect_backend, capabilities_as_dict
from pjepa.logging_setup import configure_logging, LogFormat, get_logger

__all__ = ["app", "main"]


app = typer.Typer(
    name="pjepa",
    help="Persistent-JEPA: persistent graph world model for continual learning.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pjepa, version {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Print version and exit."
    ),
    log_format: str = typer.Option(
        LogFormat.HUMAN,
        "--log-format",
        help="Log format: HUMAN (default) or JSON.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
) -> None:
    """Global CLI options."""
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
        raise typer.Exit(code=2)


@app.command()
def benchmark(name: str = typer.Argument(..., help="retrieval | distortion")) -> None:
    """Run a cheap validation benchmark on the local machine.

    Args:
        name: Which benchmark to run. ``retrieval`` validates the
          (1 - 1/e) guarantee on synthetic submodular functions.
          ``distortion`` measures hyperbolic vs Euclidean distortion
          on synthetic trees.
    """
    log = get_logger(__name__)
    log.info("benchmark requested", extra={"event": "benchmark.start", "name": name})
    if name == "retrieval":
        from experiments.run_exp_a_retrieval import run as run_retrieval  # type: ignore

        result = run_retrieval()
    elif name == "distortion":
        from experiments.run_exp_b_distortion import run as run_distortion  # type: ignore

        result = run_distortion()
    else:
        typer.echo(f"unknown benchmark: {name!r}; choose 'retrieval' or 'distortion'")
        raise typer.Exit(code=2)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def pretrain(config: str = typer.Argument(..., help="Path to a YAML config file.")) -> None:
    """Pretrain a JEPA encoder using the supplied config.

    Args:
        config: Path to the YAML configuration.
    """
    log = get_logger(__name__)
    log.info("pretrain requested", extra={"event": "pretrain.start", "config": config})
    typer.echo(f"pretrain requested: config={config} (implementation in Phase 5)")


@app.command()
def train(
    dataset: str = typer.Argument(..., help="tu | cl | ogb"),
    config: str = typer.Argument(..., help="Path to a YAML config file."),
) -> None:
    """Train (supervised or continual) on the named dataset family.

    Args:
        dataset: One of ``tu``, ``cl``, or ``ogb``.
        config: Path to the YAML configuration.
    """
    log = get_logger(__name__)
    log.info("train requested", extra={"event": "train.start", "dataset": dataset, "config": config})
    typer.echo(f"train requested: dataset={dataset} config={config} (implementation in Phase 5)")


@app.command()
def eval(
    dataset: str = typer.Argument(..., help="tu | cl | ogb"),
    run_dir: str = typer.Argument(..., help="Path to a checkpoint directory."),
) -> None:
    """Evaluate a saved checkpoint on the named dataset family."""
    log = get_logger(__name__)
    log.info(
        "eval requested",
        extra={"event": "eval.start", "dataset": dataset, "run_dir": run_dir},
    )
    typer.echo(f"eval requested: dataset={dataset} run_dir={run_dir} (implementation in Phase 5)")


def main() -> None:
    """Entry point for the ``pjepa`` console script."""
    app()


if __name__ == "__main__":
    sys.exit(main())