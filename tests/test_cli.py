"""Tests for the cross-cutting CLI dispatcher."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pjepa.cli.app import (
    EXIT_CONFIG,
    EXIT_DISPATCH,
    EXIT_RUNTIME,
    RUNNERS,
    app,
)

__all__ = [
    "test_app_lists_expected_subcommands",
    "test_baseline_smoke_accepts_yaml_path",
    "test_benchmark_distortion_dispatches",
    "test_benchmark_rejects_unknown_name",
    "test_exit_codes_are_distinct",
    "test_hardware_command_runs",
    "test_pretrain_command_reports_resolved_config",
    "test_runners_table_covers_every_command",
    "test_train_rejects_unknown_dataset",
    "test_version_flag_prints_version",
]


runner = CliRunner()


def test_version_flag_prints_version() -> None:
    """`pjepa --version` prints the package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "pjepa" in result.stdout


def test_app_lists_expected_subcommands() -> None:
    """The Typer app advertises every documented subcommand."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "hardware",
        "doctor",
        "benchmark",
        "pretrain",
        "train",
        "tune",
        "baseline-smoke",
        "decoupling",
        "ablation",
        "sensitivity",
        "aggregate",
    ):
        assert command in result.stdout


def test_exit_codes_are_distinct() -> None:
    """Distinct exit codes let CI tell config / dispatch / runtime failures apart."""
    assert EXIT_CONFIG != EXIT_DISPATCH != EXIT_RUNTIME != EXIT_CONFIG
    assert {EXIT_CONFIG, EXIT_DISPATCH, EXIT_RUNTIME} == {2, 3, 4}


def test_runners_table_covers_every_command() -> None:
    """The single RUNNERS table is the source of truth for dispatch."""
    assert "train.tu" in RUNNERS
    assert "train.cl" in RUNNERS
    assert "train.ogb" in RUNNERS
    assert RUNNERS["train.ogb"][1] == "run_ogb_experiment"
    assert "tune.tu" in RUNNERS
    assert "decoupling" in RUNNERS
    assert "ablation" in RUNNERS
    assert "sensitivity" in RUNNERS
    assert "benchmark.retrieval" in RUNNERS
    assert "benchmark.distortion" in RUNNERS
    assert "benchmark.encoder-ablation" in RUNNERS


def test_hardware_command_runs() -> None:
    """The hardware command runs and prints a backend line."""
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "backend=" in result.stdout


def test_benchmark_distortion_dispatches() -> None:
    """The benchmark command dispatches to the named experiment runner."""
    result = runner.invoke(app, ["benchmark", "distortion"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "rows" in payload or "all_pass" in payload


def test_benchmark_rejects_unknown_name() -> None:
    """The benchmark command rejects names outside the documented set."""
    result = runner.invoke(app, ["benchmark", "does-not-exist"])
    assert result.exit_code == EXIT_CONFIG
    assert "unknown benchmark" in result.stdout


def test_train_rejects_unknown_dataset() -> None:
    """The train command rejects dataset families outside the documented set."""
    result = runner.invoke(app, ["train", "unknown", "configs/default.yaml"])
    assert result.exit_code == EXIT_CONFIG
    assert "unknown dataset" in result.stdout


def test_pretrain_command_reports_resolved_config(tmp_path: Path) -> None:
    """The pretrain command returns a JSON report and resolves the config epoch count."""
    cfg = tmp_path / "pretrain.yaml"
    cfg.write_text("training:\n  epochs: 7\n", encoding="utf-8")
    result = runner.invoke(app, ["pretrain", str(cfg)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "pretrain"
    assert payload["epochs"] == 7
    assert "smoke" in payload


def test_baseline_smoke_accepts_yaml_path(tmp_path: Path) -> None:
    """`baseline-smoke` reads the YAML and reports the resolved configuration."""
    cfg = tmp_path / "gcn.yaml"
    cfg.write_text(
        "training:\n  epochs: 3\nmodel:\n  hidden_dim: 32\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["baseline-smoke", "gcn", str(cfg)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["baseline"] == "gcn"
    smoke = payload["smoke"]
    assert smoke["ran"] is True
    assert smoke["hidden_dim"] == 32


def test_aggregate_emits_canonical_files(tmp_path: Path) -> None:
    """`pjepa aggregate` writes the canonical artefacts even when empty."""
    result = runner.invoke(app, ["aggregate", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "all_runs.jsonl").exists()
    assert (tmp_path / "tables" / "all_runs.csv").exists()
    assert (tmp_path / "tables" / "summary.md").exists()
    summary = (tmp_path / "tables" / "summary.md").read_text(encoding="utf-8")
    assert "No run results found" in summary


def test_pretrain_missing_config_does_not_raise(tmp_path: Path) -> None:
    """Pretrain still produces a report when the YAML is missing or malformed."""
    bogus = tmp_path / "missing.yaml"
    result = runner.invoke(app, ["pretrain", str(bogus)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "pretrain"
    assert payload["epochs"] == 2
    assert "smoke" in payload
