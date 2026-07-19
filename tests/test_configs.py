"""Tests for the bundled YAML configurations."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pjepa.config import load_config

__all__ = [
    "test_baseline_configs_have_baseline_field",
    "test_cl_config_has_required_fields",
    "test_default_config_loads",
    "test_eight_baseline_configs_present",
    "test_ogb_config_has_required_fields",
    "test_tu_config_has_required_fields",
    "test_yaml_configs_are_parseable",
]


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
EXPECTED_CONFIGS: tuple[str, ...] = (
    "default.yaml",
    "tu.yaml",
    "cl.yaml",
    "ogb.yaml",
    "baseline_gcn.yaml",
    "baseline_gin.yaml",
    "baseline_graphmae.yaml",
    "baseline_graphcl.yaml",
    "baseline_infograph.yaml",
    "baseline_naive.yaml",
    "baseline_ewc.yaml",
    "baseline_gem.yaml",
)


def test_eight_baseline_configs_present() -> None:
    """The eight advertised baseline YAML files exist."""
    for filename in EXPECTED_CONFIGS:
        assert (CONFIG_DIR / filename).exists(), f"missing config: {filename}"
    baseline_files = sorted(p.name for p in CONFIG_DIR.glob("baseline_*.yaml"))
    assert len(baseline_files) == 8


def test_yaml_configs_are_parseable() -> None:
    """Every YAML config loads with ``load_config``."""
    for filename in EXPECTED_CONFIGS:
        path = CONFIG_DIR / filename
        loaded = load_config(path)
        assert isinstance(loaded, dict)
        assert loaded, f"{filename} should not be empty"


def test_default_config_loads() -> None:
    """The default config has the documented top-level sections."""
    loaded = load_config(CONFIG_DIR / "default.yaml")
    assert "experiment" in loaded
    assert "training" in loaded
    assert "model" in loaded
    assert "pjepa" in loaded


def test_tu_config_has_required_fields() -> None:
    """The TU config has the fields consumed by the TU experiment runner."""
    loaded = load_config(CONFIG_DIR / "tu.yaml")
    assert loaded["experiment"]["datasets"]
    assert "PROTEINS" in loaded["experiment"]["datasets"]
    assert "lr" in loaded["training"]
    assert "B" in loaded["pjepa"]
    assert "beta_ib" in loaded["pjepa"]


def test_cl_config_has_required_fields() -> None:
    """The CL config has the fields consumed by the CL experiment runner."""
    loaded = load_config(CONFIG_DIR / "cl.yaml")
    assert loaded["experiment"]["datasets"]
    assert loaded["experiment"]["n_tasks"] == 2
    assert "lr" in loaded["training"]


def test_ogb_config_has_required_fields() -> None:
    """The OGB config has the fields consumed by the OGB experiment runner."""
    loaded = load_config(CONFIG_DIR / "ogb.yaml")
    assert loaded["experiment"]["dataset"] == "ogbn-arxiv"
    assert loaded["training"]["epochs"] >= 10


@pytest.mark.parametrize("filename", sorted(p.name for p in CONFIG_DIR.glob("baseline_*.yaml")))
def test_baseline_configs_have_baseline_field(filename: str) -> None:
    """Every baseline config advertises a ``baseline`` key."""
    path = CONFIG_DIR / filename
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    assert isinstance(raw, dict)
    assert "experiment" in raw
    assert "baseline" in raw["experiment"], f"{filename} missing baseline field"
    assert raw["experiment"]["baseline"]
