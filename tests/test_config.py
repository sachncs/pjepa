"""Tests for pjepa.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pjepa.config import ConfigSchema, load_config, merge_configs, save_config
from pjepa.exceptions import ConfigError

__all__ = [
    "test_happy_load_yaml",
    "test_bad_missing_file",
    "test_bad_yaml_not_mapping",
    "test_bad_required_section_missing",
    "test_round_trip_save_load",
    "test_merge_deep",
    "test_merge_type_collision",
    "test_schema_invalid_identifier",
    "test_save_to_missing_directory",
]


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_happy_load_yaml(tmp_path: Path) -> None:
    """A well-formed YAML file loads as a dict."""
    path = _write_yaml(tmp_path, "training:\n  epochs: 100\n")
    config = load_config(path)
    assert config["training"]["epochs"] == 100


def test_bad_missing_file(tmp_path: Path) -> None:
    """A missing path raises ConfigError."""
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_bad_yaml_not_mapping(tmp_path: Path) -> None:
    """A YAML whose root is a list raises ConfigError."""
    path = _write_yaml(tmp_path, "- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_bad_required_section_missing(tmp_path: Path) -> None:
    """A required section absent from the file raises ConfigError."""
    path = _write_yaml(tmp_path, "training:\n  epochs: 10\n")
    schema = ConfigSchema(required=("experiment",))
    with pytest.raises(ConfigError):
        load_config(path, schema)


def test_round_trip_save_load(tmp_path: Path) -> None:
    """Saving and reloading yields an equivalent configuration."""
    target = tmp_path / "out.yaml"
    config = {"training": {"epochs": 50}, "model": {"hidden_dim": 64}}
    save_config(config, target)
    loaded = load_config(target)
    assert loaded == config


def test_merge_deep() -> None:
    """Nested mappings are merged recursively."""
    merged = merge_configs({"a": {"b": 1, "c": 2}}, {"a": {"c": 3, "d": 4}})
    assert merged == {"a": {"b": 1, "c": 3, "d": 4}}


def test_merge_type_collision() -> None:
    """A type collision between mapping and non-mapping raises ConfigError."""
    with pytest.raises(ConfigError):
        merge_configs({"a": {"b": 1}}, {"a": "scalar"})


def test_schema_invalid_identifier() -> None:
    """An invalid section name raises ValueError at construction time."""
    with pytest.raises(ValueError):
        ConfigSchema(required=("1bad",))


def test_save_to_missing_directory(tmp_path: Path) -> None:
    """Saving to a non-existent directory raises ConfigError."""
    with pytest.raises(ConfigError):
        save_config({}, tmp_path / "missing" / "cfg.yaml")


def test_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    """An empty YAML file yields an empty dict, not None."""
    path = _write_yaml(tmp_path, "")
    assert load_config(path) == {}