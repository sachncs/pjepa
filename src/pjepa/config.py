"""Configuration loading and saving for the ``pjepa`` package.

Configurations are YAML files. A caller may supply a tuple of required
top-level section names; any deviation raises
:class:`pjepa.exceptions.ConfigError`. The implementation intentionally
avoids pulling Pydantic into the runtime surface so the core library
keeps a tiny dependency footprint. The loader is permissive about
*shape* (unknown top-level sections are allowed) but strict about
*type* (the root must remain a mapping; lists and scalars cannot stand
in for it).

Example configuration::

    experiment:
      name: tu_proteins_baseline
      dataset: PROTEINS
      seed_split: 0
      seed_model: 42
    training:
      epochs: 200
      batch_size: 32
      optimizer: adamw
      lr: 5.0e-4
      weight_decay: 1.0e-5
    model:
      hidden_dim: 128
      num_layers: 4
    pjepa:
      B: 64
      beta_ib: 1.0e-2
      lambda_mdl: 1.0e-3
      gamma_forward: 1.0e-4

This module is **synchronous** and **side-effect-free** outside the
filesystem. Concurrent calls from multiple workers to
:func:`save_config` against the same path may race; pass distinct paths
if that matters.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pjepa.exceptions import ConfigError

__all__ = ["load_config", "merge_configs", "save_config"]


def read_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file into a ``dict``; missing files raise :class:`ConfigError`.

    Args:
        path: Filesystem path to the YAML document.

    Returns:
        The parsed mapping, or ``{}`` for an empty file.

    Raises:
        ConfigError: If the file is missing, if PyYAML is not
            installed, or if the YAML root is not a mapping.
    """
    try:
        import yaml  # PyYAML is declared in pyproject [dependencies].
    except ImportError as exc:
        raise ConfigError(
            "load_config: PyYAML is not installed; install with `pip install pyyaml`"
        ) from exc
    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"load_config: file does not exist: {path}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"load_config: top-level YAML in {path} must be a mapping; got {type(loaded).__name__}"
        )
    return loaded


def load_config(
    path: str | os.PathLike[str],
    required_sections: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load and optionally validate a YAML configuration file.

    The returned dictionary shares its nested structure with the file;
    mutating it mutates the parsed view but not the file on disk.

    Args:
        path: Path to a YAML configuration file.
        required_sections: Optional tuple of section names that must be
            present in the configuration. ``None`` accepts any
            well-formed mapping.

    Returns:
        The loaded configuration as a ``dict``.

    Raises:
        ConfigError: If the file cannot be read, parsed, or fails
            validation, or if PyYAML is not installed.

    Example:
        >>> cfg = load_config("configs/tu.yaml")
        >>> cfg["training"]["epochs"]
        200
    """
    config = read_yaml_file(Path(path))
    if required_sections:
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"load_config: required section {section!r} missing from {path}")
    return config


def merge_configs(*configs: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge configurations, with later configurations taking precedence.

    Nested mappings are merged recursively. Non-mapping values are
    *overwritten* by the latest occurrence — there is no list-append
    semantics. The returned dictionary contains only string keys.

    Args:
        *configs: One or more mapping objects, in increasing order of
            precedence.

    Returns:
        A new dictionary containing the merged configuration.

    Raises:
        ConfigError: If a value collides between a mapping and a
            non-mapping at the same key.

    Example:
        >>> merge_configs({"a": {"b": 1}}, {"a": {"c": 2}})
        {'a': {'b': 1, 'c': 2}}
    """
    result: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            elif key in result and isinstance(result[key], dict) is not isinstance(value, dict):
                raise ConfigError(f"merge_configs: type collision for key {key!r}")
            else:
                result[key] = value
    return result


def save_config(config: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Write a configuration to a YAML file.

    The function opens the destination in write mode, truncating any
    existing file. The write is **not** crash-consistent: no fsync is
    issued and the rename is not atomic, so a process crash between
    open and close can leave a half-written file. Callers that need
    durability should write to a sibling temp file and ``os.replace``
    it into place.

    Args:
        config: The configuration to serialise.
        path: Destination path.

    Returns:
        None.

    Raises:
        ConfigError: If PyYAML is not installed or the parent
            directory does not exist.

    Example:
        >>> save_config({"training": {"epochs": 50}}, "configs/min.yaml")
    """
    try:
        import yaml  # PyYAML is declared in pyproject [dependencies].
    except ImportError as exc:
        raise ConfigError(
            "save_config: PyYAML is not installed; install with `pip install pyyaml`"
        ) from exc
    target = Path(path)
    if not target.parent.exists():
        raise ConfigError(f"save_config: parent directory does not exist: {target.parent}")
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(config), fh, sort_keys=False)
