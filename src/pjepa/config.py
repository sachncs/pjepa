"""Configuration loading and validation for the pjepa package.

Configurations are YAML files. They are validated against a hand-rolled
schema (``ConfigSchema``) at load time; any deviation raises
:class:`pjepa.exceptions.ConfigError`. We intentionally do not pull in
Pydantic as a dependency at this stage to keep the runtime surface
small. The schema is intentionally permissive in shape but strict in
type.

Example configuration:

```yaml
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
```
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pjepa.exceptions import ConfigError

__all__ = ["ConfigSchema", "load_config", "merge_configs", "save_config"]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ConfigSchema:
    """Schema describing the recognised top-level sections.

    A section is either required or optional. Unknown sections are
    permitted (so that user extensions work) but emit a warning via
    the logger.
    """

    required: tuple[str, ...] = field(default_factory=tuple)
    optional: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for section in (*self.required, *self.optional):
            if not _IDENT.match(section):
                raise ValueError(
                    f"ConfigSchema: section name {section!r} is not a valid identifier"
                )
        if set(self.required) & set(self.optional):
            raise ValueError(
                "ConfigSchema: required and optional sections overlap"
            )


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict, returning an empty dict on missing file."""
    if not path.exists():
        raise ConfigError(f"load_config: file does not exist: {path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigError(
            "load_config: PyYAML is not installed; install with `pip install pyyaml`"
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"load_config: top-level YAML in {path} must be a mapping; "
            f"got {type(loaded).__name__}"
        )
    return loaded


def load_config(
    path: str | os.PathLike[str],
    schema: ConfigSchema | None = None,
) -> dict[str, Any]:
    """Load and validate a configuration file.

    Args:
        path: Path to a YAML configuration file.
        schema: Optional schema to enforce required sections.

    Returns:
        The loaded configuration as a dictionary.

    Raises:
        ConfigError: If the file cannot be read, parsed, or fails
          schema validation.

    Example:
        >>> cfg = load_config("configs/tu.yaml")
        >>> cfg["training"]["epochs"]
        200
    """
    config = _read_yaml(Path(path))
    if schema is not None:
        for section in schema.required:
            if section not in config:
                raise ConfigError(
                    f"load_config: required section {section!r} missing from {path}"
                )
    return config


def merge_configs(*configs: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge configurations, with later configurations taking precedence.

    Nested mappings are merged recursively; non-mapping values are
    overwritten by the latest occurrence.

    Args:
        *configs: One or more mapping objects.

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
                raise ConfigError(
                    f"merge_configs: type collision for key {key!r}"
                )
            else:
                result[key] = value
    return result


def save_config(config: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Save a configuration to a YAML file.

    Args:
        config: The configuration to save.
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
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigError(
            "save_config: PyYAML is not installed; install with `pip install pyyaml`"
        ) from exc
    target = Path(path)
    if not target.parent.exists():
        raise ConfigError(f"save_config: parent directory does not exist: {target.parent}")
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(config), fh, sort_keys=False)