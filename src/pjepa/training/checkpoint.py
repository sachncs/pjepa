"""Checkpoint save/load with sharded persistent graph support.

A checkpoint is a directory containing one ``.pt`` file per component
plus a ``metadata.json`` summary. The interface is intentionally
simple so it can be reused by both pretraining and continual-learning
loops.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pjepa.exceptions import CheckpointError

__all__ = ["Checkpoint", "save_checkpoint", "load_checkpoint"]


@dataclass(frozen=True)
class Checkpoint:
    """In-memory representation of a training checkpoint.

    Attributes:
        encoder_state: The encoder state dict.
        predictor_state: The predictor state dict.
        target_state: The target encoder state dict.
        optimizer_state: The optimiser state dict.
        epoch: Epoch at which the checkpoint was taken.
        loss: Mean epoch loss.
        extras: Optional additional state to persist.
    """

    encoder_state: dict[str, torch.Tensor]
    predictor_state: dict[str, torch.Tensor]
    target_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, torch.Tensor]
    epoch: int
    loss: float
    extras: dict[str, object] = field(default_factory=dict)


def save_checkpoint(
    checkpoint: Checkpoint,
    directory: str | os.PathLike[str],
    run_id: str,
) -> Path:
    """Save a checkpoint to ``directory/run_id``.

    Args:
        checkpoint: The checkpoint to save.
        directory: Parent directory.
        run_id: Unique identifier for this checkpoint.

    Returns:
        The path of the saved checkpoint directory.

    Raises:
        CheckpointError: If the parent directory does not exist or
          the save fails.
    """
    target = Path(directory) / run_id
    parent = Path(directory)
    if not parent.exists():
        raise CheckpointError(f"save_checkpoint: directory does not exist: {parent}")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CheckpointError(f"save_checkpoint: cannot create {target}: {exc}") from exc
    torch.save(checkpoint.encoder_state, target / "encoder.pt")
    torch.save(checkpoint.predictor_state, target / "predictor.pt")
    torch.save(checkpoint.target_state, target / "target.pt")
    torch.save(checkpoint.optimizer_state, target / "optimizer.pt")
    metadata = {
        "epoch": checkpoint.epoch,
        "loss": checkpoint.loss,
        "extras": {k: _serialise(v) for k, v in checkpoint.extras.items()},
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def load_checkpoint(
    path: str | os.PathLike[str],
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> Checkpoint:
    """Load a checkpoint from ``path``.

    Args:
        path: Path to the checkpoint directory created by
          :func:`save_checkpoint`.
        optimizer: Optional optimiser whose state to populate.
        device: Optional device to map tensors onto.

    Returns:
        A populated :class:`Checkpoint`.

    Raises:
        CheckpointError: If the directory is missing or malformed.
    """
    target = Path(path)
    if not target.is_dir():
        raise CheckpointError(f"load_checkpoint: not a directory: {target}")
    try:
        encoder_state = torch.load(target / "encoder.pt", map_location=device, weights_only=True)
        predictor_state = torch.load(target / "predictor.pt", map_location=device, weights_only=True)
        target_state = torch.load(target / "target.pt", map_location=device, weights_only=True)
        optimizer_state = torch.load(
            target / "optimizer.pt", map_location=device, weights_only=True
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise CheckpointError(f"load_checkpoint: missing or malformed files: {exc}") from exc
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    return Checkpoint(
        encoder_state=encoder_state,
        predictor_state=predictor_state,
        target_state=target_state,
        optimizer_state=optimizer_state,
        epoch=int(metadata.get("epoch", 0)),
        loss=float(metadata.get("loss", 0.0)),
        extras=metadata.get("extras", {}),
    )


def _serialise(value: object) -> object:
    """Serialise values that are not natively JSON-encodable."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialise(v) for k, v in value.items()}
    return repr(value)