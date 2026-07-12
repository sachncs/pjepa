"""Structured logging for the pjepa package.

The library never uses ``print`` for status output. Instead, every
module obtains a logger via :func:`get_logger` and emits structured
records through the configuration established by :func:`configure_logging`.
Two formats are supported:

* ``HUMAN`` — coloured, human-readable lines on stderr (default for dev).
* ``JSON``  — one JSON object per line on stderr (default for CI/prod).

Both modes route through Python's standard :mod:`logging` framework so
that third-party libraries (e.g. PyTorch, PyG) participate in the same
hierarchy.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Final

__all__ = [
    "LogFormat",
    "configure_logging",
    "get_logger",
    "log_event",
]


class LogFormat(str):
    """String constants for the supported log formats."""

    HUMAN: Final[str] = "HUMAN"
    JSON: Final[str] = "JSON"


_CONFIGURED = False


def configure_logging(level: str = "INFO", fmt: str = LogFormat.HUMAN) -> None:
    """Configure the root logger for the pjepa package.

    The function is idempotent: calling it multiple times replaces the
    handlers on the ``pjepa`` logger but leaves the root logger
    untouched so that PyTorch and other libraries retain their default
    behaviour.

    Args:
        level: A standard logging level name (``"DEBUG"``, ``"INFO"``,
          ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``).
        fmt: Either :data:`LogFormat.HUMAN` or :data:`LogFormat.JSON`.

    Returns:
        None.

    Raises:
        ValueError: If ``fmt`` is neither ``HUMAN`` nor ``JSON``.

    Example:
        >>> configure_logging("INFO", LogFormat.JSON)
    """
    global _CONFIGURED
    if fmt not in (LogFormat.HUMAN, LogFormat.JSON):
        raise ValueError(f"configure_logging: unknown format {fmt!r}")

    package_logger = logging.getLogger("pjepa")
    package_logger.setLevel(level.upper())
    package_logger.propagate = False

    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    if fmt == LogFormat.JSON:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_HumanFormatter())
    package_logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``pjepa`` namespace.

    The configuration is initialised lazily to :data:`LogFormat.HUMAN`
    on first use so that test modules and notebooks that import
    ``pjepa`` never fail because of a missing logging configuration.

    Args:
        name: A dotted module path. Typically ``__name__``.

    Returns:
        A configured :class:`logging.Logger` instance.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("ready")
    """
    if not _CONFIGURED:
        configure_logging()
    if not name.startswith("pjepa"):
        name = f"pjepa.{name}"
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured event with arbitrary keyword fields.

    Args:
        logger: A logger obtained from :func:`get_logger`.
        event: A short snake_case event identifier such as
          ``"experiment.completed"``.
        **fields: Arbitrary structured fields attached to the event.

    Returns:
        None.

    Example:
        >>> log = get_logger(__name__)
        >>> log_event(log, "checkpoint.saved", path="/tmp/ckpt.pt")
    """
    extras = {"event": event, **fields}
    logger.info(event, extra=extras)


class _HumanFormatter(logging.Formatter):
    """Plain formatter emitting ``LEVEL module — message`` lines."""

    def format(self, record: logging.LogRecord) -> str:
        extras = getattr(record, "event", None)
        prefix = f"{record.levelname:<7} {record.name}"
        if extras is not None:
            extras_dict = {
                k: v
                for k, v in record.__dict__.items()
                if k
                not in (
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "event",
                    "message",
                )
            }
            extras_str = " ".join(f"{k}={v}" for k, v in extras_dict.items())
            return f"{prefix} — {extras} {extras_str}".rstrip()
        return f"{prefix} — {record.getMessage()}"


class _JsonFormatter(logging.Formatter):
    """JSON-line formatter for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "event",
                "message",
            )
        }
        payload.update(extras)
        if record.exc_info is not None:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
