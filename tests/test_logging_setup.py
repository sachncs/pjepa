"""Tests for pjepa.logging_setup."""

from __future__ import annotations

import io
import json
import logging

import pytest

from pjepa.logging_setup import LogFormat, configure_logging, get_logger, log_event

__all__ = [
    "test_bad_format_rejected",
    "test_get_logger_returns_named_logger",
    "test_human_format_renders",
    "test_json_format_is_parseable",
    "test_log_event_includes_event_and_fields",
]


def test_human_format_renders() -> None:
    """HUMAN format emits a readable string on stderr."""
    configure_logging("INFO", LogFormat.HUMAN)
    log = get_logger("test_module")
    log.info("ready")
    assert log.name == "pjepa.test_module"


def test_bad_format_rejected() -> None:
    """An unknown format raises ValueError."""
    with pytest.raises(ValueError):
        configure_logging("INFO", "XML")


def test_get_logger_returns_named_logger() -> None:
    """get_logger returns a logger under the pjepa namespace."""
    log = get_logger("foo.bar")
    assert log.name == "pjepa.foo.bar"


def test_log_event_includes_event_and_fields() -> None:
    """log_event attaches an event name and arbitrary fields."""
    captured: dict = {}

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured["message"] = record.getMessage()
            captured["event"] = getattr(record, "event", None)
            captured["dataset"] = getattr(record, "dataset", None)

    handler = Capture()
    log = get_logger("event_test")
    log.addHandler(handler)
    log_event(log, "experiment.started", dataset="PROTEINS")
    log.removeHandler(handler)
    assert captured["message"] == "experiment.started"
    assert captured["event"] == "experiment.started"
    assert captured["dataset"] == "PROTEINS"


def test_json_format_is_parseable() -> None:
    """JSON format emits parseable JSON lines."""
    from pjepa.logging_setup import JsonLogFormatter

    log = get_logger("json_test")
    log.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(JsonLogFormatter())
    log.addHandler(handler)
    try:
        log_event(log, "test.event", count=3)
        handler.flush()
    finally:
        log.removeHandler(handler)
        log.propagate = True
    line = stream.getvalue().strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["event"] == "test.event"
    assert parsed["count"] == 3


def test_default_level_is_info() -> None:
    """Calling configure_logging with no arguments yields INFO level."""
    configure_logging()
    log = get_logger("default_level")
    assert log.getEffectiveLevel() == logging.INFO
