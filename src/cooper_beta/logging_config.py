from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from .integrity import to_jsonable

LOGGER_NAMESPACE: Final = "cooper_beta"
_STRUCTURED_FIELDS: Final = (
    "run_id",
    "stage",
    "structure_filename",
    "source_path",
    "chain",
    "error_code",
)


def _utc_timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, timezone.utc).isoformat().replace("+00:00", "Z")


class _JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        document: dict[str, object] = {
            "timestamp_utc": _utc_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process_id": record.process,
            "process_name": record.processName,
        }
        for field_name in _STRUCTURED_FIELDS:
            value = getattr(record, field_name, None)
            if value not in (None, ""):
                document[field_name] = value
        if record.exc_info:
            document["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            to_jsonable(document),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = []
        for field_name in ("stage", "structure_filename", "chain", "error_code"):
            value = getattr(record, field_name, None)
            if value not in (None, ""):
                context.append(f"{field_name}={value}")
        suffix = f" ({', '.join(context)})" if context else ""
        return f"{_utc_timestamp(record)} {record.levelname} {record.name}: {record.getMessage()}{suffix}"


def _close_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except OSError:
            pass


def configure_logging(
    *,
    level: str,
    console: bool,
    jsonl_path: str | None,
) -> Path | None:
    """Configure only Cooper-Beta's logger hierarchy without mutating the root logger."""
    numeric_level = logging.getLevelName(str(level).upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level!r}.")

    logger = logging.getLogger(LOGGER_NAMESPACE)
    _close_owned_handlers(logger)
    logger.setLevel(numeric_level)
    logger.propagate = False

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(_ConsoleFormatter())
        logger.addHandler(console_handler)

    resolved_path: Path | None = None
    if jsonl_path is not None:
        resolved_path = Path(jsonl_path).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(resolved_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        file_handler = logging.FileHandler(resolved_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(_JsonLinesFormatter())
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return resolved_path
