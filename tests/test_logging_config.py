from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from cooper_beta.logging_config import configure_logging


def test_jsonl_logging_is_structured_and_never_reuses_an_existing_file(tmp_path: Path):
    log_path = tmp_path / "run.jsonl"
    resolved = configure_logging(level="INFO", console=False, jsonl_path=str(log_path))
    logging.getLogger("cooper_beta.test").error(
        "scientific failure",
        extra={
            "stage": "analysis",
            "structure_filename": "toy.pdb",
            "chain": "A",
            "error_code": "ANALYSIS_FAILED",
        },
    )
    for handler in logging.getLogger("cooper_beta").handlers:
        handler.flush()

    assert resolved == log_path.resolve()
    document = json.loads(log_path.read_text(encoding="utf-8"))
    assert document["message"] == "scientific failure"
    assert document["stage"] == "analysis"
    assert document["structure_filename"] == "toy.pdb"
    assert document["chain"] == "A"
    assert document["error_code"] == "ANALYSIS_FAILED"
    assert document["timestamp_utc"].endswith("Z")

    with pytest.raises(FileExistsError):
        configure_logging(level="INFO", console=False, jsonl_path=str(log_path))


def test_logging_configuration_does_not_mutate_root_handlers():
    root = logging.getLogger()
    before = list(root.handlers)

    configure_logging(level="WARNING", console=False, jsonl_path=None)

    assert root.handlers == before
