from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cooper_beta.config import build_config
from cooper_beta.exceptions import (
    OutputArtifactError,
    OutputArtifactExistsError,
)
from cooper_beta.integrity import file_sha256
from cooper_beta.output_artifacts import OutputArtifactTransaction
from cooper_beta.provenance import write_run_manifest
from cooper_beta.results import write_results_csv

STARTED_AT = "2026-08-11T12:00:00Z"


def test_existing_artifact_policy_error_preserves_csv_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    manifest = Path(f"{output}.manifest.json")
    output.write_text("old csv\n", encoding="utf-8")
    manifest.write_text('{"status": "complete", "old": true}\n', encoding="utf-8")

    with pytest.raises(OutputArtifactExistsError, match="Refusing to overwrite"):
        with OutputArtifactTransaction(
            output,
            write_manifest=True,
            existing_artifact_policy="error",
            started_at_utc=STARTED_AT,
        ):
            raise AssertionError("unreachable")

    assert output.read_text(encoding="utf-8") == "old csv\n"
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "status": "complete",
        "old": True,
    }


def test_failed_transaction_replaces_stale_sidecar_with_explicit_state(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    manifest = Path(f"{output}.manifest.json")
    output.write_text("old csv\n", encoding="utf-8")
    manifest.write_text('{"status": "complete", "run_id": "old"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="analysis failed"):
        with OutputArtifactTransaction(
            output,
            write_manifest=True,
            existing_artifact_policy="replace",
            started_at_utc=STARTED_AT,
            run_id="new-run",
        ) as transaction:
            running = json.loads(manifest.read_text(encoding="utf-8"))
            assert running["status"] == "running"
            assert running["run_id"] == "new-run"
            assert running["output_sha256"] is None
            assert running["artifact_binding"]["committed_by_run"] is False
            assert transaction.output_path == output.resolve()
            raise RuntimeError("analysis failed")

    failed = json.loads(manifest.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["run_id"] == "new-run"
    assert failed["failure"] == {"message": "analysis failed", "type": "RuntimeError"}
    assert failed["output_sha256"] is None
    assert output.read_text(encoding="utf-8") == "old csv\n"


def test_complete_transaction_binds_run_identity_size_and_csv_hash(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    config = build_config({"output.csv_path": str(output)})

    with OutputArtifactTransaction(
        output,
        write_manifest=True,
        existing_artifact_policy="error",
        started_at_utc=STARTED_AT,
        run_id="bound-run",
    ) as transaction:
        write_results_csv([], str(transaction.output_path))
        transaction.record_csv_commit()
        write_run_manifest(
            config=config,
            input_files=[],
            output_path=str(transaction.output_path),
            started_at_utc=STARTED_AT,
            run_id=transaction.run_id,
        )
        transaction.mark_complete()

    document = json.loads(Path(f"{output}.manifest.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["status"] == "complete"
    assert document["run_id"] == "bound-run"
    assert document["artifact_binding"] == {
        "committed_by_run": True,
        "csv_path": str(output.resolve()),
        "csv_sha256": file_sha256(output),
        "csv_size": output.stat().st_size,
        "run_id": "bound-run",
    }


def test_active_lock_rejects_a_second_process(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    child = f"""
from cooper_beta.exceptions import OutputArtifactBusyError
from cooper_beta.output_artifacts import OutputArtifactTransaction
try:
    with OutputArtifactTransaction(
        {str(output)!r},
        write_manifest=True,
        existing_artifact_policy='replace',
        started_at_utc={STARTED_AT!r},
    ):
        raise SystemExit(9)
except OutputArtifactBusyError:
    raise SystemExit(0)
"""

    with pytest.raises(RuntimeError, match="release parent lock"):
        with OutputArtifactTransaction(
            output,
            write_manifest=True,
            existing_artifact_policy="replace",
            started_at_utc=STARTED_AT,
            run_id="parent-run",
        ):
            completed = subprocess.run([sys.executable, "-c", child], check=False)
            assert completed.returncode == 0
            raise RuntimeError("release parent lock")


def test_process_crash_releases_lock_and_leaves_running_manifest(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    child = f"""
import os
from cooper_beta.output_artifacts import OutputArtifactTransaction
transaction = OutputArtifactTransaction(
    {str(output)!r},
    write_manifest=True,
    existing_artifact_policy='replace',
    started_at_utc={STARTED_AT!r},
    run_id='crashed-run',
)
transaction.__enter__()
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", child], check=True)

    manifest = Path(f"{output}.manifest.json")
    crashed = json.loads(manifest.read_text(encoding="utf-8"))
    assert crashed["status"] == "running"
    assert crashed["run_id"] == "crashed-run"

    with pytest.raises(OutputArtifactError, match="ended without"):
        with OutputArtifactTransaction(
            output,
            write_manifest=True,
            existing_artifact_policy="replace",
            started_at_utc=STARTED_AT,
            run_id="recovery-run",
        ):
            pass

    recovered = json.loads(manifest.read_text(encoding="utf-8"))
    assert recovered["status"] == "failed"
    assert recovered["run_id"] == "recovery-run"
