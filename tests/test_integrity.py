from __future__ import annotations

import json
from pathlib import Path

import pytest

from cooper_beta.exceptions import InputContentChangedError
from cooper_beta.integrity import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    freeze_input_identity,
    verified_input_snapshot,
)


def test_canonical_json_hash_is_order_independent_and_strict():
    first = {"beta": [2, 3], "alpha": {"value": 1}}
    second = {"alpha": {"value": 1}, "beta": [2, 3]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    with pytest.raises(ValueError):
        canonical_json_bytes({"invalid": float("nan")})


def test_atomic_write_json_replaces_complete_document(tmp_path: Path):
    output = tmp_path / "manifest.json"
    output.write_text("old document\n", encoding="utf-8")

    result = atomic_write_json(output, {"version": 2, "nested": {"ok": True}})

    assert result == output
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "nested": {"ok": True},
        "version": 2,
    }
    assert file_sha256(output)
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_atomic_write_json_preserves_existing_file_when_serialization_fails(tmp_path: Path):
    output = tmp_path / "manifest.json"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError):
        atomic_write_json(output, {"invalid": float("inf")})

    assert output.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_atomic_write_text_preserves_existing_file_when_writer_fails(tmp_path: Path):
    output = tmp_path / "table.csv"
    output.write_text("existing\n", encoding="utf-8")

    def failing_writer(handle) -> None:
        handle.write("partial\n")
        raise RuntimeError("abort")

    with pytest.raises(RuntimeError, match="abort"):
        atomic_write_text(output, failing_writer, newline="")

    assert output.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize(
    ("filename", "expected_suffix"),
    [
        ("input.pdb", ".pdb"),
        ("input.cif", ".cif"),
        ("input.mmcif", ".mmcif"),
        ("input.pdb.gz", ".pdb.gz"),
        ("input.cif.gz", ".cif.gz"),
        ("input.mmcif.gz", ".mmcif.gz"),
    ],
)
def test_verified_input_snapshot_preserves_format_and_cleans_up(
    tmp_path: Path,
    filename: str,
    expected_suffix: str,
):
    source = tmp_path / filename
    source.write_bytes(b"frozen coordinate bytes")
    identity = freeze_input_identity(source)
    assert identity.suffix == expected_suffix

    with verified_input_snapshot(identity) as snapshot:
        captured_path = snapshot
        assert snapshot.name.endswith(expected_suffix)
        assert snapshot.read_bytes() == source.read_bytes()
        assert snapshot.stat().st_mode & 0o222 == 0

    assert not captured_path.exists()
    assert not captured_path.parent.exists()


def test_verified_input_snapshot_rejects_bytes_different_from_frozen_identity(
    tmp_path: Path,
):
    source = tmp_path / "input.pdb"
    source.write_bytes(b"A")
    identity = freeze_input_identity(source)
    source.write_bytes(b"B")

    with pytest.raises(InputContentChangedError, match="before its private snapshot"):
        with verified_input_snapshot(identity):
            pass
