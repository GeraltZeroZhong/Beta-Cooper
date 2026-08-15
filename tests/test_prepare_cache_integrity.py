from __future__ import annotations

import json
from pathlib import Path

import pytest

import cooper_beta.prepare_cache as prepare_cache
from cooper_beta.config import AppConfig, build_config
from cooper_beta.exceptions import InputContentChangedError
from cooper_beta.integrity import canonical_json_sha256, freeze_input_identity
from cooper_beta.prepare_cache import (
    PREPARE_CACHE_VERSION,
    build_prepare_cache_key,
    load_prepare_payloads,
    prepare_cache_path,
    store_prepare_payloads,
)


def _config(tmp_path: Path, executable: Path | None = None) -> AppConfig:
    return build_config(
        {
            "runtime.dssp_bin_path": str(executable) if executable is not None else None,
            "runtime.prepare_cache_enabled": True,
            "runtime.prepare_cache_dir": str(tmp_path / "cache"),
        }
    )


def _payloads(input_file: Path) -> list[dict[str, object]]:
    return [
        {
            "filename": "toy.pdb",
            "source_path": str(input_file),
            "chain": "A",
            "residues_data": [
                {
                    "res_id": 1,
                    "coord": [0.0, 1.0, 2.0],
                    "dssp_assignment_available": True,
                    "is_sheet": True,
                    "strand_node_id": "strand_0",
                    "polymer_index": 0,
                    "peptide_bond_distance_to_previous_angstrom": None,
                    "chain": "A",
                    "resseq": 1,
                    "icode": "",
                    "hetfield": "",
                    "res_uid": {"chain": "A", "hetfield": "", "resseq": 1, "icode": ""},
                }
            ],
            "strand_graph": {
                "author_chain_id": "A",
                "nodes": [
                    {
                        "node_id": "strand_0",
                        "start_polymer_index": 0,
                        "end_polymer_index": 0,
                    }
                ],
                "edges": [],
            },
            "degraded": False,
            "degradation_code": "",
            "degradation_reason": "",
        }
    ]


def _degraded_payloads(input_file: Path) -> list[dict[str, object]]:
    payloads = _payloads(input_file)
    payloads[0]["degraded"] = True
    payloads[0]["degradation_code"] = "DSSP_FAILED"
    payloads[0]["degradation_reason"] = "temporary DSSP failure"
    return payloads


def test_prepare_cache_key_tracks_preparation_source_fingerprint(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")
    executable = tmp_path / "mkdssp"
    executable.write_text("dssp\n", encoding="utf-8")
    executable.chmod(0o755)
    cfg = _config(tmp_path, executable)

    monkeypatch.setattr(
        prepare_cache,
        "_preparation_source_state",
        lambda: {"algorithm": "sha256", "sha256": "source-a", "files": []},
    )
    first_key = build_prepare_cache_key(str(input_file), cfg)
    monkeypatch.setattr(
        prepare_cache,
        "_preparation_source_state",
        lambda: {"algorithm": "sha256", "sha256": "source-b", "files": []},
    )
    second_key = build_prepare_cache_key(str(input_file), cfg)

    assert first_key != second_key


def test_prepare_cache_key_distinguishes_compressed_coordinate_formats(tmp_path: Path):
    pdb_file = tmp_path / "toy.pdb.gz"
    cif_file = tmp_path / "toy.cif.gz"
    identical_bytes = b"identical compressed payload"
    pdb_file.write_bytes(identical_bytes)
    cif_file.write_bytes(identical_bytes)
    cfg = _config(tmp_path)

    assert build_prepare_cache_key(str(pdb_file), cfg) != build_prepare_cache_key(
        str(cif_file), cfg
    )


def test_prepare_cache_envelope_records_and_revalidates_key_payload(tmp_path: Path):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")
    cfg = _config(tmp_path)
    payloads = _payloads(input_file)

    store_prepare_payloads(str(input_file), cfg, payloads)
    cache_path = prepare_cache_path(str(input_file), cfg)
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))

    assert envelope["cache_version"] == PREPARE_CACHE_VERSION
    assert envelope["cache_key"] == cache_path.stem
    source_state = envelope["cache_key_payload"]["prepare"]["producer"]["source"]
    assert source_state["sha256"]
    source_paths = {entry["path"] for entry in source_state["files"]}
    assert source_paths >= {
        "loader.py",
        "dssp_adapter.py",
        "strand_graph.py",
        "polymer_sequence.py",
        "models.py",
        "runtime.py",
        "prepare_cache.py",
        "config.py",
        "conf/config.yaml",
    }
    assert load_prepare_payloads(str(input_file), cfg) == payloads

    envelope["cache_key_payload"]["prepare"]["structure_loading"]["model_id"] = 999
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    assert load_prepare_payloads(str(input_file), cfg) is None
    assert not cache_path.exists()


def test_prepare_cache_rejects_tampered_payload_content(tmp_path: Path):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")
    cfg = _config(tmp_path)
    payloads = _payloads(input_file)
    store_prepare_payloads(str(input_file), cfg, payloads)
    cache_path = prepare_cache_path(str(input_file), cfg)
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    envelope["payloads"][0]["chain"] = "B"
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    assert load_prepare_payloads(str(input_file), cfg) is None
    assert not cache_path.exists()


def test_prepare_cache_rejects_schema_invalid_payload_even_with_matching_digest(
    tmp_path: Path,
):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")
    cfg = _config(tmp_path)
    store_prepare_payloads(str(input_file), cfg, _payloads(input_file))
    cache_path = prepare_cache_path(str(input_file), cfg)
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    envelope["payloads"][0]["residues_data"][0]["is_sheet"] = "false"
    envelope["payloads_sha256"] = canonical_json_sha256(envelope["payloads"])
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    assert load_prepare_payloads(str(input_file), cfg) is None
    assert not cache_path.exists()


def test_prepare_cache_never_stores_degraded_payloads(tmp_path: Path):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")
    cfg = _config(tmp_path)

    store_prepare_payloads(str(input_file), cfg, _degraded_payloads(input_file))

    assert not prepare_cache_path(str(input_file), cfg).exists()


def test_prepare_cache_discards_stale_degraded_payloads(tmp_path: Path):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")
    cfg = _config(tmp_path)
    store_prepare_payloads(str(input_file), cfg, _payloads(input_file))
    cache_path = prepare_cache_path(str(input_file), cfg)
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    envelope["payloads"] = _degraded_payloads(input_file)
    envelope["payloads_sha256"] = canonical_json_sha256(envelope["payloads"])
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    assert load_prepare_payloads(str(input_file), cfg) is None
    assert not cache_path.exists()


def test_prepare_cache_hit_revalidates_frozen_input_before_return(tmp_path: Path):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("OLD\n", encoding="utf-8")
    cfg = _config(tmp_path)
    frozen = freeze_input_identity(input_file)
    store_prepare_payloads(
        str(input_file),
        cfg,
        _payloads(input_file),
        input_identity=frozen,
    )
    input_file.write_text("NEW\n", encoding="utf-8")

    with pytest.raises(InputContentChangedError, match="Input content changed during the run"):
        load_prepare_payloads(
            str(input_file),
            cfg,
            input_identity=frozen,
        )


def test_prepare_cache_store_removes_entry_if_input_changes_during_publication(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("OLD\n", encoding="utf-8")
    cfg = _config(tmp_path)
    frozen = freeze_input_identity(input_file)
    cache_path = prepare_cache_path(str(input_file), cfg)
    original_atomic_write_json = prepare_cache.atomic_write_json

    def mutate_after_write(path, value, *, indent=2):
        written_path = original_atomic_write_json(path, value, indent=indent)
        input_file.write_text("NEW\n", encoding="utf-8")
        return written_path

    monkeypatch.setattr(prepare_cache, "atomic_write_json", mutate_after_write)

    with pytest.raises(InputContentChangedError, match="Input content changed during the run"):
        store_prepare_payloads(
            str(input_file),
            cfg,
            _payloads(input_file),
            input_identity=frozen,
        )

    assert not cache_path.exists()
