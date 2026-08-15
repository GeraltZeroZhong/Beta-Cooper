from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import cooper_beta.provenance as provenance
from cooper_beta.bootstrap import RuntimeBootstrapState
from cooper_beta.config import build_config
from cooper_beta.constants import NATIVE_THREAD_ENV_NAMES
from cooper_beta.exceptions import InputContentChangedError
from cooper_beta.integrity import canonical_json_sha256, file_sha256, freeze_input_identity
from cooper_beta.provenance import build_run_manifest, write_run_manifest


def test_run_manifest_records_input_file_hash(tmp_path: Path):
    input_file = tmp_path / "model.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")

    manifest = build_run_manifest(
        config=build_config(),
        input_files=[str(input_file)],
        output_path=str(tmp_path / "results.csv"),
    )

    state = manifest["input_file_state"][0]
    assert state["path"] == str(input_file.resolve())
    assert state["exists"] is True
    assert state["size"] == len("HEADER\n")
    assert isinstance(state["sha256"], str)
    assert len(state["sha256"]) == 64
    assert isinstance(manifest["input_set_hash"], str)
    assert len(manifest["input_set_hash"]) == 64
    assert isinstance(manifest["input_inventory_hash"], str)


def test_input_set_hash_is_content_based_and_path_portable(tmp_path: Path):
    first = tmp_path / "first.pdb"
    second = tmp_path / "nested" / "renamed.pdb"
    second.parent.mkdir()
    first.write_text("HEADER\n", encoding="utf-8")
    second.write_bytes(first.read_bytes())

    first_manifest = build_run_manifest(
        config=build_config(),
        input_files=[str(first)],
        output_path=None,
    )
    second_manifest = build_run_manifest(
        config=build_config(),
        input_files=[str(second)],
        output_path=None,
    )

    assert first_manifest["input_set_hash"] == second_manifest["input_set_hash"]
    assert first_manifest["input_inventory_hash"] != second_manifest["input_inventory_hash"]


def test_run_manifest_records_versions_output_dssp_and_actual_thread_state(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "model.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")
    output_file = tmp_path / "results.csv"
    output_file.write_text("filename,result\nmodel.pdb,BARREL\n", encoding="utf-8")
    dssp = tmp_path / "mkdssp"
    dssp.write_text("#!/bin/sh\nprintf 'mkdssp version 4.5.3\\n'\n", encoding="utf-8")
    dssp.chmod(0o755)

    monkeypatch.setattr(
        provenance,
        "_source_project",
        lambda: (None, {"name": "cooper-beta", "version": "source-version"}),
    )
    monkeypatch.setattr(
        provenance,
        "_package_version",
        lambda distribution: (
            "installed-version" if distribution == "cooper-beta" else f"test-{distribution}"
        ),
    )
    expected_thread_environment = {
        env_name: f"threads-{index}"
        for index, env_name in enumerate(NATIVE_THREAD_ENV_NAMES, start=1)
    }
    for env_name, value in expected_thread_environment.items():
        monkeypatch.setenv(env_name, value)
    monkeypatch.setenv("PYTHONHASHSEED", "1234")
    monkeypatch.setattr(
        provenance,
        "_thread_pool_state",
        lambda: [{"prefix": "test-blas", "num_threads": 1}],
    )
    monkeypatch.setattr(
        provenance,
        "runtime_bootstrap_state",
        lambda: RuntimeBootstrapState(native_threads_per_process=1),
    )

    manifest = build_run_manifest(
        config=build_config({"runtime.dssp_bin_path": str(dssp)}),
        input_files=[str(input_file)],
        output_path=str(output_file),
    )

    assert manifest["schema_version"] == 1
    generated = str(manifest["generated_at_utc"])
    assert generated.endswith("Z")
    assert datetime.fromisoformat(generated.replace("Z", "+00:00")).tzinfo == timezone.utc
    assert manifest["project"] == {
        "name": "cooper-beta",
        "source_version": "source-version",
        "installed_distribution_version": "installed-version",
    }
    assert manifest["output_file_state"]["sha256"] == file_sha256(output_file)
    assert manifest["executables"]["dssp_resolved_path"] == str(dssp.resolve())
    assert manifest["executables"]["dssp_sha256"] == file_sha256(dssp)
    assert manifest["runtime"]["thread_environment"] == expected_thread_environment
    assert manifest["runtime"]["native_thread_pools"] == [{"prefix": "test-blas", "num_threads": 1}]
    assert manifest["runtime"]["native_thread_policy"] == {
        "requested_threads_per_process": 1,
        "applied_limit_in_parent_process": 1,
        "applied_limit_matches_request": True,
        "environment_matches_request": False,
        "loaded_pools_within_request": True,
        "worker_initializers_reapply_same_limit": True,
    }
    assert manifest["runtime"]["python_hash_policy"] == {
        "environment_value_observed": "1234",
        "runtime_assignment_performed": False,
        "current_interpreter_hash_secret_reconfigured": False,
        "algorithm_depends_on_hash_iteration_order": False,
    }
    producer_identity = manifest["producer_identity"]
    assert manifest["producer_identity_hash"] == canonical_json_sha256(producer_identity)
    assert producer_identity["dssp"] == {
        "version": "mkdssp version 4.5.3",
        "sha256": file_sha256(dssp),
    }
    assert producer_identity["runtime"]["native_thread_pools"] == [{"prefix": "test-blas"}]
    assert "git" not in producer_identity


def test_write_run_manifest_writes_strict_json_sidecar(tmp_path: Path, monkeypatch):
    output_file = tmp_path / "results.csv"
    output_file.write_text("filename,result\n", encoding="utf-8")
    monkeypatch.setattr(
        provenance,
        "_source_project",
        lambda: (None, {"name": "cooper-beta", "version": "source-version"}),
    )

    manifest_path = write_run_manifest(
        config=build_config(),
        input_files=[],
        output_path=str(output_file),
    )

    assert manifest_path == Path(f"{output_file}.manifest.json")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["output_file_state"]["sha256"] == file_sha256(output_file)
    assert document["generated_at_utc"].endswith("Z")


def test_run_manifest_can_skip_input_hashes_and_records_resolved_workers(tmp_path: Path):
    input_file = tmp_path / "model.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")

    manifest = build_run_manifest(
        config=build_config(),
        input_files=[str(input_file)],
        output_path=None,
        hash_input_files=False,
        resolved_analysis_workers=4,
        resolved_prepare_workers=2,
    )

    assert manifest["input_file_hashing_enabled"] is False
    assert manifest["input_file_state"][0]["exists"] is True
    assert manifest["input_file_state"][0]["sha256"] is None
    assert manifest["input_set_hash"] is None
    assert manifest["runtime"]["resolved_analysis_workers"] == 4
    assert manifest["runtime"]["resolved_prepare_workers"] == 2


def test_manifest_uses_frozen_input_identity_and_verified_mode_rejects_changes(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "model.pdb"
    input_file.write_text("OLD\n", encoding="utf-8")
    frozen = freeze_input_identity(input_file)
    monkeypatch.setattr(
        provenance,
        "freeze_input_identities",
        lambda paths: (_ for _ in ()).throw(AssertionError("must not recapture")),
    )

    manifest = build_run_manifest(
        config=build_config(),
        input_files=[str(input_file)],
        input_identities=[frozen],
        input_identities_verified=True,
        output_path=None,
    )

    assert manifest["input_file_state"][0]["sha256"] == frozen.sha256
    assert manifest["input_identity_policy"] == {
        "algorithm": "sha256",
        "polymer_position_policy": ("selected-model-mmcif-label-seq-or-pdb-unique-seqres"),
        "dssp_residue_coverage_policy": (
            "selected-model-declared-polypeptide-ca-residues-with-finite-n-ca-c-o"
        ),
        "frozen_before_parsing": True,
        "verified_before_artifact_publication": True,
        "hash_redacted_from_manifest": False,
    }

    input_file.write_text("NEW\n", encoding="utf-8")
    with pytest.raises(InputContentChangedError, match="Input content changed during the run"):
        build_run_manifest(
            config=build_config(),
            input_files=[str(input_file)],
            input_identities=[frozen],
            input_identities_verified=True,
            output_path=None,
        )
