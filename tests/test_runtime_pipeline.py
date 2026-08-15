from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import cooper_beta.pipeline as pipeline
from cooper_beta.config import build_config
from cooper_beta.constants import DEFAULT_RESULT_COLUMNS
from cooper_beta.exceptions import InputContentChangedError, InputValidationError
from cooper_beta.integrity import file_sha256, freeze_input_identity
from cooper_beta.pipeline import (
    _ordered_result_rows,
    _prepare_error_rows,
    apply_runtime_overrides,
    detect,
    discover_input_files,
    resolve_analysis_worker_count,
    resolve_prepare_worker_count,
    run_pipeline,
    run_pipeline_result,
)
from cooper_beta.results import print_results_summary, write_results_csv


def _complete_result_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "filename": "toy.pdb",
        "source_path": "/input/toy.pdb",
        "author_chain_id": "A",
        "result": "NON_BARREL",
        "result_stage": "decision",
        "dssp_unassigned_residue_count": 0,
        "strand_count": 8,
        "strand_adjacency_count": 7,
        "cycle_strand_count": 8,
        "cycle_strand_fraction": 1.0,
        "cycle_rank": 1,
        "reason": "test fixture",
        "error_code": "",
        "degraded": False,
    }
    assert set(row) == set(DEFAULT_RESULT_COLUMNS)
    row.update(overrides)
    if row["result"] == "BARREL":
        row["strand_adjacency_count"] = 8
    return row


def test_entrypoint_imports_do_not_eagerly_import_numerical_workers():
    script = """
import sys
import cooper_beta.pipeline
import cooper_beta.cli
assert 'cooper_beta.execution' not in sys.modules
assert 'numpy' not in sys.modules
assert 'pandas' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_runtime_overrides_replace_frozen_config_without_mutating_source(tmp_path: Path):
    cfg = build_config()

    updated = apply_runtime_overrides(
        cfg,
        input_path=str(tmp_path / "input"),
        workers=3,
        prepare_workers=2,
        out_csv=str(tmp_path / "results.csv"),
    )

    assert updated is not cfg
    assert updated.input.path == str(tmp_path / "input")
    assert updated.runtime.workers == 3
    assert updated.runtime.prepare_workers == 2
    assert updated.output.csv_path == str(tmp_path / "results.csv")
    assert cfg.input.path == ""
    assert cfg.runtime.workers is None
    assert cfg.runtime.prepare_workers is None
    assert cfg.output.csv_path == "cooper_beta_results.csv"


def test_bootstrap_has_no_random_configuration_path() -> None:
    import cooper_beta.bootstrap as bootstrap

    assert not hasattr(bootstrap, "configure_random_environment")
    assert not hasattr(bootstrap.runtime_bootstrap_state(), "random_seed")


def test_thread_configuration_limits_already_loaded_native_runtimes() -> None:
    script = """
import os
import numpy as np
from threadpoolctl import threadpool_info
from cooper_beta.bootstrap import configure_thread_environment, runtime_bootstrap_state

np.dot(np.ones((4, 4)), np.ones((4, 4)))
configure_thread_environment(1)
pools = threadpool_info()
assert all(pool['num_threads'] <= 1 for pool in pools)
assert runtime_bootstrap_state().native_threads_per_process == 1
for name in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    assert os.environ[name] == '1'
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_thread_configuration_fails_closed_when_runtime_ignores_limit(monkeypatch) -> None:
    import threadpoolctl

    from cooper_beta.bootstrap import configure_thread_environment
    from cooper_beta.constants import NATIVE_THREAD_ENV_NAMES

    with pytest.raises(ValueError, match="greater than zero"):
        configure_thread_environment(0)

    for name in NATIVE_THREAD_ENV_NAMES:
        monkeypatch.setenv(name, "baseline")
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda *, limits: object())
    monkeypatch.setattr(
        threadpoolctl,
        "threadpool_info",
        lambda: [{"prefix": "test-blas", "num_threads": 3}],
    )
    with pytest.raises(RuntimeError, match="test-blas=3"):
        configure_thread_environment(1)


def test_prepare_failure_is_serialized_as_complete_result_row(tmp_path: Path):
    from cooper_beta.preparation import PrepareFailure

    source_path = str(tmp_path / "bad.pdb")
    rows = _prepare_error_rows(
        [
            PrepareFailure(
                source_path=source_path,
                error_code="STRUCTURE_PARSE_FAILED",
                message="invalid coordinates",
            )
        ]
    )

    assert set(rows[0]) == set(DEFAULT_RESULT_COLUMNS)
    assert rows[0]["filename"] == "bad.pdb"
    assert rows[0]["source_path"] == source_path
    assert rows[0]["result"] == "ERROR"
    assert rows[0]["result_stage"] == "preparation"
    assert rows[0]["dssp_unassigned_residue_count"] == 0
    assert rows[0]["error_code"] == "STRUCTURE_PARSE_FAILED"
    assert rows[0]["reason"] == "invalid coordinates"


def test_prepare_worker_default_heuristic_follows_analysis_workers():
    assert resolve_prepare_worker_count(None, 1) == 1
    assert resolve_prepare_worker_count(None, 2) == 2
    assert resolve_prepare_worker_count(None, 4) == 4
    assert resolve_prepare_worker_count(None, 7) == 7
    assert resolve_prepare_worker_count(None, 8) == 8


def test_analysis_worker_override_is_respected():
    assert resolve_analysis_worker_count(6, cpu_reserve=1) == 6


def test_run_pipeline_resolves_dssp_path_once(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(tmp_path),
            "output.csv_path": str(output_file),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )
    calls = []

    def fake_require_dssp_binary(explicit_path):
        calls.append(explicit_path)
        return "/opt/dssp/mkdssp"

    def fake_iter_prepared_payload_batches(files, cfg_arg, prepare_workers, **kwargs):
        assert kwargs["show_progress"] is True
        assert files == [str(input_file)]
        assert prepare_workers == 1
        assert cfg_arg.runtime.dssp_bin_path == "/opt/dssp/mkdssp"
        yield [{"filename": "toy.pdb", "chain": "A", "residues_data": []}]

    def fake_run_analysis_stream(payload_batches, cfg_arg, workers, *, on_results=None, **kwargs):
        assert kwargs["show_progress"] is True
        batches = list(payload_batches)
        assert batches == [[{"filename": "toy.pdb", "chain": "A", "residues_data": []}]]
        assert workers == 1
        assert cfg_arg.runtime.dssp_bin_path == "/opt/dssp/mkdssp"
        rows = [_complete_result_row(filename="toy.pdb", author_chain_id="A")]
        if on_results is not None:
            on_results(rows)
        return rows

    monkeypatch.setattr("cooper_beta.pipeline.require_dssp_binary", fake_require_dssp_binary)
    monkeypatch.setattr(
        "cooper_beta.pipeline.iter_prepared_payload_batches",
        fake_iter_prepared_payload_batches,
    )
    monkeypatch.setattr("cooper_beta.pipeline.run_analysis_stream", fake_run_analysis_stream)
    monkeypatch.setattr("cooper_beta.pipeline.print_results_summary", lambda *args, **kwargs: None)

    rows = run_pipeline(cfg)

    assert rows == [_complete_result_row(filename="toy.pdb", author_chain_id="A")]
    assert calls == [None]


def test_run_pipeline_configures_environment_before_worker_dispatch(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")
    cfg = build_config(
        {
            "input.path": str(input_file),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
            "runtime.native_threads_per_process": 2,
        }
    )
    events: list[str] = []

    monkeypatch.setattr(
        pipeline,
        "configure_thread_environment",
        lambda value: events.append(f"threads:{value}"),
    )
    monkeypatch.setattr(
        pipeline,
        "require_dssp_binary",
        lambda value: events.append("dssp") or "/opt/dssp/mkdssp",
    )

    def fake_iter(*args, **kwargs):
        del args, kwargs
        events.append("prepare")
        yield [{"filename": "toy.pdb", "chain": "A", "residues_data": []}]

    def fake_analyze(payload_batches, *args, **kwargs):
        del args, kwargs
        events.append("analysis")
        list(payload_batches)
        return []

    monkeypatch.setattr(pipeline, "iter_prepared_payload_batches", fake_iter)
    monkeypatch.setattr(pipeline, "run_analysis_stream", fake_analyze)

    run_pipeline_result(
        cfg,
        write_csv=False,
        print_summary=False,
        strict_input=True,
        show_progress=False,
    )

    assert events == ["threads:2", "dssp", "analysis", "prepare"]


def test_detect_returns_structured_result_without_csv(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(tmp_path),
            "output.csv_path": str(output_file),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )

    def fake_iter_prepared_payload_batches(files, cfg_arg, prepare_workers, **kwargs):
        assert kwargs["show_progress"] is False
        assert files == [str(input_file)]
        assert prepare_workers == 1
        yield [{"filename": "toy.pdb", "chain": "A", "residues_data": []}]

    def fake_run_analysis_stream(payload_batches, cfg_arg, workers, *, on_results=None, **kwargs):
        assert kwargs["show_progress"] is False
        assert list(payload_batches) == [
            [{"filename": "toy.pdb", "chain": "A", "residues_data": []}]
        ]
        assert workers == 1
        assert on_results is None
        return [_complete_result_row(filename="toy.pdb", author_chain_id="A", reason="short")]

    monkeypatch.setattr(
        "cooper_beta.pipeline.require_dssp_binary", lambda explicit_path: "/opt/dssp/mkdssp"
    )
    monkeypatch.setattr(
        "cooper_beta.pipeline.iter_prepared_payload_batches",
        fake_iter_prepared_payload_batches,
    )
    monkeypatch.setattr("cooper_beta.pipeline.run_analysis_stream", fake_run_analysis_stream)

    result = detect(str(tmp_path), config=cfg, write_csv=False, print_summary=False)

    assert result.output_path is None
    assert result.input_files == [str(input_file)]
    assert result.result_counts == {"NON_BARREL": 1}
    assert result.rows[0].filename == "toy.pdb"
    assert result.rows[0].reason == "short"
    assert not output_file.exists()


def test_run_pipeline_result_rejects_missing_input_by_default(tmp_path: Path) -> None:
    cfg = build_config(
        {
            "input.path": str(tmp_path / "missing.pdb"),
            "output.csv_path": str(tmp_path / "results.csv"),
        }
    )

    with pytest.raises(InputValidationError, match="does not exist"):
        run_pipeline_result(cfg, write_csv=False, print_summary=False, show_progress=False)


def test_result_publication_cannot_replace_a_frozen_input(tmp_path: Path) -> None:
    input_file = tmp_path / "input.pdb"
    original = b"HEADER    SCIENTIFIC INPUT\n"
    input_file.write_bytes(original)
    identity = freeze_input_identity(input_file)
    cfg = build_config(
        {
            "input.path": str(input_file),
            "output.csv_path": str(input_file),
            "output.existing_artifact_policy": "replace",
        }
    )

    with pytest.raises(InputValidationError, match="disjoint.*input structure"):
        pipeline._publish_results(
            [],
            cfg,
            input_files=[str(input_file)],
            analysis_workers=1,
            prepare_workers=1,
            started_at_utc="2026-01-01T00:00:00Z",
            input_identities=[identity],
        )

    assert input_file.read_bytes() == original
    assert not Path(f"{input_file}.manifest.json").exists()


def test_log_path_cannot_collide_with_a_frozen_input(tmp_path: Path) -> None:
    input_file = tmp_path / "input.pdb"
    original = b"HEADER    SCIENTIFIC INPUT\n"
    input_file.write_bytes(original)
    cfg = build_config(
        {
            "input.path": str(input_file),
            "runtime.log_jsonl_path": str(input_file),
            "output.csv_path": str(tmp_path / "result.csv"),
        }
    )

    with pytest.raises(InputValidationError, match="disjoint.*input structure"):
        run_pipeline_result(cfg, write_csv=False, print_summary=False, show_progress=False)

    assert input_file.read_bytes() == original


def test_detect_rejects_config_and_overrides_together():
    with pytest.raises(TypeError, match="overrides"):
        detect("input.pdb", config=build_config(), overrides={"runtime.workers": 1})


def test_run_pipeline_reports_all_prepare_failures(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "bad.pdb"
    input_file.write_text("not a structure\n")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(tmp_path),
            "output.csv_path": str(output_file),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )
    exceptions = __import__("cooper_beta.exceptions", fromlist=["InputValidationError"])

    def fake_iter_prepared_payload_batches(
        files,
        cfg_arg,
        prepare_workers,
        *,
        on_failures=None,
        **kwargs,
    ):
        del files, cfg_arg, prepare_workers, kwargs
        if on_failures is not None:
            from cooper_beta.preparation import PrepareFailure

            on_failures(
                [
                    PrepareFailure(
                        source_path=str(input_file),
                        error_code="STRUCTURE_PARSE_FAILED",
                        message="parse failed",
                    )
                ]
            )
        if False:
            yield []

    monkeypatch.setattr(
        "cooper_beta.pipeline.require_dssp_binary", lambda explicit_path: "/opt/dssp/mkdssp"
    )
    monkeypatch.setattr(
        "cooper_beta.pipeline.iter_prepared_payload_batches",
        fake_iter_prepared_payload_batches,
    )

    with pytest.raises(exceptions.InputValidationError, match="failed during preparation"):
        run_pipeline_result(cfg, print_summary=False, strict_input=True, show_progress=False)

    csv_text = output_file.read_text(encoding="utf-8")
    assert "bad.pdb" in csv_text
    assert "parse failed" in csv_text


def test_run_pipeline_writes_empty_csv_when_no_payloads_are_produced(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "empty.pdb"
    input_file.write_text("HEADER\n")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(tmp_path),
            "output.csv_path": str(output_file),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )

    def fake_iter_prepared_payload_batches(files, cfg_arg, prepare_workers, **kwargs):
        del files, cfg_arg, prepare_workers, kwargs
        if False:
            yield []

    monkeypatch.setattr(
        "cooper_beta.pipeline.require_dssp_binary", lambda explicit_path: "/opt/dssp/mkdssp"
    )
    monkeypatch.setattr(
        "cooper_beta.pipeline.iter_prepared_payload_batches",
        fake_iter_prepared_payload_batches,
    )

    result = run_pipeline_result(cfg, print_summary=False, strict_input=True, show_progress=False)

    assert result.rows == []
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8").startswith(
        "filename,source_path,author_chain_id,result"
    )
    assert output_file.with_suffix(".csv.manifest.json").exists()


def test_run_pipeline_manifest_records_worker_resolution_and_respects_hash_setting(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "empty.pdb"
    input_file.write_text("HEADER\n")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(input_file),
            "output.csv_path": str(output_file),
            "output.hash_input_files": False,
            "runtime.workers": 2,
            "runtime.prepare_workers": 3,
        }
    )

    def fake_iter(*args, **kwargs):
        del args, kwargs
        if False:
            yield []

    monkeypatch.setattr(pipeline, "require_dssp_binary", lambda value: "/opt/dssp/mkdssp")
    monkeypatch.setattr(pipeline, "iter_prepared_payload_batches", fake_iter)

    run_pipeline_result(
        cfg,
        print_summary=False,
        strict_input=True,
        show_progress=False,
    )

    document = json.loads(Path(f"{output_file}.manifest.json").read_text(encoding="utf-8"))
    assert document["input_file_hashing_enabled"] is False
    assert document["input_file_state"][0]["sha256"] is None
    assert document["input_identity_policy"] == {
        "algorithm": "sha256",
        "polymer_position_policy": ("selected-model-mmcif-label-seq-or-pdb-unique-seqres"),
        "dssp_residue_coverage_policy": (
            "selected-model-declared-polypeptide-ca-residues-with-finite-n-ca-c-o"
        ),
        "frozen_before_parsing": True,
        "verified_before_artifact_publication": True,
        "hash_redacted_from_manifest": True,
    }
    assert document["runtime"]["resolved_analysis_workers"] == 2
    assert document["runtime"]["resolved_prepare_workers"] == 3
    assert document["status"] == "complete"
    assert document["artifact_binding"] == {
        "committed_by_run": True,
        "csv_path": str(output_file.resolve()),
        "csv_sha256": file_sha256(output_file),
        "csv_size": output_file.stat().st_size,
        "run_id": document["run_id"],
    }
    native_policy = document["runtime"]["native_thread_policy"]
    assert native_policy["requested_threads_per_process"] == 1
    assert native_policy["applied_limit_in_parent_process"] == 1
    assert native_policy["applied_limit_matches_request"] is True
    assert native_policy["environment_matches_request"] is True
    assert native_policy["loaded_pools_within_request"] is True
    assert all(pool["num_threads"] <= 1 for pool in document["runtime"]["native_thread_pools"])
    assert document["runtime"]["python_hash_policy"]["runtime_assignment_performed"] is False
    assert (
        document["runtime"]["python_hash_policy"]["algorithm_depends_on_hash_iteration_order"]
        is False
    )


def test_run_pipeline_rejects_input_changed_after_preparation_even_when_hashes_are_redacted(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "changing.pdb"
    input_file.write_text("OLD\n", encoding="utf-8")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(input_file),
            "output.csv_path": str(output_file),
            "output.hash_input_files": False,
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )

    def mutating_batches(*args, **kwargs):
        del args
        assert kwargs["input_identities"][0].sha256
        input_file.write_text("NEW\n", encoding="utf-8")
        if False:
            yield []

    monkeypatch.setattr(pipeline, "require_dssp_binary", lambda value: "/opt/dssp/mkdssp")
    monkeypatch.setattr(pipeline, "iter_prepared_payload_batches", mutating_batches)

    with pytest.raises(InputContentChangedError, match="Input content changed during the run"):
        run_pipeline_result(
            cfg,
            print_summary=False,
            strict_input=True,
            show_progress=False,
        )

    assert not output_file.exists()
    assert not Path(f"{output_file}.manifest.json").exists()


def test_run_pipeline_respects_disabled_manifest(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "empty.pdb"
    input_file.write_text("HEADER\n")
    output_file = tmp_path / "results.csv"
    manifest_file = Path(f"{output_file}.manifest.json")
    manifest_file.write_text('{"stale": true}\n', encoding="utf-8")
    cfg = build_config(
        {
            "input.path": str(input_file),
            "output.csv_path": str(output_file),
            "output.write_manifest": False,
            "output.existing_artifact_policy": "replace",
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )

    def fake_iter(*args, **kwargs):
        del args, kwargs
        if False:
            yield []

    monkeypatch.setattr(pipeline, "require_dssp_binary", lambda value: "/opt/dssp/mkdssp")
    monkeypatch.setattr(pipeline, "iter_prepared_payload_batches", fake_iter)

    run_pipeline_result(
        cfg,
        print_summary=False,
        strict_input=True,
        show_progress=False,
    )

    assert output_file.exists()
    assert not manifest_file.exists()


def test_manifest_failure_writes_failed_sidecar_bound_to_committed_csv(tmp_path: Path, monkeypatch):
    output_file = tmp_path / "results.csv"
    output_file.write_text("new output\n", encoding="utf-8")
    manifest_file = Path(f"{output_file}.manifest.json")
    manifest_file.write_text('{"stale": true}\n', encoding="utf-8")
    cfg = build_config(
        {
            "output.csv_path": str(output_file),
            "output.existing_artifact_policy": "replace",
        }
    )
    monkeypatch.setattr(
        pipeline,
        "write_run_manifest",
        lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        pipeline._publish_results(
            [],
            cfg,
            input_files=[],
            analysis_workers=1,
            prepare_workers=1,
            started_at_utc="2026-01-01T00:00:00Z",
            input_identities=[],
        )

    document = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["failure"] == {"message": "disk full", "type": "OSError"}
    assert document["output_sha256"] == file_sha256(output_file)
    assert document["artifact_binding"] == {
        "committed_by_run": True,
        "csv_path": str(output_file.resolve()),
        "csv_sha256": file_sha256(output_file),
        "csv_size": output_file.stat().st_size,
        "run_id": document["run_id"],
    }


def test_discover_input_files_is_case_insensitive(tmp_path: Path):
    mixed_case = tmp_path / "example.mmCIF"
    mixed_case.write_text("data_example\n")
    compressed = tmp_path / "compressed.PDB.GZ"
    compressed.write_bytes(b"compressed")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("ignore me\n")

    allowed = [".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz"]
    assert discover_input_files(str(tmp_path), allowed, strict=True) == [
        str(compressed),
        str(mixed_case),
    ]
    assert discover_input_files(str(compressed), allowed, strict=True) == [str(compressed)]


def test_run_analysis_stream_batches_payloads_and_invokes_sink(monkeypatch):
    from cooper_beta.bootstrap import configure_thread_environment

    cfg = build_config({"runtime.analysis_batch_size": 2})
    configure_thread_environment(cfg.runtime.native_threads_per_process)
    import cooper_beta.execution as execution

    payloads = [
        {"filename": f"toy-{index}.pdb", "chain": "A", "residues_data": []} for index in range(5)
    ]
    analyzed_batches = []

    def fake_analyze_payload_batch(payload_batch, cfg_arg):
        assert cfg_arg is cfg
        analyzed_batches.append([payload["filename"] for payload in payload_batch])
        return [
            {
                "filename": payload["filename"],
                "author_chain_id": payload["chain"],
                "result": "NON_BARREL",
            }
            for payload in payload_batch
        ]

    written_rows = []
    monkeypatch.setattr(
        execution,
        "analyze_payload_batch",
        fake_analyze_payload_batch,
    )

    rows = execution.run_analysis_stream(
        [payloads],
        cfg,
        workers=1,
        on_results=written_rows.extend,
    )

    assert analyzed_batches == [
        ["toy-0.pdb", "toy-1.pdb"],
        ["toy-2.pdb", "toy-3.pdb"],
        ["toy-4.pdb"],
    ]
    assert rows == written_rows
    assert len(rows) == 5


def test_print_results_summary_limits_console_but_writes_full_csv(tmp_path: Path, capsys):
    rows = [
        _complete_result_row(
            filename="a.pdb",
            author_chain_id="A",
            result="NON_BARREL",
            result_stage="decision",
        ),
        _complete_result_row(
            filename="b.pdb",
            author_chain_id="A",
            result="NON_BARREL",
            result_stage="decision",
        ),
        _complete_result_row(
            filename="c.pdb",
            author_chain_id="A",
            result="BARREL",
            result_stage="decision",
        ),
    ]
    output_file = tmp_path / "results.csv"

    print_results_summary(rows, str(output_file), summary_limit=1)

    captured = capsys.readouterr().out
    assert "Rows: 3" in captured
    assert "omitted 2 row(s)" in captured
    assert "a.pdb" in captured
    assert "b.pdb" not in captured
    assert output_file.read_text().count("\n") == 4


def test_write_results_csv_writes_schema_for_empty_results(tmp_path: Path):
    output_file = tmp_path / "empty.csv"

    write_results_csv([], str(output_file))

    header = output_file.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("filename,source_path,author_chain_id,result,result_stage")


def test_ordered_result_rows_follow_input_file_and_author_chain_order():
    rows = [
        {"filename": "b.pdb", "author_chain_id": "B", "result": "NON_BARREL"},
        {"filename": "a.pdb", "author_chain_id": "C", "result": "NON_BARREL"},
        {"filename": "a.pdb", "author_chain_id": "A", "result": "BARREL"},
    ]

    ordered = _ordered_result_rows(rows, ["/input/a.pdb", "/input/b.pdb"])

    assert [(row["filename"], row["author_chain_id"]) for row in ordered] == [
        ("a.pdb", "A"),
        ("a.pdb", "C"),
        ("b.pdb", "B"),
    ]


def test_ordered_result_rows_use_source_path_for_duplicate_basenames(tmp_path: Path):
    first = tmp_path / "first" / "same.pdb"
    second = tmp_path / "second" / "same.pdb"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("HEADER first\n")
    second.write_text("HEADER second\n")

    rows = [
        {
            "filename": "same.pdb",
            "source_path": str(second),
            "author_chain_id": "A",
            "result": "NON_BARREL",
        },
        {
            "filename": "same.pdb",
            "source_path": str(first),
            "author_chain_id": "A",
            "result": "BARREL",
        },
    ]

    ordered = _ordered_result_rows(rows, [str(first), str(second)])

    assert [row["source_path"] for row in ordered] == [str(first), str(second)]
