from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from importlib.metadata import version
from pathlib import Path

import pytest

from cooper_beta.constants import DEFAULT_RESULT_COLUMNS


def _detector_row(
    filename: str,
    source_path: str,
    chain: str,
    result: str,
) -> dict[str, object]:
    is_error = result == "ERROR"
    row: dict[str, object] = {
        "filename": filename,
        "source_path": source_path,
        "author_chain_id": chain,
        "result": result,
        "result_stage": "preparation" if is_error else "decision",
        "dssp_unassigned_residue_count": 0,
        "strand_count": 0 if is_error else 8,
        "strand_adjacency_count": 0 if is_error else 8,
        "cycle_strand_count": 0 if is_error else 8,
        "cycle_strand_fraction": 0.0 if is_error else 1.0,
        "cycle_rank": 0 if is_error else 1,
        "reason": "parse failed" if is_error else "Strand adjacency count below minimum",
        "error_code": "STRUCTURE_PARSE_FAILED" if is_error else "",
        "degraded": False,
    }
    assert set(row) == set(DEFAULT_RESULT_COLUMNS)
    return row


def test_import_package():
    pkg = importlib.import_module("cooper_beta")
    assert hasattr(pkg, "AppConfig")
    assert hasattr(pkg, "build_config")
    assert hasattr(pkg, "detect")
    assert not hasattr(pkg, "extract_chain_slices")


def test_source_and_installed_distribution_versions_match():
    pkg = importlib.import_module("cooper_beta")
    source_version = importlib.import_module("cooper_beta._version").__version__

    assert pkg.__version__ == source_version == version("cooper-beta")


def test_import_pipeline():
    pipe = importlib.import_module("cooper_beta.pipeline")
    assert hasattr(pipe, "detect")
    assert not hasattr(pipe, "main")


def test_build_config_supports_explicit_nested_overrides():
    config_mod = importlib.import_module("cooper_beta.config")

    cfg = config_mod.build_config(
        {
            "runtime.dssp_bin_path": "/tmp/mkdssp",
            "rules.cycle_strand_count_fraction.minimum_fraction": 0.75,
        }
    )

    assert cfg.runtime.dssp_bin_path == "/tmp/mkdssp"
    assert cfg.rules.cycle_strand_count_fraction.minimum_fraction == pytest.approx(0.75)


def test_build_config_resolves_frozen_scientific_defaults():
    config_mod = importlib.import_module("cooper_beta.config")

    cfg = config_mod.build_config()

    assert cfg.rules.strand_adjacency_count.minimum == 8
    assert cfg.rules.cycle_strand_count_fraction.minimum_count == 4
    assert cfg.rules.cycle_strand_count_fraction.minimum_fraction == pytest.approx(0.05)
    assert cfg.rules.cycle_rank.minimum == 1
    with pytest.raises(FrozenInstanceError):
        cfg.rules.cycle_rank.minimum = 2


def test_build_config_rejects_invalid_user_values():
    config_mod = importlib.import_module("cooper_beta.config")
    exceptions = importlib.import_module("cooper_beta.exceptions")

    with pytest.raises(exceptions.ConfigValidationError, match="minimum_fraction"):
        config_mod.build_config({"rules.cycle_strand_count_fraction.minimum_fraction": 1.5})


def test_require_dssp_binary_reports_help(monkeypatch: pytest.MonkeyPatch):
    runtime = importlib.import_module("cooper_beta.runtime")

    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="Cooper-Beta requires DSSP"):
        runtime.require_dssp_binary(None)


def test_require_dssp_binary_rejects_invalid_explicit_path(monkeypatch: pytest.MonkeyPatch):
    runtime = importlib.import_module("cooper_beta.runtime")

    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/mkdssp")

    with pytest.raises(RuntimeError, match="Configured DSSP executable"):
        runtime.require_dssp_binary("/definitely/missing/mkdssp")


@pytest.mark.parametrize(
    ("resolved_path", "version_output"),
    [
        ("/opt/dssp/minimum/mkdssp", "mkdssp version 4.5.3\n"),
        ("/opt/dssp/newer/mkdssp", "mkdssp version 4.6.1\n"),
    ],
)
def test_require_dssp_binary_accepts_supported_versions(
    resolved_path: str,
    version_output: str,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = importlib.import_module("cooper_beta.runtime")
    monkeypatch.setattr(runtime, "find_dssp_binary", lambda _: resolved_path)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=version_output, stderr=""
        ),
    )

    assert runtime.require_dssp_binary(None) == resolved_path


def test_require_dssp_binary_rejects_dssp_older_than_public_minimum(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = importlib.import_module("cooper_beta.runtime")
    resolved_path = "/opt/dssp/unsupported-4.2.2/mkdssp"
    monkeypatch.setattr(runtime, "find_dssp_binary", lambda _: resolved_path)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="mkdssp version 4.2.2\n", stderr=""
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"Unsupported DSSP version 4\.2\.2.*requires mkdssp 4\.5\.3 or newer",
    ):
        runtime.require_dssp_binary(None)


def test_require_dssp_binary_rejects_unparseable_version_output(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = importlib.import_module("cooper_beta.runtime")
    resolved_path = "/opt/dssp/unparseable/mkdssp"
    monkeypatch.setattr(runtime, "find_dssp_binary", lambda _: resolved_path)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="DSSP 4.6.1\n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match=r"expected `mkdssp version X\.Y\.Z`"):
        runtime.require_dssp_binary(None)


def test_dssp_version_query_is_reused_for_the_resolved_executable(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = importlib.import_module("cooper_beta.runtime")
    provenance = importlib.import_module("cooper_beta.provenance")
    resolved_path = "/opt/dssp/query-once/mkdssp"
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(runtime, "find_dssp_binary", lambda _: resolved_path)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="mkdssp version 4.5.3\n",
            stderr="",
        )

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    assert runtime.require_dssp_binary(None) == resolved_path
    assert runtime.require_dssp_binary(None) == resolved_path
    summary = runtime.runtime_summary(None)
    manifest_version = provenance.dssp_version(resolved_path)

    assert summary["dssp"] == f"{resolved_path} (mkdssp version 4.5.3)"
    assert manifest_version == "mkdssp version 4.5.3"
    assert calls == [
        (
            [resolved_path, "--version"],
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": runtime.DSSP_VERSION_QUERY_TIMEOUT_SECONDS,
            },
        )
    ]


def test_locked_setup_help_states_the_dssp_release():
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["bash", str(root / "scripts" / "setup_env.sh"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "DSSP 4.5.3" in completed.stdout


def test_cli_requires_explicit_input_path(capsys: pytest.CaptureFixture[str]):
    cli = importlib.import_module("cooper_beta.cli")

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert exc_info.value.code == 2
    assert "input path is required" in capsys.readouterr().err


def test_cli_accepts_hydra_override_before_path(capsys: pytest.CaptureFixture[str]):
    cli = importlib.import_module("cooper_beta.cli")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["runtime.workers=1", "missing.pdb"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Input path does not exist" in stderr
    assert "input path is required" not in stderr


def test_cli_reports_bad_output_path_without_traceback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    cli = importlib.import_module("cooper_beta.cli")
    pipeline = importlib.import_module("cooper_beta.pipeline")
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")

    monkeypatch.setattr(pipeline, "require_dssp_binary", lambda explicit_path: "/opt/dssp/mkdssp")

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(input_file), "--out", str(tmp_path)])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Error:" in stderr
    assert "Output CSV path points to a directory" in stderr
    assert "Traceback" not in stderr


def test_cli_check_env_rejects_invalid_configured_dssp(
    capsys: pytest.CaptureFixture[str],
):
    cli = importlib.import_module("cooper_beta.cli")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--check-env", "runtime.dssp_bin_path=/definitely/missing/mkdssp"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Configured DSSP executable" in stderr
    assert "Traceback" not in stderr


def test_evaluate_rejects_invalid_metric_level_before_detector_runs(tmp_path: Path):
    runner = importlib.import_module("cooper_beta.evaluation.runner")

    with pytest.raises(ValueError, match="metric_level"):
        runner.evaluate(
            true_dir=tmp_path / "true",
            false_dir=tmp_path / "false",
            workers=1,
            prepare_workers=1,
            save_dir=tmp_path / "out",
            metric_level="sample",
            tag="bad",
        )


def test_evaluation_file_outputs_preserve_error_files(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    runner = importlib.import_module("cooper_beta.evaluation.runner")
    metrics = importlib.import_module("cooper_beta.evaluation.metrics")

    positive = pd.DataFrame(
        [
            _detector_row(
                "broken.pdb",
                str(tmp_path / "true" / "broken.pdb"),
                "",
                "ERROR",
            )
        ]
    )
    negative = pd.DataFrame(
        [
            _detector_row(
                "ok.pdb",
                str(tmp_path / "false" / "ok.pdb"),
                "A",
                "NON_BARREL",
            )
        ]
    )

    _, _, aggregated = runner.save_outputs(positive, negative, tmp_path / "out", "unit")
    error_row = aggregated.loc[aggregated["filename"].eq("broken.pdb")].iloc[0]
    assert int(error_row["error_chains_n"]) == int(error_row["chains_n"]) == 1

    with pytest.raises(metrics.MetricInputError, match="detector ERROR"):
        metrics.compute_file_metrics(aggregated)

    _, extra = metrics.compute_file_metrics(aggregated, error_policy="exclude")
    assert extra["metric_error_policy"] == "exclude"
    assert extra["excluded_positive_error_files"] == 1
    assert extra["positive_error_files"] == 1
    assert extra["n_positive_files"] == 0
    assert extra["n_negative_files"] == 1
    assert extra["positive_file_coverage"] == 0.0


def test_path_executed_module_entrypoints_have_helpful_smoke_output():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}

    version = subprocess.run(
        [sys.executable, str(root / "src" / "cooper_beta" / "__main__.py"), "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert version.returncode == 0
    assert "cooper-beta" in version.stdout

    help_result = subprocess.run(
        [
            sys.executable,
            str(root / "src" / "cooper_beta" / "evaluation" / "__main__.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert help_result.returncode == 0
    assert "--metric-level" in help_result.stdout
