from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cooper_beta.constants import DEFAULT_RESULT_COLUMNS, RESULT_ERROR
from cooper_beta.evaluation import app, runner
from cooper_beta.evaluation.metrics import (
    CandidateTruth,
    MetricInputError,
    compute_chain_metrics,
    compute_confusion_metrics,
    compute_file_metrics,
    print_metrics,
)
from cooper_beta.integrity import file_sha256


def _row(
    filename: str,
    chain: str,
    result: str,
    *,
    source_path: str | None = None,
) -> dict[str, object]:
    error = result == RESULT_ERROR
    return {
        "filename": filename,
        "source_path": source_path or filename,
        "author_chain_id": chain,
        "result": result,
        "result_stage": "preparation"
        if error and not chain
        else ("worker" if error else "decision"),
        "dssp_unassigned_residue_count": 0,
        "strand_count": 0 if error else 8,
        "strand_adjacency_count": 0 if error else 8,
        "cycle_strand_count": 0 if error else 8,
        "cycle_strand_fraction": 0.0 if error else 1.0,
        "cycle_rank": 0 if error else 1,
        "reason": "fixture",
        "error_code": "PREPARATION_FAILED" if error else "",
        "degraded": False,
    }


def _truth(values: dict[str, str]) -> tuple[CandidateTruth, ...]:
    return tuple(
        CandidateTruth(filename, filename, "0" * 64, chain) for filename, chain in values.items()
    )


def _write_truth_manifest(path: Path, values: dict[Path, str]) -> Path:
    pd.DataFrame(
        [
            {
                "filename": structure.name,
                "source_path": str(structure.resolve()),
                "structure_sha256": file_sha256(structure),
                "target_author_chain_id": chain,
            }
            for structure, chain in values.items()
        ]
    ).to_csv(path, index=False)
    return path


def test_cli_forwards_manifests_and_error_policy(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"chain_csv": "chain.csv", "file_csv": "file.csv"}

    monkeypatch.setattr(app, "evaluate", fake_evaluate)
    app._run(
        [
            "--positives",
            str(tmp_path / "positive"),
            "--negatives",
            str(tmp_path / "negative"),
            "--metric-error-policy",
            "exclude",
        ]
    )

    assert captured["metric_error_policy"] == "exclude"


def test_chain_metrics_select_only_manifest_targets() -> None:
    positives = pd.DataFrame(
        [_row("positive.pdb", "A", "BARREL"), _row("positive.pdb", "B", "NON_BARREL")]
    )
    negatives = pd.DataFrame([_row("negative.pdb", "N", "NON_BARREL")])

    metrics, extra = compute_chain_metrics(
        positives,
        negatives,
        positive_candidate_truth=_truth({"positive.pdb": "A"}),
        negative_candidate_truth=_truth({"negative.pdb": "N"}),
    )

    assert (metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]) == (1, 1, 0, 0)
    assert extra["positive_partner_rows_unlabeled"] == 1


def test_chain_metrics_require_symmetric_target_manifests() -> None:
    frame = pd.DataFrame([_row("x.pdb", "A", "NON_BARREL")])
    with pytest.raises(MetricInputError, match="negative manifest"):
        compute_chain_metrics(
            frame,
            frame,
            positive_candidate_truth=_truth({"x.pdb": "A"}),
            negative_candidate_truth=None,
        )


def test_error_policy_is_strict_or_explicitly_excluded() -> None:
    positives = pd.DataFrame([_row("ok.pdb", "A", "BARREL"), _row("bad.pdb", "A", "ERROR")])
    negatives = pd.DataFrame([_row("negative.pdb", "A", "NON_BARREL")])
    positive_truth = _truth({"ok.pdb": "A", "bad.pdb": "A"})
    negative_truth = _truth({"negative.pdb": "A"})

    with pytest.raises(MetricInputError, match="ERROR"):
        compute_chain_metrics(
            positives,
            negatives,
            positive_candidate_truth=positive_truth,
            negative_candidate_truth=negative_truth,
        )
    metrics, extra = compute_chain_metrics(
        positives,
        negatives,
        positive_candidate_truth=positive_truth,
        negative_candidate_truth=negative_truth,
        error_policy="exclude",
    )
    assert metrics["TP"] == 1
    assert extra["n_positive_used"] == 1


def test_file_metrics_use_any_chain_prediction_and_error_coverage() -> None:
    frame = pd.DataFrame(
        [
            {"split": "positive", "pred_barrel_any": True, "error_chains_n": 0},
            {"split": "positive", "pred_barrel_any": False, "error_chains_n": 1},
            {"split": "negative", "pred_barrel_any": False, "error_chains_n": 0},
        ]
    )
    with pytest.raises(MetricInputError, match="ERROR"):
        compute_file_metrics(frame)
    metrics, extra = compute_file_metrics(frame, error_policy="exclude")
    assert (metrics["TP"], metrics["TN"]) == (1, 1)
    assert extra["overall_file_coverage"] == pytest.approx(2 / 3)


def test_save_outputs_uses_lean_detector_schema(tmp_path: Path) -> None:
    positives = pd.DataFrame([_row("positive.pdb", "A", "BARREL")])
    negatives = pd.DataFrame([_row("negative.pdb", "A", "NON_BARREL")])

    chain_csv, file_csv, aggregated = runner.save_outputs(
        positives,
        negatives,
        tmp_path,
        "lean",
        positive_candidate_truth=_truth({"positive.pdb": "A"}),
        negative_candidate_truth=_truth({"negative.pdb": "A"}),
    )

    assert Path(chain_csv).is_file() and Path(file_csv).is_file()
    assert aggregated.set_index("split").loc["positive", "pred_barrel_any"]


def test_save_outputs_rejects_extra_detector_columns(tmp_path: Path) -> None:
    positive = pd.DataFrame([dict(_row("positive.pdb", "A", "BARREL"), unexpected_column=1.0)])
    negative = pd.DataFrame([_row("negative.pdb", "A", "NON_BARREL")])
    with pytest.raises(ValueError, match="fixed public schema"):
        runner.save_outputs(positive, negative, tmp_path, "drift")


def test_truth_manifest_loader_requires_exact_current_schema(tmp_path: Path) -> None:
    structure = tmp_path / "sample.cif"
    structure.write_text("data_sample\n", encoding="utf-8")
    manifest = _write_truth_manifest(tmp_path / "truth.csv", {structure: "A"})
    loaded = runner.load_positive_manifest(manifest)
    assert loaded[0].target_author_chain_id == "A"

    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"filename": structure.name, "target_author_chain_id": "A"}]).to_csv(
        bad, index=False
    )
    with pytest.raises(ValueError, match="exactly these columns"):
        runner.load_positive_manifest(bad)


def test_confusion_metrics_keep_undefined_values_null(capsys) -> None:
    metrics = compute_confusion_metrics(0, 0, 5, 5)
    assert metrics["precision"] is None
    assert metrics["mcc"] is None
    print_metrics("fixture", metrics)
    assert "undefined" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [1.5, float("nan"), True, "1"])
def test_confusion_metrics_reject_non_integer_counts(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        compute_confusion_metrics(bad, 0, 0, 0)


def test_confusion_metrics_accept_numpy_integer_counts() -> None:
    metrics = compute_confusion_metrics(np.int64(1), np.int64(0), np.int64(1), np.int64(0))
    assert metrics["accuracy"] == 1.0


def test_evaluate_rejects_invalid_metric_level_before_running(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metric_level"):
        runner.evaluate(
            tmp_path,
            tmp_path / "negative",
            None,
            None,
            tmp_path / "output",
            "invalid",
            "tag",
        )


def test_evaluate_runs_one_complete_paired_workflow(tmp_path: Path, monkeypatch) -> None:
    positive_dir = tmp_path / "positive"
    negative_dir = tmp_path / "negative"
    positive_dir.mkdir()
    negative_dir.mkdir()
    positive_structure = positive_dir / "positive.cif"
    negative_structure = negative_dir / "negative.cif"
    positive_structure.write_text("data_positive\n", encoding="utf-8")
    negative_structure.write_text("data_negative\n", encoding="utf-8")
    positive_manifest = _write_truth_manifest(
        tmp_path / "positive_truth.csv", {positive_structure: "A"}
    )
    negative_manifest = _write_truth_manifest(
        tmp_path / "negative_truth.csv", {negative_structure: "A"}
    )

    frames = {
        positive_dir.resolve(): pd.DataFrame(
            [_row(positive_structure.name, "A", "BARREL", source_path=str(positive_structure))]
        ),
        negative_dir.resolve(): pd.DataFrame(
            [_row(negative_structure.name, "A", "NON_BARREL", source_path=str(negative_structure))]
        ),
    }

    def fake_run_detector(folder: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        return frames[folder.resolve()]

    def fake_validate(_manifest: Path, *, expected_output: Path) -> dict[str, object]:
        structure = (
            positive_structure
            if expected_output.name.startswith("positive")
            else negative_structure
        )
        return {
            "scientific_config_hash": "1" * 64,
            "producer_identity_hash": "2" * 64,
            "_validated_input_paths": [str(structure.resolve())],
            "_validated_input_hashes": [file_sha256(structure)],
        }

    monkeypatch.setattr(runner, "run_detector", fake_run_detector)
    monkeypatch.setattr(runner, "validate_detector_artifact_manifest", fake_validate)
    result = runner.evaluate(
        positive_dir,
        negative_dir,
        workers=1,
        prepare_workers=1,
        save_dir=tmp_path / "evaluation",
        metric_level="both",
        tag="paired",
        print_metric_tables=False,
        positive_manifest=positive_manifest,
        negative_manifest=negative_manifest,
    )

    assert result["chain_TP"] == 1
    assert result["chain_TN"] == 1
    assert result["file_TP"] == 1
    assert result["file_TN"] == 1
    manifest = pd.read_json(result["evaluation_manifest"], typ="series")
    assert manifest["status"] == "complete"


def test_public_schema_has_only_the_three_graph_rule_inputs() -> None:
    assert tuple(DEFAULT_RESULT_COLUMNS[6:11]) == (
        "strand_count",
        "strand_adjacency_count",
        "cycle_strand_count",
        "cycle_strand_fraction",
        "cycle_rank",
    )
