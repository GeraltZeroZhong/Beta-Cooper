from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cooper_beta.config import build_config
from cooper_beta.constants import DEFAULT_RESULT_COLUMNS
from cooper_beta.integrity import canonical_json_sha256, file_sha256
from cooper_beta.provenance import (
    build_run_manifest,
    resolved_config_sections,
    scientific_producer_identity,
)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/nested_grouped_decision_experiment.py"
_SPEC = importlib.util.spec_from_file_location("nested_grouped_decision_experiment", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
_SPEC.loader.exec_module(experiment)


def test_confusion_metrics_serialize_true_undefined_values_as_null() -> None:
    metrics = experiment.confusion_metrics(
        np.asarray([0], dtype=int),
        np.asarray([0], dtype=int),
    )

    assert metrics["recall"] is None
    assert metrics["precision"] is None
    assert metrics["f1"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["mcc"] is None
    assert metrics["specificity"] == 1.0
    assert metrics["undefined_metric_policy"] == "serialize_null_and_mark_undefined"
    serialized = json.dumps(metrics, allow_nan=False)
    assert '"recall": null' in serialized


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        (np.asarray([0.9, 0.1]), np.asarray([1, 0]), "non-boolean binary integer"),
        (np.asarray([0.0, 1.0]), np.asarray([1, 0]), "non-boolean binary integer"),
        (np.asarray([False, True]), np.asarray([1, 0]), "non-boolean binary integer"),
        (np.asarray(["0", "1"]), np.asarray([1, 0]), "non-boolean binary integer"),
        (np.asarray([0, 2]), np.asarray([1, 0]), "either 0 or 1"),
    ],
)
def test_confusion_metrics_rejects_non_integer_or_non_binary_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(experiment.ExperimentInputError, match=message):
        experiment.confusion_metrics(y_true, y_pred)


def _result_row(
    source_path: Path,
    chain: str,
    *,
    score: float,
    result: str = "NON_BARREL",
) -> dict[str, object]:
    if result == "BARREL":
        adjacency_count = 8 + int(score >= 0.75) + int(score >= 1.0)
        cycle_count = 8 + int(score >= 0.75)
        cycle_rank = 1 + int(score >= 1.0)
    else:
        adjacency_count = 4 + int(round(score * 8))
        cycle_count = 0 if score == 0.0 else 4
        cycle_rank = 0 if cycle_count == 0 else 1
    strand_count = 10
    return {
        "filename": source_path.name,
        "source_path": str(source_path.resolve()),
        "author_chain_id": chain,
        "result": result,
        "result_stage": "decision",
        "dssp_unassigned_residue_count": 2,
        "strand_count": strand_count,
        "strand_adjacency_count": adjacency_count,
        "cycle_strand_count": cycle_count,
        "cycle_strand_fraction": cycle_count / strand_count,
        "cycle_rank": cycle_rank,
        "reason": "test",
        "error_code": "",
        "degraded": False,
    }


def _invalid_result_row(source_path: Path, chain: str) -> dict[str, object]:
    row = _result_row(source_path, chain, score=0.0)
    row.update(
        {
            "result": "UNKNOWN",
            "strand_adjacency_count": 7,
            "cycle_strand_count": 0,
            "cycle_strand_fraction": 0.0,
            "cycle_rank": 0,
            "reason": "invalid fixture",
        }
    )
    return row


def _write_detector_artifact(
    root: Path,
    split: str,
    structures: list[Path],
    rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    csv_path = (root / f"{split}.csv").resolve()
    dataframe = pd.DataFrame(rows, columns=DEFAULT_RESULT_COLUMNS)
    dataframe.to_csv(csv_path, index=False)
    config = build_config(
        {
            "input.path": str(root.resolve()),
            "output.csv_path": str(csv_path),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
            "runtime.prepare_cache_enabled": False,
            "output.summary_limit": 0,
        }
    )
    manifest = build_run_manifest(
        config=config,
        input_files=[str(path.resolve()) for path in structures],
        output_path=str(csv_path),
        hash_input_files=True,
        run_id=f"nested-fixture-{split}",
    )
    manifest_path = root / f"{split}.csv.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path, manifest_path


def _rewrite_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def _rebind_detector_csv(manifest_path: Path, csv_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_state = manifest["output_file_state"]
    binding = manifest["artifact_binding"]
    assert isinstance(output_state, dict)
    assert isinstance(binding, dict)
    stat = csv_path.stat()
    digest = file_sha256(csv_path)
    output_state.update(
        {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
    )
    manifest["output_sha256"] = digest
    binding["csv_size"] = stat.st_size
    binding["csv_sha256"] = digest
    _rewrite_manifest(manifest_path, manifest)


def _refresh_scientific_config_identity(manifest: dict[str, object]) -> None:
    config = manifest["config"]
    config_document, scientific, execution, io = resolved_config_sections(config)
    manifest["config"] = config_document
    manifest["config_hash"] = canonical_json_sha256(config_document)
    manifest["config_partitions"] = {
        "scientific": scientific,
        "execution": execution,
        "io": io,
    }
    manifest["scientific_config_hash"] = canonical_json_sha256(scientific)
    manifest["execution_config_hash"] = canonical_json_sha256(execution)
    manifest["io_config_hash"] = canonical_json_sha256(io)


def _refresh_producer_identity(manifest: dict[str, object]) -> None:
    identity = scientific_producer_identity(manifest)
    manifest["producer_identity"] = identity
    manifest["producer_identity_hash"] = canonical_json_sha256(identity)


def _write_labels(
    path: Path,
    structures: list[Path],
    *,
    groups: list[str],
    chains: list[str] | None = None,
) -> Path:
    target_author_chain_ids = chains or ["A"] * len(structures)
    pd.DataFrame(
        {
            "filename": [structure.name for structure in structures],
            "source_path": [str(structure.resolve()) for structure in structures],
            "target_author_chain_id": target_author_chain_ids,
            "group_id": groups,
        },
        columns=experiment.LABEL_MANIFEST_COLUMNS,
    ).to_csv(path, index=False)
    return path


def _make_inputs(
    tmp_path: Path,
    *,
    samples_per_class: int = 1,
    include_positive_partner: bool = False,
) -> experiment.InputSpecification:
    positive_structures: list[Path] = []
    negative_structures: list[Path] = []
    positive_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    for index in range(samples_per_class):
        positive = tmp_path / f"positive-{index}.pdb"
        negative = tmp_path / f"negative-{index}.pdb"
        positive.write_text(f"POSITIVE {index}\n", encoding="utf-8")
        negative.write_text(f"NEGATIVE {index}\n", encoding="utf-8")
        positive_structures.append(positive)
        negative_structures.append(negative)
        positive_rows.append(
            _result_row(positive, "A", score=0.5 + 0.25 * (index % 3), result="BARREL")
        )
        negative_rows.append(_result_row(negative, "A", score=0.25 * (index % 2)))
        if include_positive_partner:
            positive_rows.append(_result_row(positive, "B", score=1.0, result="BARREL"))
    positive_csv, positive_detector_manifest = _write_detector_artifact(
        tmp_path, "positive", positive_structures, positive_rows
    )
    negative_csv, negative_detector_manifest = _write_detector_artifact(
        tmp_path, "negative", negative_structures, negative_rows
    )
    positive_labels = _write_labels(
        tmp_path / "positive-labels.csv",
        positive_structures,
        groups=[f"positive-family-{index // 2}" for index in range(samples_per_class)],
    )
    negative_labels = _write_labels(
        tmp_path / "negative-labels.csv",
        negative_structures,
        groups=[f"negative-family-{index // 2}" for index in range(samples_per_class)],
    )
    return experiment.InputSpecification(
        positive_csv=positive_csv,
        positive_detector_manifest=positive_detector_manifest,
        positive_label_manifest=positive_labels,
        negative_csv=negative_csv,
        negative_detector_manifest=negative_detector_manifest,
        negative_label_manifest=negative_labels,
        grouping_method="external-test-family-clusters",
    )


def _parameters() -> experiment.ExperimentParameters:
    return experiment.ExperimentParameters(
        algorithm="logistic_regression",
        c_values=(0.5, 1.0),
        threshold_min=0.25,
        threshold_max=0.75,
        threshold_count=3,
        outer_folds=3,
        inner_folds=2,
        deployment_folds=3,
        objective="mcc",
        outer_split_seed=13,
        inner_split_seed=17,
        deployment_split_seed=19,
        model_seed=23,
        max_iter=1000,
        tolerance=1e-6,
        class_weight="balanced",
        solver="liblinear",
        penalty="l2",
        l1_ratio=0.0,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1.0,
        scaler_with_mean=True,
        scaler_with_std=True,
        native_threads=1,
    )


def test_detector_csv_hash_must_match_paired_manifest(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    with inputs.positive_csv.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(experiment.ExperimentInputError, match="does not match"):
        experiment.load_experiment_samples(inputs)


def test_detector_csv_must_match_the_exact_public_schema(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    dataframe = pd.read_csv(inputs.positive_csv, keep_default_na=False)
    dataframe["unexpected_column"] = 1
    dataframe.to_csv(inputs.positive_csv, index=False)
    _rebind_detector_csv(inputs.positive_detector_manifest, inputs.positive_csv)

    with pytest.raises(experiment.ExperimentInputError, match="fixed public schema/order"):
        experiment.load_experiment_samples(inputs)


def test_only_manifest_target_author_chain_is_labeled_and_partner_is_ignored(
    tmp_path: Path,
):
    inputs = _make_inputs(tmp_path, include_positive_partner=True)

    samples, _ = experiment.load_experiment_samples(inputs)

    assert len(samples) == 2
    assert set(samples["target_author_chain_id"]) == {"A"}
    assert samples.loc[samples["split"].eq("positive"), "strand_adjacency_count"].item() < 10


def test_unknown_detector_result_is_rejected(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    positive_source = tmp_path / "positive-0.pdb"
    pd.DataFrame(
        [_invalid_result_row(positive_source, "A")],
        columns=DEFAULT_RESULT_COLUMNS,
    ).to_csv(inputs.positive_csv, index=False)
    _rebind_detector_csv(inputs.positive_detector_manifest, inputs.positive_csv)

    with pytest.raises(experiment.ExperimentInputError, match="Unknown detection result"):
        experiment.load_experiment_samples(inputs)


def test_any_detector_error_fails_instead_of_being_dropped(tmp_path: Path):
    inputs = _make_inputs(tmp_path, include_positive_partner=True)
    dataframe = pd.read_csv(inputs.positive_csv, keep_default_na=False)
    partner_mask = dataframe["author_chain_id"].eq("B")
    dataframe.loc[partner_mask, "result"] = "ERROR"
    dataframe.loc[partner_mask, "result_stage"] = "worker"
    dataframe.loc[partner_mask, "error_code"] = "TEST_ERROR"
    dataframe.to_csv(inputs.positive_csv, index=False)
    _rebind_detector_csv(inputs.positive_detector_manifest, inputs.positive_csv)

    with pytest.raises(experiment.ExperimentInputError, match="error policy is fail"):
        experiment.load_experiment_samples(inputs)


def test_label_manifest_requires_external_group_id_column(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    labels = pd.read_csv(inputs.positive_label_manifest).drop(columns="group_id")
    labels.to_csv(inputs.positive_label_manifest, index=False)

    with pytest.raises(experiment.ExperimentInputError, match="exactly these ordered columns"):
        experiment.load_experiment_samples(inputs)


def test_split_detector_scientific_configs_must_match(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    manifest = json.loads(inputs.negative_detector_manifest.read_text(encoding="utf-8"))
    manifest["config"]["rules"]["strand_adjacency_count"]["minimum"] = 9
    _refresh_scientific_config_identity(manifest)
    _rewrite_manifest(inputs.negative_detector_manifest, manifest)

    with pytest.raises(experiment.ExperimentInputError, match="same scientific configuration"):
        experiment.load_experiment_samples(inputs)


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_nested_rejects_noncomplete_detector_manifest(tmp_path: Path, status: str):
    inputs = _make_inputs(tmp_path)
    manifest = json.loads(inputs.positive_detector_manifest.read_text(encoding="utf-8"))
    manifest["status"] = status
    _rewrite_manifest(inputs.positive_detector_manifest, manifest)

    with pytest.raises(experiment.ExperimentInputError, match="completed Cooper-Beta run"):
        experiment.load_experiment_samples(inputs)


def test_nested_rejects_detector_manifest_without_artifact_binding(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    manifest = json.loads(inputs.positive_detector_manifest.read_text(encoding="utf-8"))
    del manifest["artifact_binding"]
    _rewrite_manifest(inputs.positive_detector_manifest, manifest)

    with pytest.raises(experiment.ExperimentInputError, match="artifact_binding"):
        experiment.load_experiment_samples(inputs)


def test_nested_rejects_forged_scientific_config_hash(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    manifest = json.loads(inputs.positive_detector_manifest.read_text(encoding="utf-8"))
    manifest["scientific_config_hash"] = "0" * 64
    _rewrite_manifest(inputs.positive_detector_manifest, manifest)

    with pytest.raises(experiment.ExperimentInputError, match="embedded config"):
        experiment.load_experiment_samples(inputs)


@pytest.mark.parametrize("producer_component", ["source", "dssp"])
def test_nested_rejects_different_scientific_producers(tmp_path: Path, producer_component: str):
    inputs = _make_inputs(tmp_path)
    manifest = json.loads(inputs.negative_detector_manifest.read_text(encoding="utf-8"))
    if producer_component == "source":
        package_content = manifest["source"]["package_content"]
        files = package_content["files"]
        files[0]["sha256"] = "0" * 64
        package_content["combined_sha256"] = canonical_json_sha256(files)
    else:
        manifest["executables"]["dssp_version"] = "different-dssp-version"
        manifest["executables"]["dssp_sha256"] = "1" * 64
    _refresh_producer_identity(manifest)
    _rewrite_manifest(inputs.negative_detector_manifest, manifest)

    with pytest.raises(experiment.ExperimentInputError, match="producer identity hash"):
        experiment.load_experiment_samples(inputs)


def test_nested_accepts_current_detector_artifacts(tmp_path: Path):
    inputs = _make_inputs(tmp_path)

    samples, provenance = experiment.load_experiment_samples(inputs)

    manifest = json.loads(inputs.positive_detector_manifest.read_text(encoding="utf-8"))
    assert len(samples) == 2
    assert provenance["positive"]["scientific_config_hash"] == manifest["scientific_config_hash"]
    assert provenance["positive"]["producer_identity_hash"] == manifest["producer_identity_hash"]


def test_nested_experiment_keeps_groups_out_of_outer_test_and_separates_deployment(
    tmp_path: Path,
):
    inputs = _make_inputs(tmp_path, samples_per_class=6)
    output_dir = tmp_path / "experiment"

    run_manifest_path = experiment.run_experiment(
        inputs,
        _parameters(),
        output_dir=output_dir,
    )

    assignments = pd.read_csv(output_dir / "outer_fold_assignments.csv")
    predictions = pd.read_csv(output_dir / "outer_predictions.csv")
    inner_choices = pd.read_csv(output_dir / "inner_choices.csv")
    outer_summary = json.loads(
        (output_dir / "outer_summary_metrics.json").read_text(encoding="utf-8")
    )
    deployment_choice = json.loads(
        (output_dir / "deployment_choice.json").read_text(encoding="utf-8")
    )
    deployment_model = json.loads(
        (output_dir / "deployment_model.json").read_text(encoding="utf-8")
    )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    assert assignments.groupby("group_id")["outer_fold"].nunique().eq(1).all()
    assert len(predictions) == len(assignments) == 12
    assert predictions["sample_id"].is_unique
    assert len(inner_choices) == 3
    assert {"mcc_defined", "f1_defined", "balanced_accuracy_defined"}.issubset(
        inner_choices.columns
    )
    assert outer_summary["scope"] == "pooled_outer_test_predictions"
    assert "not an unbiased performance estimate" in deployment_choice["interpretation"]
    assert run_manifest["input_contract"]["error_policy"] == "fail_on_any_detector_ERROR_row"
    assert run_manifest["schema_version"] == 1
    assert deployment_model["schema_version"] == 1
    assert deployment_model["feature_columns"] == list(experiment.RULE_MEASUREMENT_COLUMNS)
    assert "dssp_unassigned_residue_count" not in experiment.RULE_MEASUREMENT_COLUMNS
    assert run_manifest["status"] == "complete"
    assert run_manifest["phase"] == "complete"
    assert run_manifest["started_at_utc"].endswith("Z")
    assert run_manifest["completed_at_utc"].endswith("Z")
    assert run_manifest["scientific_design"]["score_calibration_claim"] == "none"
    assert run_manifest["parameters"]["penalty"] == "l2"
    assert run_manifest["parameters"]["l1_ratio"] == 0.0
    assert run_manifest["parameters"]["intercept_scaling"] == 1.0
    assert run_manifest["script"]["sha256"] == file_sha256(experiment.__file__)
    for state in run_manifest["outputs"].values():
        assert state["sha256"] == file_sha256(state["path"])


def test_nested_experiment_failure_manifest_records_phase_error_and_partial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = _make_inputs(tmp_path, samples_per_class=6)
    output_dir = tmp_path / "failed_experiment"
    original_write_csv = experiment._atomic_write_csv
    write_calls = 0

    def fail_during_artifact_writes(path: Path, dataframe: pd.DataFrame):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            running = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
            assert running["status"] == "running"
            assert running["phase"] == "writing_artifacts"
            raise OSError("deliberate artifact write failure")
        return original_write_csv(path, dataframe)

    monkeypatch.setattr(experiment, "_atomic_write_csv", fail_during_artifact_writes)

    with pytest.raises(OSError, match="deliberate artifact"):
        experiment.run_experiment(inputs, _parameters(), output_dir=output_dir)

    document = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["phase"] == "writing_artifacts"
    assert document["failed_at_utc"].endswith("Z")
    assert document["error"] == {
        "type": "OSError",
        "message": "deliberate artifact write failure",
    }
    assert set(document["outputs"]) == {"outer_fold_assignments.csv"}
    state = document["outputs"]["outer_fold_assignments.csv"]
    assert state["sha256"] == file_sha256(state["path"])
