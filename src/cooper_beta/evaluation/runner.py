from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in minimal installations
    pd = None

from cooper_beta import pipeline
from cooper_beta.constants import RESULT_BARREL, RESULT_ERROR
from cooper_beta.integrity import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    file_sha256,
)
from cooper_beta.provenance import (
    RUN_MANIFEST_KIND,
    RUN_MANIFEST_SCHEMA_VERSION,
    resolved_config_sections,
    scientific_producer_identity,
)

from .metrics import (
    CHAIN_GROUND_TRUTH_SCHEMA,
    CHAIN_GROUND_TRUTH_UNAVAILABLE,
    FILE_GROUND_TRUTH_SCHEMA,
    FILE_PREDICTION_SCHEMA,
    TRUTH_MANIFEST_COLUMNS,
    CandidateTruth,
    MetricErrorPolicy,
    assign_manifest_candidate_truth,
    compute_chain_metrics,
    compute_file_metrics,
    ensure_columns,
    print_metrics,
    resolve_error_policy,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a SHA-256 string.")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"{label} is not a valid SHA-256 digest.")
    return normalized


def _input_inventory(document: Mapping[str, object]) -> dict[str, str]:
    paths = document.get("_validated_input_paths")
    hashes = document.get("_validated_input_hashes")
    if not isinstance(paths, list) or not isinstance(hashes, list):
        raise RuntimeError("Detector manifest has not been validated.")
    return dict(zip((str(path) for path in paths), (str(value) for value in hashes), strict=True))


def validate_detector_artifact_manifest(
    manifest_path: Path,
    *,
    expected_output: Path,
) -> dict[str, object]:
    """Validate the scientific identity, inputs, and CSV bound by a detector manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    expected_output = expected_output.expanduser().resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Detector manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(document, dict):
        raise RuntimeError(f"Detector manifest must contain a JSON object: {manifest_path}")
    if document.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Detector manifest schema_version must be {RUN_MANIFEST_SCHEMA_VERSION}."
        )
    if document.get("manifest_kind") != RUN_MANIFEST_KIND or document.get("status") != "complete":
        raise RuntimeError("Detector manifest must describe a completed Cooper-Beta run.")
    run_id = document.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RuntimeError("Detector manifest run_id must be a non-empty string.")

    config = document.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Detector manifest config must be an object.")
    config_document, scientific, execution, io = resolved_config_sections(config)
    expected_hashes = {
        "config_hash": canonical_json_sha256(config_document),
        "scientific_config_hash": canonical_json_sha256(scientific),
        "execution_config_hash": canonical_json_sha256(execution),
        "io_config_hash": canonical_json_sha256(io),
    }
    for name, expected_hash in expected_hashes.items():
        if _sha256(document.get(name), label=name) != expected_hash:
            raise RuntimeError(f"Detector manifest {name} does not match its embedded config.")

    try:
        producer_identity = scientific_producer_identity(document)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Detector manifest producer provenance is invalid.") from exc
    if document.get("producer_identity") != producer_identity:
        raise RuntimeError("Detector manifest producer_identity does not match its provenance.")
    producer_hash = _sha256(document.get("producer_identity_hash"), label="producer_identity_hash")
    if producer_hash != canonical_json_sha256(producer_identity):
        raise RuntimeError("Detector manifest producer_identity_hash is inconsistent.")

    if document.get("input_file_hashing_enabled") is not True:
        raise RuntimeError("Evaluation requires detector input hashing.")
    input_files = document.get("input_files")
    input_states = document.get("input_file_state")
    if not isinstance(input_files, list) or not isinstance(input_states, list):
        raise RuntimeError("Detector manifest input files and states must be lists.")
    if not input_files or len(input_files) != len(input_states):
        raise RuntimeError("Detector manifest must contain one state per input file.")

    paths: list[str] = []
    hashes: list[str] = []
    for position, (path_value, state_value) in enumerate(
        zip(input_files, input_states, strict=True), start=1
    ):
        if not isinstance(path_value, str) or not isinstance(state_value, dict):
            raise RuntimeError(f"Detector input {position} has an invalid path or state.")
        path = Path(path_value).expanduser().resolve()
        recorded_path = Path(str(state_value.get("path", ""))).expanduser().resolve()
        if recorded_path != path or not path.is_file():
            raise RuntimeError(f"Detector input {position} is unavailable or has the wrong path.")
        digest = _sha256(state_value.get("sha256"), label=f"detector input {position}")
        if file_sha256(path) != digest:
            raise RuntimeError(f"Detector input {position} has changed since detection.")
        paths.append(str(path))
        hashes.append(digest)
    if len(paths) != len(set(paths)):
        raise RuntimeError("Detector manifest contains duplicate input paths.")

    binding = document.get("artifact_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("Detector manifest artifact_binding must be an object.")
    bound_path = Path(str(binding.get("csv_path", ""))).expanduser().resolve()
    bound_hash = _sha256(binding.get("csv_sha256"), label="artifact_binding.csv_sha256")
    if (
        binding.get("run_id") != run_id
        or binding.get("committed_by_run") is not True
        or bound_path != expected_output
        or not expected_output.is_file()
        or file_sha256(expected_output) != bound_hash
    ):
        raise RuntimeError("Detector artifact_binding does not match the result CSV.")

    document["producer_identity"] = producer_identity
    document["producer_identity_hash"] = producer_hash
    document["_validated_input_paths"] = paths
    document["_validated_input_hashes"] = hashes
    return document


def _validate_split_compatibility(
    positive: Mapping[str, object],
    negative: Mapping[str, object],
) -> dict[str, object]:
    if positive["scientific_config_hash"] != negative["scientific_config_hash"]:
        raise ValueError("Positive and negative runs use different scientific configurations.")
    if positive["producer_identity_hash"] != negative["producer_identity_hash"]:
        raise ValueError("Positive and negative runs use different detector implementations.")
    positive_inputs = _input_inventory(positive)
    negative_inputs = _input_inventory(negative)
    if set(positive_inputs) & set(negative_inputs):
        raise ValueError("Positive and negative splits contain the same input path.")
    positive_hashes = list(positive_inputs.values())
    negative_hashes = list(negative_inputs.values())
    if len(positive_hashes) != len(set(positive_hashes)):
        raise ValueError("Positive split contains byte-identical repeated structures.")
    if len(negative_hashes) != len(set(negative_hashes)):
        raise ValueError("Negative split contains byte-identical repeated structures.")
    if set(positive_hashes) & set(negative_hashes):
        raise ValueError("Positive and negative splits contain byte-identical structures.")
    return {
        "scientific_config_hash": positive["scientific_config_hash"],
        "producer_identity_hash": positive["producer_identity_hash"],
        "positive_structure_count": len(positive_inputs),
        "negative_structure_count": len(negative_inputs),
    }


def _load_candidate_manifest(path: Path, *, label: str) -> tuple[CandidateTruth, ...]:
    if pd is None:
        raise RuntimeError("pandas is required to run evaluation.")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(str(column) for column in dataframe.columns) != TRUTH_MANIFEST_COLUMNS:
        raise ValueError(
            f"{label} manifest must have exactly these columns in order: "
            + ", ".join(TRUTH_MANIFEST_COLUMNS)
        )
    if dataframe.empty:
        raise ValueError(f"{label} manifest contains no truth records.")
    records: list[CandidateTruth] = []
    for row_number, row in enumerate(dataframe.to_dict(orient="records"), start=2):
        try:
            records.append(
                CandidateTruth(
                    filename=str(row["filename"]),
                    source_path=str(row["source_path"]),
                    structure_sha256=str(row["structure_sha256"]),
                    target_author_chain_id=str(row["target_author_chain_id"]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"{label} manifest row {row_number} is invalid: {exc}") from exc
    if len({record.source_path for record in records}) != len(records):
        raise ValueError(f"{label} manifest contains duplicate source paths.")
    if len({record.filename for record in records}) != len(records):
        raise ValueError(f"{label} manifest contains duplicate filenames.")
    return tuple(records)


def load_positive_manifest(path: Path) -> tuple[CandidateTruth, ...]:
    """Load canonical positive target-chain identities."""
    return _load_candidate_manifest(path, label="Positive")


def load_negative_manifest(path: Path) -> tuple[CandidateTruth, ...]:
    """Load canonical negative target-chain identities."""
    return _load_candidate_manifest(path, label="Negative")


def _archive_truth_manifest(
    source: Path,
    destination: Path,
    *,
    label: str,
) -> tuple[CandidateTruth, ...]:
    source = source.expanduser().resolve()
    shutil.copyfile(source, destination)
    return _load_candidate_manifest(destination, label=label)


def run_detector(
    folder: Path,
    workers: int | None,
    prepare_workers: int | None,
    detector_overrides: dict[str, object] | list[str] | None = None,
    *,
    output_csv: Path,
) -> pd.DataFrame:
    """Run the detector and retain its CSV, manifest, and logs."""
    if pd is None:
        raise RuntimeError("pandas is required to run evaluation.")
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(f"{output_csv}.manifest.json")
    stdout_path = Path(f"{output_csv}.stdout.log")
    stderr_path = Path(f"{output_csv}.stderr.log")
    existing = [
        path for path in (output_csv, manifest_path, stdout_path, stderr_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite detector artifacts: " + ", ".join(map(str, existing))
        )
    with (
        stdout_path.open("x", encoding="utf-8") as stdout_handle,
        stderr_path.open("x", encoding="utf-8") as stderr_handle,
        contextlib.redirect_stdout(stdout_handle),
        contextlib.redirect_stderr(stderr_handle),
    ):
        pipeline.detect(
            str(folder),
            workers=workers,
            prepare_workers=prepare_workers,
            output=str(output_csv),
            overrides=detector_overrides,
            write_csv=True,
            print_summary=True,
            show_progress=True,
            strict_input=True,
        )
    if not output_csv.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Detector did not produce its CSV and manifest for {folder}.")
    return pd.read_csv(output_csv, keep_default_na=False)


def _validated_rows(dataframe: pd.DataFrame, *, label: str) -> pd.DataFrame:
    validated = ensure_columns(dataframe)
    if validated.empty:
        raise ValueError(f"{label} detector results contain no rows.")
    for column in ("filename", "source_path", "author_chain_id", "result"):
        validated[column] = validated[column].fillna("").astype(str).str.strip()
    if validated["source_path"].eq("").any():
        raise ValueError(f"{label} detector results contain an empty source_path.")
    validated["source_path"] = validated["source_path"].map(
        lambda value: str(Path(value).expanduser().resolve())
    )
    expected_names = validated["source_path"].map(lambda value: Path(value).name)
    if validated["filename"].ne(expected_names).any():
        raise ValueError(f"{label} detector filenames do not match their source paths.")
    duplicate = validated[["source_path", "author_chain_id"]].duplicated()
    if duplicate.any():
        raise ValueError(f"{label} detector results contain duplicate structure/chain rows.")
    return validated


def _validate_truth_binding(
    dataframe: pd.DataFrame,
    document: Mapping[str, object],
    truth: Sequence[CandidateTruth] | None,
    *,
    label: str,
) -> dict[str, int | bool]:
    inventory = _input_inventory(document)
    observed_paths = set(dataframe["source_path"])
    if observed_paths != set(inventory):
        raise ValueError(f"{label} detector rows do not exactly cover the manifest inputs.")
    result: dict[str, int | bool] = {
        "detector_rows": len(dataframe),
        "structure_count": len(inventory),
        "truth_bound": truth is not None,
    }
    if truth is None:
        return result
    truth_by_path = {record.source_path: record for record in truth}
    if set(truth_by_path) != set(inventory):
        raise ValueError(f"{label} truth rows do not exactly cover the detector inputs.")
    mismatches = [
        path for path, record in truth_by_path.items() if record.structure_sha256 != inventory[path]
    ]
    if mismatches:
        raise ValueError(f"{label} truth hashes do not match the detector inputs.")
    result["truth_rows"] = len(truth)
    return result


def _annotate_truth(
    dataframe: pd.DataFrame,
    truth: Sequence[CandidateTruth] | None,
    *,
    split: str,
    truth_value: int,
    schema: str,
) -> pd.DataFrame:
    if truth is not None:
        annotated, _ = assign_manifest_candidate_truth(
            dataframe,
            truth,
            truth_value=truth_value,
            split_name=split,
            ground_truth_schema=schema,
        )
        return annotated
    annotated = dataframe.copy()
    annotated["target_author_chain_id"] = ""
    annotated["truth_structure_sha256"] = ""
    annotated["ground_truth_role"] = f"{split}_file_unlabeled_chain"
    annotated["chain_ground_truth_schema"] = CHAIN_GROUND_TRUTH_UNAVAILABLE
    annotated["use_for_chain_metrics"] = False
    annotated["chain_y_true"] = pd.NA
    return annotated


def save_outputs(
    positive_raw: pd.DataFrame,
    negative_raw: pd.DataFrame,
    save_dir: Path,
    tag: str,
    *,
    positive_candidate_truth: Sequence[CandidateTruth] | None = None,
    negative_candidate_truth: Sequence[CandidateTruth] | None = None,
) -> tuple[str, str, pd.DataFrame]:
    """Write chain-level annotations and one any-chain row per input structure."""
    if (positive_candidate_truth is None) != (negative_candidate_truth is None):
        raise ValueError("Positive and negative target manifests must be provided together.")
    save_dir.mkdir(parents=True, exist_ok=True)
    positive = _validated_rows(positive_raw, label="Positive")
    negative = _validated_rows(negative_raw, label="Negative")
    schema = (
        CHAIN_GROUND_TRUTH_SCHEMA
        if positive_candidate_truth is not None
        else CHAIN_GROUND_TRUTH_UNAVAILABLE
    )
    positive = _annotate_truth(
        positive,
        positive_candidate_truth,
        split="positive",
        truth_value=1,
        schema=schema,
    )
    negative = _annotate_truth(
        negative,
        negative_candidate_truth,
        split="negative",
        truth_value=0,
        schema=schema,
    )

    def decorate(dataframe: pd.DataFrame, *, split: str, truth_value: int) -> pd.DataFrame:
        decorated = dataframe.copy()
        decorated["split"] = split
        decorated["file_y_true"] = truth_value
        decorated["file_id"] = decorated["source_path"]
        decorated["file_ground_truth_schema"] = FILE_GROUND_TRUTH_SCHEMA
        decorated["file_prediction_schema"] = FILE_PREDICTION_SCHEMA
        decorated["pred_barrel"] = decorated["result"].eq(RESULT_BARREL)
        decorated["is_error"] = decorated["result"].eq(RESULT_ERROR)
        return decorated

    combined = pd.concat(
        [
            decorate(positive, split="positive", truth_value=1),
            decorate(negative, split="negative", truth_value=0),
        ],
        ignore_index=True,
    )
    combined["chain_y_true"] = pd.to_numeric(combined["chain_y_true"], errors="coerce").astype(
        "Int64"
    )
    combined["y_true"] = combined["chain_y_true"]
    chain_output = save_dir / f"eval_chain_results_{tag}.csv"
    file_output = save_dir / f"eval_file_results_{tag}.csv"
    if chain_output.exists() or file_output.exists():
        raise FileExistsError(f"Evaluation outputs already exist for tag {tag!r}.")
    atomic_write_text(chain_output, lambda handle: combined.to_csv(handle, index=False), newline="")

    aggregated = combined.groupby(["split", "file_y_true", "file_id"], as_index=False).agg(
        filename=("filename", "first"),
        source_path=("source_path", "first"),
        file_ground_truth_schema=("file_ground_truth_schema", "first"),
        file_prediction_schema=("file_prediction_schema", "first"),
        target_author_chain_id=("target_author_chain_id", "first"),
        truth_structure_sha256=("truth_structure_sha256", "first"),
        pred_barrel_any=("pred_barrel", "max"),
        error_chains_n=("is_error", "sum"),
        chains_n=("result", "size"),
    )
    aggregated["y_true"] = aggregated["file_y_true"]
    atomic_write_text(
        file_output, lambda handle: aggregated.to_csv(handle, index=False), newline=""
    )
    return str(chain_output), str(file_output), aggregated


def _input_directory(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} input is not a directory: {resolved}")
    return resolved


def _evaluation_tag(tag: str) -> str:
    value = str(tag).strip()
    if not value or Path(value).name != value or "\\" in value:
        raise ValueError("tag must be one filename component")
    return value


def evaluate(
    true_dir: Path,
    false_dir: Path,
    workers: int | None,
    prepare_workers: int | None,
    save_dir: Path,
    metric_level: str,
    tag: str,
    detector_overrides: dict[str, object] | list[str] | None = None,
    print_metric_tables: bool = True,
    *,
    metric_error_policy: MetricErrorPolicy | str = MetricErrorPolicy.STRICT,
    positive_manifest: Path | None = None,
    negative_manifest: Path | None = None,
) -> dict[str, object]:
    """Run both splits, bind optional chain truth, and compute requested metrics."""
    if metric_level not in {"chain", "file", "both"}:
        raise ValueError("metric_level must be one of: chain, file, both")
    if pd is None:
        raise RuntimeError("pandas is required (pip install 'cooper-beta[eval]').")
    error_policy = resolve_error_policy(metric_error_policy)
    tag = _evaluation_tag(tag)
    positive_dir = _input_directory(Path(true_dir), label="Positive")
    negative_dir = _input_directory(Path(false_dir), label="Negative")
    if (
        positive_dir == negative_dir
        or positive_dir in negative_dir.parents
        or negative_dir in positive_dir.parents
    ):
        raise ValueError("Positive and negative directories must be disjoint and non-nested.")

    positive_manifest = positive_manifest.expanduser().resolve() if positive_manifest else None
    negative_manifest = negative_manifest.expanduser().resolve() if negative_manifest else None
    if (positive_manifest is None) != (negative_manifest is None):
        raise ValueError("Positive and negative target manifests must be provided together.")
    chain_requested = metric_level in {"chain", "both"}
    if chain_requested and positive_manifest is None:
        raise ValueError("Chain metrics require both target manifests.")

    save_dir = save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    detector_dir = save_dir / f"detector_runs_{tag}"
    suite_path = save_dir / f"evaluation_manifest_{tag}.json"
    chain_output = save_dir / f"eval_chain_results_{tag}.csv"
    file_output = save_dir / f"eval_file_results_{tag}.csv"
    if any(path.exists() for path in (detector_dir, suite_path, chain_output, file_output)):
        raise FileExistsError(f"Evaluation artifacts already exist for tag {tag!r}.")
    detector_dir.mkdir()
    positive_csv = detector_dir / "positive_results.csv"
    negative_csv = detector_dir / "negative_results.csv"
    archived_positive = detector_dir / "positive_truth.csv" if positive_manifest else None
    archived_negative = detector_dir / "negative_truth.csv" if negative_manifest else None
    suite: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "tag": tag,
        "metric_level": metric_level,
        "metric_error_policy": error_policy.value,
        "inputs": {
            "positive_directory": str(positive_dir),
            "negative_directory": str(negative_dir),
            "positive_truth": str(positive_manifest) if positive_manifest else None,
            "negative_truth": str(negative_manifest) if negative_manifest else None,
            "archived_positive_truth": str(archived_positive) if archived_positive else None,
            "archived_negative_truth": str(archived_negative) if archived_negative else None,
        },
        "detector": {
            "overrides": detector_overrides,
            "positive_csv": str(positive_csv),
            "positive_manifest": f"{positive_csv}.manifest.json",
            "negative_csv": str(negative_csv),
            "negative_manifest": f"{negative_csv}.manifest.json",
        },
        "outputs": {"chain_csv": str(chain_output), "file_csv": str(file_output)},
    }
    atomic_write_json(suite_path, suite, indent=2)

    try:
        positive_truth = (
            _archive_truth_manifest(positive_manifest, archived_positive, label="Positive")
            if positive_manifest is not None and archived_positive is not None
            else None
        )
        negative_truth = (
            _archive_truth_manifest(negative_manifest, archived_negative, label="Negative")
            if negative_manifest is not None and archived_negative is not None
            else None
        )
        positive_frame = _validated_rows(
            run_detector(
                positive_dir,
                workers,
                prepare_workers,
                detector_overrides,
                output_csv=positive_csv,
            ),
            label="Positive",
        )
        negative_frame = _validated_rows(
            run_detector(
                negative_dir,
                workers,
                prepare_workers,
                detector_overrides,
                output_csv=negative_csv,
            ),
            label="Negative",
        )
        positive_document = validate_detector_artifact_manifest(
            Path(f"{positive_csv}.manifest.json"), expected_output=positive_csv
        )
        negative_document = validate_detector_artifact_manifest(
            Path(f"{negative_csv}.manifest.json"), expected_output=negative_csv
        )
        suite["split_compatibility"] = _validate_split_compatibility(
            positive_document, negative_document
        )
        suite["truth_binding"] = {
            "positive": _validate_truth_binding(
                positive_frame, positive_document, positive_truth, label="Positive"
            ),
            "negative": _validate_truth_binding(
                negative_frame, negative_document, negative_truth, label="Negative"
            ),
        }
        chain_csv, file_csv, aggregated = save_outputs(
            positive_frame,
            negative_frame,
            save_dir,
            tag,
            positive_candidate_truth=positive_truth,
            negative_candidate_truth=negative_truth,
        )

        row: dict[str, object] = {
            "tag": tag,
            "chain_csv": chain_csv,
            "file_csv": file_csv,
            "evaluation_manifest": str(suite_path),
            "positive_detector_csv": str(positive_csv),
            "positive_detector_manifest": f"{positive_csv}.manifest.json",
            "negative_detector_csv": str(negative_csv),
            "negative_detector_manifest": f"{negative_csv}.manifest.json",
            "metric_error_policy": error_policy.value,
            "chain_metrics_computed": False,
            "file_metrics_computed": False,
            "positive_detector_rows": len(positive_frame),
            "negative_detector_rows": len(negative_frame),
        }
        if isinstance(detector_overrides, dict):
            row.update(detector_overrides)
        computed: list[str] = []
        if chain_requested:
            assert positive_truth is not None and negative_truth is not None
            metrics, extra = compute_chain_metrics(
                positive_frame,
                negative_frame,
                positive_candidate_truth=positive_truth,
                negative_candidate_truth=negative_truth,
                error_policy=error_policy,
            )
            row["chain_metrics_computed"] = True
            row.update({f"chain_{name}": value for name, value in {**metrics, **extra}.items()})
            computed.append("chain")
            if print_metric_tables:
                print_metrics("=== Chain-level target chains ===", metrics)
        if metric_level in {"file", "both"}:
            metrics, extra = compute_file_metrics(aggregated, error_policy=error_policy)
            row["file_metrics_computed"] = True
            row.update({f"file_{name}": value for name, value in {**metrics, **extra}.items()})
            computed.append("file")
            if print_metric_tables:
                print_metrics("=== File-level any-chain prediction ===", metrics)
        row["metric_levels_computed"] = ",".join(computed)
    except Exception as exc:
        suite.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        atomic_write_json(suite_path, suite, indent=2)
        raise

    suite.update({"status": "complete", "completed_at_utc": _utc_now(), "metrics": row})
    atomic_write_json(suite_path, suite, indent=2)
    return row
