#!/usr/bin/env python3
"""Run a leakage-resistant Cooper-Beta decision-model experiment.

The observation unit is exactly one manifest-selected target chain per input
structure.  Detector outputs are consumed as immutable artifacts: this script
never runs the detector, invents labels for partner chains, imputes features, or
silently excludes errors.

Performance estimation uses nested StratifiedGroupKFold.  Hyperparameter and
threshold selection happen only inside each outer training partition.  A
separate full-data tuning pass selects the deployable model; its tuning score is
not reported as a performance estimate.
"""

from __future__ import annotations

import argparse
import inspect
import os
import platform
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int_]
StringArray = NDArray[np.str_]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cooper_beta._version import __version__ as source_package_version  # noqa: E402
from cooper_beta.constants import (  # noqa: E402
    DEFAULT_RESULT_COLUMNS,
    RESULT_BARREL,
    RESULT_ERROR,
    RESULT_NON_BARREL,
    RULE_MEASUREMENT_COLUMNS,
)
from cooper_beta.evaluation.runner import validate_detector_artifact_manifest  # noqa: E402
from cooper_beta.integrity import (  # noqa: E402
    atomic_write_json,
    file_sha256,
)
from cooper_beta.models import DetectionResult  # noqa: E402

LABEL_MANIFEST_COLUMNS = (
    "filename",
    "source_path",
    "target_author_chain_id",
    "group_id",
)
VALID_OBJECTIVES = ("mcc", "f1", "balanced_accuracy")
OUTPUT_FILENAMES = (
    "outer_fold_assignments.csv",
    "inner_candidate_metrics.csv",
    "inner_choices.csv",
    "outer_predictions.csv",
    "outer_fold_metrics.csv",
    "outer_summary_metrics.json",
    "deployment_inner_candidate_metrics.csv",
    "deployment_choice.json",
    "deployment_model.json",
    "run_manifest.json",
)


class ExperimentInputError(ValueError):
    """Raised when an input violates the scientific experiment contract."""


@dataclass(frozen=True)
class ExperimentParameters:
    algorithm: str
    c_values: tuple[float, ...]
    threshold_min: float
    threshold_max: float
    threshold_count: int
    outer_folds: int
    inner_folds: int
    deployment_folds: int
    objective: str
    outer_split_seed: int
    inner_split_seed: int
    deployment_split_seed: int
    model_seed: int
    max_iter: int
    tolerance: float
    class_weight: str
    solver: str
    penalty: str
    l1_ratio: float
    dual: bool
    fit_intercept: bool
    intercept_scaling: float
    scaler_with_mean: bool
    scaler_with_std: bool
    native_threads: int

    def validate(self) -> None:
        if self.algorithm != "logistic_regression":
            raise ExperimentInputError(
                "Only the explicit logistic_regression algorithm is supported."
            )
        if not self.c_values:
            raise ExperimentInputError("At least one LogisticRegression C value is required.")
        if any(not isfinite(value) or value <= 0.0 for value in self.c_values):
            raise ExperimentInputError("Every C value must be finite and greater than zero.")
        if len(set(self.c_values)) != len(self.c_values):
            raise ExperimentInputError("C values must be unique.")
        if not (0.0 <= self.threshold_min < self.threshold_max <= 1.0):
            raise ExperimentInputError("Threshold bounds must satisfy 0 <= minimum < maximum <= 1.")
        if self.threshold_count < 2:
            raise ExperimentInputError("Threshold count must be at least two.")
        if min(self.outer_folds, self.inner_folds, self.deployment_folds) < 2:
            raise ExperimentInputError("Every cross-validation fold count must be at least two.")
        if self.objective not in VALID_OBJECTIVES:
            raise ExperimentInputError(f"Unknown selection objective: {self.objective!r}.")
        if (
            min(
                self.outer_split_seed,
                self.inner_split_seed,
                self.deployment_split_seed,
                self.model_seed,
            )
            < 0
        ):
            raise ExperimentInputError("Random seeds must be non-negative integers.")
        if self.max_iter <= 0:
            raise ExperimentInputError("LogisticRegression max_iter must be positive.")
        if not isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ExperimentInputError("LogisticRegression tolerance must be finite and positive.")
        if self.class_weight != "balanced":
            raise ExperimentInputError("The fixed estimator requires class_weight='balanced'.")
        if self.solver != "liblinear":
            raise ExperimentInputError("The fixed estimator requires solver='liblinear'.")
        if self.penalty != "l2":
            raise ExperimentInputError("The fixed estimator requires penalty='l2'.")
        if self.l1_ratio != 0.0:
            raise ExperimentInputError("The fixed L2 estimator requires l1_ratio=0.0.")
        if self.dual:
            raise ExperimentInputError("The fixed estimator requires dual=False.")
        if not self.fit_intercept:
            raise ExperimentInputError("The fixed estimator requires fit_intercept=True.")
        if not isfinite(self.intercept_scaling) or self.intercept_scaling <= 0.0:
            raise ExperimentInputError(
                "LogisticRegression intercept_scaling must be finite and positive."
            )
        if not self.scaler_with_mean or not self.scaler_with_std:
            raise ExperimentInputError(
                "The fixed preprocessing contract requires centered, variance-scaled features."
            )
        if self.native_threads <= 0:
            raise ExperimentInputError("Native thread limit must be positive.")


@dataclass(frozen=True)
class DetectorArtifact:
    split: str
    csv_path: Path
    manifest_path: Path
    dataframe: pd.DataFrame
    manifest: dict[str, Any]
    input_states: dict[Path, dict[str, Any]]
    csv_sha256: str
    manifest_sha256: str
    scientific_config_hash: str
    producer_identity_hash: str


@dataclass(frozen=True)
class InputSpecification:
    positive_csv: Path
    positive_detector_manifest: Path
    positive_label_manifest: Path
    negative_csv: Path
    negative_detector_manifest: Path
    negative_label_manifest: Path
    grouping_method: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _require_sha256(value: object, *, context: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ExperimentInputError(f"{context} is not a valid SHA-256 digest.")
    return normalized


def _read_public_detector_csv(path: Path, *, split: str) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(path, dtype=object, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ExperimentInputError(f"Cannot read {split} detector CSV: {path}") from exc
    if tuple(dataframe.columns) != DEFAULT_RESULT_COLUMNS:
        raise ExperimentInputError(
            f"{split} detector CSV does not match the fixed public schema/order; "
            f"expected {list(DEFAULT_RESULT_COLUMNS)!r}, got {list(dataframe.columns)!r}."
        )
    if dataframe.empty:
        raise ExperimentInputError(f"{split} detector CSV contains no rows.")

    for row_number, row in enumerate(dataframe.to_dict(orient="records"), start=2):
        try:
            DetectionResult.from_row(row)
        except (TypeError, ValueError) as exc:
            raise ExperimentInputError(
                f"{split} detector CSV row {row_number} violates the public result schema: {exc}"
            ) from exc

    errors = dataframe.loc[dataframe["result"].astype(str).eq(RESULT_ERROR)]
    if not errors.empty:
        preview = ", ".join(
            f"{row.filename}:{row.author_chain_id or '<file>'}:{row.error_code}"
            for row in errors.head(5).itertuples(index=False)
        )
        raise ExperimentInputError(
            f"{split} detector CSV contains {len(errors)} ERROR row(s); the experiment error "
            f"policy is fail: {preview}"
        )

    normalized_sources = dataframe["source_path"].map(
        lambda value: str(Path(str(value)).expanduser().resolve())
    )
    author_chain_ids = dataframe["author_chain_id"].astype(str)
    duplicated = pd.DataFrame(
        {"source_path": normalized_sources, "author_chain_id": author_chain_ids}
    ).duplicated(keep=False)
    if duplicated.any():
        preview = ", ".join(
            f"{source}:{author_chain_id}"
            for source, author_chain_id in zip(
                normalized_sources[duplicated].head(5),
                author_chain_ids[duplicated].head(5),
                strict=True,
            )
        )
        raise ExperimentInputError(
            f"{split} detector CSV has duplicate source_path/author_chain_id observations: "
            f"{preview}"
        )
    dataframe = dataframe.copy()
    dataframe["_resolved_source_path"] = normalized_sources
    return dataframe


def validate_detector_artifact(
    csv_path: Path,
    manifest_path: Path,
    *,
    split: str,
) -> DetectorArtifact:
    csv_path = csv_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))
    try:
        manifest = validate_detector_artifact_manifest(
            manifest_path,
            expected_output=csv_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExperimentInputError(f"Invalid {split} detector artifact: {exc}") from exc
    config = manifest["config"]
    assert isinstance(config, dict)  # validated by validate_detector_artifact_manifest
    csv_digest = file_sha256(csv_path)

    output_config = config.get("output")
    if not isinstance(output_config, dict):
        raise ExperimentInputError(f"{split} detector output config must be an object.")
    configured_csv = Path(str(output_config.get("csv_path", ""))).expanduser().resolve()
    if configured_csv != csv_path:
        raise ExperimentInputError(f"{split} detector config csv_path does not identify its CSV.")
    if output_config.get("write_manifest") is not True:
        raise ExperimentInputError(f"{split} detector config did not require a run manifest.")
    if output_config.get("hash_input_files") is not True:
        raise ExperimentInputError(f"{split} detector config did not require input hashes.")
    input_files = manifest["input_files"]
    input_file_state = manifest["input_file_state"]
    assert isinstance(input_files, list)
    assert isinstance(input_file_state, list)
    states: dict[Path, dict[str, Any]] = {}
    seen_hashes: set[str] = set()
    for position, (input_value, state_value) in enumerate(
        zip(input_files, input_file_state, strict=True), start=1
    ):
        if not isinstance(state_value, dict):
            raise ExperimentInputError(f"{split} input state {position} must be an object.")
        input_path = Path(str(input_value)).expanduser().resolve()
        digest = _require_sha256(
            state_value.get("sha256"), context=f"{split} input {position} SHA-256"
        )
        if digest in seen_hashes:
            raise ExperimentInputError(
                f"{split} detector inputs include byte-identical repeated structures ({digest})."
            )
        seen_hashes.add(digest)
        states[input_path] = {**state_value, "sha256": digest, "path": str(input_path)}

    dataframe = _read_public_detector_csv(csv_path, split=split)
    observed_sources = {Path(value) for value in dataframe["_resolved_source_path"]}
    unknown_sources = sorted(observed_sources - set(states), key=str)
    if unknown_sources:
        raise ExperimentInputError(
            f"{split} detector CSV contains source_path values absent from its manifest: "
            + ", ".join(str(path) for path in unknown_sources[:5])
        )
    missing_observations = sorted(set(states) - observed_sources, key=str)
    if missing_observations:
        raise ExperimentInputError(
            f"{split} detector CSV has no row for manifest input(s): "
            + ", ".join(str(path) for path in missing_observations[:5])
        )

    return DetectorArtifact(
        split=split,
        csv_path=csv_path,
        manifest_path=manifest_path,
        dataframe=dataframe,
        manifest=manifest,
        input_states=states,
        csv_sha256=csv_digest,
        manifest_sha256=file_sha256(manifest_path),
        scientific_config_hash=str(manifest["scientific_config_hash"]),
        producer_identity_hash=str(manifest["producer_identity_hash"]),
    )


def _load_label_manifest(path: Path, *, split: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    try:
        dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ExperimentInputError(f"Cannot read {split} label manifest: {path}") from exc
    if tuple(dataframe.columns) != LABEL_MANIFEST_COLUMNS:
        raise ExperimentInputError(
            f"{split} label manifest must have exactly these ordered columns: "
            f"{list(LABEL_MANIFEST_COLUMNS)!r}."
        )
    if dataframe.empty:
        raise ExperimentInputError(f"{split} label manifest is empty.")
    cleaned = dataframe.copy()
    for column in LABEL_MANIFEST_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip()
        if cleaned[column].eq("").any():
            raise ExperimentInputError(f"{split} label manifest contains an empty {column}.")
    cleaned["_resolved_source_path"] = cleaned["source_path"].map(
        lambda value: str(
            (path.parent / value).resolve()
            if not Path(value).expanduser().is_absolute()
            else Path(value).expanduser().resolve()
        )
    )
    duplicated = cleaned["_resolved_source_path"].duplicated(keep=False)
    if duplicated.any():
        preview = ", ".join(cleaned.loc[duplicated, "source_path"].head(5))
        raise ExperimentInputError(
            f"{split} label manifest must define exactly one target chain per file: {preview}"
        )
    return cleaned


def _detector_identity(artifact: DetectorArtifact) -> dict[str, Any]:
    producer_identity = artifact.manifest.get("producer_identity")
    if not isinstance(producer_identity, dict):  # validated by shared validator
        raise ExperimentInputError(f"{artifact.split} producer_identity must be an object.")
    return {
        "scientific_config_hash": artifact.scientific_config_hash,
        "producer_identity_hash": artifact.producer_identity_hash,
        "producer_identity": producer_identity,
    }


def _validate_paired_detector_identity(
    positive: DetectorArtifact, negative: DetectorArtifact
) -> None:
    if positive.scientific_config_hash != negative.scientific_config_hash:
        raise ExperimentInputError(
            "Positive and negative detector artifacts do not share the same scientific "
            "configuration hash."
        )
    if positive.producer_identity_hash != negative.producer_identity_hash:
        raise ExperimentInputError(
            "Positive and negative detector artifacts do not share the same scientific "
            "producer identity hash."
        )
    positive_hashes = {state["sha256"] for state in positive.input_states.values()}
    negative_hashes = {state["sha256"] for state in negative.input_states.values()}
    path_overlap = set(positive.input_states) & set(negative.input_states)
    hash_overlap = positive_hashes & negative_hashes
    if path_overlap or hash_overlap:
        raise ExperimentInputError(
            "Positive/negative split leakage detected by input path or structure SHA-256."
        )


def _select_manifest_samples(
    artifact: DetectorArtifact,
    label_manifest_path: Path,
    *,
    label: int,
) -> pd.DataFrame:
    labels = _load_label_manifest(label_manifest_path, split=artifact.split)
    label_paths = {Path(value) for value in labels["_resolved_source_path"]}
    detector_paths = set(artifact.input_states)
    if label_paths != detector_paths:
        missing = sorted(detector_paths - label_paths, key=str)
        extra = sorted(label_paths - detector_paths, key=str)
        raise ExperimentInputError(
            f"{artifact.split} label manifest and detector inputs differ; "
            f"missing={list(map(str, missing[:5]))!r}, extra={list(map(str, extra[:5]))!r}."
        )

    detector = artifact.dataframe
    selected_rows: list[dict[str, Any]] = []
    for manifest_row in labels.to_dict(orient="records"):
        source_path = Path(str(manifest_row["_resolved_source_path"]))
        target_author_chain_id = str(manifest_row["target_author_chain_id"])
        matching = detector.loc[
            detector["_resolved_source_path"].eq(str(source_path))
            & detector["author_chain_id"].astype(str).eq(target_author_chain_id)
        ]
        if len(matching) != 1:
            raise ExperimentInputError(
                f"{artifact.split} target must resolve to exactly one detector row: "
                f"{source_path}:{target_author_chain_id}; found {len(matching)}."
            )
        result_row = matching.iloc[0]
        target_result = str(result_row["result"])
        if target_result not in {RESULT_BARREL, RESULT_NON_BARREL}:
            raise ExperimentInputError(
                f"{artifact.split} target {source_path}:{target_author_chain_id} must have a "
                f"decision-stage BARREL/NON_BARREL result; observed {target_result!r}."
            )
        filename = str(manifest_row["filename"])
        if str(result_row["filename"]) != filename:
            raise ExperimentInputError(
                f"{artifact.split} label filename does not match detector row for {source_path}."
            )
        if Path(filename).name != source_path.name:
            raise ExperimentInputError(
                f"{artifact.split} label filename does not match source_path basename: "
                f"{filename!r} != {source_path.name!r}."
            )
        input_digest = str(artifact.input_states[source_path]["sha256"])
        sample: dict[str, Any] = {
            "sample_id": f"sha256:{input_digest}:author_chain:{target_author_chain_id}",
            "split": artifact.split,
            "filename": filename,
            "source_path": str(source_path),
            "structure_sha256": input_digest,
            "target_author_chain_id": target_author_chain_id,
            "group_id": str(manifest_row["group_id"]),
            "y_true": int(label),
        }
        for feature in RULE_MEASUREMENT_COLUMNS:
            try:
                value = float(result_row[feature])
            except (TypeError, ValueError) as exc:
                raise ExperimentInputError(
                    f"Feature {feature!r} is not numeric for {sample['sample_id']}."
                ) from exc
            if not isfinite(value):
                raise ExperimentInputError(
                    f"Feature {feature!r} is NaN or infinite for {sample['sample_id']}."
                )
            sample[feature] = value
        selected_rows.append(sample)
    return pd.DataFrame(selected_rows)


def load_experiment_samples(inputs: InputSpecification) -> tuple[pd.DataFrame, dict[str, Any]]:
    grouping_method = inputs.grouping_method.strip()
    if not grouping_method:
        raise ExperimentInputError(
            "grouping_method is required and must identify an external homology/family method."
        )
    positive = validate_detector_artifact(
        inputs.positive_csv, inputs.positive_detector_manifest, split="positive"
    )
    negative = validate_detector_artifact(
        inputs.negative_csv, inputs.negative_detector_manifest, split="negative"
    )
    _validate_paired_detector_identity(positive, negative)
    positive_samples = _select_manifest_samples(positive, inputs.positive_label_manifest, label=1)
    negative_samples = _select_manifest_samples(negative, inputs.negative_label_manifest, label=0)
    samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
    if samples["sample_id"].duplicated().any():
        raise ExperimentInputError("Selected target-chain sample IDs are not unique.")
    if set(samples["y_true"]) != {0, 1}:
        raise ExperimentInputError("Both positive and negative target-chain samples are required.")
    values = samples[list(RULE_MEASUREMENT_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ExperimentInputError(
            "Feature matrix contains NaN or infinity; imputation is forbidden."
        )
    provenance = {
        "grouping_method": grouping_method,
        "positive": _artifact_provenance(
            positive, inputs.positive_label_manifest.expanduser().resolve()
        ),
        "negative": _artifact_provenance(
            negative, inputs.negative_label_manifest.expanduser().resolve()
        ),
        "paired_detector_identity": _detector_identity(positive),
    }
    return samples, provenance


def _artifact_provenance(artifact: DetectorArtifact, label_manifest_path: Path) -> dict[str, Any]:
    return {
        "detector_csv": str(artifact.csv_path),
        "detector_csv_sha256": artifact.csv_sha256,
        "detector_manifest": str(artifact.manifest_path),
        "detector_manifest_sha256": artifact.manifest_sha256,
        "label_manifest": str(label_manifest_path),
        "label_manifest_sha256": file_sha256(label_manifest_path),
        "detector_config_hash": artifact.manifest["config_hash"],
        "scientific_config_hash": artifact.scientific_config_hash,
        "producer_identity_hash": artifact.producer_identity_hash,
        "detector_config": artifact.manifest["config"],
        "input_files": [
            {
                "path": str(path),
                "size": int(state["size"]),
                "sha256": state["sha256"],
            }
            for path, state in sorted(artifact.input_states.items(), key=lambda item: str(item[0]))
        ],
    }


def confusion_metrics(
    y_true: IntegerArray, y_pred: IntegerArray
) -> dict[str, int | float | bool | str | None]:
    raw_true = np.asarray(y_true)
    raw_predicted = np.asarray(y_pred)
    if (
        raw_true.ndim != 1
        or raw_predicted.ndim != 1
        or raw_true.shape != raw_predicted.shape
        or raw_true.size == 0
    ):
        raise ExperimentInputError("Metrics require equal, non-empty one-dimensional arrays.")

    def validated_binary_integers(values: IntegerArray, *, label: str) -> IntegerArray:
        normalized: list[int] = []
        for position, value in enumerate(values.tolist()):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ExperimentInputError(
                    f"{label} position {position} must be a non-boolean binary integer."
                )
            numeric = int(value)
            if numeric not in {0, 1}:
                raise ExperimentInputError(f"{label} position {position} must be either 0 or 1.")
            normalized.append(numeric)
        return np.asarray(normalized, dtype=int)

    true = validated_binary_integers(raw_true, label="y_true")
    predicted = validated_binary_integers(raw_predicted, label="y_pred")
    tp = int(np.sum((true == 1) & (predicted == 1)))
    fp = int(np.sum((true == 0) & (predicted == 1)))
    tn = int(np.sum((true == 0) & (predicted == 0)))
    fn = int(np.sum((true == 1) & (predicted == 0)))

    def ratio(numerator: float, denominator: float) -> tuple[float | None, bool]:
        return (float(numerator / denominator), True) if denominator else (None, False)

    recall, recall_defined = ratio(tp, tp + fn)
    precision, precision_defined = ratio(tp, tp + fp)
    specificity, specificity_defined = ratio(tn, tn + fp)
    accuracy, accuracy_defined = ratio(tp + tn, true.size)
    f1, f1_defined = ratio(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = (
        float((recall + specificity) / 2.0)
        if recall is not None and specificity is not None
        else None
    )
    balanced_accuracy_defined = recall_defined and specificity_defined
    mcc_denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc, mcc_defined = ratio((tp * tn) - (fp * fn), mcc_denominator)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "recall": recall,
        "recall_defined": recall_defined,
        "precision": precision,
        "precision_defined": precision_defined,
        "specificity": specificity,
        "specificity_defined": specificity_defined,
        "accuracy": accuracy,
        "accuracy_defined": accuracy_defined,
        "f1": f1,
        "f1_defined": f1_defined,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_defined": balanced_accuracy_defined,
        "mcc": mcc,
        "mcc_defined": mcc_defined,
        "undefined_metric_policy": "serialize_null_and_mark_undefined",
    }


def _validated_group_splits(
    features: FloatArray,
    labels: IntegerArray,
    groups: StringArray,
    *,
    n_splits: int,
    seed: int,
    context: str,
) -> list[tuple[IntegerArray, IntegerArray]]:
    if len(np.unique(groups)) < n_splits:
        raise ExperimentInputError(
            f"{context} requires at least {n_splits} distinct group_id values."
        )
    for label in (0, 1):
        group_count = len(np.unique(groups[labels == label]))
        if group_count < n_splits:
            raise ExperimentInputError(
                f"{context} requires at least {n_splits} groups containing class {label}; "
                f"found {group_count}."
            )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            splits = list(splitter.split(features, labels, groups=groups))
        except (ValueError, UserWarning) as exc:
            raise ExperimentInputError(f"Cannot construct {context}: {exc}") from exc
    if len(splits) != n_splits:
        raise ExperimentInputError(f"{context} did not produce exactly {n_splits} folds.")
    seen_validation: set[int] = set()
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_groups = set(groups[train_indices])
        validation_groups = set(groups[validation_indices])
        if train_groups & validation_groups:
            raise RuntimeError(f"{context} fold {fold} leaks group_id values.")
        if set(labels[train_indices]) != {0, 1} or set(labels[validation_indices]) != {0, 1}:
            raise ExperimentInputError(
                f"{context} fold {fold} does not contain both classes in train and validation."
            )
        overlap = seen_validation & set(map(int, validation_indices))
        if overlap:
            raise RuntimeError(f"{context} validation assignments overlap: {sorted(overlap)[:5]}")
        seen_validation.update(map(int, validation_indices))
    if seen_validation != set(range(len(labels))):
        raise RuntimeError(f"{context} does not assign every observation exactly once.")
    return splits


def _build_model(c_value: float, parameters: ExperimentParameters) -> Pipeline:
    penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
    regularization_api = (
        {"l1_ratio": parameters.l1_ratio}
        if penalty_default == "deprecated"
        else {"penalty": parameters.penalty}
    )
    return Pipeline(
        [
            (
                "scale",
                StandardScaler(
                    with_mean=parameters.scaler_with_mean,
                    with_std=parameters.scaler_with_std,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    dual=parameters.dual,
                    solver=parameters.solver,
                    class_weight=parameters.class_weight,
                    fit_intercept=parameters.fit_intercept,
                    intercept_scaling=parameters.intercept_scaling,
                    max_iter=parameters.max_iter,
                    tol=parameters.tolerance,
                    random_state=parameters.model_seed,
                    **regularization_api,
                ),
            ),
        ]
    )


def _fit_model(
    model: Pipeline, features: FloatArray, labels: IntegerArray, *, context: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(features, labels)
        except ConvergenceWarning as exc:
            raise RuntimeError(f"LogisticRegression did not converge during {context}.") from exc


def _positive_scores(model: Pipeline, features: FloatArray) -> FloatArray:
    scores = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise RuntimeError("LogisticRegression produced a non-finite positive-class probability.")
    return scores


def _selection_key(row: dict[str, Any], objective: str) -> tuple[Any, ...]:
    def descending_metric(name: str) -> float:
        if not bool(row[f"{name}_defined"]):
            return 0.0
        value = row[name]
        if value is None:
            raise RuntimeError(f"Metric {name!r} is marked defined but has a null value.")
        return -float(value)

    return (
        not bool(row[f"{objective}_defined"]),
        descending_metric(objective),
        not bool(row["balanced_accuracy_defined"]),
        descending_metric("balanced_accuracy"),
        not bool(row["f1_defined"]),
        descending_metric("f1"),
        float(row["C"]),
        abs(float(row["threshold"]) - 0.5),
        float(row["threshold"]),
    )


def tune_on_grouped_oof(
    features: FloatArray,
    labels: IntegerArray,
    groups: StringArray,
    *,
    parameters: ExperimentParameters,
    n_splits: int,
    split_seed: int,
    context: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits = _validated_group_splits(
        features,
        labels,
        groups,
        n_splits=n_splits,
        seed=split_seed,
        context=context,
    )
    thresholds = np.linspace(
        parameters.threshold_min,
        parameters.threshold_max,
        num=parameters.threshold_count,
        dtype=float,
    )
    candidate_rows: list[dict[str, Any]] = []
    for c_value in parameters.c_values:
        oof_scores = np.full(labels.shape, np.nan, dtype=float)
        for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
            model = _build_model(c_value, parameters)
            _fit_model(
                model,
                features[train_indices],
                labels[train_indices],
                context=f"{context} fold {fold}, C={c_value:g}",
            )
            oof_scores[validation_indices] = _positive_scores(model, features[validation_indices])
        if not np.isfinite(oof_scores).all():
            raise RuntimeError(f"{context} left missing/non-finite OOF scores for C={c_value:g}.")
        for threshold in thresholds:
            metrics = confusion_metrics(labels, (oof_scores >= threshold).astype(int))
            candidate_rows.append(
                {
                    "C": float(c_value),
                    "threshold": float(threshold),
                    "inner_folds": int(n_splits),
                    "inner_split_seed": int(split_seed),
                    **metrics,
                }
            )
    eligible = [row for row in candidate_rows if bool(row[f"{parameters.objective}_defined"])]
    if not eligible:
        raise ExperimentInputError(
            f"{context} cannot select {parameters.objective}: it is undefined for every candidate."
        )
    selected = min(eligible, key=lambda row: _selection_key(row, parameters.objective))
    selected_identity = (selected["C"], selected["threshold"])
    for row in candidate_rows:
        row["selected"] = (row["C"], row["threshold"]) == selected_identity
    return dict(selected), pd.DataFrame(candidate_rows)


def run_nested_cv(
    samples: pd.DataFrame,
    parameters: ExperimentParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = samples[list(RULE_MEASUREMENT_COLUMNS)].to_numpy(dtype=float)
    labels = samples["y_true"].to_numpy(dtype=int)
    groups = samples["group_id"].to_numpy(dtype=str)
    outer_splits = _validated_group_splits(
        features,
        labels,
        groups,
        n_splits=parameters.outer_folds,
        seed=parameters.outer_split_seed,
        context="outer nested CV",
    )
    assignments = samples[
        [
            "sample_id",
            "split",
            "filename",
            "source_path",
            "structure_sha256",
            "target_author_chain_id",
            "group_id",
            "y_true",
        ]
    ].copy()
    assignments["outer_fold"] = 0
    prediction_rows: list[dict[str, Any]] = []
    inner_rows: list[pd.DataFrame] = []
    choice_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for outer_fold, (train_indices, test_indices) in enumerate(outer_splits, start=1):
        assignments.loc[test_indices, "outer_fold"] = outer_fold
        chosen, candidates = tune_on_grouped_oof(
            features[train_indices],
            labels[train_indices],
            groups[train_indices],
            parameters=parameters,
            n_splits=parameters.inner_folds,
            split_seed=parameters.inner_split_seed,
            context=f"outer fold {outer_fold} inner selection",
        )
        candidates.insert(0, "outer_fold", outer_fold)
        inner_rows.append(candidates)
        choice_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_train_n": int(len(train_indices)),
                "outer_test_n": int(len(test_indices)),
                "outer_train_groups_n": int(len(set(groups[train_indices]))),
                "outer_test_groups_n": int(len(set(groups[test_indices]))),
                "objective": parameters.objective,
                **chosen,
            }
        )
        model = _build_model(float(chosen["C"]), parameters)
        _fit_model(
            model,
            features[train_indices],
            labels[train_indices],
            context=f"outer fold {outer_fold} final fit",
        )
        scores = _positive_scores(model, features[test_indices])
        predictions = (scores >= float(chosen["threshold"])).astype(int)
        fold_metrics = confusion_metrics(labels[test_indices], predictions)
        fold_metric_rows.append(
            {
                "outer_fold": outer_fold,
                "n": int(len(test_indices)),
                "C_selected_inside_outer_train": float(chosen["C"]),
                "threshold_selected_inside_outer_train": float(chosen["threshold"]),
                **fold_metrics,
            }
        )
        for local_position, sample_index in enumerate(test_indices):
            sample = samples.iloc[int(sample_index)]
            prediction_rows.append(
                {
                    "outer_fold": outer_fold,
                    "sample_id": sample["sample_id"],
                    "split": sample["split"],
                    "filename": sample["filename"],
                    "source_path": sample["source_path"],
                    "structure_sha256": sample["structure_sha256"],
                    "target_author_chain_id": sample["target_author_chain_id"],
                    "group_id": sample["group_id"],
                    "y_true": int(sample["y_true"]),
                    "model_score": float(scores[local_position]),
                    "threshold": float(chosen["threshold"]),
                    "y_pred": int(predictions[local_position]),
                    "C": float(chosen["C"]),
                }
            )
    if (assignments["outer_fold"] == 0).any():
        raise RuntimeError("Outer CV did not assign every sample to a test fold.")
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["outer_fold", "sample_id"], kind="stable"
    )
    if len(predictions) != len(samples) or predictions["sample_id"].duplicated().any():
        raise RuntimeError("Outer CV must produce exactly one prediction per sample.")
    summary = {
        "scope": "pooled_outer_test_predictions",
        "interpretation": (
            "Nested-CV performance estimate; every score was produced by a model whose "
            "hyperparameter and threshold selection excluded that outer test fold."
        ),
        "observation_unit": "one manifest-selected target chain per structure file",
        "n": int(len(predictions)),
        "groups_n": int(samples["group_id"].nunique()),
        **confusion_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["y_pred"].to_numpy(dtype=int),
        ),
    }
    return (
        assignments.sort_values(["outer_fold", "sample_id"], kind="stable"),
        pd.concat(inner_rows, ignore_index=True),
        pd.DataFrame(choice_rows),
        pd.DataFrame(fold_metric_rows),
        {"summary": summary, "predictions": predictions},
    )


def fit_deployment_model(
    samples: pd.DataFrame,
    parameters: ExperimentParameters,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    features = samples[list(RULE_MEASUREMENT_COLUMNS)].to_numpy(dtype=float)
    labels = samples["y_true"].to_numpy(dtype=int)
    groups = samples["group_id"].to_numpy(dtype=str)
    selected, candidates = tune_on_grouped_oof(
        features,
        labels,
        groups,
        parameters=parameters,
        n_splits=parameters.deployment_folds,
        split_seed=parameters.deployment_split_seed,
        context="full-data deployment selection",
    )
    model = _build_model(float(selected["C"]), parameters)
    _fit_model(model, features, labels, context="full-data deployment fit")
    scaler = model.named_steps["scale"]
    estimator = model.named_steps["model"]
    estimator_parameters = estimator.get_params(deep=False)
    choice = {
        "interpretation": (
            "Full-data grouped OOF tuning used only to choose the deployable model. Its "
            "selection metrics are not an unbiased performance estimate."
        ),
        "objective": parameters.objective,
        "selection_folds": parameters.deployment_folds,
        "selection_split_seed": parameters.deployment_split_seed,
        **selected,
    }
    model_document = {
        "schema_version": 1,
        "algorithm": "sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression)",
        "score_semantics": (
            "positive-class predict_proba score under balanced class weights; not asserted "
            "to be prevalence-calibrated"
        ),
        "trained_on_all_labeled_samples": True,
        "performance_estimate_source": "outer_summary_metrics.json, not deployment tuning",
        "feature_columns": list(RULE_MEASUREMENT_COLUMNS),
        "threshold": float(selected["threshold"]),
        "pipeline_parameters": {
            "scale": {
                "with_mean": parameters.scaler_with_mean,
                "with_std": parameters.scaler_with_std,
            },
            "model": {
                "C": float(selected["C"]),
                "penalty": parameters.penalty,
                "l1_ratio": parameters.l1_ratio,
                "dual": parameters.dual,
                "solver": parameters.solver,
                "class_weight": parameters.class_weight,
                "fit_intercept": parameters.fit_intercept,
                "intercept_scaling": parameters.intercept_scaling,
                "max_iter": parameters.max_iter,
                "tol": parameters.tolerance,
                "random_state": parameters.model_seed,
            },
        },
        "resolved_sklearn_estimator_parameters": {
            key: value
            for key, value in estimator_parameters.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "fitted_parameters": {
            "classes": [int(value) for value in estimator.classes_],
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "scaler_variance": [float(value) for value in scaler.var_],
            "coefficient": [[float(value) for value in row] for row in estimator.coef_],
            "intercept": [float(value) for value in estimator.intercept_],
            "iterations": [int(value) for value in estimator.n_iter_],
        },
    }
    return candidates, choice, model_document


def _atomic_write_csv(path: Path, dataframe: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            dataframe.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    return path


def _ensure_fresh_output_directory(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ExperimentInputError(f"Output path is not a directory: {output_dir}")
    existing = (
        sorted(output_dir.iterdir(), key=lambda path: path.name) if output_dir.exists() else []
    )
    if existing:
        raise ExperimentInputError(
            "Experiment output directory must be empty; refusing to mix or overwrite artifacts: "
            + ", ".join(str(path) for path in existing[:10])
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _artifact_states(output_dir: Path) -> dict[str, dict[str, Any]]:
    states = _existing_artifact_states(output_dir)
    expected = set(OUTPUT_FILENAMES) - {"run_manifest.json"}
    missing = sorted(expected - set(states))
    if missing:
        raise RuntimeError(f"Experiment output artifact(s) missing: {', '.join(missing)}")
    return states


def _existing_artifact_states(output_dir: Path) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for filename in OUTPUT_FILENAMES:
        if filename == "run_manifest.json":
            continue
        path = output_dir / filename
        if not path.is_file():
            continue
        states[filename] = {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return states


def run_experiment(
    inputs: InputSpecification,
    parameters: ExperimentParameters,
    *,
    output_dir: Path,
) -> Path:
    parameters.validate()
    script_path = Path(__file__).resolve()
    script_digest = file_sha256(script_path)
    samples, input_provenance = load_experiment_samples(inputs)
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "run_manifest.json"
    started_at_utc = _utc_now()
    dependencies = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cooper-beta-source": source_package_version,
        "cooper-beta-installed-distribution": _distribution_version("cooper-beta"),
        "numpy": _distribution_version("numpy"),
        "pandas": _distribution_version("pandas"),
        "scikit-learn": _distribution_version("scikit-learn"),
        "scipy": _distribution_version("scipy"),
        "threadpoolctl": _distribution_version("threadpoolctl"),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "phase": "initialization",
        "started_at_utc": started_at_utc,
        "generated_at_utc": started_at_utc,
        "script": {
            "path": str(script_path),
            "sha256": script_digest,
        },
        "dependencies": dependencies,
        "runtime": {
            "native_thread_limit": parameters.native_threads,
            "effective_threadpools_during_fit": None,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "parameters": asdict(parameters),
        "input_contract": {
            "detector_csv_columns": list(DEFAULT_RESULT_COLUMNS),
            "label_manifest_columns": list(LABEL_MANIFEST_COLUMNS),
            "feature_columns": list(RULE_MEASUREMENT_COLUMNS),
            "observation_unit": "one manifest-selected target chain per structure file",
            "error_policy": "fail_on_any_detector_ERROR_row",
            "missing_or_nonfinite_feature_policy": "fail_without_imputation",
            "undefined_metric_policy": "serialize_null_and_mark_undefined",
            "partner_chain_policy": "ignore_unlabeled_partner_chains",
            "input_integrity_policy": (
                "verify detector CSV, detector manifest, resolved structure files, config "
                "partition hashes, native producer identity hash, artifact binding, and "
                "positive/negative path/content disjointness"
            ),
        },
        "scientific_design": {
            "algorithm_selection": "fixed LogisticRegression; no import-dependent candidates",
            "preprocessing": "StandardScaler fitted within every training partition",
            "grouping": inputs.grouping_method.strip(),
            "nested_estimation": (
                "outer test folds are untouched by inner C and threshold selection"
            ),
            "selection_rule": (
                f"maximize defined {parameters.objective}; then defined balanced_accuracy, "
                "defined f1, smaller C, threshold closest to 0.5, smaller threshold"
            ),
            "outer_performance_artifact": "outer_summary_metrics.json",
            "deployment_selection": (
                "separate full-data grouped OOF tuning; not a performance estimate"
            ),
            "score_calibration_claim": "none",
        },
        "inputs": input_provenance,
        "sample_counts": {
            "total": int(len(samples)),
            "positive": int((samples["y_true"] == 1).sum()),
            "negative": int((samples["y_true"] == 0).sum()),
            "groups": int(samples["group_id"].nunique()),
        },
        "outputs": {},
    }
    _ensure_fresh_output_directory(output_dir)
    atomic_write_json(manifest_path, manifest, indent=2)

    try:
        with threadpool_limits(limits=parameters.native_threads):
            manifest["runtime"]["effective_threadpools_during_fit"] = threadpool_info()
            manifest["phase"] = "nested_cross_validation"
            manifest["updated_at_utc"] = _utc_now()
            atomic_write_json(manifest_path, manifest, indent=2)
            assignments, inner_candidates, inner_choices, outer_fold_metrics, outer = run_nested_cv(
                samples, parameters
            )

            manifest["phase"] = "deployment_model_selection"
            manifest["updated_at_utc"] = _utc_now()
            atomic_write_json(manifest_path, manifest, indent=2)
            deployment_candidates, deployment_choice, deployment_model = fit_deployment_model(
                samples, parameters
            )

        manifest["phase"] = "writing_artifacts"
        manifest["updated_at_utc"] = _utc_now()
        atomic_write_json(manifest_path, manifest, indent=2)
        _atomic_write_csv(output_dir / "outer_fold_assignments.csv", assignments)
        _atomic_write_csv(output_dir / "inner_candidate_metrics.csv", inner_candidates)
        _atomic_write_csv(output_dir / "inner_choices.csv", inner_choices)
        _atomic_write_csv(output_dir / "outer_predictions.csv", outer["predictions"])
        _atomic_write_csv(output_dir / "outer_fold_metrics.csv", outer_fold_metrics)
        atomic_write_json(output_dir / "outer_summary_metrics.json", outer["summary"], indent=2)
        _atomic_write_csv(
            output_dir / "deployment_inner_candidate_metrics.csv", deployment_candidates
        )
        atomic_write_json(output_dir / "deployment_choice.json", deployment_choice, indent=2)
        atomic_write_json(output_dir / "deployment_model.json", deployment_model, indent=2)

        manifest["phase"] = "artifact_verification"
        manifest["updated_at_utc"] = _utc_now()
        manifest["outputs"] = _artifact_states(output_dir)
        atomic_write_json(manifest_path, manifest, indent=2)
        if file_sha256(script_path) != script_digest:
            raise RuntimeError("Experiment script changed while the run was in progress.")

        completed_at_utc = _utc_now()
        manifest["status"] = "complete"
        manifest["phase"] = "complete"
        manifest["completed_at_utc"] = completed_at_utc
        manifest["generated_at_utc"] = completed_at_utc
        manifest["updated_at_utc"] = completed_at_utc
        atomic_write_json(manifest_path, manifest, indent=2)
        return manifest_path
    except Exception as exc:
        failed_at_utc = _utc_now()
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = failed_at_utc
        manifest["updated_at_utc"] = failed_at_utc
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["outputs"] = _existing_artifact_states(output_dir)
        atomic_write_json(manifest_path, manifest, indent=2)
        raise


def _parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/nested_grouped_decision_experiment.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Estimate a fixed logistic Cooper-Beta decision model with nested, stratified, "
            "homology-grouped cross-validation. Each structure contributes the single target "
            "chain selected by its label manifest."
        ),
        epilog=(
            "Output: fold assignments, candidate metrics, predictions, the deployment model, and "
            "run_manifest.json in an initially empty directory. The outer folds estimate "
            "performance. Inner folds choose C and the probability threshold using only "
            "outer-training samples. A separate grouped fit produces the deployment model. "
            "Invalid input contracts exit with status 2; unexpected fitting failures exit nonzero."
        ),
    )
    parser.add_argument(
        "--positive-csv",
        type=Path,
        required=True,
        metavar="CSV",
        help="Cooper-Beta results CSV for structures labeled positive.",
    )
    parser.add_argument(
        "--positive-detector-manifest",
        type=Path,
        required=True,
        metavar="JSON",
        help="Completed Cooper-Beta output manifest paired with --positive-csv.",
    )
    parser.add_argument(
        "--positive-label-manifest",
        type=Path,
        required=True,
        metavar="CSV",
        help=(
            "Positive selection CSV with ordered columns filename, source_path, "
            "target_author_chain_id, group_id and one selected chain per structure."
        ),
    )
    parser.add_argument(
        "--negative-csv",
        type=Path,
        required=True,
        metavar="CSV",
        help="Cooper-Beta results CSV for structures labeled negative.",
    )
    parser.add_argument(
        "--negative-detector-manifest",
        type=Path,
        required=True,
        metavar="JSON",
        help="Completed Cooper-Beta output manifest paired with --negative-csv.",
    )
    parser.add_argument(
        "--negative-label-manifest",
        type=Path,
        required=True,
        metavar="CSV",
        help=(
            "Negative selection CSV with ordered columns filename, source_path, "
            "target_author_chain_id, group_id and one selected chain per structure."
        ),
    )
    parser.add_argument(
        "--grouping-method",
        required=True,
        metavar="DESCRIPTION",
        help="Homology or family clustering method and parameters used to assign group_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIRECTORY",
        help="Empty directory to create or use for all experiment artifacts.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("logistic_regression",),
        default="logistic_regression",
        help="Classifier family; the experiment currently defines logistic regression.",
    )
    parser.add_argument(
        "--c-values",
        nargs="+",
        type=_parse_positive_float,
        default=[0.1, 1.0, 10.0],
        metavar="C",
        help="Candidate inverse L2-regularization strengths evaluated in each inner search.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
        metavar="PROBABILITY",
        help="Inclusive lower bound of the evenly spaced decision-threshold grid.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.95,
        metavar="PROBABILITY",
        help="Inclusive upper bound of the evenly spaced decision-threshold grid.",
    )
    parser.add_argument(
        "--threshold-count",
        type=int,
        default=91,
        metavar="N",
        help="Number of probability thresholds in the grid, including both bounds.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
        metavar="N",
        help="Grouped folds used for the outer performance estimate.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
        metavar="N",
        help="Grouped folds used to tune each outer training partition.",
    )
    parser.add_argument(
        "--deployment-folds",
        type=int,
        default=5,
        metavar="N",
        help="Grouped folds used to tune the model fitted to the complete dataset.",
    )
    parser.add_argument(
        "--objective",
        choices=VALID_OBJECTIVES,
        default="mcc",
        help="Primary inner-validation metric used to choose C and the threshold.",
    )
    parser.add_argument(
        "--outer-split-seed",
        type=int,
        default=13,
        metavar="SEED",
        help="Random seed for outer grouped fold assignment.",
    )
    parser.add_argument(
        "--inner-split-seed",
        type=int,
        default=1313,
        metavar="SEED",
        help="Base random seed for inner searches within outer folds.",
    )
    parser.add_argument(
        "--deployment-split-seed",
        type=int,
        default=131313,
        metavar="SEED",
        help="Random seed for full-dataset deployment tuning folds.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=17,
        metavar="SEED",
        help="Random seed passed to every logistic-regression fit.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        metavar="N",
        help="Maximum liblinear iterations per fit.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        metavar="VALUE",
        help="Positive liblinear stopping tolerance.",
    )
    parser.add_argument(
        "--native-threads",
        type=int,
        default=1,
        metavar="N",
        help="Native numerical-library thread limit during the experiment.",
    )
    return parser


def _parameters_from_args(args: argparse.Namespace) -> ExperimentParameters:
    return ExperimentParameters(
        algorithm=args.algorithm,
        c_values=tuple(float(value) for value in args.c_values),
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        threshold_count=int(args.threshold_count),
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        deployment_folds=int(args.deployment_folds),
        objective=str(args.objective),
        outer_split_seed=int(args.outer_split_seed),
        inner_split_seed=int(args.inner_split_seed),
        deployment_split_seed=int(args.deployment_split_seed),
        model_seed=int(args.model_seed),
        max_iter=int(args.max_iter),
        tolerance=float(args.tolerance),
        class_weight="balanced",
        solver="liblinear",
        penalty="l2",
        l1_ratio=0.0,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1.0,
        scaler_with_mean=True,
        scaler_with_std=True,
        native_threads=int(args.native_threads),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = InputSpecification(
        positive_csv=args.positive_csv,
        positive_detector_manifest=args.positive_detector_manifest,
        positive_label_manifest=args.positive_label_manifest,
        negative_csv=args.negative_csv,
        negative_detector_manifest=args.negative_detector_manifest,
        negative_label_manifest=args.negative_label_manifest,
        grouping_method=args.grouping_method,
    )
    try:
        manifest_path = run_experiment(
            inputs,
            _parameters_from_args(args),
            output_dir=args.output_dir,
        )
    except (ExperimentInputError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Experiment complete. Strict run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
