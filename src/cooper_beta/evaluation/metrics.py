from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in minimal installations
    pd = None

from cooper_beta.constants import (
    DEFAULT_RESULT_COLUMNS,
    RESULT_BARREL,
    RESULT_ERROR,
    RESULT_STAGE_PREPARATION,
)
from cooper_beta.models import DetectionResult

CHAIN_GROUND_TRUTH_SCHEMA = "manifest_target_positive_vs_manifest_target_negative"
CHAIN_GROUND_TRUTH_UNAVAILABLE = "unavailable_without_paired_target_manifests"
FILE_GROUND_TRUTH_SCHEMA = "directory_file_label"
FILE_PREDICTION_SCHEMA = "barrel_if_any_chain_is_barrel"
TRUTH_MANIFEST_COLUMNS = (
    "filename",
    "source_path",
    "structure_sha256",
    "target_author_chain_id",
)


@dataclass(frozen=True, slots=True)
class CandidateTruth:
    """One immutable target-chain identity from a canonical truth manifest."""

    filename: str
    source_path: str
    structure_sha256: str
    target_author_chain_id: str

    def __post_init__(self) -> None:
        filename = str(self.filename).strip()
        source_value = str(self.source_path).strip()
        structure_sha256 = str(self.structure_sha256).strip().lower()
        target_author_chain_id = str(self.target_author_chain_id).strip()
        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
        ):
            raise ValueError("Truth-manifest filename must be a non-empty basename.")
        if not source_value:
            raise ValueError("Truth-manifest source_path must not be empty.")
        source_path = str(Path(source_value).expanduser().resolve())
        if Path(source_path).name != filename:
            raise ValueError("Truth-manifest filename must equal the basename of its source_path.")
        if len(structure_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in structure_sha256
        ):
            raise ValueError("Truth-manifest structure_sha256 must be a SHA-256 digest.")
        if not target_author_chain_id:
            raise ValueError("Truth-manifest target_author_chain_id must not be empty.")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "structure_sha256", structure_sha256)
        object.__setattr__(self, "target_author_chain_id", target_author_chain_id)

    def as_row(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "source_path": self.source_path,
            "structure_sha256": self.structure_sha256,
            "target_author_chain_id": self.target_author_chain_id,
        }


class MetricErrorPolicy(str, Enum):
    """How detector ERROR observations are handled by metric computation."""

    STRICT = "strict"
    EXCLUDE = "exclude"


class MetricInputError(ValueError):
    """Raised when evaluation inputs cannot support the requested metric."""


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required to compute evaluation metrics.")


def resolve_error_policy(policy: MetricErrorPolicy | str) -> MetricErrorPolicy:
    if isinstance(policy, MetricErrorPolicy):
        return policy
    try:
        return MetricErrorPolicy(str(policy).strip().lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in MetricErrorPolicy)
        raise ValueError(f"metric error policy must be one of: {choices}") from exc


def ensure_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Require the fixed public detector schema without imputing scientific data."""
    _require_pandas()
    if tuple(dataframe.columns) != DEFAULT_RESULT_COLUMNS:
        raise MetricInputError(
            "Detector results do not match the fixed public schema/order; "
            f"expected {list(DEFAULT_RESULT_COLUMNS)!r}, got {list(dataframe.columns)!r}."
        )
    validated = dataframe.copy()
    for row_index, row in enumerate(
        validated[list(DEFAULT_RESULT_COLUMNS)].to_dict(orient="records")
    ):
        try:
            DetectionResult.from_row(row)
        except (TypeError, ValueError) as exc:
            raise MetricInputError(
                f"Detector result row {row_index} violates the public schema: {exc}"
            ) from exc
    return validated


def _error_mask(dataframe: pd.DataFrame) -> pd.Series:
    if "result" not in dataframe.columns:
        raise MetricInputError("Metric input is missing required column: result")
    return dataframe["result"].fillna("").astype(str).str.strip().str.upper().eq(RESULT_ERROR)


def _coerce_boolean(series: pd.Series, *, label: str) -> pd.Series:
    normalized = series.map(
        {
            True: True,
            False: False,
            "true": True,
            "false": False,
            "True": True,
            "False": False,
            "TRUE": True,
            "FALSE": False,
            "1": True,
            "0": False,
        }
    )
    if normalized.isna().any():
        invalid = series.loc[normalized.isna()].astype(str).unique().tolist()
        preview = ", ".join(repr(value) for value in invalid[:5])
        raise MetricInputError(f"{label} contains non-boolean value(s): {preview}")
    return normalized.astype(bool)


def _coverage(used: int, total: int) -> float:
    return float(used / total) if total else 0.0


def _apply_row_outcome_policies(
    dataframe: pd.DataFrame,
    *,
    error_policy: MetricErrorPolicy,
    label: str,
) -> tuple[pd.DataFrame, int]:
    error_mask = _error_mask(dataframe)
    error_count = int(error_mask.sum())
    if error_count and error_policy is MetricErrorPolicy.STRICT:
        raise MetricInputError(
            f"{label} contains {error_count} detector ERROR observation(s); "
            "rerun with metric_error_policy='exclude' only if exclusion is scientifically justified."
        )
    excluded_mask = pd.Series(False, index=dataframe.index)
    if error_policy is MetricErrorPolicy.EXCLUDE:
        excluded_mask |= error_mask
    return dataframe.loc[~excluded_mask].copy(), error_count


def assign_manifest_candidate_truth(
    dataframe: pd.DataFrame,
    candidate_truth: Sequence[CandidateTruth],
    *,
    truth_value: int,
    split_name: str,
    ground_truth_schema: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate and annotate exactly one manifest-defined candidate per file.

    This is the sole implementation of manifest-to-detector matching used by
    both output annotation and metric computation. A candidate may be absent
    only when the detector emitted one file-level ``preparation`` ERROR row
    with an empty chain; a chain-specific/partner ERROR is never relabelled.
    """
    if truth_value not in {0, 1}:
        raise ValueError("truth_value must be 0 or 1")
    annotated = dataframe.copy()
    if not candidate_truth:
        raise MetricInputError(
            f"Chain-level metrics require a non-empty {split_name} candidate manifest."
        )
    normalized_manifest: dict[str, CandidateTruth] = {}
    seen_filenames: set[str] = set()
    for entry in candidate_truth:
        if not isinstance(entry, CandidateTruth):
            raise MetricInputError(
                f"{split_name.capitalize()} candidate truth must contain only CandidateTruth "
                "records from the canonical four-column schema."
            )
        if entry.source_path in normalized_manifest:
            raise MetricInputError(
                f"{split_name.capitalize()} manifest contains duplicate source_path: "
                f"{entry.source_path!r}."
            )
        if entry.filename in seen_filenames:
            raise MetricInputError(
                f"{split_name.capitalize()} manifest contains duplicate filename: "
                f"{entry.filename!r}."
            )
        normalized_manifest[entry.source_path] = entry
        seen_filenames.add(entry.filename)

    annotated["filename"] = annotated["filename"].astype(str).str.strip()
    annotated["source_path"] = annotated["source_path"].astype(str).str.strip()
    annotated["author_chain_id"] = annotated["author_chain_id"].astype(str).str.strip()
    if annotated["source_path"].eq("").any():
        raise MetricInputError(
            f"{split_name.capitalize()} detector results contain an empty source_path; "
            "canonical truth matching never falls back to filename."
        )
    resolved_source_paths = annotated["source_path"].map(
        lambda value: str(Path(value).expanduser().resolve())
    )
    annotated["source_path"] = resolved_source_paths
    observed_identities = pd.DataFrame(
        {"filename": annotated["filename"], "source_path": resolved_source_paths}
    ).drop_duplicates()
    inconsistent_sources = observed_identities.groupby("source_path")["filename"].nunique()
    inconsistent_sources = inconsistent_sources[inconsistent_sources.gt(1)]
    if not inconsistent_sources.empty:
        preview = ", ".join(str(value) for value in inconsistent_sources.index[:5])
        raise MetricInputError(
            f"{split_name.capitalize()} detector results map one source_path to multiple "
            f"filenames ({preview})."
        )
    ambiguous_filenames = observed_identities.groupby("filename")["source_path"].nunique()
    ambiguous_filenames = ambiguous_filenames[ambiguous_filenames.gt(1)]
    if not ambiguous_filenames.empty:
        preview = ", ".join(str(value) for value in ambiguous_filenames.index[:5])
        raise MetricInputError(
            f"{split_name.capitalize()} detector results map the same filename to multiple "
            f"source paths ({preview})."
        )

    observed_files = set(resolved_source_paths)
    manifest_files = set(normalized_manifest)
    missing_manifest = sorted(observed_files - manifest_files)
    missing_results = sorted(manifest_files - observed_files)
    if missing_manifest:
        raise MetricInputError(
            f"{split_name.capitalize()} result source_path(s) missing from manifest: "
            + ", ".join(missing_manifest[:5])
        )
    if missing_results:
        raise MetricInputError(
            f"{split_name.capitalize()} manifest source_path(s) missing from detector results: "
            + ", ".join(missing_results[:5])
        )

    candidate_role = f"{split_name}_candidate_chain"
    partner_role = f"{split_name}_partner_unlabeled"
    unavailable_role = f"{split_name}_candidate_unavailable_error"
    annotated["target_author_chain_id"] = resolved_source_paths.map(
        {source: entry.target_author_chain_id for source, entry in normalized_manifest.items()}
    )
    annotated["truth_structure_sha256"] = resolved_source_paths.map(
        {source: entry.structure_sha256 for source, entry in normalized_manifest.items()}
    )
    annotated["ground_truth_role"] = partner_role
    annotated["chain_ground_truth_schema"] = ground_truth_schema
    annotated["use_for_chain_metrics"] = False
    annotated["chain_y_true"] = pd.NA

    partner_rows = 0
    partner_error_rows = 0
    unavailable_error_files = 0
    result_values = annotated["result"].astype(str).str.strip().str.upper()
    result_stages = annotated["result_stage"].astype(str).str.strip().str.lower()
    for source_path, truth in sorted(normalized_manifest.items()):
        filename = truth.filename
        expected_chain = truth.target_author_chain_id
        file_mask = annotated["source_path"].eq(source_path)
        observed_filenames = set(annotated.loc[file_mask, "filename"])
        if observed_filenames != {filename}:
            raise MetricInputError(
                f"Detector filename for {source_path!r} does not match truth-manifest "
                f"filename {filename!r}."
            )
        chain_mask = file_mask & annotated["author_chain_id"].eq(expected_chain)
        match_count = int(chain_mask.sum())
        if match_count > 1:
            raise MetricInputError(
                f"Detector results contain duplicate candidate-chain rows for "
                f"{filename}:{expected_chain}."
            )
        if match_count == 1:
            target_mask = chain_mask
            role = candidate_role
        else:
            target_mask = (
                file_mask
                & result_values.eq(RESULT_ERROR)
                & annotated["author_chain_id"].eq("")
                & result_stages.eq(RESULT_STAGE_PREPARATION)
            )
            if int(file_mask.sum()) != 1 or int(target_mask.sum()) != 1:
                raise MetricInputError(
                    f"Candidate chain {expected_chain!r} for {filename!r} is absent. Only "
                    f"one file-level {RESULT_STAGE_PREPARATION} ERROR row with an empty "
                    "chain can represent an unavailable candidate; partner-chain ERROR "
                    "rows cannot be relabeled."
                )
            unavailable_error_files += 1
            role = unavailable_role
        annotated.loc[target_mask, "ground_truth_role"] = role
        annotated.loc[target_mask, "use_for_chain_metrics"] = True
        annotated.loc[target_mask, "chain_y_true"] = truth_value
        partner_mask = file_mask & ~target_mask
        partner_rows += int(partner_mask.sum())
        partner_error_rows += int((partner_mask & result_values.eq(RESULT_ERROR)).sum())

    return annotated, {
        "candidate_files_total": int(len(normalized_manifest)),
        "rows_total": int(len(annotated)),
        "partner_rows_unlabeled": partner_rows,
        "partner_error_rows_unlabeled": partner_error_rows,
        "candidate_unavailable_error_files": unavailable_error_files,
    }


def _candidate_rows(
    dataframe: pd.DataFrame,
    candidate_truth: Sequence[CandidateTruth],
    *,
    truth_value: int,
    split_name: str,
    ground_truth_schema: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    annotated, counts = assign_manifest_candidate_truth(
        dataframe,
        candidate_truth,
        truth_value=truth_value,
        split_name=split_name,
        ground_truth_schema=ground_truth_schema,
    )
    selected = annotated.loc[annotated["use_for_chain_metrics"]].copy()
    prefix = "positive" if truth_value == 1 else "negative"
    return selected, {f"{prefix}_{key}": value for key, value in counts.items()}


def compute_confusion_metrics(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> dict[str, bool | float | int | str | None]:
    raw_counts = (tp, fp, tn, fn)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_counts):
        raise MetricInputError("Confusion-matrix counts must be non-boolean integers.")
    tp, fp, tn, fn = (int(value) for value in raw_counts)
    if any(value < 0 for value in (tp, fp, tn, fn)):
        raise MetricInputError("Confusion-matrix counts must be non-negative integers.")
    if tp + fp + tn + fn == 0:
        raise MetricInputError("Confusion-matrix metrics require at least one observation.")

    def nullable_ratio(numerator: float, denominator: float) -> float | None:
        return (numerator / denominator) if denominator else None

    recall_defined = tp + fn > 0
    precision_defined = tp + fp > 0
    specificity_defined = tn + fp > 0
    recall = nullable_ratio(tp, tp + fn)
    precision = nullable_ratio(tp, tp + fp)
    specificity = nullable_ratio(tn, tn + fp)
    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    f1_denominator = 2 * tp + fp + fn
    f1_defined = f1_denominator > 0
    f1 = nullable_ratio(2 * tp, f1_denominator)
    balanced_accuracy_defined = recall_defined and specificity_defined
    balanced_accuracy = (
        0.5 * (recall + specificity) if recall is not None and specificity is not None else None
    )
    mcc_denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc_defined = mcc_denominator > 0
    mcc = nullable_ratio((tp * tn) - (fp * fn), mcc_denominator)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "accuracy_defined": True,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "recall_defined": recall_defined,
        "precision_defined": precision_defined,
        "specificity_defined": specificity_defined,
        "f1_defined": f1_defined,
        "balanced_accuracy_defined": balanced_accuracy_defined,
        "mcc_defined": mcc_defined,
        "undefined_metric_policy": "serialize_null_and_mark_undefined",
    }


def print_metrics(
    title: str,
    metrics: Mapping[str, bool | float | int | str | None],
) -> None:
    def formatted(name: str) -> str:
        defined_key = f"{name}_defined"
        if defined_key in metrics and not bool(metrics[defined_key]):
            return "undefined"
        value = metrics[name]
        if value is None:
            return "undefined"
        return f"{float(value):.4f}"

    if title:
        print(title)
    print("Confusion matrix:")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}")
    print(f"  FN={metrics['FN']}  TN={metrics['TN']}\n")
    print("Metrics:")
    print(f"  Recall      : {formatted('recall')}")
    print(f"  Precision   : {formatted('precision')}")
    print(f"  F1          : {formatted('f1')}")
    print(f"  Specificity : {formatted('specificity')}")
    print(f"  Accuracy    : {formatted('accuracy')}")
    print(f"  Bal. Acc.   : {formatted('balanced_accuracy')}")
    print(f"  MCC         : {formatted('mcc')}\n")


def compute_chain_metrics(
    positive_dataframe: pd.DataFrame,
    negative_dataframe: pd.DataFrame,
    *,
    positive_candidate_truth: Sequence[CandidateTruth] | None,
    negative_candidate_truth: Sequence[CandidateTruth] | None = None,
    error_policy: MetricErrorPolicy | str = MetricErrorPolicy.STRICT,
) -> tuple[
    dict[str, bool | float | int | str | None],
    dict[str, float | int | str],
]:
    """
    Compute chain-level metrics at manifest-defined chain sampling units.

    Partner chains are deliberately unlabeled. Positive and negative observations
    use the same manifest-selected, one-target-chain-per-file sampling unit.
    """
    if positive_candidate_truth is None:
        raise MetricInputError(
            "Chain-level metrics require positive and negative manifests containing "
            "filename, source_path, structure_sha256, and target_author_chain_id."
        )
    if negative_candidate_truth is None:
        raise MetricInputError(
            "Chain-level metrics require a negative manifest so positive and negative "
            "observations use the same target-chain sampling unit."
        )
    policy = resolve_error_policy(error_policy)
    positive = ensure_columns(positive_dataframe)
    negative_all = ensure_columns(negative_dataframe)
    positive_detector_errors = int(_error_mask(positive).sum())
    negative_detector_errors = int(_error_mask(negative_all).sum())
    if positive_detector_errors and policy is MetricErrorPolicy.STRICT:
        raise MetricInputError(
            f"Positive results contain {positive_detector_errors} detector ERROR row(s), "
            "including candidate or unlabeled partner rows; strict metrics require none."
        )
    if negative_detector_errors and policy is MetricErrorPolicy.STRICT:
        raise MetricInputError(
            f"Negative results contain {negative_detector_errors} detector ERROR row(s), "
            "including candidate or unlabeled partner rows; strict metrics require none."
        )
    ground_truth_schema = CHAIN_GROUND_TRUTH_SCHEMA
    positive_candidates, selection_extra = _candidate_rows(
        positive,
        positive_candidate_truth,
        truth_value=1,
        split_name="positive",
        ground_truth_schema=ground_truth_schema,
    )
    negative, negative_counts = _candidate_rows(
        negative_all,
        negative_candidate_truth,
        truth_value=0,
        split_name="negative",
        ground_truth_schema=ground_truth_schema,
    )
    if negative.empty:
        raise MetricInputError("Chain-level metrics require at least one negative-chain row.")

    positive_used, positive_errors = _apply_row_outcome_policies(
        positive_candidates,
        error_policy=policy,
        label="Positive candidate chains",
    )
    negative_used, negative_errors = _apply_row_outcome_policies(
        negative,
        error_policy=policy,
        label="Negative chains",
    )

    def predicted_barrel(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.upper().eq(RESULT_BARREL)

    true_positive_mask = predicted_barrel(positive_used["result"])
    false_positive_mask = predicted_barrel(negative_used["result"])
    tp = int(true_positive_mask.sum())
    fn = int((~true_positive_mask).sum())
    fp = int(false_positive_mask.sum())
    tn = int((~false_positive_mask).sum())

    positive_total = int(len(positive_candidates))
    negative_total = int(len(negative))
    used_total = int(len(positive_used) + len(negative_used))
    total = positive_total + negative_total
    metrics = compute_confusion_metrics(tp, fp, tn, fn)
    extra: dict[str, float | int | str] = {
        **selection_extra,
        **negative_counts,
        "ground_truth_schema": ground_truth_schema,
        "metric_error_policy": policy.value,
        "positive_candidate_error_n": int(positive_errors),
        "negative_chain_error_n": int(negative_errors),
        "n_positive_total": positive_total,
        "n_negative_total": negative_total,
        "n_positive_used": int(len(positive_used)),
        "n_negative_used": int(len(negative_used)),
        "positive_coverage": _coverage(len(positive_used), positive_total),
        "negative_coverage": _coverage(len(negative_used), negative_total),
        "overall_coverage": _coverage(used_total, total),
    }
    return metrics, extra


def compute_file_metrics(
    aggregated_dataframe: pd.DataFrame,
    *,
    error_policy: MetricErrorPolicy | str = MetricErrorPolicy.STRICT,
) -> tuple[
    dict[str, bool | float | int | str | None],
    dict[str, float | int | str],
]:
    """Compute file metrics with explicit ERROR handling."""
    _require_pandas()
    policy = resolve_error_policy(error_policy)
    required = {"split", "pred_barrel_any", "error_chains_n"}
    missing = sorted(required - set(aggregated_dataframe.columns))
    if missing:
        raise MetricInputError(
            f"File-level metric input is missing column(s): {', '.join(missing)}"
        )

    dataframe = aggregated_dataframe.copy()
    dataframe["split"] = dataframe["split"].fillna("").astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(dataframe["split"]) - {"positive", "negative"})
    if invalid_splits:
        preview = ", ".join(repr(value) for value in invalid_splits[:5])
        raise MetricInputError(f"File-level split contains invalid value(s): {preview}")
    if not dataframe["split"].eq("positive").any() or not dataframe["split"].eq("negative").any():
        raise MetricInputError(
            "File-level metrics require at least one positive and one negative file."
        )
    dataframe["pred_barrel_any"] = _coerce_boolean(
        dataframe["pred_barrel_any"],
        label="File-level pred_barrel_any",
    )
    error_counts = pd.to_numeric(dataframe["error_chains_n"], errors="coerce")
    invalid_error_counts = error_counts.isna() | error_counts.lt(0) | error_counts.mod(1).ne(0)
    if invalid_error_counts.any():
        raise MetricInputError("File-level error_chains_n must contain non-negative integers.")
    error_mask = error_counts.gt(0)

    error_files = int(error_mask.sum())
    if error_files and policy is MetricErrorPolicy.STRICT:
        raise MetricInputError(
            f"File-level input contains {error_files} file(s) with detector ERROR rows; "
            "rerun with metric_error_policy='exclude' only if exclusion is scientifically justified."
        )
    excluded_mask = pd.Series(False, index=dataframe.index)
    if policy is MetricErrorPolicy.EXCLUDE:
        excluded_mask |= error_mask
    used = dataframe.loc[~excluded_mask].copy()
    error_excluded = (
        error_mask
        if policy is MetricErrorPolicy.EXCLUDE
        else pd.Series(False, index=dataframe.index)
    )

    positives = used[used["split"] == "positive"].copy()
    negatives = used[used["split"] == "negative"].copy()
    all_positives = dataframe[dataframe["split"] == "positive"]
    all_negatives = dataframe[dataframe["split"] == "negative"]
    tp = int(positives["pred_barrel_any"].sum()) if len(positives) else 0
    fn = int(len(positives) - tp) if len(positives) else 0
    fp = int(negatives["pred_barrel_any"].sum()) if len(negatives) else 0
    tn = int(len(negatives) - fp) if len(negatives) else 0

    positive_errors = int(((dataframe["split"] == "positive") & error_mask).sum())
    negative_errors = int(((dataframe["split"] == "negative") & error_mask).sum())
    total = int(len(dataframe))
    metrics = compute_confusion_metrics(tp, fp, tn, fn)
    extra: dict[str, float | int | str] = {
        "ground_truth_schema": FILE_GROUND_TRUTH_SCHEMA,
        "prediction_schema": FILE_PREDICTION_SCHEMA,
        "metric_error_policy": policy.value,
        "n_positive_files_total": int(len(all_positives)),
        "n_negative_files_total": int(len(all_negatives)),
        "n_positive_files": int(len(positives)),
        "n_negative_files": int(len(negatives)),
        "positive_error_files": positive_errors,
        "negative_error_files": negative_errors,
        "excluded_positive_error_files": int(
            ((dataframe["split"] == "positive") & error_excluded).sum()
        ),
        "excluded_negative_error_files": int(
            ((dataframe["split"] == "negative") & error_excluded).sum()
        ),
        "positive_file_coverage": _coverage(len(positives), len(all_positives)),
        "negative_file_coverage": _coverage(len(negatives), len(all_negatives)),
        "overall_file_coverage": _coverage(len(used), total),
    }
    return metrics, extra
