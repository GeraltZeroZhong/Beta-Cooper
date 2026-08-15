from __future__ import annotations

import csv
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast

from cooper_beta.integrity import (
    atomic_write_json,
    canonical_json_sha256,
    freeze_input_identity,
)

METRIC_LEVELS = frozenset({"file", "chain", "both"})
TARGET_MANIFEST_FIELDS = ("relative_path", "author_chain_id")
PREDICTION_RESULTS = frozenset({"BARREL", "NON_BARREL"})
FILE_FIELDS = [
    "split",
    "y_true",
    "relative_path",
    "filename",
    "source_file",
    "decision_score_max",
    "pred_barrel_any",
    "chains_n",
]
SUMMARY_FIELDS = [
    "level",
    "n_used",
    "TP",
    "FP",
    "TN",
    "FN",
    "recall",
    "recall_defined",
    "precision",
    "precision_defined",
    "f1",
    "f1_defined",
    "specificity",
    "specificity_defined",
    "accuracy",
    "accuracy_defined",
    "balanced_accuracy",
    "balanced_accuracy_defined",
    "mcc",
    "mcc_defined",
]
MANIFEST_SCHEMA_VERSION = 1
_RUN_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class FileState(TypedDict):
    path: str
    size: int
    sha256: str


class InventoryEntry(FileState):
    relative_path: str


class ArtifactEntry(TypedDict):
    relative_path: str
    size: int
    sha256: str


class CheckoutState(TypedDict):
    path: str
    inventory: list[InventoryEntry]
    inventory_sha256: str
    excluded_path_components: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_run_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def fresh_run_directory(output_root: Path, *, baseline: str, tag: str) -> Path:
    normalized_tag = str(tag).strip()
    if not _RUN_TAG.fullmatch(normalized_tag):
        raise ValueError(
            "tag must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'."
        )
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{baseline}_{normalized_tag}_{utc_run_token()}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def _require_finite(value: object, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite value.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite(item, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, context=f"{context}[{index}]")


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> Path:
    materialized = [dict(row) for row in rows]
    _require_finite(materialized, context=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)
    return path


def file_state(path: Path) -> FileState:
    identity = freeze_input_identity(path)
    return {
        "path": identity.path,
        "size": identity.size,
        "sha256": identity.sha256,
    }


def files_inventory(files: Sequence[Path], *, root: Path) -> list[InventoryEntry]:
    resolved_root = root.expanduser().resolve()
    inventory: list[InventoryEntry] = []
    for path in sorted(path.expanduser().resolve() for path in files):
        try:
            relative_path = path.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Input file {path} is outside declared root {resolved_root}."
            ) from exc
        state = file_state(path)
        inventory.append(
            {
                "relative_path": relative_path,
                "path": state["path"],
                "size": state["size"],
                "sha256": state["sha256"],
            }
        )
    if not inventory:
        raise ValueError(f"No supported structure files found in {resolved_root}.")
    return inventory


def validate_disjoint_labeled_inventories(
    positive: Sequence[Mapping[str, object]],
    negative: Sequence[Mapping[str, object]],
) -> None:
    positive_paths = {str(item["path"]) for item in positive}
    negative_paths = {str(item["path"]) for item in negative}
    shared_paths = sorted(positive_paths & negative_paths)
    positive_hashes = {str(item["sha256"]) for item in positive}
    negative_hashes = {str(item["sha256"]) for item in negative}
    shared_hashes = sorted(positive_hashes & negative_hashes)
    if shared_paths or shared_hashes:
        raise ValueError(
            "Positive and negative structure inventories must be disjoint by resolved path and "
            f"content hash; shared_paths={shared_paths[:10]!r}, shared_hashes={shared_hashes[:10]!r}."
        )


def artifact_inventory(run_dir: Path, *, manifest_path: Path) -> list[ArtifactEntry]:
    resolved_run_dir = run_dir.resolve()
    resolved_manifest = manifest_path.resolve()
    artifacts: list[ArtifactEntry] = []
    for path in sorted(resolved_run_dir.rglob("*")):
        if not path.is_file() or path.resolve() == resolved_manifest:
            continue
        state = file_state(path)
        artifacts.append(
            {
                "relative_path": path.relative_to(resolved_run_dir).as_posix(),
                "size": state["size"],
                "sha256": state["sha256"],
            }
        )
    return artifacts


def executable_identity(command: str | Path) -> FileState:
    raw = os.fspath(command)
    resolved_command = shutil.which(raw)
    if resolved_command is None:
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            resolved_command = str(candidate)
    if resolved_command is None:
        raise FileNotFoundError(f"External executable not found: {raw}")
    return file_state(Path(resolved_command))


def checkout_identity(root: Path) -> CheckoutState:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(str(resolved))
    files = [
        path
        for path in resolved.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(resolved).parts
    ]
    inventory = files_inventory(files, root=resolved)
    return {
        "path": str(resolved),
        "inventory": inventory,
        "inventory_sha256": canonical_json_sha256(inventory),
        "excluded_path_components": [".git"],
    }


def initialize_manifest(
    manifest_path: Path,
    *,
    baseline: str,
    script_path: Path,
    supplied_parameters: Mapping[str, object],
) -> dict[str, object]:
    def json_safe(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return {"non_finite_float_repr": repr(value)}
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    document: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "baseline": baseline,
        "status": "running",
        "phase": "initialization",
        "started_at_utc": utc_now(),
        "script": file_state(script_path),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "parameters_supplied": json_safe(supplied_parameters),
        "parameters": None,
        "inputs": None,
        "external_software": None,
        "metric_sampling": None,
        "outputs": {"artifacts": [], "artifacts_sha256": canonical_json_sha256([])},
    }
    atomic_write_json(manifest_path, document)
    return document


def update_running_manifest(
    manifest_path: Path,
    document: dict[str, object],
    *,
    phase: str,
) -> None:
    document["phase"] = phase
    atomic_write_json(manifest_path, document)


def fail_manifest(
    manifest_path: Path,
    document: dict[str, object],
    *,
    phase: str,
    error: Exception,
) -> None:
    artifacts = artifact_inventory(manifest_path.parent, manifest_path=manifest_path)
    document.update(
        {
            "status": "failed",
            "phase": phase,
            "failed_at_utc": utc_now(),
            "error": {"type": type(error).__name__, "message": str(error)},
            "outputs": {
                "artifacts": artifacts,
                "artifacts_sha256": canonical_json_sha256(artifacts),
            },
        }
    )
    atomic_write_json(manifest_path, document)


def complete_manifest(manifest_path: Path, document: dict[str, object]) -> None:
    artifacts = artifact_inventory(manifest_path.parent, manifest_path=manifest_path)
    document.update(
        {
            "status": "complete",
            "phase": "complete",
            "completed_at_utc": utc_now(),
            "outputs": {
                "artifacts": artifacts,
                "artifacts_sha256": canonical_json_sha256(artifacts),
            },
        }
    )
    atomic_write_json(manifest_path, document)


def validate_metric_contract(
    metric_level: str,
    positive_manifest: Path | None,
    negative_manifest: Path | None,
) -> None:
    if metric_level not in METRIC_LEVELS:
        raise ValueError(f"metric_level must be one of {sorted(METRIC_LEVELS)!r}.")
    if (positive_manifest is None) != (negative_manifest is None):
        raise ValueError(
            "positive_target_manifest and negative_target_manifest must either both be "
            "provided or both be omitted."
        )
    if metric_level in {"chain", "both"} and positive_manifest is None:
        raise ValueError(
            "Chain metrics require paired positive and negative target-chain manifests."
        )


def read_target_manifest(
    path: Path,
    *,
    split_root: Path,
    structure_files: Sequence[Path],
) -> dict[str, str]:
    resolved_manifest = path.expanduser().resolve()
    with resolved_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TARGET_MANIFEST_FIELDS:
            raise ValueError(
                f"Target-chain manifest must have exactly these columns in order: "
                f"{','.join(TARGET_MANIFEST_FIELDS)}."
            )
        rows = list(reader)

    root = split_root.expanduser().resolve()
    expected = {
        structure.expanduser().resolve().relative_to(root).as_posix()
        for structure in structure_files
    }
    targets: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        raw_relative = str(row.get("relative_path", "")).strip().replace("\\", "/")
        relative = Path(raw_relative)
        author_chain_id = str(row.get("author_chain_id", "")).strip()
        if (
            not raw_relative
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in {".", ""}
        ):
            raise ValueError(
                f"Target-chain manifest row {row_number} has an invalid relative_path."
            )
        normalized = relative.as_posix()
        if normalized in targets:
            raise ValueError(
                f"Target-chain manifest must contain exactly one target per file; duplicate: "
                f"{normalized!r}."
            )
        if not author_chain_id:
            raise ValueError(f"Target-chain manifest row {row_number} has a blank author_chain_id.")
        targets[normalized] = author_chain_id
    if set(targets) != expected:
        missing = sorted(expected - set(targets))
        extra = sorted(set(targets) - expected)
        raise ValueError(
            "Target-chain manifest must match the structure inventory exactly; "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}."
        )
    return targets


def validate_prediction_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("External baseline produced no chain predictions.")
    sample_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(
                f"External baseline returned a blank or duplicate sample_id: {sample_id!r}."
            )
        sample_ids.add(sample_id)
        result = str(row.get("result", ""))
        if result not in PREDICTION_RESULTS:
            raise ValueError(
                f"External baseline sample {sample_id!r} returned invalid/ERROR result {result!r}."
            )
        score = row.get("decision_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(
                f"External baseline sample {sample_id!r} returned a non-finite decision score."
            )
        if bool(row.get("is_error")):
            raise ValueError(f"External baseline sample {sample_id!r} has ERROR status.")
        if bool(row.get("is_filtered_out")):
            raise ValueError(
                f"External baseline sample {sample_id!r} has FILTERED_OUT status under the "
                "fixed strict policy."
            )
        if bool(row.get("is_skip")):
            raise ValueError(
                f"External baseline sample {sample_id!r} has SKIP status under the fixed "
                "strict policy."
            )


def validate_generated_source_coverage(
    source_paths: Sequence[str],
    *,
    structure_files: Sequence[Path],
    context: str,
) -> None:
    generated = {Path(path).expanduser().resolve() for path in source_paths}
    expected = {path.expanduser().resolve() for path in structure_files}
    if generated != expected:
        missing = sorted(str(path) for path in expected - generated)
        extra = sorted(str(path) for path in generated - expected)
        raise ValueError(
            f"Every {context} structure must yield at least one eligible chain; "
            f"missing={missing[:10]!r}, unexpected={extra[:10]!r}."
        )


def file_rows_from_chain_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    split: str,
    split_root: Path,
    structure_files: Sequence[Path],
) -> list[dict[str, object]]:
    root = split_root.expanduser().resolve()
    grouped: dict[Path, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[Path(str(row["source_file"])).expanduser().resolve()].append(row)
    expected = {path.expanduser().resolve() for path in structure_files}
    if set(grouped) != expected:
        missing = sorted(str(path) for path in expected - set(grouped))
        extra = sorted(str(path) for path in set(grouped) - expected)
        raise ValueError(
            "Every input structure must yield at least one successful chain prediction; "
            f"missing={missing[:10]!r}, unexpected={extra[:10]!r}."
        )
    y_true = 1 if split == "positive" else 0
    output: list[dict[str, object]] = []
    for source in sorted(grouped):
        group = grouped[source]
        scores = [float(cast(int | float, row["decision_score"])) for row in group]
        output.append(
            {
                "split": split,
                "y_true": y_true,
                "relative_path": source.relative_to(root).as_posix(),
                "filename": source.name,
                "source_file": str(source),
                "decision_score_max": max(scores),
                "pred_barrel_any": any(str(row["result"]) == "BARREL" for row in group),
                "chains_n": len(group),
            }
        )
    return output


def apply_target_chain_labels(
    rows: Sequence[Mapping[str, object]],
    *,
    split: str,
    split_root: Path,
    targets: Mapping[str, str] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root = split_root.expanduser().resolve()
    predictions: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    matched: set[str] = set()
    for original in rows:
        row = dict(original)
        source = Path(str(row["source_file"])).expanduser().resolve()
        relative_path = source.relative_to(root).as_posix()
        target_author_chain_id = targets.get(relative_path) if targets is not None else None
        is_target = (
            target_author_chain_id is not None
            and str(row["author_chain_id"]) == target_author_chain_id
        )
        row["relative_path"] = relative_path
        row["is_target_author_chain"] = is_target
        row["y_true"] = (1 if split == "positive" else 0) if is_target else ""
        row["split"] = split if is_target else ""
        row["use_for_metrics"] = is_target
        predictions.append(row)
        if is_target:
            if relative_path in matched:
                raise ValueError(
                    f"Multiple generated predictions matched target chain for {relative_path!r}."
                )
            matched.add(relative_path)
            target_rows.append(dict(row))
    if targets is not None and matched != set(targets):
        missing = sorted(set(targets) - matched)
        raise ValueError(
            "Every manifest target chain must yield exactly one successful prediction; "
            f"missing={missing[:10]!r}."
        )
    return predictions, target_rows


def _division(numerator: float, denominator: float) -> tuple[float | None, bool]:
    return (numerator / denominator, True) if denominator else (None, False)


def metrics(rows: Sequence[Mapping[str, object]], *, prediction_field: str) -> dict[str, object]:
    tp = fp = tn = fn = 0
    for row in rows:
        if (
            bool(row.get("is_error"))
            or bool(row.get("is_filtered_out"))
            or bool(row.get("is_skip"))
        ):
            raise ValueError(
                "ERROR, FILTERED_OUT, and SKIP observations are forbidden in strict baseline "
                "metrics."
            )
        y_true = row.get("y_true")
        if y_true not in {0, 1}:
            raise ValueError("Every metric observation must have an explicit binary truth label.")
        prediction = row.get(prediction_field)
        if not isinstance(prediction, bool):
            raise ValueError("Every metric observation must have an explicit boolean prediction.")
        if y_true == 1 and prediction:
            tp += 1
        elif y_true == 0 and prediction:
            fp += 1
        elif y_true == 0:
            tn += 1
        else:
            fn += 1

    recall, recall_defined = _division(tp, tp + fn)
    precision, precision_defined = _division(tp, tp + fp)
    specificity, specificity_defined = _division(tn, tn + fp)
    accuracy, accuracy_defined = _division(tp + tn, tp + fp + tn + fn)
    f1, f1_defined = _division(2 * tp, (2 * tp) + fp + fn)
    if recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2
        balanced_accuracy_defined = True
    else:
        balanced_accuracy = None
        balanced_accuracy_defined = False
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc, mcc_defined = _division((tp * tn) - (fp * fn), denominator)
    return {
        "n_used": tp + fp + tn + fn,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "recall": recall,
        "recall_defined": recall_defined,
        "precision": precision,
        "precision_defined": precision_defined,
        "f1": f1,
        "f1_defined": f1_defined,
        "specificity": specificity,
        "specificity_defined": specificity_defined,
        "accuracy": accuracy,
        "accuracy_defined": accuracy_defined,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_defined": balanced_accuracy_defined,
        "mcc": mcc,
        "mcc_defined": mcc_defined,
    }


def summary_rows(
    *,
    metric_level: str,
    file_rows: Sequence[Mapping[str, object]],
    target_chain_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    if metric_level in {"file", "both"}:
        output.append({"level": "file", **metrics(file_rows, prediction_field="pred_barrel_any")})
    if metric_level in {"chain", "both"}:
        output.append(
            {"level": "chain", **metrics(target_chain_rows, prediction_field="pred_barrel")}
        )
    return output


def summary_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(SUMMARY_FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in SUMMARY_FIELDS) + " |")
    return "\n".join(lines) + "\n"
