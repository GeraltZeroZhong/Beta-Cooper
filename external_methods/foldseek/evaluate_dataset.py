from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cooper_beta.integrity import canonical_json_sha256
from external_methods.evaluation_common import (
    FILE_FIELDS,
    SUMMARY_FIELDS,
    apply_target_chain_labels,
    atomic_write_csv,
    atomic_write_text,
    complete_manifest,
    executable_identity,
    fail_manifest,
    file_rows_from_chain_rows,
    file_state,
    files_inventory,
    fresh_run_directory,
    initialize_manifest,
    read_target_manifest,
    summary_markdown,
    summary_rows,
    update_running_manifest,
    validate_disjoint_labeled_inventories,
    validate_generated_source_coverage,
    validate_metric_contract,
    validate_prediction_rows,
)
from external_methods.foldseek.runner import (
    BASELINE_NAME,
    DEFAULT_ALIGNMENT_TYPE,
    DEFAULT_EVALUE,
    DEFAULT_MAX_SEQS,
    DEFAULT_MIN_QUERY_COVERAGE,
    DEFAULT_MIN_TARGET_COVERAGE,
    DEFAULT_SCORE_MODE,
    DEFAULT_SCORE_THRESHOLD,
    SUPPORTED_SCORE_MODES,
    FoldseekResult,
    run_baseline,
)
from external_methods.foldseek.structures import (
    DEFAULT_MIN_RESIDUES,
    GeneratedStructureChain,
    GeneratedStructureSet,
    discover_structure_files,
    foldseek_query_aliases,
    generate_structure_chains,
)

CHAIN_FIELDS = [
    "filename",
    "relative_path",
    "author_chain_id",
    "source_file",
    "sample_id",
    "baseline",
    "result",
    "pred_barrel",
    "decision_score",
    "decision_threshold",
    "chain_residue_count",
    "is_error",
    "is_filtered_out",
    "is_skip",
    "is_target_author_chain",
    "use_for_metrics",
    "y_true",
    "split",
    "pdb_id",
    "alignment_type",
    "score_mode",
    "score_threshold",
    "min_query_coverage",
    "min_target_coverage",
    "reference_policy",
    "hit_count",
    "eligible_hit_count",
    "ignored_target_hit_count",
    "best_target",
    "best_target_source_file",
    "best_target_pdb_id",
    "qlen",
    "tlen",
    "alnlen",
    "qcov",
    "tcov",
    "qtmscore",
    "ttmscore",
    "alntmscore",
    "evalue",
    "bits",
]
GROUP_MANIFEST_FIELDS = ("split", "relative_path", "author_chain_id", "group_id")
GROUP_MANIFEST_SPLITS = {"positive", "negative", "reference"}
DEFAULT_REFERENCE_POLICY = "explicit_reference_exact_content_family_group_and_same_pdb_excluded"
NORMALIZED_CHAIN_HASH_POLICY = "sha256_over_generated_single_chain_pdb_bytes"
DEFAULT_METRIC_LEVEL = "file"
NORMALIZED_FIELDS = [
    "baseline",
    "sample_id",
    "result",
    "score",
    "decision_rule",
    "score_mode",
    "score_threshold",
    "min_query_coverage",
    "min_target_coverage",
    "hit_count",
    "eligible_hit_count",
    "ignored_target_hit_count",
    "best_target",
    "qlen",
    "tlen",
    "alnlen",
    "qcov",
    "tcov",
    "qtmscore",
    "ttmscore",
    "alntmscore",
    "evalue",
    "bits",
]


@dataclass(frozen=True)
class SplitRun:
    split_name: str
    generated: GeneratedStructureSet
    results: list[FoldseekResult]


class LeakageQuery(TypedDict):
    sample_id: str
    query_chain_sha256: str | None
    same_pdb_reference_candidates_n: int
    curated_group_reference_candidates_n: int
    exact_content_reference_candidates_n: int
    exact_content_reference_ids: list[str]
    combined_ignored_reference_candidates_n: int
    observed_ignored_hits_n: int | None


class LeakageSplitState(TypedDict):
    queries: list[LeakageQuery]
    query_count: int
    exact_content_reference_candidates_excluded_total: int
    combined_reference_candidates_excluded_total: int
    observed_ignored_hits_total: int | None


class ReferenceLeakageControl(TypedDict):
    policy: str
    hash_algorithm: str
    hash_policy: str
    identity_failure_policy: str
    generated_chain_content_inventories: dict[str, dict[str, object]]
    splits: dict[str, LeakageSplitState]
    exact_content_reference_candidates_excluded_total: int
    combined_reference_candidates_excluded_total: int
    observed_ignored_hits_total: int | None


def _pdb_id_from_filename(filename: str) -> str:
    name = Path(filename).name
    while Path(name).suffix.lower() in {".gz", ".pdb", ".cif", ".mmcif"}:
        name = Path(name).stem
    return name.split("_", 1)[0].upper()


def _metadata_by_sample(
    generated: GeneratedStructureSet,
) -> dict[str, GeneratedStructureChain]:
    metadata = {record.sample_id: record for record in generated.records}
    if len(metadata) != len(generated.records):
        raise ValueError("Generated Foldseek chains contain duplicate sample IDs.")
    return metadata


def _generated_chain_content_inventory(
    generated: GeneratedStructureSet,
    *,
    label: str,
) -> dict[str, object]:
    """Freeze exact identities for every normalized single-chain PDB artifact."""

    chain_root = Path(generated.chain_dir).expanduser().resolve()
    if not chain_root.is_dir():
        raise ValueError(f"{label} generated chain directory is unavailable: {chain_root}")
    metadata = _metadata_by_sample(generated)
    records: list[dict[str, object]] = []
    observed_paths: set[Path] = set()
    for sample_id, record in sorted(metadata.items()):
        chain_path = Path(record.chain_path).expanduser().resolve()
        try:
            relative_path = chain_path.relative_to(chain_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"{label} normalized chain artifact is outside its declared directory: {chain_path}"
            ) from exc
        if chain_path in observed_paths:
            raise ValueError(
                f"{label} generated records reuse normalized chain artifact {chain_path}."
            )
        observed_paths.add(chain_path)
        if chain_path.suffix.lower() != ".pdb" or not chain_path.is_file():
            raise ValueError(
                f"{label} normalized chain artifact must be a regular PDB file: {chain_path}"
            )
        state = file_state(chain_path)
        if int(state["size"]) <= 0 or len(str(state["sha256"])) != 64:
            raise ValueError(
                f"{label} normalized chain artifact has no reliable SHA-256 identity: {chain_path}"
            )
        records.append(
            {
                "sample_id": sample_id,
                "relative_path": relative_path,
                "size": state["size"],
                "sha256": state["sha256"],
            }
        )

    actual_paths = {
        path.resolve()
        for path in chain_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdb"
    }
    if actual_paths != observed_paths:
        missing = sorted(str(path) for path in observed_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - observed_paths)
        raise ValueError(
            f"{label} normalized chain inventory is not represented exactly by generated "
            f"records; missing={missing[:10]!r}, extra={extra[:10]!r}."
        )
    return {
        "label": label,
        "hash_algorithm": "sha256",
        "hash_policy": NORMALIZED_CHAIN_HASH_POLICY,
        "chain_dir": str(chain_root),
        "records": records,
        "inventory_sha256": canonical_json_sha256(records),
    }


def _content_hashes_by_sample(inventory: Mapping[str, object]) -> dict[str, str]:
    if inventory.get("hash_policy") != NORMALIZED_CHAIN_HASH_POLICY:
        raise ValueError("Generated chain inventory has an unknown content-hash policy.")
    raw_records = inventory.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Generated chain inventory records are unavailable.")
    hashes: dict[str, str] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("Generated chain inventory contains a malformed record.")
        sample_id = str(raw_record.get("sample_id", "")).strip()
        digest = str(raw_record.get("sha256", "")).strip()
        if not sample_id or sample_id in hashes or len(digest) != 64:
            raise ValueError(
                "Generated chain inventory contains an unreliable sample/hash identity."
            )
        hashes[sample_id] = digest
    if not hashes:
        raise ValueError("Generated chain inventory contains no content identities.")
    return hashes


def _target_ids_by_content_hash(
    records: Sequence[GeneratedStructureChain],
    content_hashes: Mapping[str, str],
) -> dict[str, set[str]]:
    sample_ids = {record.sample_id for record in records}
    if sample_ids != set(content_hashes):
        raise ValueError(
            "Generated chain records and content-hash identities do not match exactly."
        )
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[content_hashes[record.sample_id]].add(record.sample_id)
    return grouped


def _target_ids_by_pdb(
    records: Sequence[GeneratedStructureChain],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[_pdb_id_from_filename(Path(record.source_path).name)].add(record.sample_id)
    return grouped


def _group_assignments(path: Path) -> dict[tuple[str, str, str], str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or [])
        if fields != GROUP_MANIFEST_FIELDS:
            raise ValueError(
                "Group manifest must have exactly these columns in order: "
                f"{','.join(GROUP_MANIFEST_FIELDS)}."
            )
        rows = list(reader)

    assignments: dict[tuple[str, str, str], str] = {}
    for row_number, row in enumerate(rows, start=2):
        split = str(row.get("split", "")).strip().lower()
        relative_path_text = str(row.get("relative_path", "")).strip().replace("\\", "/")
        author_chain_id = str(row.get("author_chain_id", "")).strip()
        group_id = str(row.get("group_id", "")).strip()
        relative_path = Path(relative_path_text)
        if split not in GROUP_MANIFEST_SPLITS:
            raise ValueError(
                f"Group manifest row {row_number} has invalid split {split!r}; "
                f"expected one of {sorted(GROUP_MANIFEST_SPLITS)!r}."
            )
        if (
            not relative_path_text
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path_text in {".", "./"}
        ):
            raise ValueError(f"Group manifest row {row_number} has an invalid relative_path.")
        if not group_id:
            raise ValueError(f"Group manifest row {row_number} has a blank group_id.")
        if not author_chain_id:
            raise ValueError(f"Group manifest row {row_number} has a blank author_chain_id.")
        key = (split, relative_path.as_posix(), author_chain_id)
        if key in assignments:
            raise ValueError(f"Group manifest contains duplicate assignment for {key!r}.")
        assignments[key] = group_id
    if not assignments:
        raise ValueError("Group manifest must contain at least one assignment.")
    return assignments


def _relative_source_path(record: GeneratedStructureChain, input_root: Path) -> str:
    root = input_root.expanduser().resolve()
    source = Path(record.source_path).expanduser().resolve()
    try:
        return source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Generated source {source} is outside declared input root {root}."
        ) from exc


def _group_ids_for_records(
    records: Sequence[GeneratedStructureChain],
    *,
    split: str,
    input_root: Path,
    assignments: dict[tuple[str, str, str], str],
) -> dict[str, str]:
    groups: dict[str, str] = {}
    missing: list[tuple[str, str, str]] = []
    for record in records:
        key = (split, _relative_source_path(record, input_root), record.chain_id)
        group_id = assignments.get(key)
        if group_id is None:
            missing.append(key)
        else:
            groups[record.sample_id] = group_id
    if missing:
        preview = ", ".join(repr(key) for key in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(
            f"Group manifest has no assignment for generated chain(s): {preview}{suffix}"
        )
    return groups


def _target_ids_by_group(
    records: Sequence[GeneratedStructureChain],
    group_ids: dict[str, str],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[group_ids[record.sample_id]].add(record.sample_id)
    return grouped


def _ignore_details_for_queries(
    query_records: Sequence[GeneratedStructureChain],
    query_group_ids: dict[str, str],
    reference_ids_by_group: dict[str, set[str]],
    reference_ids_by_pdb: dict[str, set[str]],
    *,
    query_content_hashes: Mapping[str, str] | None = None,
    reference_ids_by_content_hash: Mapping[str, set[str]] | None = None,
) -> tuple[dict[str, set[str]], list[LeakageQuery]]:
    if (query_content_hashes is None) != (reference_ids_by_content_hash is None):
        raise ValueError(
            "Query and reference normalized-chain content identities must be supplied together."
        )
    if query_content_hashes is not None and {record.sample_id for record in query_records} != set(
        query_content_hashes
    ):
        raise ValueError("Query records and normalized-chain content identities do not match.")
    ignored: dict[str, set[str]] = {}
    details: list[LeakageQuery] = []
    for record in query_records:
        pdb_id = _pdb_id_from_filename(Path(record.source_path).name)
        same_pdb_ids = set(reference_ids_by_pdb.get(pdb_id, set()))
        same_group_ids = set(reference_ids_by_group.get(query_group_ids[record.sample_id], set()))
        query_digest = (
            str(query_content_hashes[record.sample_id]) if query_content_hashes is not None else ""
        )
        exact_content_ids = (
            set(reference_ids_by_content_hash.get(query_digest, set()))
            if reference_ids_by_content_hash is not None
            else set()
        )
        target_ids = same_pdb_ids | same_group_ids | exact_content_ids
        if target_ids:
            ignored[record.sample_id] = target_ids
        details.append(
            {
                "sample_id": record.sample_id,
                "query_chain_sha256": query_digest or None,
                "same_pdb_reference_candidates_n": len(same_pdb_ids),
                "curated_group_reference_candidates_n": len(same_group_ids),
                "exact_content_reference_candidates_n": len(exact_content_ids),
                "exact_content_reference_ids": sorted(exact_content_ids),
                "combined_ignored_reference_candidates_n": len(target_ids),
                "observed_ignored_hits_n": None,
            }
        )
    return ignored, details


def _ignore_map_for_queries(
    query_records: Sequence[GeneratedStructureChain],
    query_group_ids: dict[str, str],
    reference_ids_by_group: dict[str, set[str]],
    reference_ids_by_pdb: dict[str, set[str]],
    *,
    query_content_hashes: Mapping[str, str] | None = None,
    reference_ids_by_content_hash: Mapping[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    ignored, _ = _ignore_details_for_queries(
        query_records,
        query_group_ids,
        reference_ids_by_group,
        reference_ids_by_pdb,
        query_content_hashes=query_content_hashes,
        reference_ids_by_content_hash=reference_ids_by_content_hash,
    )
    return ignored


def _chain_rows_for_split(
    run: SplitRun,
    *,
    reference_metadata: dict[str, GeneratedStructureChain],
    alignment_type: int,
    reference_policy: str,
) -> list[dict[str, object]]:
    metadata = _metadata_by_sample(run.generated)
    result_by_sample = {result.sample_id: result for result in run.results}
    if len(result_by_sample) != len(run.results):
        raise ValueError("Foldseek returned duplicate sample IDs.")
    missing_samples = sorted(set(metadata) - set(result_by_sample))
    unexpected_samples = sorted(set(result_by_sample) - set(metadata))
    if missing_samples or unexpected_samples:
        raise ValueError(
            "Foldseek result identity does not match generated queries; "
            f"missing={missing_samples[:10]!r}, unexpected={unexpected_samples[:10]!r}."
        )

    rows: list[dict[str, object]] = []
    for sample_id, record in metadata.items():
        result = result_by_sample[sample_id]
        filename = Path(record.source_path).name
        best_target_record = reference_metadata.get(result.best_target or "")
        best_target_source_file = (
            best_target_record.source_path if best_target_record is not None else ""
        )
        rows.append(
            {
                "filename": filename,
                "relative_path": "",
                "author_chain_id": record.chain_id,
                "result": result.result,
                "pred_barrel": result.result == "BARREL",
                "decision_score": result.score,
                "decision_threshold": result.score_threshold,
                "chain_residue_count": record.n_residues,
                "y_true": "",
                "split": "",
                "is_error": False,
                "is_filtered_out": False,
                "is_skip": False,
                "use_for_metrics": False,
                "is_target_author_chain": False,
                "sample_id": result.sample_id,
                "baseline": BASELINE_NAME,
                "source_file": str(Path(record.source_path).resolve()),
                "pdb_id": _pdb_id_from_filename(filename),
                "alignment_type": alignment_type,
                "score_mode": result.score_mode,
                "score_threshold": result.score_threshold,
                "min_query_coverage": result.min_query_coverage,
                "min_target_coverage": result.min_target_coverage,
                "reference_policy": reference_policy,
                "hit_count": result.hit_count,
                "eligible_hit_count": result.eligible_hit_count,
                "ignored_target_hit_count": result.ignored_target_hit_count,
                "best_target": result.best_target,
                "best_target_source_file": best_target_source_file,
                "best_target_pdb_id": (
                    _pdb_id_from_filename(Path(best_target_source_file).name)
                    if best_target_source_file
                    else ""
                ),
                "qlen": result.qlen,
                "tlen": result.tlen,
                "alnlen": result.alnlen,
                "qcov": result.qcov,
                "tcov": result.tcov,
                "qtmscore": result.qtmscore,
                "ttmscore": result.ttmscore,
                "alntmscore": result.alntmscore,
                "evalue": result.evalue,
                "bits": result.bits,
            }
        )
    validate_prediction_rows(rows)
    return rows


def _default_foldseek_executable() -> str:
    local_binary = Path("tools/foldseek/bin/foldseek")
    if local_binary.exists():
        return str(local_binary.resolve())
    return os.environ.get("FOLDSEEK_BIN", "foldseek")


def _validate_parameters(
    *,
    min_residues: int,
    alignment_type: int,
    score_mode: str,
    score_threshold: float,
    min_query_coverage: float,
    min_target_coverage: float,
    evalue: float,
    max_seqs: int,
    timeout: float | None,
    reference_policy: str,
) -> None:
    if isinstance(min_residues, bool) or not isinstance(min_residues, int) or min_residues <= 0:
        raise ValueError("min_residues must be a positive integer.")
    if isinstance(alignment_type, bool) or not isinstance(alignment_type, int):
        raise ValueError("alignment_type must be an integer.")
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"score_mode must be one of {sorted(SUPPORTED_SCORE_MODES)!r}.")
    for label, value in (
        ("score_threshold", score_threshold),
        ("min_query_coverage", min_query_coverage),
        ("min_target_coverage", min_target_coverage),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{label} must be finite and within [0, 1].")
    if (
        isinstance(evalue, bool)
        or not isinstance(evalue, (int, float))
        or not math.isfinite(float(evalue))
        or evalue <= 0
    ):
        raise ValueError("evalue must be finite and > 0.")
    if isinstance(max_seqs, bool) or not isinstance(max_seqs, int) or max_seqs <= 0:
        raise ValueError("max_seqs must be a positive integer.")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("timeout must be finite and > 0 when provided.")
    if not str(reference_policy).strip():
        raise ValueError("reference_policy must not be blank.")


def run_dataset(
    positive_dir: Path,
    negative_dir: Path,
    output_root: Path,
    *,
    reference_dir: Path,
    group_manifest: Path,
    foldseek_executable: str | Path | None,
    positive_target_manifest: Path | None,
    negative_target_manifest: Path | None,
    metric_level: str,
    min_residues: int,
    create_index: bool,
    alignment_type: int,
    score_mode: str,
    score_threshold: float,
    min_query_coverage: float,
    min_target_coverage: float,
    evalue: float,
    max_seqs: int,
    extra_args: Sequence[str] | None,
    timeout: float | None,
    tag: str,
    reference_policy: str = DEFAULT_REFERENCE_POLICY,
) -> Path:
    executable = foldseek_executable or _default_foldseek_executable()
    supplied_parameters = {
        "positive_dir": str(positive_dir),
        "negative_dir": str(negative_dir),
        "output_root": str(output_root),
        "reference_dir": str(reference_dir),
        "group_manifest": str(group_manifest),
        "foldseek_executable": os.fspath(executable),
        "positive_target_manifest": (
            str(positive_target_manifest) if positive_target_manifest is not None else None
        ),
        "negative_target_manifest": (
            str(negative_target_manifest) if negative_target_manifest is not None else None
        ),
        "metric_level": metric_level,
        "min_residues": min_residues,
        "create_index": create_index,
        "alignment_type": alignment_type,
        "score_mode": score_mode,
        "score_threshold": score_threshold,
        "min_query_coverage": min_query_coverage,
        "min_target_coverage": min_target_coverage,
        "evalue": evalue,
        "max_seqs": max_seqs,
        "extra_args": list(extra_args or []),
        "timeout": timeout,
        "tag": tag,
        "reference_policy": reference_policy,
    }
    run_dir = fresh_run_directory(output_root, baseline=BASELINE_NAME, tag=tag)
    manifest_path = run_dir / "evaluation_manifest.json"
    manifest = initialize_manifest(
        manifest_path,
        baseline=BASELINE_NAME,
        script_path=Path(__file__),
        supplied_parameters=supplied_parameters,
    )
    phase = "validation"
    try:
        update_running_manifest(manifest_path, manifest, phase=phase)
        validate_metric_contract(metric_level, positive_target_manifest, negative_target_manifest)
        _validate_parameters(
            min_residues=min_residues,
            alignment_type=alignment_type,
            score_mode=score_mode,
            score_threshold=score_threshold,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
            evalue=evalue,
            max_seqs=max_seqs,
            timeout=timeout,
            reference_policy=reference_policy,
        )
        if not isinstance(create_index, bool):
            raise ValueError("create_index must be a boolean.")
        if any(not isinstance(value, str) or not value for value in (extra_args or [])):
            raise ValueError("extra_args must contain only non-empty strings.")

        roots = {
            "positive": positive_dir.expanduser().resolve(),
            "negative": negative_dir.expanduser().resolve(),
            "reference": reference_dir.expanduser().resolve(),
        }
        for label, root in roots.items():
            if not root.is_dir():
                raise NotADirectoryError(f"{label} input must be a directory: {root}")
        structure_files = {split: discover_structure_files(root) for split, root in roots.items()}
        inventories = {
            split: files_inventory(files, root=roots[split])
            for split, files in structure_files.items()
        }
        validate_disjoint_labeled_inventories(inventories["positive"], inventories["negative"])
        group_state = file_state(group_manifest)
        target_states = {
            "positive": (
                file_state(positive_target_manifest)
                if positive_target_manifest is not None
                else None
            ),
            "negative": (
                file_state(negative_target_manifest)
                if negative_target_manifest is not None
                else None
            ),
        }
        targets = {
            "positive": (
                read_target_manifest(
                    positive_target_manifest,
                    split_root=roots["positive"],
                    structure_files=structure_files["positive"],
                )
                if positive_target_manifest is not None
                else None
            ),
            "negative": (
                read_target_manifest(
                    negative_target_manifest,
                    split_root=roots["negative"],
                    structure_files=structure_files["negative"],
                )
                if negative_target_manifest is not None
                else None
            ),
        }
        tool_identity = executable_identity(executable)
        manifest["parameters"] = {
            **supplied_parameters,
            "positive_dir": str(roots["positive"]),
            "negative_dir": str(roots["negative"]),
            "reference_dir": str(roots["reference"]),
            "group_manifest": str(Path(group_manifest).expanduser().resolve()),
            "foldseek_executable": tool_identity["path"],
            "filtered_out_policy": "strict",
        }
        code_dependencies = {
            "evaluation_common": file_state(
                Path(__file__).resolve().parents[1] / "evaluation_common.py"
            ),
            "runner": file_state(Path(__file__).with_name("runner.py")),
            "structures": file_state(Path(__file__).with_name("structures.py")),
        }
        manifest["code_dependencies"] = code_dependencies
        manifest["inputs"] = {
            "structure_inventories": inventories,
            "group_manifest": group_state,
            "target_chain_manifests": target_states,
        }
        manifest["external_software"] = {"foldseek": tool_identity}
        manifest["metric_sampling"] = {
            "file": "one_directory_labeled_structure_file_any_chain_prediction",
            "chain": (
                "one_manifest_target_chain_per_positive_and_negative_file"
                if targets["positive"] is not None
                else None
            ),
            "partner_chain_labels": "unlabeled",
            "error_policy": "fail_closed",
            "filtered_out_policy": "strict",
        }
        phase = "input_preparation"
        update_running_manifest(manifest_path, manifest, phase=phase)

        reference_generated = generate_structure_chains(
            roots["reference"], run_dir / "reference_chains", min_residues=min_residues
        )
        validate_generated_source_coverage(
            [record.source_path for record in reference_generated.records],
            structure_files=structure_files["reference"],
            context="reference",
        )
        generated_chain_inventories: dict[str, dict[str, object]] = {
            "reference": _generated_chain_content_inventory(
                reference_generated,
                label="reference",
            )
        }
        reference_content_hashes = _content_hashes_by_sample(
            generated_chain_inventories["reference"]
        )
        reference_ids_by_content_hash = _target_ids_by_content_hash(
            reference_generated.records,
            reference_content_hashes,
        )
        reference_metadata = _metadata_by_sample(reference_generated)
        group_assignments = _group_assignments(Path(group_state["path"]))
        reference_group_ids = _group_ids_for_records(
            reference_generated.records,
            split="reference",
            input_root=roots["reference"],
            assignments=group_assignments,
        )
        reference_ids_by_group = _target_ids_by_group(
            reference_generated.records, reference_group_ids
        )
        reference_ids_by_pdb = _target_ids_by_pdb(reference_generated.records)
        target_aliases = foldseek_query_aliases(reference_generated.records)

        generated_by_split: dict[str, GeneratedStructureSet] = {}
        query_group_ids_by_split: dict[str, dict[str, str]] = {}
        query_content_hashes_by_split: dict[str, dict[str, str]] = {}
        for split_name in ("positive", "negative"):
            split_output = run_dir / split_name
            input_dir = roots[split_name]
            generated = (
                reference_generated
                if input_dir == roots["reference"]
                else generate_structure_chains(
                    input_dir,
                    split_output / "query_chains",
                    min_residues=min_residues,
                )
            )
            validate_generated_source_coverage(
                [record.source_path for record in generated.records],
                structure_files=structure_files[split_name],
                context=split_name,
            )
            generated_by_split[split_name] = generated
            generated_chain_inventories[split_name] = _generated_chain_content_inventory(
                generated,
                label=split_name,
            )
            query_content_hashes_by_split[split_name] = _content_hashes_by_sample(
                generated_chain_inventories[split_name]
            )
            query_group_ids_by_split[split_name] = _group_ids_for_records(
                generated.records,
                split=split_name,
                input_root=input_dir,
                assignments=group_assignments,
            )

        used_group_keys = {
            (
                split_name,
                _relative_source_path(record, roots[split_name]),
                record.chain_id,
            )
            for split_name, generated in {
                "reference": reference_generated,
                **generated_by_split,
            }.items()
            for record in generated.records
        }
        if set(group_assignments) != used_group_keys:
            extra = sorted(set(group_assignments) - used_group_keys)
            missing = sorted(used_group_keys - set(group_assignments))
            raise ValueError(
                "Group manifest must match all generated chain identities exactly; "
                f"missing={missing[:10]!r}, extra={extra[:10]!r}."
            )

        ignore_maps: dict[str, dict[str, set[str]]] = {}
        leakage_queries_by_split: dict[str, list[LeakageQuery]] = {}
        for split_name in ("positive", "negative"):
            ignore_map, query_details = _ignore_details_for_queries(
                generated_by_split[split_name].records,
                query_group_ids_by_split[split_name],
                reference_ids_by_group,
                reference_ids_by_pdb,
                query_content_hashes=query_content_hashes_by_split[split_name],
                reference_ids_by_content_hash=reference_ids_by_content_hash,
            )
            ignore_maps[split_name] = ignore_map
            leakage_queries_by_split[split_name] = query_details

        all_leakage_queries = [
            query
            for split_name in ("positive", "negative")
            for query in leakage_queries_by_split[split_name]
        ]
        leakage_split_states: dict[str, LeakageSplitState] = {
            split_name: {
                "queries": leakage_queries_by_split[split_name],
                "query_count": len(leakage_queries_by_split[split_name]),
                "exact_content_reference_candidates_excluded_total": sum(
                    query["exact_content_reference_candidates_n"]
                    for query in leakage_queries_by_split[split_name]
                ),
                "combined_reference_candidates_excluded_total": sum(
                    query["combined_ignored_reference_candidates_n"]
                    for query in leakage_queries_by_split[split_name]
                ),
                "observed_ignored_hits_total": None,
            }
            for split_name in ("positive", "negative")
        }
        reference_leakage_control: ReferenceLeakageControl = {
            "policy": (
                "exclude_union_of_exact_normalized_chain_content_same_pdb_and_curated_group"
            ),
            "hash_algorithm": "sha256",
            "hash_policy": NORMALIZED_CHAIN_HASH_POLICY,
            "identity_failure_policy": "fail_closed",
            "generated_chain_content_inventories": generated_chain_inventories,
            "splits": leakage_split_states,
            "exact_content_reference_candidates_excluded_total": sum(
                query["exact_content_reference_candidates_n"] for query in all_leakage_queries
            ),
            "combined_reference_candidates_excluded_total": sum(
                query["combined_ignored_reference_candidates_n"] for query in all_leakage_queries
            ),
            "observed_ignored_hits_total": None,
        }
        manifest["reference_leakage_control"] = reference_leakage_control
        update_running_manifest(manifest_path, manifest, phase="reference_leakage_control")

        split_runs: list[SplitRun] = []
        for split_name in ("positive", "negative"):
            phase = f"external_run_{split_name}"
            update_running_manifest(manifest_path, manifest, phase=phase)
            split_output = run_dir / split_name
            generated = generated_by_split[split_name]
            results = run_baseline(
                generated.chain_dir,
                reference_generated.chain_dir,
                foldseek_executable=tool_identity["path"],
                work_dir=split_output / "foldseek_work",
                output_path=None,
                query_ids=[record.sample_id for record in generated.records],
                query_aliases=foldseek_query_aliases(generated.records),
                target_aliases=target_aliases,
                ignore_target_ids_by_query=ignore_maps[split_name],
                build_target_db=True,
                create_index=create_index,
                alignment_type=alignment_type,
                score_mode=score_mode,
                score_threshold=score_threshold,
                min_query_coverage=min_query_coverage,
                min_target_coverage=min_target_coverage,
                evalue=evalue,
                max_seqs=max_seqs,
                extra_args=extra_args,
                timeout=timeout,
            )
            atomic_write_csv(
                split_output / "normalized.csv",
                NORMALIZED_FIELDS,
                [result.as_row() for result in results],
            )
            result_by_sample = {result.sample_id: result for result in results}
            if len(result_by_sample) != len(results):
                raise ValueError("Foldseek returned duplicate sample IDs.")
            for query in leakage_queries_by_split[split_name]:
                sample_id = str(query["sample_id"])
                if sample_id not in result_by_sample:
                    raise ValueError(
                        f"Foldseek returned no result for leakage-audited query {sample_id!r}."
                    )
                query["observed_ignored_hits_n"] = result_by_sample[
                    sample_id
                ].ignored_target_hit_count
            split_leakage_state = reference_leakage_control["splits"][split_name]
            split_leakage_state["observed_ignored_hits_total"] = sum(
                cast(int, query["observed_ignored_hits_n"])
                for query in leakage_queries_by_split[split_name]
            )
            observed_totals = [
                reference_leakage_control["splits"][name]["observed_ignored_hits_total"]
                for name in ("positive", "negative")
            ]
            reference_leakage_control["observed_ignored_hits_total"] = (
                sum(value for value in observed_totals if value is not None)
                if all(value is not None for value in observed_totals)
                else None
            )
            split_runs.append(SplitRun(split_name, generated, results))
            update_running_manifest(manifest_path, manifest, phase=phase)

        phase = "metric_construction"
        update_running_manifest(manifest_path, manifest, phase=phase)
        all_predictions: list[dict[str, object]] = []
        all_target_rows: list[dict[str, object]] = []
        all_file_rows: list[dict[str, object]] = []
        for split_run in split_runs:
            raw_rows = _chain_rows_for_split(
                split_run,
                reference_metadata=reference_metadata,
                alignment_type=alignment_type,
                reference_policy=reference_policy,
            )
            predictions, target_rows = apply_target_chain_labels(
                raw_rows,
                split=split_run.split_name,
                split_root=roots[split_run.split_name],
                targets=targets[split_run.split_name],
            )
            all_predictions.extend(predictions)
            all_target_rows.extend(target_rows)
            all_file_rows.extend(
                file_rows_from_chain_rows(
                    raw_rows,
                    split=split_run.split_name,
                    split_root=roots[split_run.split_name],
                    structure_files=structure_files[split_run.split_name],
                )
            )
        metric_rows = summary_rows(
            metric_level=metric_level,
            file_rows=all_file_rows,
            target_chain_rows=all_target_rows,
        )

        phase = "output_writing"
        update_running_manifest(manifest_path, manifest, phase=phase)
        atomic_write_csv(run_dir / "chain_predictions.csv", CHAIN_FIELDS, all_predictions)
        atomic_write_csv(run_dir / "file_results.csv", FILE_FIELDS, all_file_rows)
        if metric_level in {"chain", "both"}:
            atomic_write_csv(run_dir / "target_chain_results.csv", CHAIN_FIELDS, all_target_rows)
        atomic_write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, metric_rows)
        atomic_write_text(run_dir / "summary.md", summary_markdown(metric_rows))

        phase = "provenance_verification"
        update_running_manifest(manifest_path, manifest, phase=phase)
        current_generated_chain_inventories = {
            "reference": _generated_chain_content_inventory(
                reference_generated,
                label="reference",
            ),
            **{
                split_name: _generated_chain_content_inventory(
                    generated_by_split[split_name],
                    label=split_name,
                )
                for split_name in ("positive", "negative")
            },
        }
        if current_generated_chain_inventories != generated_chain_inventories:
            raise RuntimeError(
                "A normalized query/reference chain artifact changed during Foldseek evaluation."
            )
        current_inventories = {
            split: files_inventory(files, root=roots[split])
            for split, files in structure_files.items()
        }
        if current_inventories != inventories:
            raise RuntimeError("A structure input changed during Foldseek evaluation.")
        if file_state(Path(group_state["path"])) != group_state:
            raise RuntimeError("The group manifest changed during Foldseek evaluation.")
        for split in ("positive", "negative"):
            state = target_states[split]
            if state is not None and file_state(Path(state["path"])) != state:
                raise RuntimeError(f"The {split} target-chain manifest changed during evaluation.")
        if executable_identity(Path(str(tool_identity["path"]))) != tool_identity:
            raise RuntimeError("The Foldseek executable changed during evaluation.")
        if file_state(Path(__file__)) != manifest["script"]:
            raise RuntimeError("The Foldseek evaluator script changed during evaluation.")
        current_code_dependencies = {
            "evaluation_common": file_state(
                Path(__file__).resolve().parents[1] / "evaluation_common.py"
            ),
            "runner": file_state(Path(__file__).with_name("runner.py")),
            "structures": file_state(Path(__file__).with_name("structures.py")),
        }
        if current_code_dependencies != code_dependencies:
            raise RuntimeError("A Foldseek evaluator code dependency changed during evaluation.")
        complete_manifest(manifest_path, manifest)
    except Exception as exc:
        fail_manifest(manifest_path, manifest, phase=phase, error=exc)
        raise

    print(f"Chain predictions: {len(all_predictions)}")
    print(f"File observations: {len(all_file_rows)}")
    print(f"Target-chain observations: {len(all_target_rows)}")
    print(f"Run directory: {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/foldseek/evaluate_dataset.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate the Foldseek global-TMalign baseline on labeled Cooper-Beta structure "
            "directories using an explicit reference panel and homology-group assignments."
        ),
        epilog=(
            "Output: a new timestamped directory under --out-dir with generated chains, raw and "
            "normalized predictions, file and optional target-chain metrics, and run_manifest.json. "
            "Arguments after -- are passed to Foldseek easy-search. Invalid input or subprocess "
            "failures exit with status 2."
        ),
    )
    parser.add_argument(
        "--positive-dir",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing structures labeled positive.",
    )
    parser.add_argument(
        "--negative-dir",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing structures labeled negative.",
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing the curated Foldseek reference structures.",
    )
    parser.add_argument(
        "--group-manifest",
        required=True,
        metavar="CSV",
        help=(
            "CSV with split,relative_path,author_chain_id,group_id for every generated positive, "
            "negative, and reference chain."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="DIRECTORY",
        help="Output root; a fresh timestamped run directory is created.",
    )
    parser.add_argument(
        "--foldseek",
        default=_default_foldseek_executable(),
        metavar="COMMAND",
        help="Foldseek executable path or command name.",
    )
    parser.add_argument(
        "--metric-level",
        choices=["file", "chain", "both"],
        default=DEFAULT_METRIC_LEVEL,
        help="Metric granularity; file metrics use directory labels and any-positive-chain output.",
    )
    parser.add_argument(
        "--positive-target-manifest",
        metavar="CSV",
        help=(
            "Positive target-selection CSV with exactly relative_path,author_chain_id; required "
            "with the negative target manifest for chain metrics."
        ),
    )
    parser.add_argument(
        "--negative-target-manifest",
        metavar="CSV",
        help=(
            "Negative target-selection CSV with exactly relative_path,author_chain_id; required "
            "with the positive target manifest for chain metrics."
        ),
    )
    parser.add_argument(
        "--tag",
        required=True,
        metavar="NAME",
        help="Run label included in the output directory name.",
    )
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        metavar="RESIDUES",
        help="Minimum alpha-carbon residue count required to evaluate a chain.",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Build a Foldseek index for the generated reference database before searching.",
    )
    parser.add_argument(
        "--alignment-type",
        type=int,
        default=DEFAULT_ALIGNMENT_TYPE,
        metavar="N",
        help="Foldseek --alignment-type value; 1 selects global TMalign.",
    )
    parser.add_argument(
        "--score-mode",
        default=DEFAULT_SCORE_MODE,
        choices=sorted(SUPPORTED_SCORE_MODES),
        help="Per-hit score used for the threshold decision.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        metavar="FRACTION",
        help="Inclusive minimum selected hit score for a BARREL decision.",
    )
    parser.add_argument(
        "--min-query-coverage",
        type=float,
        default=DEFAULT_MIN_QUERY_COVERAGE,
        metavar="FRACTION",
        help="Inclusive minimum query coverage for eligible hits, from 0 to 1.",
    )
    parser.add_argument(
        "--min-target-coverage",
        type=float,
        default=DEFAULT_MIN_TARGET_COVERAGE,
        metavar="FRACTION",
        help="Inclusive minimum target coverage for eligible hits, from 0 to 1.",
    )
    parser.add_argument(
        "--evalue",
        type=float,
        default=DEFAULT_EVALUE,
        metavar="EVALUE",
        help="Maximum E-value passed to Foldseek easy-search.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=DEFAULT_MAX_SEQS,
        metavar="N",
        help="Maximum candidate sequences passed to Foldseek --max-seqs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum elapsed time for each Foldseek subprocess; omit for no timeout.",
    )
    return parser


def _parse_args_and_passthrough(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw_args:
        return parser.parse_args(raw_args), []
    passthrough_index = raw_args.index("--")
    return (
        parser.parse_args(raw_args[:passthrough_index]),
        raw_args[passthrough_index + 1 :],
    )


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_args = _parse_args_and_passthrough(build_arg_parser(), argv)
    run_dataset(
        Path(args.positive_dir),
        Path(args.negative_dir),
        Path(args.out_dir),
        reference_dir=Path(args.reference_dir),
        group_manifest=Path(args.group_manifest),
        foldseek_executable=args.foldseek,
        positive_target_manifest=(
            Path(args.positive_target_manifest) if args.positive_target_manifest else None
        ),
        negative_target_manifest=(
            Path(args.negative_target_manifest) if args.negative_target_manifest else None
        ),
        metric_level=args.metric_level,
        min_residues=args.min_residues,
        create_index=args.create_index,
        alignment_type=args.alignment_type,
        score_mode=args.score_mode,
        score_threshold=args.score_threshold,
        min_query_coverage=args.min_query_coverage,
        min_target_coverage=args.min_target_coverage,
        evalue=args.evalue,
        max_seqs=args.max_seqs,
        extra_args=extra_args,
        timeout=args.timeout,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
