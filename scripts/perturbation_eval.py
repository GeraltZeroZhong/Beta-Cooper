#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypedDict

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cooper_beta._version import __version__ as source_package_version  # noqa: E402
from cooper_beta.evaluation.metrics import (  # noqa: E402
    TRUTH_MANIFEST_COLUMNS,
    CandidateTruth,
)
from cooper_beta.integrity import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    freeze_input_identity,
    verified_input_snapshot,
)
from cooper_beta.polymer_sequence import declared_polymer_sequences  # noqa: E402

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
DEFAULT_NOISE_SIGMAS = "0,0.25,0.5,1.0,1.5,2.0"
DEFAULT_NOISE_SEEDS = "0"
DEFAULT_METRIC_LEVEL = "file"
DEFAULT_METRIC_ERROR_POLICY = "strict"
DEFAULT_NOISE_ATOMS = "ca"
DEFAULT_SAVE_DIR = "eval_outputs"
DEFAULT_WORKERS = None
DEFAULT_PREPARE_WORKERS = None
DEFAULT_MAX_FILES_PER_SPLIT = None
DEFAULT_SUBSET_SEED = 0
DEFAULT_POSITIVE_MANIFEST = None
DEFAULT_NEGATIVE_MANIFEST = None
EVALUATED_STRUCTURE_RETENTION_POLICY = "always_persist"
SUITE_MANIFEST_SCHEMA_VERSION = 1
STABLE_SEED_DIGEST_BYTES = 8
RNG_SEED_MODULUS = 2**32
NOISE_VECTOR_DIMENSIONS = 3
NOISE_MEAN_ANGSTROM = 0.0
RNG_BIT_GENERATOR = "PCG64"

FloatArray = NDArray[np.float64]
BooleanArray = NDArray[np.bool_]


class PerturbationSuiteManifest(TypedDict, total=False):
    """Mutable manifest fields written throughout one perturbation suite."""

    schema_version: int
    status: str
    phase: str
    started_at_utc: str
    run_token_utc: str
    output_dir: str
    script: dict[str, str]
    software: dict[str, object]
    parameters: dict[str, object]
    inputs: dict[str, object]
    metric_sampling: dict[str, object]
    artifact_policy: dict[str, object]
    experiments: list[dict[str, object]]
    outputs: dict[str, object]
    current_experiment: str
    completed_at_utc: str
    experiment_count: int
    failed_at_utc: str
    error: dict[str, str]


class PerturbationIdentityInvariants(TypedDict, total=False):
    """Identity checks recorded for one generated structure."""

    policy: str
    preserved: bool
    non_coordinate_bytes_exact: bool
    complete_polymer_sequences_equal: bool
    source: dict[str, object]
    generated: dict[str, object]


class PerturbationDetails(TypedDict):
    """Coordinate perturbation applied to one structure."""

    format: str
    requested_sigma_angstrom: float
    atoms_policy: str
    realized_delta: dict[str, object]
    identity_invariants: PerturbationIdentityInvariants


class PerturbedStructureRecord(TypedDict):
    """Persistent inventory record for one generated structure."""

    source_path: str
    source_sha256: str
    relative_path: str
    generated_path: str
    generated_size: int
    generated_sha256: str
    perturbation: PerturbationDetails


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_run_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _strict_optional_positive_int(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer when provided.")
    return value


def _strict_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer.")
    return int(value)


def _strict_float_values(
    values: list[float],
    *,
    label: str,
    allow_zero: bool,
) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must contain at least one value.")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{label} must contain only numeric values.")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{label} values must be finite.")
        if number < 0.0 or (number == 0.0 and not allow_zero):
            comparison = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"{label} values must be {comparison}.")
        normalized.append(number)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must not contain duplicate values.")
    return normalized


def _strict_int_values(values: list[int], *, label: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must contain at least one value.")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label} must contain only integers.")
        normalized.append(int(value))
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must not contain duplicate values.")
    return normalized


def _require_finite_row(row: dict[str, object], *, experiment: str) -> None:
    for key, value in row.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
            raise ValueError(
                f"Evaluation {experiment!r} returned non-finite summary field {key!r}."
            )


def _atomic_write_dataframe_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
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


def _truth_manifest_state(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _freeze_truth_manifest(
    path: Path,
    *,
    label: str,
) -> tuple[tuple[CandidateTruth, ...], dict[str, object]]:
    from cooper_beta.evaluation.runner import load_negative_manifest, load_positive_manifest

    identity = freeze_input_identity(path)
    with verified_input_snapshot(identity) as snapshot:
        if label == "positive":
            candidates = load_positive_manifest(snapshot)
        elif label == "negative":
            candidates = load_negative_manifest(snapshot)
        else:  # pragma: no cover - implementation contract
            raise ValueError(f"Unknown truth-manifest split: {label}")
    return candidates, {
        "path": identity.path,
        "size": identity.size,
        "mtime_ns": identity.mtime_ns,
        "sha256": identity.sha256,
        "snapshot_policy": "descriptor_verified_private_copy",
    }


def _parse_float_list(raw: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("Expected a comma-separated list of numeric values.") from exc
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_int_list(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("Expected a comma-separated list of integer values.") from exc
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _level_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _discover_structures(folder: Path) -> list[Path]:
    folder = folder.resolve()
    if not folder.exists():
        raise FileNotFoundError(str(folder))
    if folder.is_file():
        return [folder]
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES
    )


def _relative_structure_path(source_root: Path, source: Path) -> str:
    source_root = source_root.resolve()
    source = source.resolve()
    if source_root.is_file():
        return source.name
    return source.relative_to(source_root).as_posix()


def _structure_inventory(source_root: Path) -> list[dict[str, object]]:
    resolved_root = source_root.expanduser().resolve()
    files = _discover_structures(resolved_root)
    if not files:
        raise ValueError(f"No structure files found in {resolved_root}")
    inventory: list[dict[str, object]] = []
    for source in files:
        stat = source.stat()
        inventory.append(
            {
                "source_path": str(source.resolve()),
                "relative_path": _relative_structure_path(resolved_root, source),
                "size": int(stat.st_size),
                "sha256": file_sha256(source),
            }
        )
    return inventory


def _select_inventory(
    inventory: list[dict[str, object]],
    *,
    limit: int | None,
    seed: int,
    split: str,
) -> list[dict[str, object]]:
    if limit is None or limit >= len(inventory):
        return [dict(item) for item in inventory]
    split_seed = _stable_seed(seed, f"subset-selection:{split}")
    generator = np.random.Generator(np.random.PCG64(split_seed))
    selected_indices = sorted(
        int(index) for index in generator.choice(len(inventory), size=limit, replace=False).tolist()
    )
    return [dict(inventory[index]) for index in selected_indices]


def _archive_inventory(
    inventory: list[dict[str, object]],
    *,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    archived: list[dict[str, object]] = []
    for item in inventory:
        source = Path(str(item["source_path"])).resolve()
        destination = output_dir / str(item["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Archive destination collision: {destination}")
        shutil.copy2(source, destination)
        archived_digest = file_sha256(destination)
        if archived_digest != item["sha256"]:
            raise RuntimeError(f"Structure changed while it was being archived: {source}")
        archived.append(
            {
                **item,
                "archived_path": str(destination.resolve()),
                "archived_size": int(destination.stat().st_size),
                "archived_sha256": archived_digest,
            }
        )
    return archived


def _prepare_persistent_base_inputs(
    positive_dir: Path,
    negative_dir: Path,
    *,
    archive_root: Path,
    max_files_per_split: int | None,
    subset_seed: int,
) -> tuple[Path, Path, dict[str, object]]:
    positive_inventory = _structure_inventory(positive_dir)
    negative_inventory = _structure_inventory(negative_dir)
    selected_positive = _select_inventory(
        positive_inventory,
        limit=max_files_per_split,
        seed=subset_seed,
        split="positive",
    )
    selected_negative = _select_inventory(
        negative_inventory,
        limit=max_files_per_split,
        seed=subset_seed,
        split="negative",
    )
    positive_archive = archive_root / "base" / "positive"
    negative_archive = archive_root / "base" / "negative"
    archived_positive = _archive_inventory(selected_positive, output_dir=positive_archive)
    archived_negative = _archive_inventory(selected_negative, output_dir=negative_archive)
    state: dict[str, object] = {
        "selection": {
            "algorithm": "numpy.random.Generator(PCG64).choice_without_replacement",
            "ordering_before_sampling": "lexicographic_resolved_path",
            "selected_indices_sorted_after_sampling": True,
            "base_seed": subset_seed,
            "split_seed_derivation": "blake2b(<base_seed>\\0subset-selection:<split>)",
            "limit_per_split": max_files_per_split,
        },
        "positive": {
            "source_root": str(positive_dir.resolve()),
            "full_inventory": positive_inventory,
            "full_inventory_sha256": canonical_json_sha256(positive_inventory),
            "selected_inventory": archived_positive,
            "selected_inventory_sha256": canonical_json_sha256(archived_positive),
            "archive_dir": str(positive_archive.resolve()),
        },
        "negative": {
            "source_root": str(negative_dir.resolve()),
            "full_inventory": negative_inventory,
            "full_inventory_sha256": canonical_json_sha256(negative_inventory),
            "selected_inventory": archived_negative,
            "selected_inventory_sha256": canonical_json_sha256(archived_negative),
            "archive_dir": str(negative_archive.resolve()),
        },
    }
    return positive_archive, negative_archive, state


def _write_effective_truth_manifest(
    *,
    source_state: dict[str, object],
    candidate_truth: tuple[CandidateTruth, ...],
    full_inventory: list[dict[str, object]],
    selected_inventory: list[dict[str, object]],
    destination: Path,
    label: str,
) -> dict[str, object]:
    truth_by_source = {record.source_path: record for record in candidate_truth}
    if len(truth_by_source) != len(candidate_truth):
        raise ValueError(f"{label} truth contains duplicate source_path identities.")
    full_by_source = {
        str(Path(str(item["source_path"])).expanduser().resolve()): item for item in full_inventory
    }
    expected = set(full_by_source)
    supplied = set(truth_by_source)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing[:5]))
        if extra:
            details.append("unexpected " + ", ".join(extra[:5]))
        raise ValueError(
            f"{label} truth manifest must match the complete source-path inventory exactly ("
            + "; ".join(details)
            + ")."
        )
    for source_path, item in full_by_source.items():
        truth = truth_by_source[source_path]
        if truth.filename != Path(source_path).name:
            raise ValueError(
                f"{label} truth filename does not match source_path basename: {source_path}"
            )
        if truth.structure_sha256 != str(item["sha256"]):
            raise ValueError(
                f"{label} truth structure_sha256 does not match source inventory: {source_path}"
            )

    rows: list[dict[str, str]] = []
    for item in sorted(selected_inventory, key=lambda value: str(value["archived_path"])):
        source_path = str(Path(str(item["source_path"])).expanduser().resolve())
        truth = truth_by_source[source_path]
        archived_path = str(Path(str(item["archived_path"])).expanduser().resolve())
        rows.append(
            CandidateTruth(
                filename=Path(archived_path).name,
                source_path=archived_path,
                structure_sha256=str(item["archived_sha256"]),
                target_author_chain_id=truth.target_author_chain_id,
            ).as_row()
        )
    _atomic_write_dataframe_csv(
        pd.DataFrame(rows, columns=list(TRUTH_MANIFEST_COLUMNS)),
        destination,
    )
    effective_state = _truth_manifest_state(destination)
    assert effective_state is not None
    return {
        "policy": "exact_selected_archived_path_hash_identities_from_frozen_complete_manifest",
        "source": source_state,
        "source_row_count": len(candidate_truth),
        "effective": effective_state,
        "effective_row_count": len(rows),
        "effective_rows_sha256": canonical_json_sha256(rows),
        "selected_source_paths": [row["source_path"] for row in rows],
    }


def _prepare_effective_truth_manifests(
    *,
    positive_candidates: tuple[CandidateTruth, ...],
    negative_candidates: tuple[CandidateTruth, ...],
    positive_manifest_state: dict[str, object],
    negative_manifest_state: dict[str, object],
    archive_state: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    effective_positive = output_dir / "positive_manifest.csv"
    effective_negative = output_dir / "negative_manifest.csv"
    positive_archive = archive_state["positive"]
    negative_archive = archive_state["negative"]
    if not isinstance(positive_archive, dict) or not isinstance(negative_archive, dict):
        raise TypeError("Structure archive state has an invalid split schema.")
    positive_state = _write_effective_truth_manifest(
        source_state=positive_manifest_state,
        candidate_truth=positive_candidates,
        full_inventory=positive_archive["full_inventory"],
        selected_inventory=positive_archive["selected_inventory"],
        destination=effective_positive,
        label="Positive",
    )
    negative_state = _write_effective_truth_manifest(
        source_state=negative_manifest_state,
        candidate_truth=negative_candidates,
        full_inventory=negative_archive["full_inventory"],
        selected_inventory=negative_archive["selected_inventory"],
        destination=effective_negative,
        label="Negative",
    )
    return (
        effective_positive,
        effective_negative,
        {
            "policy": "persistent_normalized_manifests_exactly_matching_archived_inputs",
            "positive": positive_state,
            "negative": negative_state,
        },
    )


def _write_generated_truth_manifest(
    *,
    base_truth: tuple[CandidateTruth, ...],
    generated_inventory: list[dict[str, object]],
    destination: Path,
    label: str,
) -> dict[str, object]:
    """Rebind target-chain truth to one persisted perturbed structure inventory."""
    base_by_source = {record.source_path: record for record in base_truth}
    generated_by_source = {
        str(Path(str(item["source_path"])).expanduser().resolve()): item
        for item in generated_inventory
    }
    if set(base_by_source) != set(generated_by_source):
        missing = sorted(set(generated_by_source) - set(base_by_source))
        unexpected = sorted(set(base_by_source) - set(generated_by_source))
        details: list[str] = []
        if missing:
            details.append("missing base truth " + ", ".join(missing[:5]))
        if unexpected:
            details.append("unexpected base truth " + ", ".join(unexpected[:5]))
        raise ValueError(
            f"{label} generated inventory must exactly cover base truth ("
            + "; ".join(details)
            + ")."
        )

    rows: list[dict[str, str]] = []
    for source_path, item in sorted(generated_by_source.items()):
        base_record = base_by_source[source_path]
        if base_record.structure_sha256 != str(item["source_sha256"]):
            raise ValueError(
                f"{label} generated inventory source hash does not match base truth: {source_path}"
            )
        generated_path = str(Path(str(item["generated_path"])).expanduser().resolve())
        rows.append(
            CandidateTruth(
                filename=Path(generated_path).name,
                source_path=generated_path,
                structure_sha256=str(item["generated_sha256"]),
                target_author_chain_id=base_record.target_author_chain_id,
            ).as_row()
        )
    _atomic_write_dataframe_csv(
        pd.DataFrame(rows, columns=list(TRUTH_MANIFEST_COLUMNS)),
        destination,
    )
    state = _truth_manifest_state(destination)
    assert state is not None
    return {
        "policy": "exact_generated_path_hash_identity_rebinding",
        "manifest": state,
        "row_count": len(rows),
        "rows_sha256": canonical_json_sha256(rows),
    }


def _prepare_generated_truth_manifests(
    *,
    base_positive_manifest: Path,
    base_negative_manifest: Path,
    evaluated_inputs: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path, dict[str, object]]:
    from cooper_beta.evaluation.runner import load_negative_manifest, load_positive_manifest

    positive_inventory = evaluated_inputs.get("positive_inventory")
    negative_inventory = evaluated_inputs.get("negative_inventory")
    if not isinstance(positive_inventory, list) or not isinstance(negative_inventory, list):
        raise TypeError("Generated perturbation inventory has an invalid split schema.")
    output_dir.mkdir(parents=True, exist_ok=False)
    positive_path = output_dir / "positive_manifest.csv"
    negative_path = output_dir / "negative_manifest.csv"
    positive_state = _write_generated_truth_manifest(
        base_truth=load_positive_manifest(base_positive_manifest),
        generated_inventory=positive_inventory,
        destination=positive_path,
        label="Positive",
    )
    negative_state = _write_generated_truth_manifest(
        base_truth=load_negative_manifest(base_negative_manifest),
        generated_inventory=negative_inventory,
        destination=negative_path,
        label="Negative",
    )
    return (
        positive_path,
        negative_path,
        {
            "policy": "per_experiment_generated_path_hash_truth",
            "positive": positive_state,
            "negative": negative_state,
        },
    )


def _stable_seed(base_seed: int, relative_name: str) -> int:
    payload = f"{int(base_seed)}\0{relative_name}".encode()
    digest = hashlib.blake2b(payload, digest_size=STABLE_SEED_DIGEST_BYTES).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % RNG_SEED_MODULUS


_MMCIF_COORDINATE_COLUMNS = (
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
)
_MMCIF_RESIDUE_IDENTITY_COLUMNS = (
    "_atom_site.group_PDB",
    "_atom_site.label_comp_id",
    "_atom_site.label_asym_id",
    "_atom_site.label_entity_id",
    "_atom_site.label_seq_id",
    "_atom_site.pdbx_PDB_ins_code",
    "_atom_site.auth_comp_id",
    "_atom_site.auth_asym_id",
    "_atom_site.auth_seq_id",
    "_atom_site.pdbx_PDB_model_num",
)


def _normalized_mmcif_value(value: object) -> str | list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def _mmcif_column(
    document: dict[str, object],
    name: str,
    *,
    expected_length: int | None = None,
) -> list[str]:
    if name not in document:
        raise ValueError(f"mmCIF input is missing required column {name!r}.")
    value = document[name]
    values = [str(item) for item in value] if isinstance(value, (list, tuple)) else [str(value)]
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"mmCIF column {name!r} has {len(values)} rows; expected {expected_length}."
        )
    return values


def _finite_coordinate_matrix(document: dict[str, object]) -> FloatArray:
    columns = [_mmcif_column(document, name) for name in _MMCIF_COORDINATE_COLUMNS]
    lengths = {len(column) for column in columns}
    if len(lengths) != 1 or not columns[0]:
        raise ValueError("mmCIF coordinate columns must have one equal, non-zero row count.")
    try:
        coordinates = np.asarray(columns, dtype=float).T
    except ValueError as exc:
        raise ValueError("mmCIF Cartesian coordinates must be numeric.") from exc
    if coordinates.shape[1:] != (NOISE_VECTOR_DIMENSIONS,) or not np.all(np.isfinite(coordinates)):
        raise ValueError("mmCIF Cartesian coordinates must be finite three-vectors.")
    return coordinates


def _mmcif_identity_state(document: dict[str, object]) -> dict[str, object]:
    coordinates = _finite_coordinate_matrix(document)
    atom_count = int(coordinates.shape[0])
    non_coordinate_document = {
        key: _normalized_mmcif_value(document[key])
        for key in sorted(document)
        if key not in _MMCIF_COORDINATE_COLUMNS
    }
    atom_identity = {
        key: value
        for key, value in non_coordinate_document.items()
        if key.startswith("_atom_site.")
    }
    if not atom_identity:
        raise ValueError("mmCIF input has no non-coordinate `_atom_site` identity columns.")

    residue_columns: dict[str, list[str]] = {}
    for name in _MMCIF_RESIDUE_IDENTITY_COLUMNS:
        if name in document:
            residue_columns[name] = _mmcif_column(
                document,
                name,
                expected_length=atom_count,
            )
    if not residue_columns:
        raise ValueError("mmCIF input has no residue-identity columns.")
    residue_inventory: list[list[str]] = []
    seen_residues: set[tuple[str, ...]] = set()
    for row_index in range(atom_count):
        identity = tuple(residue_columns[name][row_index] for name in residue_columns)
        if identity not in seen_residues:
            seen_residues.add(identity)
            residue_inventory.append(list(identity))

    return {
        "atom_count": atom_count,
        "residue_count": len(residue_inventory),
        "non_coordinate_category_sha256": canonical_json_sha256(non_coordinate_document),
        "atom_identity_sha256": canonical_json_sha256(atom_identity),
        "residue_inventory_sha256": canonical_json_sha256(
            {
                "columns": list(residue_columns),
                "rows": residue_inventory,
            }
        ),
        "entity_poly_present": any(key.startswith("_entity_poly.") for key in document),
    }


_PDB_COORDINATE_RECORDS = frozenset({b"ATOM  ", b"HETATM"})
_PDB_XYZ_PLACEHOLDER = b"<PDB_XYZ_COLUMNS_31_TO_54>"


def _pdb_coordinate_state(
    payload: bytes,
) -> tuple[list[bytes], FloatArray, tuple[str, ...], bytes]:
    """Parse fixed-column PDB coordinates while retaining exact non-XYZ bytes."""

    lines = payload.splitlines(keepends=True)
    coordinates: list[tuple[float, float, float]] = []
    atom_names: list[str] = []
    identity_lines: list[bytes] = []
    for line_number, line in enumerate(lines, start=1):
        if line[:6] not in _PDB_COORDINATE_RECORDS:
            identity_lines.append(line)
            continue
        if len(line) < 54:
            raise ValueError(
                f"PDB coordinate record at line {line_number} is shorter than 54 columns."
            )
        try:
            coordinate = (
                float(line[30:38].decode("ascii")),
                float(line[38:46].decode("ascii")),
                float(line[46:54].decode("ascii")),
            )
            atom_name = line[12:16].decode("ascii").strip().upper()
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"PDB coordinate record at line {line_number} has invalid fixed-column data."
            ) from exc
        if not atom_name or not all(np.isfinite(value) for value in coordinate):
            raise ValueError(
                f"PDB coordinate record at line {line_number} has an invalid atom or coordinate."
            )
        coordinates.append(coordinate)
        atom_names.append(atom_name)
        identity_lines.append(line[:30] + _PDB_XYZ_PLACEHOLDER + line[54:])
    if not coordinates:
        raise ValueError("PDB input contains no ATOM/HETATM coordinate records.")
    return (
        lines,
        np.asarray(coordinates, dtype=float),
        tuple(atom_names),
        b"".join(identity_lines),
    )


def _format_pdb_coordinate(value: float, *, line_number: int) -> bytes:
    field = f"{float(value):8.3f}"
    if len(field) != 8:
        raise ValueError(
            f"Perturbed PDB coordinate at line {line_number} does not fit the 8.3 field."
        )
    return field.encode("ascii")


def _pdb_payload_with_coordinates(lines: list[bytes], coordinates: FloatArray) -> bytes:
    output_lines: list[bytes] = []
    coordinate_index = 0
    for line_number, line in enumerate(lines, start=1):
        if line[:6] not in _PDB_COORDINATE_RECORDS:
            output_lines.append(line)
            continue
        coordinate = coordinates[coordinate_index]
        coordinate_index += 1
        xyz = b"".join(
            _format_pdb_coordinate(float(value), line_number=line_number) for value in coordinate
        )
        output_lines.append(line[:30] + xyz + line[54:])
    if coordinate_index != int(coordinates.shape[0]):
        raise RuntimeError("PDB coordinate replacement did not consume every coordinate row.")
    return b"".join(output_lines)


def _complete_sequence_identity(path: Path) -> list[dict[str, object]]:
    return [
        {
            "author_chain_id": record.author_chain_id,
            "label_asym_id": record.label_asym_id,
            "entity_id": record.entity_id,
            "sequence": record.sequence,
            "monomer_ids": list(record.monomer_ids),
            "sequence_source": record.sequence_source,
        }
        for record in declared_polymer_sequences(path)
    ]


def _atomic_save_structure(destination: Path, save: Callable[[str], object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        save(temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)


def _delta_summary(
    before: FloatArray, after: FloatArray, selected: BooleanArray
) -> dict[str, object]:
    if before.shape != after.shape:
        raise RuntimeError(
            f"Coordinate row count changed during perturbation: {before.shape} != {after.shape}."
        )
    deltas = np.linalg.norm(after - before, axis=1)
    selected_deltas = deltas[selected]
    if not np.all(np.isfinite(deltas)):
        raise RuntimeError("Perturbation produced non-finite realized coordinate deltas.")
    if selected_deltas.size:
        minimum = float(np.min(selected_deltas))
        maximum = float(np.max(selected_deltas))
        mean = float(np.mean(selected_deltas))
        root_mean_square = float(np.sqrt(np.mean(np.square(selected_deltas))))
        total = float(np.sum(selected_deltas))
        total_squares = float(np.sum(np.square(selected_deltas)))
    else:
        minimum = maximum = mean = root_mean_square = total = total_squares = 0.0
    return {
        "total_atom_count": int(deltas.size),
        "selected_atom_count": int(selected_deltas.size),
        "unselected_atom_count": int(deltas.size - selected_deltas.size),
        "selected_changed_atom_count": int(np.count_nonzero(selected_deltas)),
        "unselected_changed_atom_count": int(np.count_nonzero(deltas[~selected])),
        "selected_delta_angstrom": {
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "rms": root_mean_square,
            "sum": total,
            "sum_squares": total_squares,
        },
    }


def _perturb_structure_file(
    source: Path,
    destination: Path,
    *,
    sigma: float,
    seed: int,
    relative_name: str,
    atoms: str,
) -> PerturbationDetails:
    if atoms not in {"ca", "all"}:
        raise ValueError("atoms must be one of: ca, all.")
    if not np.isfinite(float(sigma)) or float(sigma) < 0.0:
        raise ValueError("sigma must be finite and >= 0.")
    rng = np.random.Generator(np.random.PCG64(_stable_seed(seed, relative_name)))
    suffix = source.suffix.lower()
    identity_invariants: PerturbationIdentityInvariants
    if suffix in {".cif", ".mmcif"}:
        document = dict(MMCIF2Dict(str(source)))
        before = _finite_coordinate_matrix(document)
        mmcif_atom_names: list[str] | None = None
        for name_column in ("_atom_site.label_atom_id", "_atom_site.auth_atom_id"):
            if name_column in document:
                mmcif_atom_names = _mmcif_column(
                    document,
                    name_column,
                    expected_length=before.shape[0],
                )
                break
        if mmcif_atom_names is None:
            raise ValueError("mmCIF input has no atom-name identity column.")
        selected = np.asarray(
            [atoms == "all" or name.strip().upper() == "CA" for name in mmcif_atom_names],
            dtype=bool,
        )
        source_identity = _mmcif_identity_state(document)
        perturbed = before.copy()
        perturbed[selected] += rng.normal(
            loc=NOISE_MEAN_ANGSTROM,
            scale=float(sigma),
            size=(int(np.count_nonzero(selected)), NOISE_VECTOR_DIMENSIONS),
        )
        for column_index, name in enumerate(_MMCIF_COORDINATE_COLUMNS):
            document[name] = [format(float(value), ".17g") for value in perturbed[:, column_index]]
        writer = MMCIFIO()
        writer.set_dict(document)
        _atomic_save_structure(destination, writer.save)
        generated_document = dict(MMCIF2Dict(str(destination)))
        generated_identity = _mmcif_identity_state(generated_document)
        after = _finite_coordinate_matrix(generated_document)
        preserved = source_identity == generated_identity
        if not preserved:
            raise RuntimeError(
                "mmCIF identity changed during coordinate perturbation; refusing the generated "
                "structure."
            )
        identity_invariants = {
            "policy": "all_mmcif_fields_except_atom_site_Cartn_xyz_semantically_identical",
            "preserved": True,
            "source": source_identity,
            "generated": generated_identity,
        }
        format_name = "mmcif"
    else:
        source_payload = source.read_bytes()
        source_lines, before, pdb_atom_names, source_non_xyz_payload = _pdb_coordinate_state(
            source_payload
        )
        source_complete_sequences = _complete_sequence_identity(source)
        selected = np.asarray(
            [atoms == "all" or atom_name == "CA" for atom_name in pdb_atom_names],
            dtype=bool,
        )
        perturbed = before.copy()
        perturbed[selected] += rng.normal(
            loc=NOISE_MEAN_ANGSTROM,
            scale=float(sigma),
            size=(int(np.count_nonzero(selected)), NOISE_VECTOR_DIMENSIONS),
        )
        generated_payload = _pdb_payload_with_coordinates(source_lines, perturbed)

        def save_pdb_payload(path: str) -> None:
            with open(path, "wb") as handle:
                handle.write(generated_payload)

        _atomic_save_structure(destination, save_pdb_payload)
        _, after, generated_atom_names, generated_non_xyz_payload = _pdb_coordinate_state(
            destination.read_bytes()
        )
        generated_complete_sequences = _complete_sequence_identity(destination)
        if generated_atom_names != pdb_atom_names:
            raise RuntimeError("PDB atom-name order changed during coordinate perturbation.")
        non_xyz_bytes_preserved = generated_non_xyz_payload == source_non_xyz_payload
        complete_sequences_preserved = generated_complete_sequences == source_complete_sequences
        if not non_xyz_bytes_preserved or not complete_sequences_preserved:
            raise RuntimeError(
                "PDB non-coordinate records or complete polymer declarations changed during "
                "coordinate perturbation."
            )
        source_identity = {
            "atom_count": int(before.shape[0]),
            "complete_polymer_sequences": source_complete_sequences,
        }
        generated_identity = {
            "atom_count": int(after.shape[0]),
            "complete_polymer_sequences": generated_complete_sequences,
        }
        preserved = source_identity == generated_identity
        if not preserved:
            raise RuntimeError(
                "PDB atom count or complete sequence identity changed during perturbation."
            )
        identity_invariants = {
            "policy": (
                "all_bytes_except_atom_hetatm_xyz_columns_exact_and_complete_sequences_equal"
            ),
            "preserved": True,
            "non_coordinate_bytes_exact": non_xyz_bytes_preserved,
            "complete_polymer_sequences_equal": complete_sequences_preserved,
            "source": source_identity,
            "generated": generated_identity,
        }
        format_name = "pdb"

    return {
        "format": format_name,
        "requested_sigma_angstrom": float(sigma),
        "atoms_policy": atoms,
        "realized_delta": _delta_summary(before, after, selected),
        "identity_invariants": identity_invariants,
    }


def _write_perturbed_split(
    source_dir: Path,
    output_dir: Path,
    *,
    sigma: float,
    seed: int,
    atoms: str,
) -> list[PerturbedStructureRecord]:
    source_dir = source_dir.resolve()
    files = _discover_structures(source_dir)
    if not files:
        raise ValueError(f"No structure files found in {source_dir}")

    generated: list[PerturbedStructureRecord] = []
    for index, source in enumerate(files):
        if source_dir.is_file():
            relative = source.name
        else:
            try:
                relative = str(source.resolve().relative_to(source_dir))
            except ValueError:
                relative = source.name
        destination = output_dir / relative
        if destination.exists():
            destination = output_dir / f"{index:05d}_{source.name}"
        source_digest_before = file_sha256(source)
        perturbation = _perturb_structure_file(
            source,
            destination,
            sigma=sigma,
            seed=seed,
            relative_name=relative,
            atoms=atoms,
        )
        source_digest_after = file_sha256(source)
        if source_digest_after != source_digest_before:
            raise RuntimeError(f"Structure changed while it was being perturbed: {source}")
        generated.append(
            {
                "source_path": str(source.resolve()),
                "source_sha256": source_digest_before,
                "relative_path": relative,
                "generated_path": str(destination.resolve()),
                "generated_size": int(destination.stat().st_size),
                "generated_sha256": file_sha256(destination),
                "perturbation": perturbation,
            }
        )
    return generated


def _noise_input_dirs(
    positive_dir: Path,
    negative_dir: Path,
    *,
    sigma: float,
    seed: int,
    atoms: str,
    archive_root: Path,
    experiment_name: str,
) -> tuple[Path, Path, dict[str, object]]:
    root = archive_root / "noise" / experiment_name
    root.mkdir(parents=True, exist_ok=False)
    perturbed_positive = root / "positive"
    perturbed_negative = root / "negative"
    positive_inventory = _write_perturbed_split(
        positive_dir,
        perturbed_positive,
        sigma=sigma,
        seed=seed,
        atoms=atoms,
    )
    negative_inventory = _write_perturbed_split(
        negative_dir,
        perturbed_negative,
        sigma=sigma,
        seed=seed,
        atoms=atoms,
    )
    return (
        perturbed_positive,
        perturbed_negative,
        {
            "kind": "persistent_generated_noise_archive",
            "positive_dir": str(perturbed_positive.resolve()),
            "negative_dir": str(perturbed_negative.resolve()),
            "positive_inventory": positive_inventory,
            "positive_inventory_sha256": canonical_json_sha256(positive_inventory),
            "negative_inventory": negative_inventory,
            "negative_inventory_sha256": canonical_json_sha256(negative_inventory),
            "identity_invariants": {
                "policy": "every_generated_structure_preserves_atom_and_residue_identity",
                "all_preserved": all(
                    bool(item["perturbation"]["identity_invariants"]["preserved"])
                    for item in [*positive_inventory, *negative_inventory]
                ),
            },
        },
    )


def _metric_text(row: dict[str, object], level: str, metric: str) -> str:
    defined_key = f"{level}_{metric}_defined"
    if defined_key in row and not bool(row[defined_key]):
        return "undefined"
    value = row.get(f"{level}_{metric}")
    if value is None:
        return "unavailable"
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"Metric {level}_{metric} must be numeric when defined")
    return f"{float(value):.4f}"


def _summarize_row(row: dict[str, object]) -> str:
    parts: list[str] = []
    if bool(row.get("chain_metrics_computed")) or "chain_TP" in row:
        parts.append(
            "chain: "
            f"R={_metric_text(row, 'chain', 'recall')} "
            f"P={_metric_text(row, 'chain', 'precision')} "
            f"F1={_metric_text(row, 'chain', 'f1')} "
            f"MCC={_metric_text(row, 'chain', 'mcc')}"
        )
    if bool(row.get("file_metrics_computed")) or "file_TP" in row:
        parts.append(
            "file: "
            f"R={_metric_text(row, 'file', 'recall')} "
            f"P={_metric_text(row, 'file', 'precision')} "
            f"F1={_metric_text(row, 'file', 'f1')} "
            f"MCC={_metric_text(row, 'file', 'mcc')}"
        )
    return " | ".join(parts) if parts else "metrics unavailable"


def _ordered_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "exp",
        "perturbation_mode",
        "coordinate_noise_sigma",
        "coordinate_noise_seed",
        "coordinate_noise_atoms",
        "chain_recall",
        "chain_precision",
        "chain_specificity",
        "chain_f1",
        "chain_balanced_accuracy",
        "chain_mcc",
        "chain_TP",
        "chain_FP",
        "chain_TN",
        "chain_FN",
        "file_recall",
        "file_precision",
        "file_specificity",
        "file_f1",
        "file_balanced_accuracy",
        "file_mcc",
        "file_TP",
        "file_FP",
        "file_TN",
        "file_FN",
        "chain_accuracy",
        "file_accuracy",
        "metric_error_policy",
        "chain_csv",
        "file_csv",
    ]
    ordered_columns = [column for column in preferred_columns if column in dataframe.columns]
    ordered_columns += [column for column in dataframe.columns if column not in ordered_columns]
    return dataframe[ordered_columns]


def run_perturbation_suite(
    *,
    positive_dir: Path,
    negative_dir: Path,
    workers: int | None,
    prepare_workers: int | None,
    save_dir: Path,
    metric_level: str,
    noise_sigmas: list[float],
    noise_seeds: list[int],
    noise_atoms: str,
    max_files_per_split: int | None,
    subset_seed: int = DEFAULT_SUBSET_SEED,
    positive_manifest: Path | None = None,
    negative_manifest: Path | None = None,
    metric_error_policy: str = DEFAULT_METRIC_ERROR_POLICY,
) -> Path:
    from cooper_beta.evaluation.runner import evaluate

    if pd is None:
        raise RuntimeError("pandas is required (pip install 'cooper-beta[eval]').")

    timestamp = _utc_run_token()
    resolved_save_dir = save_dir.expanduser().resolve()
    resolved_save_dir.mkdir(parents=True, exist_ok=True)
    output_dir = resolved_save_dir / f"perturbation_{timestamp}"
    # The timestamp is part of every downstream tag. Reusing this directory would
    # therefore conflate experiments, so even an empty pre-existing directory is fatal.
    output_dir.mkdir(parents=False, exist_ok=False)
    suite_manifest_path = output_dir / "perturbation_suite_manifest.json"
    script_path = Path(__file__).resolve()
    script_digest = file_sha256(script_path)
    summary_path = output_dir / f"perturbation_summary_{timestamp}.csv"
    resolved_positive_dir = positive_dir.expanduser().resolve()
    resolved_negative_dir = negative_dir.expanduser().resolve()
    suite_manifest: PerturbationSuiteManifest = {
        "schema_version": SUITE_MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "phase": "initialization",
        "started_at_utc": _utc_now(),
        "run_token_utc": timestamp,
        "output_dir": str(output_dir),
        "script": {
            "path": str(script_path),
            "sha256": script_digest,
        },
        "software": {
            "cooper_beta": {
                "source_version": source_package_version,
                "installed_distribution_version": _package_version("cooper-beta"),
            },
            "python": sys.version,
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "biopython": _package_version("biopython"),
        },
        "parameters": {
            "positive_dir": str(resolved_positive_dir),
            "negative_dir": str(resolved_negative_dir),
            "workers_supplied": repr(workers),
            "prepare_workers_supplied": repr(prepare_workers),
            "save_dir": str(resolved_save_dir),
            "metric_level": metric_level,
            "metric_error_policy": metric_error_policy,
            "noise_sigmas_angstrom_supplied": [repr(value) for value in noise_sigmas],
            "noise_seeds_supplied": [repr(value) for value in noise_seeds],
            "noise_atoms": noise_atoms,
            "max_files_per_split_supplied": repr(max_files_per_split),
            "subset_seed_supplied": repr(subset_seed),
            "positive_manifest": (
                str(positive_manifest.expanduser().resolve())
                if positive_manifest is not None
                else None
            ),
            "negative_manifest": (
                str(negative_manifest.expanduser().resolve())
                if negative_manifest is not None
                else None
            ),
            "named_defaults": {
                "save_dir": DEFAULT_SAVE_DIR,
                "metric_level": DEFAULT_METRIC_LEVEL,
                "metric_error_policy": DEFAULT_METRIC_ERROR_POLICY,
                "noise_sigmas": DEFAULT_NOISE_SIGMAS,
                "noise_seeds": DEFAULT_NOISE_SEEDS,
                "noise_atoms": DEFAULT_NOISE_ATOMS,
                "workers": DEFAULT_WORKERS,
                "prepare_workers": DEFAULT_PREPARE_WORKERS,
                "max_files_per_split": DEFAULT_MAX_FILES_PER_SPLIT,
                "subset_seed": DEFAULT_SUBSET_SEED,
                "positive_manifest": DEFAULT_POSITIVE_MANIFEST,
                "negative_manifest": DEFAULT_NEGATIVE_MANIFEST,
            },
            "randomness": {
                "generator": "numpy.random.Generator",
                "bit_generator": RNG_BIT_GENERATOR,
                "seed_derivation_hash": "blake2b",
                "seed_derivation_digest_bytes": STABLE_SEED_DIGEST_BYTES,
                "seed_derivation_payload": "<base_seed>\\0<relative_structure_path>",
                "seed_modulus": RNG_SEED_MODULUS,
                "distribution": "independent_normal_coordinate_offsets",
                "distribution_mean_angstrom": NOISE_MEAN_ANGSTROM,
                "coordinate_vector_dimensions": NOISE_VECTOR_DIMENSIONS,
            },
            "accepted_structure_suffixes": sorted(STRUCTURE_SUFFIXES),
        },
        "inputs": {
            "positive_truth_manifest": None,
            "negative_truth_manifest": None,
            "structure_archives": None,
        },
        "metric_sampling": {
            "file": "one_directory_labeled_structure_file_any_chain_prediction",
            "chain": "unresolved_until_validation",
        },
        "artifact_policy": {
            "evaluated_structure_retention": EVALUATED_STRUCTURE_RETENTION_POLICY,
        },
        "experiments": [],
        "outputs": {
            "suite_manifest": str(suite_manifest_path),
            "summary_csv": str(summary_path),
            "summary_csv_sha256": None,
        },
    }
    atomic_write_json(suite_manifest_path, suite_manifest)

    phase = "validation"
    current_experiment: str | None = None
    try:
        suite_manifest["phase"] = phase
        workers = _strict_optional_positive_int(workers, label="workers")
        prepare_workers = _strict_optional_positive_int(prepare_workers, label="prepare_workers")
        max_files_per_split = _strict_optional_positive_int(
            max_files_per_split, label="max_files_per_split"
        )
        subset_seed = _strict_int(subset_seed, label="subset_seed")
        if metric_level not in {"chain", "file", "both"}:
            raise ValueError("metric_level must be one of: chain, file, both.")
        if metric_error_policy not in {"strict", "exclude"}:
            raise ValueError("metric_error_policy must be one of: strict, exclude.")
        if noise_atoms not in {"ca", "all"}:
            raise ValueError("noise_atoms must be one of: ca, all.")
        noise_sigmas = _strict_float_values(
            noise_sigmas,
            label="noise_sigmas",
            allow_zero=True,
        )
        noise_seeds = _strict_int_values(noise_seeds, label="noise_seeds")
        planned_experiments = [
            f"noise_sigma_{_level_token(sigma)}_seed_{seed}"
            for seed in noise_seeds
            for sigma in noise_sigmas
        ]
        if len(planned_experiments) != len(set(planned_experiments)):
            raise ValueError(
                "Perturbation values produce duplicate experiment tags at the public "
                "filename precision; choose more widely separated values."
            )
        if (positive_manifest is None) != (negative_manifest is None):
            raise ValueError(
                "positive_manifest and negative_manifest must either both be provided or both "
                "be omitted."
            )
        if metric_level in {"chain", "both"} and positive_manifest is None:
            raise ValueError(
                "Chain-level metrics require both positive_manifest and negative_manifest so "
                "positive and negative observations use the same target-chain sampling unit."
            )
        suite_manifest["metric_sampling"] = {
            "file": "one_directory_labeled_structure_file_any_chain_prediction",
            "chain": (
                "one_manifest_target_chain_per_positive_and_negative_file"
                if metric_level in {"chain", "both"}
                else "not_requested"
            ),
            "all_negative_detector_chains_allowed": False,
        }
        positive_candidates: tuple[CandidateTruth, ...] | None = None
        negative_candidates: tuple[CandidateTruth, ...] | None = None
        if positive_manifest is not None:
            positive_candidates, positive_manifest_state = _freeze_truth_manifest(
                positive_manifest,
                label="positive",
            )
        else:
            positive_manifest_state = None
        suite_manifest["inputs"] = {
            "positive_truth_manifest": positive_manifest_state,
            "negative_truth_manifest": None,
            "structure_archives": None,
        }
        atomic_write_json(suite_manifest_path, suite_manifest)
        if negative_manifest is not None:
            negative_candidates, negative_manifest_state = _freeze_truth_manifest(
                negative_manifest,
                label="negative",
            )
        else:
            negative_manifest_state = None
        suite_manifest["inputs"] = {
            "positive_truth_manifest": positive_manifest_state,
            "negative_truth_manifest": negative_manifest_state,
            "structure_archives": None,
        }
        suite_manifest["parameters"].update(
            {
                "workers": workers,
                "prepare_workers": prepare_workers,
                "noise_sigmas_angstrom": noise_sigmas,
                "noise_seeds": noise_seeds,
                "max_files_per_split": max_files_per_split,
                "subset_seed": subset_seed,
                "planned_experiments": planned_experiments,
            }
        )
        atomic_write_json(suite_manifest_path, suite_manifest)

        phase = "archive_inputs"
        suite_manifest["phase"] = phase
        atomic_write_json(suite_manifest_path, suite_manifest)
        archive_root = output_dir / "evaluated_structures"
        base_positive_dir, base_negative_dir, archive_state = _prepare_persistent_base_inputs(
            resolved_positive_dir,
            resolved_negative_dir,
            archive_root=archive_root,
            max_files_per_split=max_files_per_split,
            subset_seed=subset_seed,
        )
        suite_manifest["inputs"]["structure_archives"] = archive_state
        effective_positive_manifest = positive_manifest
        effective_negative_manifest = negative_manifest
        if positive_manifest is not None and negative_manifest is not None:
            assert positive_manifest_state is not None
            assert negative_manifest_state is not None
            assert positive_candidates is not None
            assert negative_candidates is not None
            (
                effective_positive_manifest,
                effective_negative_manifest,
                effective_manifest_state,
            ) = _prepare_effective_truth_manifests(
                positive_candidates=positive_candidates,
                negative_candidates=negative_candidates,
                positive_manifest_state=positive_manifest_state,
                negative_manifest_state=negative_manifest_state,
                archive_state=archive_state,
                output_dir=output_dir / "effective_truth_manifests",
            )
            suite_manifest["inputs"]["effective_truth_manifests"] = effective_manifest_state
        else:
            suite_manifest["inputs"]["effective_truth_manifests"] = None
        atomic_write_json(suite_manifest_path, suite_manifest)

        rows: list[dict[str, object]] = []
        print("\n=== Perturbation suite ===")
        print(f"Output dir: {output_dir}\n")

        for seed in noise_seeds:
            for sigma in noise_sigmas:
                exp = f"noise_sigma_{_level_token(sigma)}_seed_{seed}"
                current_experiment = exp
                phase = "evaluation"
                suite_manifest["phase"] = phase
                suite_manifest["current_experiment"] = exp
                atomic_write_json(suite_manifest_path, suite_manifest)
                print(f"[{exp}] atoms={noise_atoms}")
                run_positive_dir, run_negative_dir, evaluated_inputs = _noise_input_dirs(
                    base_positive_dir,
                    base_negative_dir,
                    sigma=sigma,
                    seed=seed,
                    atoms=noise_atoms,
                    archive_root=archive_root,
                    experiment_name=exp,
                )
                run_positive_manifest = effective_positive_manifest
                run_negative_manifest = effective_negative_manifest
                generated_truth_state: dict[str, object] | None = None
                if (
                    effective_positive_manifest is not None
                    and effective_negative_manifest is not None
                ):
                    (
                        run_positive_manifest,
                        run_negative_manifest,
                        generated_truth_state,
                    ) = _prepare_generated_truth_manifests(
                        base_positive_manifest=effective_positive_manifest,
                        base_negative_manifest=effective_negative_manifest,
                        evaluated_inputs=evaluated_inputs,
                        output_dir=(output_dir / "effective_truth_manifests" / "noise" / exp),
                    )
                row = evaluate(
                    true_dir=run_positive_dir,
                    false_dir=run_negative_dir,
                    workers=workers,
                    prepare_workers=prepare_workers,
                    save_dir=output_dir,
                    metric_level=metric_level,
                    tag=f"{timestamp}_{exp}",
                    detector_overrides=None,
                    print_metric_tables=False,
                    metric_error_policy=metric_error_policy,
                    positive_manifest=run_positive_manifest,
                    negative_manifest=run_negative_manifest,
                )
                row.update(
                    {
                        "exp": exp,
                        "perturbation_mode": "coordinate_noise",
                        "coordinate_noise_sigma": sigma,
                        "coordinate_noise_seed": seed,
                        "coordinate_noise_atoms": noise_atoms,
                    }
                )
                _require_finite_row(row, experiment=exp)
                evaluation_manifest = Path(str(row["evaluation_manifest"])).resolve()
                if not evaluation_manifest.is_file():
                    raise RuntimeError(
                        f"Evaluation did not produce its manifest: {evaluation_manifest}"
                    )
                suite_manifest["experiments"].append(
                    {
                        "name": exp,
                        "mode": "coordinate_noise",
                        "coordinate_noise_sigma_angstrom": sigma,
                        "coordinate_noise_seed": seed,
                        "coordinate_noise_atoms": noise_atoms,
                        "evaluated_inputs": evaluated_inputs,
                        "effective_truth_manifests": generated_truth_state,
                        "evaluation_manifest": str(evaluation_manifest),
                        "evaluation_manifest_sha256": file_sha256(evaluation_manifest),
                    }
                )
                rows.append(row)
                atomic_write_json(suite_manifest_path, suite_manifest)
                print("  " + _summarize_row(row) + "\n")

        phase = "summary"
        current_experiment = None
        suite_manifest["phase"] = phase
        suite_manifest.pop("current_experiment", None)
        atomic_write_json(suite_manifest_path, suite_manifest)
        dataframe = _ordered_summary(pd.DataFrame(rows))
        _atomic_write_dataframe_csv(dataframe, summary_path)
        suite_manifest["outputs"] = {
            "suite_manifest": str(suite_manifest_path),
            "summary_csv": str(summary_path),
            "summary_csv_sha256": file_sha256(summary_path),
        }

        if file_sha256(script_path) != script_digest:
            raise RuntimeError("Perturbation script changed while the suite was running.")
        suite_manifest["status"] = "complete"
        suite_manifest["phase"] = "complete"
        suite_manifest["completed_at_utc"] = _utc_now()
        suite_manifest["experiment_count"] = len(rows)
        atomic_write_json(suite_manifest_path, suite_manifest)
    except Exception as exc:
        suite_manifest["status"] = "failed"
        suite_manifest["phase"] = phase
        suite_manifest["failed_at_utc"] = _utc_now()
        suite_manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if current_experiment is not None:
            suite_manifest["current_experiment"] = current_experiment
        if summary_path.is_file():
            suite_manifest["outputs"] = {
                "suite_manifest": str(suite_manifest_path),
                "summary_csv": str(summary_path),
                "summary_csv_sha256": file_sha256(summary_path),
            }
        atomic_write_json(suite_manifest_path, suite_manifest)
        raise

    display_columns = [
        column
        for column in [
            "exp",
            "perturbation_mode",
            "coordinate_noise_sigma",
            "coordinate_noise_seed",
            "chain_f1",
            "chain_mcc",
            "file_f1",
            "file_mcc",
        ]
        if column in dataframe.columns
    ]
    print("=== Perturbation summary ===")
    if display_columns:
        print(dataframe[display_columns].to_string(index=False))
    print(f"\nSaved: {summary_path}\nManifest: {suite_manifest_path}\n")
    return summary_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/perturbation_eval.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Measure Cooper-Beta sensitivity to coordinate noise. "
            "Each evaluated structure and its paired truth selection are saved in the suite output."
        ),
        epilog=(
            "Output: a suite manifest, archived perturbation structures, per-run detector "
            "artifacts, and perturbation_summary.csv. Sigma values use Angstroms. "
            "Invalid inputs, policy violations, or failed detector runs exit with status 2."
        ),
    )
    parser.add_argument(
        "--positives",
        "--true",
        dest="true",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing structures labeled positive.",
    )
    parser.add_argument(
        "--negatives",
        "--false",
        dest="false",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing structures labeled negative.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help="Analysis worker processes; omit to use the detector configuration.",
    )
    parser.add_argument(
        "--prepare",
        "--prepare-workers",
        "--prep",
        type=int,
        default=DEFAULT_PREPARE_WORKERS,
        metavar="N",
        help="Preparation worker processes; omit to follow the resolved analysis count.",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        metavar="DIRECTORY",
        help="New output directory for the complete perturbation suite.",
    )
    parser.add_argument(
        "--metric-level",
        choices=["chain", "file", "both"],
        default=DEFAULT_METRIC_LEVEL,
        help="Metric granularity to compute.",
    )
    parser.add_argument(
        "--positive-manifest",
        default=DEFAULT_POSITIVE_MANIFEST,
        metavar="CSV",
        help=(
            "Canonical positive target-chain CSV with exactly filename, source_path, "
            "structure_sha256, target_author_chain_id. Required together with the "
            "negative manifest for chain/both metrics."
        ),
    )
    parser.add_argument(
        "--negative-manifest",
        default=DEFAULT_NEGATIVE_MANIFEST,
        metavar="CSV",
        help=(
            "Canonical negative target-chain CSV with exactly filename, source_path, "
            "structure_sha256, target_author_chain_id. Required together with the "
            "positive manifest for chain/both metrics."
        ),
    )
    parser.add_argument(
        "--metric-error-policy",
        choices=["strict", "exclude"],
        default=DEFAULT_METRIC_ERROR_POLICY,
        help="Treatment of detector ERROR observations: strict stops; exclude reports coverage.",
    )
    parser.add_argument(
        "--noise-sigmas",
        default=DEFAULT_NOISE_SIGMAS,
        metavar="A1,A2,...",
        help="Comma-separated coordinate-noise standard deviations in Angstroms.",
    )
    parser.add_argument(
        "--noise-seeds",
        default=DEFAULT_NOISE_SEEDS,
        metavar="S1,S2,...",
        help="Comma-separated random seeds, one complete replicate per seed and sigma.",
    )
    parser.add_argument(
        "--noise-atoms",
        choices=["ca", "all"],
        default=DEFAULT_NOISE_ATOMS,
        help="Perturb only alpha carbons (ca) or every coordinate atom (all).",
    )
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=DEFAULT_MAX_FILES_PER_SPLIT,
        metavar="N",
        help=(
            "Maximum sampled files in each label split; omit to use every input. Sampling is "
            "without replacement and the selected subset is saved."
        ),
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=DEFAULT_SUBSET_SEED,
        metavar="SEED",
        help="Base random seed for file subsampling without replacement.",
    )
    args = parser.parse_args(argv)

    try:
        noise_sigmas = _parse_float_list(args.noise_sigmas)
        noise_seeds = _parse_int_list(args.noise_seeds)

        run_perturbation_suite(
            positive_dir=Path(args.true),
            negative_dir=Path(args.false),
            workers=args.workers,
            prepare_workers=args.prepare,
            save_dir=Path(args.save_dir),
            metric_level=args.metric_level,
            noise_sigmas=noise_sigmas,
            noise_seeds=noise_seeds,
            noise_atoms=args.noise_atoms,
            max_files_per_split=args.max_files_per_split,
            subset_seed=args.subset_seed,
            positive_manifest=(
                Path(args.positive_manifest) if args.positive_manifest is not None else None
            ),
            negative_manifest=(
                Path(args.negative_manifest) if args.negative_manifest is not None else None
            ),
            metric_error_policy=args.metric_error_policy,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
