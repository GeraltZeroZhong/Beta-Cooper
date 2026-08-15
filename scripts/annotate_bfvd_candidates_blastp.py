#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median
from typing import Any, TypedDict

from cooper_beta.constants import DEFAULT_RESULT_COLUMNS
from cooper_beta.evaluation.runner import validate_detector_artifact_manifest
from cooper_beta.models import DetectionResult
from cooper_beta.polymer_sequence import (
    COMPLETE_SEQUENCE_POLICY,
    CompleteSequenceUnavailableError,
    declared_polymer_sequence_for_author_chain,
    observed_author_chain_ids,
)

BLAST_FIELDS = [
    "qseqid",
    "saccver",
    "pident",
    "length",
    "qcovs",
    "evalue",
    "bitscore",
    "sscinames",
    "sskingdoms",
    "stitle",
]
BLAST_ARTIFACT_SCHEMA_VERSION = 1
BLAST_ARTIFACT_TYPE = "cooper_beta_blastp_search"
REMOTE_BLAST_SERVICE = "NCBI BLAST Common URL API via BLAST+ -remote"
BLAST_DATABASE_IDENTITY_SUFFIXES = frozenset(
    {
        ".pdb",
        ".phr",
        ".pin",
        ".pjs",
        ".pog",
        ".pos",
        ".pot",
        ".psd",
        ".psi",
        ".psq",
        ".ptf",
        ".pto",
    }
)
BLAST_DATABASE_REQUIRED_INDEX_SUFFIX_SETS = (
    frozenset({".phr", ".pin", ".psq"}),
    frozenset({".pdb", ".pot", ".psq", ".ptf", ".pto"}),
)
SEQUENCE_TRUTH_FIELDS = (
    "relative_path",
    "author_chain_id",
    "sequence",
    "sequence_sha256",
    "sequence_source",
    "source_accession",
    "curation_evidence",
)
BFVD_SEQUENCE_COMPLETENESS_POLICY = (
    "frozen_exact_cover_complete_sequence_truth_no_coordinate_fallback"
)
BFVD_RUN_STATE_SCHEMA_VERSION = 1


class FileArtifactState(TypedDict):
    """Stable file metadata recorded in a BLAST artifact manifest."""

    path: str
    size: int
    sha256: str


class BlastInputArtifacts(TypedDict):
    """Input artifacts that define one BLAST annotation run."""

    candidate_manifest: FileArtifactState
    fasta: FileArtifactState
    results_csv: FileArtifactState
    results_manifest: FileArtifactState
    sequence_truth_manifest: FileArtifactState
    structures: list[FileArtifactState]


LOW_INFORMATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bhypothetical protein\b",
        r"\buncharacteri[sz]ed protein\b",
        r"\bputative uncharacteri[sz]ed protein\b",
        r"\bprotein of unknown function\b",
        r"\bdomain of unknown function\b",
        r"\bunknown protein\b",
        r"\bunknown function\b",
        r"\bunnamed protein product\b",
        r"\bpredicted protein\b",
        r"\bconserved hypothetical protein\b",
        r"\bUPF\d+\b",
        r"\bDUF\d+\b",
    ]
]


@dataclass
class Candidate:
    query_id: str
    filename: str
    author_chain_id: str
    result: str
    reason: str
    strand_adjacency_count: str
    cycle_strand_count: str
    cycle_strand_fraction: str
    cycle_rank: str
    source_path: str
    sequence_length: int
    sequence_status: str
    sequence_sha256: str = ""
    sequence_source: str = ""
    sequence_source_accession: str = ""
    sequence_evidence: str = ""
    sequence_completeness_policy: str = ""
    structure_declaration_status: str = ""


@dataclass(frozen=True)
class CompleteSequenceTruth:
    relative_path: str
    author_chain_id: str
    sequence: str
    sequence_sha256: str
    sequence_source: str
    source_accession: str
    curation_evidence: str

    def __post_init__(self) -> None:
        if not self.relative_path or not self.author_chain_id:
            raise ValueError("Complete-sequence truth identity fields must not be blank.")
        if not self.sequence or re.fullmatch(r"[A-Z]+", self.sequence) is None:
            raise ValueError("Complete-sequence truth must be an uppercase amino-acid sequence.")
        expected = hashlib.sha256(self.sequence.encode("ascii")).hexdigest()
        if self.sequence_sha256 != expected:
            raise ValueError("Complete-sequence truth SHA-256 does not match its sequence.")
        if not self.sequence_source or not self.source_accession or not self.curation_evidence:
            raise ValueError("Complete-sequence source, accession, and evidence must not be blank.")
        if re.search(
            r"(?:atom|coordinate|observed[_ -]?residue|ca[_ -]?residue)",
            self.sequence_source,
            re.IGNORECASE,
        ):
            raise ValueError(
                "Coordinate-observed residues are not an admissible complete-sequence source."
            )


@dataclass
class BlastHit:
    qseqid: str
    saccver: str
    pident: float
    length: int
    qcovs: float
    evalue: float
    bitscore: float
    sscinames: str
    sskingdoms: str
    stitle: str


CANDIDATE_FIELDS = list(Candidate.__dataclass_fields__)
ANNOTATION_FIELDS = [
    "query_id",
    "filename",
    "author_chain_id",
    "source_path",
    "sequence_length",
    "sequence_status",
    "sequence_sha256",
    "sequence_source",
    "sequence_source_accession",
    "sequence_evidence",
    "sequence_completeness_policy",
    "structure_declaration_status",
    "strand_adjacency_count",
    "cycle_strand_count",
    "cycle_strand_fraction",
    "cycle_rank",
    "annotation_status",
    "annotation_label",
    "low_information_title",
    "top_saccver",
    "top_pident",
    "top_qcovs",
    "top_evalue",
    "top_bitscore",
    "top_species",
    "top_kingdom",
    "top_title",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/annotate_bfvd_candidates_blastp.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Annotate selected Cooper-Beta BFVD candidate chains with blastp. Complete protein "
            "sequences come from the supplied sequence-truth manifest."
        ),
        epilog=(
            "Output: candidate_manifest.csv, candidate_sequences.faa, blastp.tsv and its "
            "artifact metadata, blastp_annotations.csv, summary files, and run_state.json. "
            "Use --dry-run to create only the candidate and FASTA artifacts. Argument errors "
            "exit with status 2; input, BLAST, or artifact errors exit with status 1."
        ),
    )
    parser.add_argument(
        "--results",
        required=True,
        metavar="CSV",
        help="Cooper-Beta results CSV containing the candidate decisions.",
    )
    parser.add_argument(
        "--results-manifest",
        required=True,
        metavar="JSON",
        help=(
            "Completed Cooper-Beta output manifest paired with --results and containing the "
            "detector configuration and input identities."
        ),
    )
    parser.add_argument(
        "--structures",
        required=True,
        metavar="DIRECTORY",
        help="Root directory containing the BFVD PDB files named by --results.",
    )
    parser.add_argument(
        "--sequence-truth-manifest",
        required=True,
        metavar="CSV",
        help=(
            "Complete-sequence CSV covering each selected relative_path/author_chain_id pair, "
            "with sequence, source accession, and curation evidence columns."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="eval_outputs/bfvd_blastp_annotation",
        metavar="DIRECTORY",
        help="Output directory for FASTA, BLAST TSV, annotation CSV, and summaries.",
    )
    parser.add_argument(
        "--result",
        default="BARREL",
        metavar="LABEL",
        help="Cooper-Beta result label retained as a BLAST query candidate.",
    )
    parser.add_argument(
        "--min-query-length",
        type=int,
        default=20,
        metavar="RESIDUES",
        help="Minimum complete protein-sequence length included in the query FASTA.",
    )
    parser.add_argument(
        "--no-recursive-search",
        action="store_true",
        help="Search only direct relative paths under --structures, without basename fallback.",
    )
    parser.add_argument(
        "--blastp",
        default="blastp",
        metavar="COMMAND",
        help="blastp executable path or command name.",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="DATABASE",
        help=(
            "BLAST protein database name or local volume prefix. Required for new searches and "
            "reuse validation; local searches use a concrete database volume prefix."
        ),
    )
    parser.add_argument(
        "--database-release-id",
        "--db-release-id",
        dest="database_release_id",
        default=None,
        metavar="ID",
        help=(
            "Database release label recorded in output metadata; local index files identify the "
            "database contents used for reuse."
        ),
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use NCBI remote blastp; --threads applies only to local searches.",
    )
    parser.add_argument(
        "--blast-tsv",
        default=None,
        metavar="TSV",
        help="Existing BLAST outfmt 6 TSV to reuse with its matching artifact metadata.",
    )
    parser.add_argument(
        "--blast-artifact-manifest",
        default=None,
        metavar="JSON",
        help="Artifact metadata paired with --blast-tsv; omit to use <TSV>.artifact.json.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        metavar="N",
        help="blastp worker threads for a local search.",
    )
    parser.add_argument(
        "--max-target-seqs",
        type=int,
        default=10,
        metavar="N",
        help="Maximum reported BLAST hits per query.",
    )
    parser.add_argument(
        "--search-evalue",
        default="1e-5",
        metavar="EVALUE",
        help="E-value passed to blastp while generating hits.",
    )
    parser.add_argument(
        "--hit-evalue",
        type=float,
        default=1e-5,
        metavar="EVALUE",
        help="Maximum e-value for a hit to count as an annotation.",
    )
    parser.add_argument(
        "--min-pident",
        type=float,
        default=25.0,
        metavar="PERCENT",
        help="Minimum percent identity for a hit to count as an annotation.",
    )
    parser.add_argument(
        "--min-qcov",
        type=float,
        default=50.0,
        metavar="PERCENT",
        help="Minimum query coverage percentage for a hit to count as an annotation.",
    )
    parser.add_argument(
        "--entrez-query",
        default=None,
        metavar="QUERY",
        help='Optional BLAST Entrez query, e.g. "Viruses[Organism]".',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun blastp and replace the managed BLAST TSV and artifact metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write candidate and FASTA artifacts, skip blastp and hit annotation, then exit.",
    )
    return parser.parse_args(argv)


def sanitize_query_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())
    return cleaned.strip("_") or "query"


def percent(part: int, whole: int) -> str:
    if whole == 0:
        return "0.00%"
    return f"{part / whole * 100:.2f}%"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_artifact_state(path: Path, *, label: str) -> FileArtifactState:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "sha256": _file_sha256(resolved),
    }


def _strict_json_document(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"BLAST artifact manifest is not strict JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"BLAST artifact manifest must contain a JSON object: {path}")
    return document


def _normalized_positive_decimal(value: Any, *, label: str) -> str:
    try:
        numeric = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive number.") from exc
    if not numeric.is_finite() or numeric <= 0:
        raise ValueError(f"{label} must be a finite positive number.")
    return format(numeric.normalize(), "f")


def _validate_unit_interval_percent(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
        raise ValueError(f"{label} must be finite and within [0, 100].")
    return numeric


def normalized_search_parameters(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.max_target_seqs) <= 0:
        raise ValueError("--max-target-seqs must be > 0.")
    if not args.remote and int(args.threads) <= 0:
        raise ValueError("--threads must be > 0 for local BLAST searches.")
    entrez_query = str(args.entrez_query).strip() if args.entrez_query else None
    if entrez_query and not args.remote:
        raise ValueError("--entrez-query requires --remote.")
    database = str(args.db).strip() if args.db is not None else ""
    if not database:
        raise ValueError("--db is required for local and remote BLAST artifact identity.")
    return {
        "database": database,
        "entrez_query": entrez_query,
        "evalue": _normalized_positive_decimal(args.search_evalue, label="--search-evalue"),
        "max_target_seqs": int(args.max_target_seqs),
        "remote": bool(args.remote),
        "threads": None if args.remote else int(args.threads),
    }


def validate_arguments(args: argparse.Namespace) -> None:
    if int(args.min_query_length) <= 0:
        raise ValueError("--min-query-length must be > 0.")
    if not str(args.result).strip():
        raise ValueError("--result must not be empty.")
    if args.force and args.blast_tsv:
        raise ValueError("--force cannot be combined with --blast-tsv; rerun into managed output.")
    try:
        hit_evalue = float(args.hit_evalue)
    except (TypeError, ValueError) as exc:
        raise ValueError("--hit-evalue must be numeric.") from exc
    if not math.isfinite(hit_evalue) or hit_evalue < 0.0:
        raise ValueError("--hit-evalue must be finite and >= 0.")
    _validate_unit_interval_percent(args.min_pident, label="--min-pident")
    _validate_unit_interval_percent(args.min_qcov, label="--min-qcov")
    release_id = (
        str(args.database_release_id).strip() if args.database_release_id is not None else ""
    )
    if args.database_release_id is not None and not release_id:
        raise ValueError("--database-release-id must not be empty.")
    if not args.dry_run:
        normalized_search_parameters(args)


def resolve_blastp_identity(blastp: str) -> dict[str, Any]:
    resolved_command = shutil.which(blastp)
    if resolved_command is None:
        candidate = Path(blastp).expanduser()
        if candidate.is_file():
            resolved_command = str(candidate)
    if resolved_command is None:
        raise FileNotFoundError(
            f"blastp executable was not found: {blastp}. Install BLAST+ or pass --blastp."
        )
    executable = Path(resolved_command).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"blastp executable is not a file: {executable}")
    try:
        completed = subprocess.run(
            [str(executable), "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to determine blastp version: {executable}") from exc
    version = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if not version:
        raise RuntimeError(f"blastp -version produced no version text: {executable}")
    return {
        "path": str(executable),
        "sha256": _file_sha256(executable),
        "version": version,
    }


def _database_prefix_candidates(database: str) -> list[Path]:
    raw = Path(database).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute() or raw.parent != Path("."):
        candidates.append(raw.resolve())
    else:
        candidates.append((Path.cwd() / raw).resolve())
        for directory in os.environ.get("BLASTDB", "").split(os.pathsep):
            if directory.strip():
                candidates.append((Path(directory).expanduser() / raw).resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_database_identity(args: argparse.Namespace) -> dict[str, Any]:
    parameters = normalized_search_parameters(args)
    database = str(parameters["database"])
    release_id = (
        str(args.database_release_id).strip() if args.database_release_id is not None else None
    )
    if args.remote:
        return {
            "database": database,
            "kind": "remote_service_unpinned",
            "declared_release_label": release_id,
            "immutable_identity_available": False,
            "service": REMOTE_BLAST_SERVICE,
        }

    for prefix in _database_prefix_candidates(database):
        alias_files = sorted(
            path.resolve()
            for path in prefix.parent.glob(f"{prefix.name}.*")
            if path.is_file() and path.suffix.lower() == ".pal"
        )
        if prefix.is_file() and prefix.suffix.lower() == ".pal":
            alias_files = sorted({*alias_files, prefix.resolve()})
        if alias_files:
            aliases = ", ".join(str(path) for path in alias_files)
            raise ValueError(
                "Local BLAST .pal alias databases are not supported because the alias text is "
                "not an immutable identity for its DBLIST targets. Use a concrete "
                f"checksum-pinned database volume prefix instead. Found: {aliases}"
            )
        identity_files = sorted(
            path.resolve()
            for path in prefix.parent.glob(f"{prefix.name}*")
            if path.is_file() and path.suffix.lower() in BLAST_DATABASE_IDENTITY_SUFFIXES
        )
        suffixes = {path.suffix.lower() for path in identity_files}
        has_complete_index_set = any(
            required_suffixes.issubset(suffixes)
            for required_suffixes in BLAST_DATABASE_REQUIRED_INDEX_SUFFIX_SETS
        )
        if identity_files and has_complete_index_set:
            return {
                "database": database,
                "identity_files": [
                    _file_artifact_state(path, label="BLAST database identity file")
                    for path in identity_files
                ],
                "kind": "local_index_hashes",
                "declared_release_label": release_id,
                "immutable_identity_available": True,
                "resolved_prefix": str(prefix),
            }
    raise ValueError(
        "Unable to establish checksum-pinned local BLAST database identity. Pass a resolvable "
        "database prefix with key index files; --database-release-id is diagnostic only."
    )


def build_blast_artifact_context(
    args: argparse.Namespace,
    *,
    fasta_path: Path,
    candidate_manifest: Path,
    results_path: Path,
    results_manifest_path: Path,
    detector_manifest: Mapping[str, object],
    sequence_truth_manifest: Path,
    candidates: list[Candidate],
) -> dict[str, Any]:
    parameters = normalized_search_parameters(args)
    tool = resolve_blastp_identity(str(args.blastp))
    database = resolve_database_identity(args)
    database_identity = dict(database)
    declared_release_label = database_identity.pop("declared_release_label", None)
    structure_paths = sorted(
        {
            str(Path(candidate.source_path).expanduser().resolve())
            for candidate in candidates
            if candidate.source_path
        }
    )
    inputs: BlastInputArtifacts = {
        "candidate_manifest": _file_artifact_state(
            candidate_manifest,
            label="candidate manifest",
        ),
        "fasta": _file_artifact_state(fasta_path, label="candidate FASTA"),
        "results_csv": _file_artifact_state(results_path, label="results CSV"),
        "results_manifest": _file_artifact_state(
            results_manifest_path,
            label="detector results manifest",
        ),
        "sequence_truth_manifest": _file_artifact_state(
            sequence_truth_manifest,
            label="complete-sequence truth manifest",
        ),
        "structures": [
            _file_artifact_state(Path(path), label="candidate structure")
            for path in structure_paths
        ],
    }
    identity = {
        "blast_fields": list(BLAST_FIELDS),
        "database": database_identity,
        "fasta_sha256": inputs["fasta"]["sha256"],
        "sequence_completeness_policy": BFVD_SEQUENCE_COMPLETENESS_POLICY,
        "structure_declaration_policy": COMPLETE_SEQUENCE_POLICY,
        "sequence_truth_manifest_sha256": inputs["sequence_truth_manifest"]["sha256"],
        "detector_provenance": {
            "scientific_config_hash": str(detector_manifest["scientific_config_hash"]),
            "producer_identity_hash": str(detector_manifest["producer_identity_hash"]),
            "results_manifest_sha256": inputs["results_manifest"]["sha256"],
        },
        "search_parameters": parameters,
        "tool": tool,
    }
    return {
        "artifact_key": _json_sha256(identity),
        "diagnostics": {
            "declared_database_release_label": declared_release_label,
        },
        "identity": identity,
        "inputs": inputs,
        "reuse_allowed": bool(
            not args.remote
            and database.get("kind") == "local_index_hashes"
            and database.get("immutable_identity_available") is True
        ),
    }


def blast_artifact_manifest_path(args: argparse.Namespace, blast_tsv: Path) -> Path:
    if args.blast_artifact_manifest:
        return Path(args.blast_artifact_manifest).expanduser().resolve()
    return Path(f"{blast_tsv}.artifact.json").expanduser().resolve()


def build_blast_artifact_manifest(
    context: dict[str, Any],
    *,
    blast_tsv: Path,
    command: list[str],
) -> dict[str, Any]:
    return {
        "artifact_key": context["artifact_key"],
        "artifact_type": BLAST_ARTIFACT_TYPE,
        "diagnostics": context["diagnostics"],
        "generated_at_utc": _utc_now(),
        "identity": context["identity"],
        "inputs": context["inputs"],
        "outputs": {
            "blast_tsv": _file_artifact_state(blast_tsv, label="BLAST TSV"),
        },
        "reproduction_command": list(command),
        "reuse_allowed": bool(context["reuse_allowed"]),
        "schema_version": BLAST_ARTIFACT_SCHEMA_VERSION,
    }


def validate_blast_artifact_reuse(
    manifest_path: Path,
    *,
    blast_tsv: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise ValueError(
            f"Existing BLAST TSV has no artifact manifest: {manifest_path}. "
            "Rerun explicitly with --force."
        )
    document = _strict_json_document(manifest_path)
    required = {
        "artifact_key",
        "artifact_type",
        "diagnostics",
        "generated_at_utc",
        "identity",
        "inputs",
        "outputs",
        "reproduction_command",
        "reuse_allowed",
        "schema_version",
    }
    if set(document) != required:
        raise ValueError(
            "BLAST artifact manifest fields do not match the strict schema; rerun with --force."
        )
    if document["schema_version"] != BLAST_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported BLAST artifact manifest schema; rerun with --force.")
    if document["artifact_type"] != BLAST_ARTIFACT_TYPE:
        raise ValueError("BLAST artifact manifest has the wrong artifact type.")
    if not isinstance(document["diagnostics"], dict):
        raise ValueError("BLAST artifact manifest diagnostics must be an object.")
    timestamp = str(document["generated_at_utc"])
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("BLAST artifact manifest has an invalid generated_at_utc.") from exc
    utc_offset = parsed_timestamp.utcoffset()
    if (
        not timestamp.endswith("Z")
        or parsed_timestamp.tzinfo is None
        or utc_offset is None
        or utc_offset.total_seconds() != 0
    ):
        raise ValueError("BLAST artifact manifest generated_at_utc must be UTC with a Z suffix.")
    if not isinstance(document["identity"], dict):
        raise ValueError("BLAST artifact manifest identity must be an object.")
    recorded_key = str(document["artifact_key"])
    if recorded_key != _json_sha256(document["identity"]):
        raise ValueError("BLAST artifact manifest artifact_key is inconsistent.")
    if document["identity"] != context["identity"]:
        raise ValueError(
            "Existing BLAST artifact identity does not match FASTA, search parameters, "
            "blastp, or database identity; rerun explicitly with --force."
        )
    if recorded_key != context["artifact_key"]:
        raise ValueError("Existing BLAST artifact key does not match; rerun with --force.")
    if document["inputs"] != context["inputs"]:
        raise ValueError("Existing BLAST artifact input hashes do not match; rerun with --force.")
    if document["reuse_allowed"] is not True or context["reuse_allowed"] is not True:
        raise ValueError(
            "BLAST artifact reuse requires a checksum-pinned local database; remote results "
            "and declared release labels are diagnostic only and are never reusable."
        )
    outputs = document["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"blast_tsv"}:
        raise ValueError("BLAST artifact manifest outputs do not match the strict schema.")
    actual_output = _file_artifact_state(blast_tsv, label="BLAST TSV")
    if outputs["blast_tsv"] != actual_output:
        raise ValueError("Existing BLAST TSV path or hash does not match its artifact manifest.")
    command = document["reproduction_command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise ValueError("BLAST artifact reproduction_command must be a non-empty string list.")
    return document


def read_candidate_rows(results_path: Path, result: str) -> tuple[list[dict[str, str]], int]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV does not exist: {results_path}")

    rows: list[dict[str, str]] = []
    with results_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if any(not str(field).strip() for field in fieldnames):
            raise ValueError("Results CSV contains an empty column name.")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("Results CSV contains duplicate column names.")
        if tuple(fieldnames) != DEFAULT_RESULT_COLUMNS:
            raise ValueError("Results CSV does not match the Cooper-Beta public schema.")

        total_rows = 0
        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            if None in row:
                raise ValueError(f"Results CSV row {row_number} has extra unnamed values.")
            normalized_result = str(row.get("result", "")).strip().upper()
            if not normalized_result:
                raise ValueError(f"Results CSV row {row_number} has an empty result.")
            if normalized_result != result.strip().upper():
                continue
            normalized = {key: value or "" for key, value in row.items()}
            try:
                DetectionResult.from_row(normalized)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Results CSV row {row_number} is invalid: {exc}") from exc
            rows.append(normalized)

    return rows, total_rows


def _safe_lookup_path(root: Path, value: str) -> Path:
    root = root.expanduser().resolve()
    candidate = Path(value.strip()).expanduser()
    if not str(candidate):
        raise ValueError("Empty filename is not a valid structure path.")
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            return resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Structure path escapes --structures: {value!r}") from exc
    if ".." in candidate.parts:
        raise ValueError(f"Unsafe structure filename in results CSV: {value!r}")
    return candidate


def _resolve_under(root: Path, relative_path: Path) -> Path | None:
    root = root.expanduser().resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def resolve_structure_path(
    structures_dir: Path,
    filename: str,
    recursive: bool,
    cache: dict[str, Path | None],
) -> Path | None:
    structures_root = structures_dir.expanduser().resolve()
    candidate = _safe_lookup_path(structures_root, filename)
    direct = _resolve_under(structures_root, candidate)
    if direct is not None and direct.is_file():
        return direct

    # A path-bearing manifest value is an exact identity, not a basename hint.
    # Recursive discovery is available only for explicitly basename-only inventories.
    if len(candidate.parts) > 1:
        return None
    if not recursive:
        return None

    basename = candidate.name
    if basename not in cache:
        matches = [match.resolve() for match in structures_root.rglob(basename) if match.is_file()]
        if len(matches) > 1:
            joined = ", ".join(str(match) for match in matches[:5])
            suffix = " ..." if len(matches) > 5 else ""
            raise ValueError(f"Ambiguous recursive match for {basename!r}: {joined}{suffix}")
        cache[basename] = matches[0] if matches else None
    return cache[basename]


def load_complete_sequence_truth(
    path: Path,
) -> dict[tuple[str, str], CompleteSequenceTruth]:
    """Load a frozen exact candidate-to-complete-sequence truth manifest."""

    resolved = path.expanduser().resolve()
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SEQUENCE_TRUTH_FIELDS:
            raise ValueError(
                "Complete-sequence truth manifest must have exactly these columns in order: "
                + ",".join(SEQUENCE_TRUTH_FIELDS)
            )
        truths: dict[tuple[str, str], CompleteSequenceTruth] = {}
        for row_number, row in enumerate(reader, start=2):
            raw_relative = str(row.get("relative_path", "")).strip().replace("\\", "/")
            relative = Path(raw_relative)
            if (
                not raw_relative
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() in {"", "."}
            ):
                raise ValueError(
                    f"Complete-sequence truth row {row_number} has an invalid relative_path."
                )
            author_chain_id = str(row.get("author_chain_id", "")).strip()
            if not author_chain_id:
                raise ValueError(
                    f"Complete-sequence truth row {row_number} has a blank author_chain_id."
                )
            truth = CompleteSequenceTruth(
                relative_path=relative.as_posix(),
                author_chain_id=author_chain_id,
                sequence=str(row.get("sequence", "")).strip(),
                sequence_sha256=str(row.get("sequence_sha256", "")).strip().lower(),
                sequence_source=str(row.get("sequence_source", "")).strip(),
                source_accession=str(row.get("source_accession", "")).strip(),
                curation_evidence=str(row.get("curation_evidence", "")).strip(),
            )
            key = (truth.relative_path, truth.author_chain_id)
            if key in truths:
                raise ValueError(f"Duplicate complete-sequence truth target {key!r}.")
            truths[key] = truth
    if not truths:
        raise ValueError("Complete-sequence truth manifest contains no targets.")
    return truths


def _relative_source_identity(source_path: Path, structures_dir: Path) -> str:
    try:
        return (
            source_path.expanduser()
            .resolve()
            .relative_to(structures_dir.expanduser().resolve())
            .as_posix()
        )
    except ValueError as exc:
        raise ValueError(
            f"Candidate structure {source_path} is outside --structures {structures_dir}."
        ) from exc


def _validate_complete_sequence_truth(
    source_path: Path,
    truth: CompleteSequenceTruth,
) -> str:
    observed_chains = observed_author_chain_ids(source_path)
    if truth.author_chain_id not in observed_chains:
        raise ValueError(
            f"Complete-sequence truth target {truth.relative_path}:{truth.author_chain_id} does not "
            f"match an observed author chain; observed={sorted(observed_chains)!r}."
        )
    try:
        declared = declared_polymer_sequence_for_author_chain(source_path, truth.author_chain_id)
    except CompleteSequenceUnavailableError:
        return "independent_frozen_truth_no_structure_declaration"
    if declared.sequence != truth.sequence:
        raise ValueError(
            f"Complete-sequence truth conflicts with the structure declaration for "
            f"{truth.relative_path}:{truth.author_chain_id}."
        )
    return f"verified_against_{declared.sequence_source}"


def build_candidates(
    selected_rows: list[dict[str, str]],
    structures_dir: Path,
    min_query_length: int,
    recursive: bool,
    *,
    sequence_truth: Mapping[tuple[str, str], CompleteSequenceTruth],
    detector_input_paths: frozenset[Path] | None = None,
) -> tuple[list[Candidate], dict[str, str], int]:
    path_cache: dict[str, Path | None] = {}
    query_id_counts: Counter[str] = Counter()
    sequences: dict[str, str] = {}
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    used_truth_targets: set[tuple[str, str]] = set()
    duplicates = 0

    for row in selected_rows:
        filename = row["filename"].strip()
        author_chain_id = row.get("author_chain_id", "").strip()
        lookup_path = row.get("source_path", "").strip() or filename
        source_path = resolve_structure_path(structures_dir, lookup_path, recursive, path_cache)
        if source_path is None:
            raise ValueError(f"Candidate structure could not be resolved exactly: {lookup_path!r}.")
        if detector_input_paths is not None and source_path.resolve() not in detector_input_paths:
            raise ValueError(
                f"Selected candidate structure is not a hash-validated detector input: "
                f"{source_path.resolve()}."
            )
        source_key = str(source_path.resolve())
        key = (source_key, author_chain_id)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)

        base_query_id = sanitize_query_id(f"{Path(filename).stem}__author_chain_{author_chain_id}")
        if row.get("source_path", "").strip():
            digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:8]
            base_query_id = sanitize_query_id(f"{base_query_id}__{digest}")
        query_id_counts[base_query_id] += 1
        query_id = base_query_id
        if query_id_counts[base_query_id] > 1:
            query_id = f"{base_query_id}__{query_id_counts[base_query_id]}"

        relative_source = _relative_source_identity(source_path, structures_dir)
        truth_key = (relative_source, author_chain_id)
        truth = sequence_truth.get(truth_key)
        if truth is None:
            raise ValueError(
                f"Complete-sequence truth manifest does not exactly cover candidate {truth_key!r}."
            )
        used_truth_targets.add(truth_key)
        declaration_status = _validate_complete_sequence_truth(source_path, truth)
        sequence = truth.sequence
        if len(sequence) < min_query_length:
            status = "too_short"
        else:
            status = "ok"
            sequences[query_id] = sequence

        candidates.append(
            Candidate(
                query_id=query_id,
                filename=filename,
                author_chain_id=author_chain_id,
                result=row.get("result", ""),
                reason=row.get("reason", ""),
                strand_adjacency_count=row.get("strand_adjacency_count", ""),
                cycle_strand_count=row.get("cycle_strand_count", ""),
                cycle_strand_fraction=row.get("cycle_strand_fraction", ""),
                cycle_rank=row.get("cycle_rank", ""),
                source_path=str(source_path),
                sequence_length=len(sequence),
                sequence_status=status,
                sequence_sha256=truth.sequence_sha256,
                sequence_source=truth.sequence_source,
                sequence_source_accession=truth.source_accession,
                sequence_evidence=truth.curation_evidence,
                sequence_completeness_policy=BFVD_SEQUENCE_COMPLETENESS_POLICY,
                structure_declaration_status=declaration_status,
            )
        )

    if used_truth_targets != set(sequence_truth):
        missing = sorted(set(sequence_truth) - used_truth_targets)
        unexpected = sorted(used_truth_targets - set(sequence_truth))
        raise ValueError(
            "Complete-sequence truth manifest must exactly cover the final unique candidate "
            f"set; extra={missing[:10]!r}, unexpected={unexpected[:10]!r}."
        )
    return candidates, sequences, duplicates


def write_fasta(sequences: dict[str, str], path: Path) -> None:
    with path.open("w") as handle:
        for query_id, sequence in sequences.items():
            handle.write(f">{query_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_candidate_manifest(candidates: list[Candidate], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def run_blastp(
    args: argparse.Namespace,
    fasta_path: Path,
    blast_tsv: Path,
    *,
    tool_identity: dict[str, Any] | None = None,
    search_parameters: dict[str, Any] | None = None,
) -> list[str]:
    tool = tool_identity or resolve_blastp_identity(str(args.blastp))
    parameters = search_parameters or normalized_search_parameters(args)
    blastp_path = str(tool["path"])

    command = [
        blastp_path,
        "-query",
        str(fasta_path),
        "-out",
        str(blast_tsv),
        "-outfmt",
        "6 " + " ".join(BLAST_FIELDS),
        "-evalue",
        str(parameters["evalue"]),
        "-max_target_seqs",
        str(parameters["max_target_seqs"]),
    ]
    if parameters["remote"]:
        command.extend(["-remote", "-db", str(parameters["database"])])
    else:
        command.extend(["-db", str(parameters["database"])])
        command.extend(["-num_threads", str(parameters["threads"])])

    if parameters["entrez_query"]:
        command.extend(["-entrez_query", str(parameters["entrez_query"])])

    env = os.environ.copy()
    if not parameters["remote"]:
        db_parent = str(Path(str(parameters["database"])).expanduser().resolve().parent)
        existing_blastdb = env.get("BLASTDB")
        env["BLASTDB"] = (
            db_parent if not existing_blastdb else f"{db_parent}{os.pathsep}{existing_blastdb}"
        )

    subprocess.run(command, check=True, env=env)
    return command


def parse_float(value: str, *, field: str = "value") -> float:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite, got {value!r}.")
    return numeric


def parse_int(value: str, *, field: str = "value") -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}.") from exc


def read_blast_hits(
    path: Path,
    *,
    expected_query_ids: set[str] | None = None,
) -> dict[str, list[BlastHit]]:
    hits: dict[str, list[BlastHit]] = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(f"BLAST TSV does not exist: {path}")

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(BLAST_FIELDS):
                raise ValueError(
                    f"BLAST TSV line {line_number} has {len(parts)} columns; "
                    f"expected exactly {len(BLAST_FIELDS)} ({', '.join(BLAST_FIELDS)})."
                )
            row = dict(zip(BLAST_FIELDS, parts, strict=True))
            if not row["qseqid"].strip() or not row["saccver"].strip():
                raise ValueError(f"BLAST TSV line {line_number} has an empty qseqid or saccver.")
            if expected_query_ids is not None and row["qseqid"] not in expected_query_ids:
                raise ValueError(
                    f"BLAST TSV line {line_number} references unknown query ID {row['qseqid']!r}."
                )
            pident = parse_float(row["pident"], field=f"line {line_number} pident")
            length = parse_int(row["length"], field=f"line {line_number} length")
            qcovs = parse_float(row["qcovs"], field=f"line {line_number} qcovs")
            evalue = parse_float(row["evalue"], field=f"line {line_number} evalue")
            bitscore = parse_float(row["bitscore"], field=f"line {line_number} bitscore")
            if not 0.0 <= pident <= 100.0:
                raise ValueError(f"BLAST TSV line {line_number} pident must be within [0, 100].")
            if length <= 0:
                raise ValueError(f"BLAST TSV line {line_number} length must be > 0.")
            if not 0.0 <= qcovs <= 100.0:
                raise ValueError(f"BLAST TSV line {line_number} qcovs must be within [0, 100].")
            if evalue < 0.0 or bitscore < 0.0:
                raise ValueError(f"BLAST TSV line {line_number} evalue and bitscore must be >= 0.")
            hits[row["qseqid"]].append(
                BlastHit(
                    qseqid=row["qseqid"],
                    saccver=row["saccver"],
                    pident=pident,
                    length=length,
                    qcovs=qcovs,
                    evalue=evalue,
                    bitscore=bitscore,
                    sscinames=row["sscinames"],
                    sskingdoms=row["sskingdoms"],
                    stitle=row["stitle"],
                )
            )

    for query_hits in hits.values():
        query_hits.sort(key=lambda hit: (-hit.bitscore, hit.evalue, -hit.pident, -hit.qcovs))
    return hits


def hit_passes(hit: BlastHit, max_evalue: float, min_pident: float, min_qcov: float) -> bool:
    return hit.evalue <= max_evalue and hit.pident >= min_pident and hit.qcovs >= min_qcov


def is_low_information_title(title: str) -> bool:
    return any(pattern.search(title) for pattern in LOW_INFORMATION_PATTERNS)


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^(?:sp|tr|ref|gb|emb|dbj|pir|prf)\|[^ ]+\s+", "", title)
    return title


def annotate_candidates(
    candidates: list[Candidate],
    hits_by_query: dict[str, list[BlastHit]],
    args: argparse.Namespace,
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    kingdom_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()
    queries_with_any_hit = 0
    queries_with_passing_hit = 0

    for candidate in candidates:
        hits = hits_by_query.get(candidate.query_id, [])
        if hits:
            queries_with_any_hit += 1

        passing_hits = [
            hit for hit in hits if hit_passes(hit, args.hit_evalue, args.min_pident, args.min_qcov)
        ]
        top_hit = passing_hits[0] if passing_hits else (hits[0] if hits else None)
        hit_status = "no_blast_hit"
        annotation_label = ""
        low_information = ""

        if candidate.sequence_status != "ok":
            hit_status = candidate.sequence_status
        elif passing_hits:
            queries_with_passing_hit += 1
            passing_top_hit = passing_hits[0]
            annotation_label = clean_title(passing_top_hit.stitle)
            low_information = str(is_low_information_title(passing_top_hit.stitle))
            hit_status = "low_information_hit" if low_information == "True" else "informative_hit"
            if passing_top_hit.sskingdoms:
                kingdom_counts[passing_top_hit.sskingdoms] += 1
            if passing_top_hit.sscinames:
                species_counts[passing_top_hit.sscinames] += 1
            if annotation_label:
                title_counts[annotation_label] += 1
        elif hits:
            hit_status = "no_passing_hit"

        status_counts[hit_status] += 1
        rows.append(
            {
                "query_id": candidate.query_id,
                "filename": candidate.filename,
                "author_chain_id": candidate.author_chain_id,
                "source_path": candidate.source_path,
                "sequence_length": candidate.sequence_length,
                "sequence_status": candidate.sequence_status,
                "sequence_sha256": candidate.sequence_sha256,
                "sequence_source": candidate.sequence_source,
                "sequence_source_accession": candidate.sequence_source_accession,
                "sequence_evidence": candidate.sequence_evidence,
                "sequence_completeness_policy": candidate.sequence_completeness_policy,
                "structure_declaration_status": candidate.structure_declaration_status,
                "strand_adjacency_count": candidate.strand_adjacency_count,
                "cycle_strand_count": candidate.cycle_strand_count,
                "cycle_strand_fraction": candidate.cycle_strand_fraction,
                "cycle_rank": candidate.cycle_rank,
                "annotation_status": hit_status,
                "annotation_label": annotation_label,
                "low_information_title": low_information,
                "top_saccver": top_hit.saccver if top_hit else "",
                "top_pident": top_hit.pident if top_hit else "",
                "top_qcovs": top_hit.qcovs if top_hit else "",
                "top_evalue": top_hit.evalue if top_hit else "",
                "top_bitscore": top_hit.bitscore if top_hit else "",
                "top_species": top_hit.sscinames if top_hit else "",
                "top_kingdom": top_hit.sskingdoms if top_hit else "",
                "top_title": top_hit.stitle if top_hit else "",
            }
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "annotation_status_counts": dict(status_counts),
        "queries_with_any_blast_hit": queries_with_any_hit,
        "queries_with_passing_hit": queries_with_passing_hit,
        "top_kingdoms": kingdom_counts.most_common(20),
        "top_species": species_counts.most_common(20),
        "top_annotation_labels": title_counts.most_common(30),
    }
    return rows, summary


def sequence_length_stats(candidates: list[Candidate]) -> dict[str, Any]:
    lengths = [
        candidate.sequence_length for candidate in candidates if candidate.sequence_status == "ok"
    ]
    if not lengths:
        return {"count": 0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(mean(lengths), 2),
        "median": round(median(lengths), 2),
    }


def build_base_summary(
    args: argparse.Namespace,
    total_input_rows: int,
    selected_rows: list[dict[str, str]],
    candidates: list[Candidate],
    sequences: dict[str, str],
    duplicates: int,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    sequence_status_counts = Counter(candidate.sequence_status for candidate in candidates)
    return {
        "results_csv": str(Path(args.results)),
        "structures_dir": str(Path(args.structures)),
        "result_filter": args.result,
        "total_input_rows": total_input_rows,
        "candidate_rows_selected": len(selected_rows),
        "unique_candidates": len(candidates),
        "duplicates_collapsed": duplicates,
        "fasta_sequences_written": len(sequences),
        "sequence_status_counts": dict(sequence_status_counts),
        "sequence_length_stats": sequence_length_stats(candidates),
        "sequence_completeness": {
            "policy": BFVD_SEQUENCE_COMPLETENESS_POLICY,
            "structure_declaration_policy": COMPLETE_SEQUENCE_POLICY,
            "truth_manifest": _file_artifact_state(
                Path(args.sequence_truth_manifest),
                label="complete-sequence truth manifest",
            ),
        },
        "blast_fields": BLAST_FIELDS,
        "hit_thresholds": {
            "max_evalue": args.hit_evalue,
            "min_pident": args.min_pident,
            "min_qcov": args.min_qcov,
        },
        "output_paths": output_paths,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
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


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    total = int(summary.get("unique_candidates", 0))
    status_counts = summary.get("annotation_status_counts", {})
    sequence_counts = summary.get("sequence_status_counts", {})
    lines = [
        "# BFVD Candidate BLASTP Annotation Summary",
        "",
        f"- Candidate rows selected: {summary.get('candidate_rows_selected', 0)}",
        f"- Unique candidate proteins: {total}",
        f"- FASTA sequences written: {summary.get('fasta_sequences_written', 0)}",
        f"- Duplicate rows collapsed: {summary.get('duplicates_collapsed', 0)}",
        f"- Sequence status counts: {sequence_counts}",
        f"- Sequence length stats: {summary.get('sequence_length_stats', {})}",
        f"- Sequence completeness: {summary.get('sequence_completeness', {})}",
        "",
        "## Annotation Counts",
        "",
        "| Category | Count | Percent of unique candidates |",
        "|---|---:|---:|",
    ]
    if status_counts:
        for status, count in sorted(status_counts.items()):
            lines.append(f"| {status} | {count} | {percent(int(count), total)} |")
    else:
        lines.append("| not_run | 0 | 0.00% |")

    lines.extend(
        [
            "",
            "## BLAST Hit Counts",
            "",
            f"- Queries with any BLAST hit: {summary.get('queries_with_any_blast_hit', 'not_run')}",
            f"- Queries with passing BLAST hit: {summary.get('queries_with_passing_hit', 'not_run')}",
            f"- Hit thresholds: {summary.get('hit_thresholds', {})}",
        ]
    )

    if summary.get("top_kingdoms"):
        lines.extend(["", "## Top Kingdoms", ""])
        for kingdom, count in summary["top_kingdoms"]:
            lines.append(f"- {kingdom}: {count}")

    if summary.get("top_species"):
        lines.extend(["", "## Top Species", ""])
        for species, count in summary["top_species"][:10]:
            lines.append(f"- {species}: {count}")

    if summary.get("top_annotation_labels"):
        lines.extend(["", "## Top Annotation Labels", ""])
        for label, count in summary["top_annotation_labels"][:15]:
            lines.append(f"- {label}: {count}")

    path.write_text("\n".join(lines) + "\n")


def print_key_numbers(summary: dict[str, Any]) -> None:
    print("[OK] Candidate rows selected:", summary.get("candidate_rows_selected", 0))
    print("[OK] Unique candidate proteins:", summary.get("unique_candidates", 0))
    print("[OK] FASTA sequences written:", summary.get("fasta_sequences_written", 0))
    print("[OK] Sequence status counts:", summary.get("sequence_status_counts", {}))
    if "annotation_status_counts" in summary:
        print("[OK] Annotation status counts:", summary["annotation_status_counts"])
        print("[OK] Queries with any BLAST hit:", summary.get("queries_with_any_blast_hit", 0))
        print("[OK] Queries with passing BLAST hit:", summary.get("queries_with_passing_hit", 0))
    print("[OK] Summary JSON:", summary["output_paths"]["summary_json"])
    print("[OK] Summary Markdown:", summary["output_paths"]["summary_md"])


def _run(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    results_manifest_path = Path(args.results_manifest).expanduser().resolve()
    structures_dir = Path(args.structures)
    sequence_truth_path = Path(args.sequence_truth_manifest).expanduser().resolve()
    out_dir = Path(args.out_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV does not exist: {results_path}")
    if not results_manifest_path.is_file():
        raise FileNotFoundError(
            f"Detector results manifest does not exist: {results_manifest_path}"
        )
    if not structures_dir.exists():
        raise FileNotFoundError(f"Structures directory does not exist: {structures_dir}")
    if not sequence_truth_path.is_file():
        raise FileNotFoundError(
            f"Complete-sequence truth manifest does not exist: {sequence_truth_path}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv = out_dir / "candidate_manifest.csv"
    fasta_path = out_dir / "candidate_sequences.faa"
    blast_tsv = Path(args.blast_tsv) if args.blast_tsv else out_dir / "blastp.tsv"
    blast_tsv = blast_tsv.expanduser().resolve()
    blast_artifact_manifest = blast_artifact_manifest_path(args, blast_tsv)
    annotation_csv = out_dir / "blastp_annotations.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    detector_manifest = validate_detector_artifact_manifest(
        results_manifest_path,
        expected_output=results_path.expanduser().resolve(),
    )
    validated_detector_inputs = detector_manifest.get("_validated_input_paths")
    if not isinstance(validated_detector_inputs, list) or not validated_detector_inputs:
        raise RuntimeError("Validated detector manifest contains no input-path identity set.")
    detector_input_paths = frozenset(
        Path(str(path)).expanduser().resolve() for path in validated_detector_inputs
    )
    selected_rows, total_input_rows = read_candidate_rows(results_path, args.result)
    sequence_truth = load_complete_sequence_truth(sequence_truth_path)
    candidates, sequences, duplicates = build_candidates(
        selected_rows=selected_rows,
        structures_dir=structures_dir,
        min_query_length=args.min_query_length,
        recursive=not args.no_recursive_search,
        sequence_truth=sequence_truth,
        detector_input_paths=detector_input_paths,
    )

    write_candidate_manifest(candidates, candidates_csv)
    write_fasta(sequences, fasta_path)

    output_paths = {
        "candidate_manifest": str(candidates_csv),
        "fasta": str(fasta_path),
        "blast_tsv": str(blast_tsv),
        "blast_artifact_manifest": str(blast_artifact_manifest),
        "annotation_csv": str(annotation_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    summary = build_base_summary(
        args=args,
        total_input_rows=total_input_rows,
        selected_rows=selected_rows,
        candidates=candidates,
        sequences=sequences,
        duplicates=duplicates,
        output_paths=output_paths,
    )
    summary["detector_provenance"] = {
        "results_manifest": _file_artifact_state(
            results_manifest_path,
            label="detector results manifest",
        ),
        "scientific_config_hash": detector_manifest["scientific_config_hash"],
        "producer_identity_hash": detector_manifest["producer_identity_hash"],
    }

    if args.dry_run:
        summary["blast_status"] = "not_run_dry_run"
        write_json(summary_json, summary)
        write_markdown_summary(summary_md, summary)
        print_key_numbers(summary)
        return 0

    if not sequences:
        raise ValueError("No FASTA sequences were written; cannot run or parse BLAST annotations.")

    context = build_blast_artifact_context(
        args,
        fasta_path=fasta_path,
        candidate_manifest=candidates_csv,
        results_path=results_path,
        results_manifest_path=results_manifest_path,
        detector_manifest=detector_manifest,
        sequence_truth_manifest=sequence_truth_path,
        candidates=candidates,
    )
    blast_command: list[str]
    artifact_document: dict[str, Any]
    reuse_existing = bool(args.blast_tsv) or (blast_tsv.exists() and not args.force)
    if reuse_existing:
        if not blast_tsv.is_file():
            raise FileNotFoundError(f"BLAST TSV does not exist: {blast_tsv}")
        artifact_document = validate_blast_artifact_reuse(
            blast_artifact_manifest,
            blast_tsv=blast_tsv,
            context=context,
        )
        blast_command = list(artifact_document["reproduction_command"])
        summary["blast_status"] = "reused_verified_artifact"
        summary["blast_cache_reused"] = True
    else:
        if blast_artifact_manifest.exists() and not args.force:
            raise ValueError(
                "BLAST artifact manifest exists without its matching TSV; rerun explicitly "
                "with --force."
            )
        blast_tsv.parent.mkdir(parents=True, exist_ok=True)
        identity = context["identity"]
        blast_command = run_blastp(
            args,
            fasta_path,
            blast_tsv,
            tool_identity=identity["tool"],
            search_parameters=identity["search_parameters"],
        )
        if not blast_tsv.is_file():
            raise RuntimeError(f"blastp completed without creating its TSV: {blast_tsv}")
        post_run_context = build_blast_artifact_context(
            args,
            fasta_path=fasta_path,
            candidate_manifest=candidates_csv,
            results_path=results_path,
            results_manifest_path=results_manifest_path,
            detector_manifest=detector_manifest,
            sequence_truth_manifest=sequence_truth_path,
            candidates=candidates,
        )
        context_fields = ("artifact_key", "identity", "inputs")
        if any(post_run_context[field] != context[field] for field in context_fields):
            blast_tsv.unlink(missing_ok=True)
            raise RuntimeError(
                "BLAST tool, database, or scientific inputs changed during search; the newly "
                "created TSV was removed and no artifact manifest was published."
            )
        context = post_run_context
        summary["blast_status"] = "ran_blastp"
        summary["blast_cache_reused"] = False

    hits_by_query = read_blast_hits(
        blast_tsv,
        expected_query_ids=set(sequences),
    )
    if not reuse_existing:
        artifact_document = build_blast_artifact_manifest(
            context,
            blast_tsv=blast_tsv,
            command=blast_command,
        )
        write_json(blast_artifact_manifest, artifact_document)
    summary["blast_artifact_key"] = context["artifact_key"]
    summary["blast_artifact_manifest_sha256"] = _file_sha256(blast_artifact_manifest)
    summary["blast_command"] = blast_command
    _, annotation_summary = annotate_candidates(candidates, hits_by_query, args, annotation_csv)
    summary.update(annotation_summary)
    write_json(summary_json, summary)
    write_markdown_summary(summary_md, summary)
    print_key_numbers(summary)
    return 0


def _managed_output_inventory(out_dir: Path, *, state_path: Path) -> list[dict[str, Any]]:
    resolved_root = out_dir.expanduser().resolve()
    resolved_state = state_path.expanduser().resolve()
    inventory: list[dict[str, Any]] = []
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file() or path.resolve() == resolved_state:
            continue
        state = _file_artifact_state(path, label="BFVD run output")
        inventory.append(
            {
                "relative_path": path.relative_to(resolved_root).as_posix(),
                "size": state["size"],
                "sha256": state["sha256"],
            }
        )
    return inventory


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_arguments(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "run_state.json"
    state: dict[str, Any] = {
        "schema_version": BFVD_RUN_STATE_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": _utc_now(),
        "parameters": vars(args),
        "sequence_completeness_policy": BFVD_SEQUENCE_COMPLETENESS_POLICY,
        "outputs": [],
    }
    write_json(state_path, state)
    try:
        return_code = _run(args)
    except BaseException as exc:
        state.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "outputs": _managed_output_inventory(out_dir, state_path=state_path),
            }
        )
        write_json(state_path, state)
        raise
    state.update(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "outputs": _managed_output_inventory(out_dir, state_path=state_path),
        }
    )
    write_json(state_path, state)
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
