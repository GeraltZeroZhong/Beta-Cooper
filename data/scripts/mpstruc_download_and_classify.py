#!/usr/bin/env python3
"""
Download all structures referenced by an mpstruc beta-barrel XML dump and classify
entries at the assembly/entity level for D1-style canonical TMBB curation.

What this script does
---------------------
1. Parse an mpstruc XML dump (master proteins + member proteins + optional related entries).
2. Download entry mmCIF files from RCSB.
3. Download available biological assembly mmCIF files.
4. Parse each entry mmCIF and compute chain/entity-level summary:
   - protein entities
   - chain-to-entity mapping
   - observed residue counts
   - beta-strand counts from _struct_sheet_range
   - preferred assembly composition
5. Produce a preliminary classification useful for D1 curation:
   - SELF_CONTAINED_MONOMER
   - SELF_CONTAINED_HOMOOLIGOMER
   - SELF_CONTAINED_WITH_PARTNER_COMPLEX
   - ASSEMBLY_FORMED_OR_OUT_OF_SCOPE
   - PARTIAL_OR_DOMAIN_ONLY
   - DESIGNED_OR_OUT_OF_SCOPE
   - NEEDS_REVIEW
6. Create symlink/copy based file organization by mpstruc subgroup and by class.

Important note
--------------
The automatic classification creates review strata, not positive ground truth.
Publication mode requires a frozen approval manifest with one exact author-chain
target and independent curation evidence per accepted structure; unapproved
automatic candidates never enter the benchmark positive set.

Recommended usage
-----------------
python mpstruc_download_and_classify.py Mpstrucis.txt \
    --out mpstruc_beta_barrels \
    --mode exploratory \
    --threads 8 \
    --download-related \
    --link-mode symlink

Dependencies
------------
Python >= 3.10
pip install lxml biopython

Optional override file
----------------------
CSV/TSV with columns:
    pdb_code,class_label,note
Use this to force classification for known special cases.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import html
import http.client
import itertools
import json
import os
import re
import shutil
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from _dataset_provenance import (
    FrozenStructureArchive,
    RunManifest,
    atomic_write_csv,
    atomic_write_json,
    input_directory_record,
    input_file_record,
    prepare_new_output_directory,
    require_exact_int,
    require_finite_real,
    require_pinned_source_id,
)

try:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        f"Biopython is required. Install with: pip install biopython\nImport error: {e}"
    ) from e


PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")
OPERATION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ClassificationParameters:
    """Scientific cutoffs used by the preliminary positive-set classifier."""

    canonical_min_beta_strands: int = 8
    assembly_formed_max_beta_strands: int = 5
    assembly_formed_min_chain_copies: int = 3
    partial_modeled_fraction: float = 0.65
    partial_keyword_modeled_fraction: float = 0.75
    partial_keyword_min_beta_strands: int = 6


class ChainInfo(TypedDict):
    """Per-chain values used to summarize one protein entity."""

    label_asym_id: str
    auth_asym_id: str
    auth_asym_mapping_status: str
    observed_residues: int
    beta_strands: int
    modeled_fraction: float


DEFAULT_CLASSIFICATION_PARAMETERS = ClassificationParameters()
CLASS_LABELS = frozenset(
    {
        "SELF_CONTAINED_MONOMER",
        "SELF_CONTAINED_HOMOOLIGOMER",
        "SELF_CONTAINED_WITH_PARTNER_COMPLEX",
        "ASSEMBLY_FORMED_OR_OUT_OF_SCOPE",
        "PARTIAL_OR_DOMAIN_ONLY",
        "DESIGNED_OR_OUT_OF_SCOPE",
        "NEEDS_REVIEW",
    }
)
APPROVAL_REQUIRED_COLUMNS = frozenset(
    {"filename", "target_author_chain_id", "group_id", "curation_evidence"}
)

# Conservative keyword rules.
OUT_OF_SCOPE_KEYWORDS = [
    "hemolysin",
    "leukocidin",
    "toxin",
    "attack complex",
    "macpf",
    "secretin",
    "tolc",
    "oprm",
    "mtre",
    "cusc",
    "cmec",
    "wza",
    "csgg",
    "letb",
    "pore-forming",
]

PARTIAL_KEYWORDS = [
    "fragment",
    "truncated",
    "isolated beta-barrel domain",
    "isolated beta barrel domain",
    "beta-barrel domain",
    "beta barrel domain",
    "translocator domain",
    "domain only",
    "c-terminal domain",
    "c terminal domain",
]

# Sometimes a known good self-contained barrel appears in complex with a partner.
# You can extend these manually if needed.
MANUAL_CLASS_OVERRIDES_DEFAULT: dict[str, tuple[str, str]] = {
    # Example:
    # "8XCJ": ("SELF_CONTAINED_WITH_PARTNER_COMPLEX", "LamB barrel + gpJ partner complex"),
    # "1WP1": ("ASSEMBLY_FORMED_OR_OUT_OF_SCOPE", "OprM-type assembly-formed barrel"),
}


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_pdb_code(code: str) -> str:
    code = (code or "").strip().upper()
    return code


def sanitize_text(x: str | None) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(x))).strip()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def safe_get(mapping: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def listify(mapping: dict[str, Any], *keys: str) -> list[str]:
    value = safe_get(mapping, *keys)
    if value is None:
        return []
    return as_list(value)


def seq_len_from_cif_string(seq: str) -> int:
    seq = seq or ""
    # Remove CIF quoting artifacts and whitespace/newlines.
    seq = seq.replace("\n", "")
    seq = re.sub(r"\s+", "", seq)
    # Keep letters only.
    seq = re.sub(r"[^A-Za-z]", "", seq)
    return len(seq)


def parse_mpstruc_xml(path: str, include_related: bool = True) -> list[dict[str, str]]:
    """
    Parse the mpstruc dump using a tolerant line-based state machine.

    Why not a strict XML parser?
    The mpstruc text dump mixes XML-like structure with HTML entities/tags and
    a number of malformed records. For this file, line-based parsing of the
    specific fields we care about is substantially more robust than full XML
    recovery parsing.
    """
    raw_lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    tag_re = re.compile(r"<([A-Za-z0-9_:-]+)>(.*?)</\1>")

    def clean_inner_text(s: str | None) -> str:
        s = s or ""
        s = html.unescape(s)
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    records: list[dict[str, str]] = []

    group_name = ""
    subgroup_name = ""
    current_protein: dict[str, Any] | None = None
    current_member: dict[str, Any] | None = None
    current_related_owner: dict[str, Any] | None = None

    def add_record(
        pdb_code: str,
        source_type: str,
        group_name: str,
        subgroup_name: str,
        name: str,
        species: str,
        taxonomic_domain: str,
        expressed_in_species: str,
        resolution: str,
        description: str,
        master_pdb_code: str,
        parent_pdb_code: str,
    ) -> None:
        code = normalize_pdb_code(pdb_code)
        if not PDB_ID_RE.match(code):
            return
        records.append(
            {
                "pdb_code": code,
                "source_type": source_type,
                "group_name": sanitize_text(group_name),
                "subgroup_name": sanitize_text(subgroup_name),
                "mpstruc_name": sanitize_text(name),
                "species": sanitize_text(species),
                "taxonomic_domain": sanitize_text(taxonomic_domain),
                "expressed_in_species": sanitize_text(expressed_in_species),
                "resolution": sanitize_text(resolution),
                "description": sanitize_text(description),
                "master_pdb_code": normalize_pdb_code(master_pdb_code),
                "parent_pdb_code": normalize_pdb_code(parent_pdb_code),
            }
        )

    def flush_member() -> None:
        nonlocal current_member
        if current_member is None or current_protein is None:
            current_member = None
            return
        master_code = normalize_pdb_code(current_protein.get("pdbCode", ""))
        parent_code = normalize_pdb_code(current_member.get("pdbCode", ""))
        merged = dict(current_protein)
        merged.update({k: v for k, v in current_member.items() if not str(k).startswith("_")})
        add_record(
            pdb_code=parent_code,
            source_type="member",
            group_name=group_name,
            subgroup_name=subgroup_name,
            name=merged.get("name", ""),
            species=merged.get("species", ""),
            taxonomic_domain=merged.get("taxonomicDomain", ""),
            expressed_in_species=merged.get("expressedInSpecies", ""),
            resolution=merged.get("resolution", ""),
            description=merged.get("description", ""),
            master_pdb_code=master_code,
            parent_pdb_code=parent_code,
        )
        if include_related:
            for rel in current_member.get("_related", []):
                add_record(
                    pdb_code=rel,
                    source_type="related_member",
                    group_name=group_name,
                    subgroup_name=subgroup_name,
                    name=merged.get("name", ""),
                    species=merged.get("species", ""),
                    taxonomic_domain=merged.get("taxonomicDomain", ""),
                    expressed_in_species=merged.get("expressedInSpecies", ""),
                    resolution="",
                    description=merged.get("description", ""),
                    master_pdb_code=master_code,
                    parent_pdb_code=parent_code,
                )
        current_member = None

    def flush_protein() -> None:
        nonlocal current_protein
        if current_protein is None:
            return
        # Flush any still-open member first.
        flush_member()
        master_code = normalize_pdb_code(current_protein.get("pdbCode", ""))
        add_record(
            pdb_code=master_code,
            source_type="master",
            group_name=group_name,
            subgroup_name=subgroup_name,
            name=current_protein.get("name", ""),
            species=current_protein.get("species", ""),
            taxonomic_domain=current_protein.get("taxonomicDomain", ""),
            expressed_in_species=current_protein.get("expressedInSpecies", ""),
            resolution=current_protein.get("resolution", ""),
            description=current_protein.get("description", ""),
            master_pdb_code=master_code,
            parent_pdb_code=master_code,
        )
        if include_related:
            for rel in current_protein.get("_related", []):
                add_record(
                    pdb_code=rel,
                    source_type="related_master",
                    group_name=group_name,
                    subgroup_name=subgroup_name,
                    name=current_protein.get("name", ""),
                    species=current_protein.get("species", ""),
                    taxonomic_domain=current_protein.get("taxonomicDomain", ""),
                    expressed_in_species=current_protein.get("expressedInSpecies", ""),
                    resolution="",
                    description=current_protein.get("description", ""),
                    master_pdb_code=master_code,
                    parent_pdb_code=master_code,
                )
        current_protein = None

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue

        # Explicit open/close markers.
        if line == "<subgroup>":
            subgroup_name = ""
            continue
        if line == "</subgroup>":
            subgroup_name = ""
            continue
        if line == "<protein>":
            # Protein tags also appear inside <memberProtein>; only open a new master
            # protein when we are not already inside a member block.
            if current_member is None:
                flush_protein()
                current_protein = {"_related": []}
            continue
        if line == "</protein>":
            if current_member is None:
                flush_protein()
            continue
        if line == "<memberProtein>":
            if current_protein is not None:
                flush_member()
                current_member = {"_related": []}
            continue
        if line == "</memberProtein>":
            flush_member()
            continue
        if line == "<relatedPdbEntries>":
            current_related_owner = (
                current_member if current_member is not None else current_protein
            )
            if current_related_owner is not None and "_related" not in current_related_owner:
                current_related_owner["_related"] = []
            continue
        if line == "</relatedPdbEntries>":
            current_related_owner = None
            continue

        m = tag_re.fullmatch(line)
        if not m:
            continue

        tag, inner = m.group(1), m.group(2)
        inner_text = clean_inner_text(inner)

        if current_related_owner is not None and tag == "pdbCode":
            current_related_owner.setdefault("_related", []).append(inner_text)
            continue

        if current_member is not None:
            current_member[tag] = inner_text
            continue
        if current_protein is not None:
            current_protein[tag] = inner_text
            continue

        if tag == "name":
            if inner_text == "TRANSMEMBRANE PROTEINS: BETA-BARREL":
                group_name = inner_text
            elif inner_text:
                subgroup_name = inner_text

    # Flush trailing open records.
    flush_protein()

    return records


def choose_representative_record(records_for_code: list[dict[str, str]]) -> dict[str, str]:
    priority = {"master": 0, "member": 1, "related_master": 2, "related_member": 3}
    rec = sorted(
        records_for_code,
        key=lambda r: (
            priority.get(r["source_type"], 99),
            r["master_pdb_code"],
            r["parent_pdb_code"],
        ),
    )[0].copy()
    rec["all_source_types"] = ";".join(sorted({r["source_type"] for r in records_for_code}))
    rec["all_subgroups"] = ";".join(
        sorted({r["subgroup_name"] for r in records_for_code if r["subgroup_name"]})
    )
    rec["all_master_pdb_codes"] = ";".join(
        sorted({r["master_pdb_code"] for r in records_for_code if r["master_pdb_code"]})
    )
    rec["source_record_count"] = str(len(records_for_code))
    return rec


def load_overrides(path: str | None) -> dict[str, tuple[str, str]]:
    overrides = dict(MANUAL_CLASS_OVERRIDES_DEFAULT)
    if not path:
        return overrides
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Override file not found: {path}")
    with p.open("r", encoding="utf-8", newline="") as fh:
        # Support CSV or TSV.
        header_line = fh.readline()
        fh.seek(0)
        dialect = csv.excel_tab if "\t" in header_line else csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        required_columns = {"pdb_code", "class_label", "note"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Override file must contain columns {sorted(required_columns)}: {path}"
            )
        seen_codes: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            code = normalize_pdb_code(row.get("pdb_code", ""))
            label = sanitize_text(row.get("class_label", ""))
            note = sanitize_text(row.get("note", ""))
            if not PDB_ID_RE.fullmatch(code):
                raise ValueError(f"Invalid PDB code in override row {line_number}: {code!r}")
            if label not in CLASS_LABELS:
                raise ValueError(f"Invalid class label in override row {line_number}: {label!r}")
            if code in seen_codes:
                raise ValueError(f"Duplicate PDB override at row {line_number}: {code}")
            seen_codes.add(code)
            overrides[code] = (label, note)
    return overrides


def load_positive_approvals(path: str) -> list[dict[str, str]]:
    """Load a frozen, human/rule-curated truth manifest without inferring labels."""
    approval_path = Path(path)
    with approval_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not APPROVAL_REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                "Positive approval manifest must contain columns "
                f"{sorted(APPROVAL_REQUIRED_COLUMNS)}"
            )
        approvals: list[dict[str, str]] = []
        seen_filenames: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            normalized = {
                column: sanitize_text(row.get(column, "")) for column in APPROVAL_REQUIRED_COLUMNS
            }
            filename = Path(normalized["filename"]).name
            code = normalize_pdb_code(Path(filename).stem)
            if not PDB_ID_RE.fullmatch(code) or Path(filename).suffix.lower() not in {
                ".cif",
                ".mmcif",
            }:
                raise ValueError(
                    f"Approval row {line_number} has invalid structure filename: {filename!r}"
                )
            for column in ("target_author_chain_id", "group_id", "curation_evidence"):
                if not normalized[column]:
                    raise ValueError(
                        f"Approval row {line_number} has blank required field {column!r}"
                    )
            filename_key = filename.casefold()
            if filename_key in seen_filenames:
                raise ValueError(
                    f"Positive approval manifest must contain exactly one target chain per "
                    f"structure file; duplicate filename {filename!r}"
                )
            seen_filenames.add(filename_key)
            approvals.append({**normalized, "filename": filename, "pdb_code": code})
    if not approvals:
        raise ValueError("Positive approval manifest contains no approved structures")
    return approvals


def match_positive_approvals(
    approvals: Sequence[dict[str, str]], entry_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Require every approved truth row to match an automatic candidate and chain."""
    candidates = {
        str(row["pdb_code"]): row
        for row in entry_rows
        if row.get("automatic_candidate_stratum") == "True"
    }
    matched: list[dict[str, Any]] = []
    for approval in approvals:
        code = approval["pdb_code"]
        candidate = candidates.get(code)
        if candidate is None:
            raise ValueError(
                f"Approved positive {code} is not an automatically stratified candidate"
            )
        allowed_auth_chains = {
            chain.strip()
            for chain in str(candidate.get("candidate_auth_asym_ids", "")).split(";")
            if chain.strip()
        }
        representative_author_chain_id = sanitize_text(
            candidate.get("representative_author_chain_id", "")
        )
        if representative_author_chain_id:
            allowed_auth_chains.add(representative_author_chain_id)
        if approval["target_author_chain_id"] not in allowed_auth_chains:
            raise ValueError(
                f"Approved target chain {approval['target_author_chain_id']!r} for {code} is not an "
                f"exact parsed author-chain identifier {sorted(allowed_auth_chains)}"
            )
        matched.append(
            {
                **approval,
                "target_author_chain_id": approval["target_author_chain_id"],
                "truth_label": "BARREL",
                "automatic_class_label": candidate.get("class_label", ""),
                "entry_cif_path": candidate.get("entry_cif_path", ""),
                "entry_cif_sha256_recorded_in_run_manifest": True,
            }
        )
    return matched


def urlopen_with_retries(
    url: str,
    timeout: int = 30,
    retries: int = 2,
    backoff: float = 1.0,
) -> http.client.HTTPResponse:
    last_err: Exception | None = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "mpstruc-download-classify/1.0",
                    "Accept": "*/*",
                },
                method="GET",
            )
            response: object = urllib.request.urlopen(req, timeout=timeout)
            if not isinstance(response, http.client.HTTPResponse):
                raise TypeError(f"HTTP response for {url} has an unsupported type")
            return response
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if i < retries:
                time.sleep(backoff * (2**i))
            else:
                raise
    raise RuntimeError(f"Unreachable retry code for URL {url}: {last_err}")


def looks_like_cif(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(256).decode("utf-8", errors="ignore").lstrip()
        return head.startswith("data_") or "_entry.id" in head or "loop_" in head
    except Exception:
        return False


def download_file_variants(
    urls: Sequence[tuple[str, str]], out_path: str, timeout: int, retries: int, backoff: float
) -> tuple[bool, str]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True, "exists"

    for url, kind in urls:
        try:
            with urlopen_with_retries(
                url, timeout=timeout, retries=retries, backoff=backoff
            ) as resp:
                status = getattr(resp, "status", None)
                if status is not None and status != 200:
                    continue
                if kind == "plain":
                    with open(tmp_path, "wb") as out:
                        while True:
                            chunk = resp.read(1024 * 256)
                            if not chunk:
                                break
                            out.write(chunk)
                elif kind == "gz":
                    with gzip.GzipFile(fileobj=resp) as gz, open(tmp_path, "wb") as out:
                        while True:
                            chunk = gz.read(1024 * 256)
                            if not chunk:
                                break
                            out.write(chunk)
                else:
                    raise ValueError(f"Unknown download kind: {kind}")

            if looks_like_cif(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, out_path)
                return True, f"downloaded:{kind}"
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                continue
            return False, f"HTTPError {e.code}: {e.reason}"
        except Exception as e:
            return False, f"Error: {e}"

    return False, "No usable file from provided URLs"


def download_entry_cif(
    code: str, out_dir: str, timeout: int, retries: int, backoff: float
) -> tuple[str, bool, str, str]:
    code = normalize_pdb_code(code)
    out_path = os.path.join(out_dir, f"{code}.cif")
    urls = [
        (f"https://files.rcsb.org/download/{code}.cif", "plain"),
        (f"https://files.rcsb.org/download/{code}.cif.gz", "gz"),
    ]
    ok, msg = download_file_variants(urls, out_path, timeout, retries, backoff)
    return code, ok, msg, out_path


def download_assembly_cif(
    code: str, assembly_id: str, out_dir: str, timeout: int, retries: int, backoff: float
) -> tuple[str, str, bool, str, str]:
    code = normalize_pdb_code(code)
    aid = str(assembly_id).strip()
    out_path = os.path.join(out_dir, f"{code}-assembly{aid}.cif")
    urls = [
        (f"https://files.rcsb.org/download/{code}-assembly{aid}.cif", "plain"),
        (f"https://files.rcsb.org/download/{code}-assembly{aid}.cif.gz", "gz"),
    ]
    ok, msg = download_file_variants(urls, out_path, timeout, retries, backoff)
    return code, aid, ok, msg, out_path


def stage_entry_cif(
    code: str, out_dir: str, archive: FrozenStructureArchive
) -> tuple[str, bool, str, str]:
    code = normalize_pdb_code(code)
    output = Path(out_dir) / f"{code}.cif"
    aliases = [
        f"{code}.cif",
        f"{code}.mmcif",
        f"{code}.cif.gz",
        f"{code}.mmcif.gz",
    ]
    source = archive.stage(aliases, output)
    return code, True, f"staged_local:{source.relative_to(archive.root)}", str(output)


def stage_assembly_cif(
    code: str,
    assembly_id: str,
    out_dir: str,
    archive: FrozenStructureArchive,
) -> tuple[str, str, bool, str, str]:
    code = normalize_pdb_code(code)
    assembly = str(assembly_id).strip()
    output = Path(out_dir) / f"{code}-assembly{assembly}.cif"
    stem = f"{code}-assembly{assembly}"
    aliases = [f"{stem}.cif", f"{stem}.mmcif", f"{stem}.cif.gz", f"{stem}.mmcif.gz"]
    source = archive.stage(aliases, output)
    return (
        code,
        assembly,
        True,
        f"staged_local:{source.relative_to(archive.root)}",
        str(output),
    )


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def expand_operation_expression(expression: str) -> set[tuple[str, ...]]:
    """Expand an mmCIF operation expression into unique transformation tuples.

    The mmCIF grammar represents Cartesian products as adjacent parenthesized
    groups, for example ``(1-3)(4,5)`` has six transformations. A single list
    such as ``1,2,3`` has three. Malformed expressions are rejected so assembly
    stoichiometry cannot silently collapse to one copy.
    """
    compact = re.sub(r"\s+", "", expression)
    if compact in {"", ".", "?"}:
        return {("1",)}

    if compact.startswith("("):
        groups = re.findall(r"\(([^()]*)\)", compact)
        if not groups or "".join(f"({group})" for group in groups) != compact:
            raise ValueError(f"Unsupported operation expression: {expression!r}")
    else:
        if "(" in compact or ")" in compact:
            raise ValueError(f"Malformed operation expression: {expression!r}")
        groups = [compact]

    expanded_groups: list[list[str]] = []
    for group in groups:
        identifiers: set[str] = set()
        for token in group.split(","):
            if not token:
                raise ValueError(f"Empty operation identifier in {expression!r}")
            numeric_range = re.fullmatch(r"(\d+)-(\d+)", token)
            if numeric_range:
                start, end = (int(value) for value in numeric_range.groups())
                if end < start:
                    raise ValueError(f"Descending operation range in {expression!r}")
                identifiers.update(str(value) for value in range(start, end + 1))
            elif OPERATION_ID_RE.fullmatch(token):
                identifiers.add(token)
            else:
                raise ValueError(f"Invalid operation identifier in {expression!r}")
        expanded_groups.append(sorted(identifiers))
    transformations = set(itertools.product(*expanded_groups))
    if not transformations:
        raise ValueError(f"Operation expression has no transformations: {expression!r}")
    return transformations


def operation_expression_multiplicity(expression: str) -> int:
    """Return the number of unique transforms in an mmCIF operation expression."""
    return len(expand_operation_expression(expression))


def parse_cif_summary(cif_path: str) -> dict[str, Any]:
    d = MMCIF2Dict(cif_path)

    entry_id = sanitize_text(
        (as_list(safe_get(d, "_entry.id")) or [Path(cif_path).stem.split("-")[0]])[0]
    ).upper()

    entity_ids = listify(d, "_entity.id")
    entity_types = listify(d, "_entity.type")
    entity_descs = listify(d, "_entity.pdbx_description")

    entity_desc_map: dict[str, str] = {}
    entity_type_map: dict[str, str] = {}
    for i, eid in enumerate(entity_ids):
        eid = sanitize_text(eid)
        entity_type_map[eid] = sanitize_text(entity_types[i]) if i < len(entity_types) else ""
        entity_desc_map[eid] = sanitize_text(entity_descs[i]) if i < len(entity_descs) else ""

    entity_poly_eids = listify(d, "_entity_poly.entity_id")
    entity_poly_types = listify(d, "_entity_poly.type")
    entity_poly_seq = listify(
        d, "_entity_poly.pdbx_seq_one_letter_code_can", "_entity_poly.pdbx_seq_one_letter_code"
    )

    entity_poly_type_map: dict[str, str] = {}
    entity_seq_len_map: dict[str, int] = {}
    for i, eid in enumerate(entity_poly_eids):
        eid = sanitize_text(eid)
        entity_poly_type_map[eid] = (
            sanitize_text(entity_poly_types[i]) if i < len(entity_poly_types) else ""
        )
        if i < len(entity_poly_seq):
            entity_seq_len_map[eid] = seq_len_from_cif_string(entity_poly_seq[i])

    # Fallback sequence lengths from entity_poly_seq loop.
    if not entity_seq_len_map:
        eps_eids = listify(d, "_entity_poly_seq.entity_id")
        for eid in eps_eids:
            eid = sanitize_text(eid)
            entity_seq_len_map[eid] = entity_seq_len_map.get(eid, 0) + 1

    struct_asym_ids = listify(d, "_struct_asym.id")
    struct_asym_entity_ids = listify(d, "_struct_asym.entity_id")
    asym_to_entity: dict[str, str] = {}
    for i, asym in enumerate(struct_asym_ids):
        asym = sanitize_text(asym)
        eid = sanitize_text(struct_asym_entity_ids[i]) if i < len(struct_asym_entity_ids) else ""
        if asym:
            asym_to_entity[asym] = eid

    protein_entity_ids = set()
    for eid, etype in entity_type_map.items():
        poly_type = entity_poly_type_map.get(eid, "").lower()
        if etype.lower() == "polymer" and "polypeptide" in poly_type:
            protein_entity_ids.add(eid)

    # Exact label_asym_id -> auth_asym_id mapping. Candidate truth is expressed
    # in author-chain identifiers, so missing mappings must never fall back to
    # label identifiers and ambiguous mappings must never select the first row.
    atom_label_asym = listify(d, "_atom_site.label_asym_id")
    atom_auth_asym = listify(d, "_atom_site.auth_asym_id")
    atom_group_pdb = listify(d, "_atom_site.group_PDB")
    atom_label_seq = listify(d, "_atom_site.label_seq_id")
    atom_comp_id = listify(d, "_atom_site.label_comp_id")

    auth_chain_candidates: dict[str, set[str]] = defaultdict(set)
    for label_column, auth_column in (
        ("_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.pdb_strand_id"),
        ("_atom_site.label_asym_id", "_atom_site.auth_asym_id"),
    ):
        label_values = listify(d, label_column)
        auth_values = listify(d, auth_column)
        if not label_values and not auth_values:
            continue
        if len(label_values) != len(auth_values):
            raise ValueError(
                f"Inconsistent mmCIF author-chain mapping columns {label_column!r} and "
                f"{auth_column!r}"
            )
        for label_value, auth_value in zip(label_values, auth_values, strict=True):
            label_asym = sanitize_text(label_value)
            auth_asym = sanitize_text(auth_value)
            is_protein_polymer_chain = asym_to_entity.get(label_asym) in protein_entity_ids
            if label_asym and is_protein_polymer_chain and auth_asym not in {"", ".", "?"}:
                auth_chain_candidates[label_asym].add(auth_asym)

    labels_by_unique_auth: dict[str, set[str]] = defaultdict(set)
    for label_asym, auth_candidates in auth_chain_candidates.items():
        if len(auth_candidates) == 1:
            labels_by_unique_auth[next(iter(auth_candidates))].add(label_asym)
    auth_chain_map = {
        label_asym: next(iter(auth_values))
        for label_asym, auth_values in auth_chain_candidates.items()
        if len(auth_values) == 1 and len(labels_by_unique_auth[next(iter(auth_values))]) == 1
    }

    def auth_mapping_status(label_asym: str) -> str:
        candidates = auth_chain_candidates.get(label_asym, set())
        if not candidates:
            return "missing"
        if len(candidates) != 1:
            return "ambiguous_multiple_author_chains"
        auth_asym = next(iter(candidates))
        if len(labels_by_unique_auth[auth_asym]) != 1:
            return "ambiguous_multiple_label_chains"
        return "exact"

    observed_residues_by_chain: dict[str, set[str]] = defaultdict(set)

    n_atoms = max(
        len(atom_label_asym),
        len(atom_auth_asym),
        len(atom_group_pdb),
        len(atom_label_seq),
        len(atom_comp_id),
    )
    for i in range(n_atoms):
        label_asym = sanitize_text(atom_label_asym[i]) if i < len(atom_label_asym) else ""
        group_pdb = sanitize_text(atom_group_pdb[i]) if i < len(atom_group_pdb) else ""
        label_seq_id = sanitize_text(atom_label_seq[i]) if i < len(atom_label_seq) else ""
        comp_id = sanitize_text(atom_comp_id[i]) if i < len(atom_comp_id) else ""
        if not label_asym:
            continue
        if group_pdb not in {"ATOM", "HETATM"}:
            continue
        if label_seq_id in {"", ".", "?"}:
            continue
        if comp_id.upper() == "HOH":
            continue
        observed_residues_by_chain[label_asym].add(label_seq_id)

    observed_len_by_chain = {
        chain: len(res_ids) for chain, res_ids in observed_residues_by_chain.items()
    }

    # Count beta-strand ranges per chain.
    ss_beg_chain = listify(
        d, "_struct_sheet_range.beg_label_asym_id", "_struct_sheet_range.beg_auth_asym_id"
    )
    ss_end_chain = listify(
        d, "_struct_sheet_range.end_label_asym_id", "_struct_sheet_range.end_auth_asym_id"
    )
    ss_beg_seq = listify(
        d, "_struct_sheet_range.beg_label_seq_id", "_struct_sheet_range.beg_auth_seq_id"
    )
    ss_end_seq = listify(
        d, "_struct_sheet_range.end_label_seq_id", "_struct_sheet_range.end_auth_seq_id"
    )
    ss_sheet_id = listify(d, "_struct_sheet_range.sheet_id")
    beta_ranges_by_chain: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    n_ss = max(
        len(ss_beg_chain), len(ss_end_chain), len(ss_beg_seq), len(ss_end_seq), len(ss_sheet_id)
    )
    for i in range(n_ss):
        beg_chain = sanitize_text(ss_beg_chain[i]) if i < len(ss_beg_chain) else ""
        end_chain = sanitize_text(ss_end_chain[i]) if i < len(ss_end_chain) else beg_chain
        if not beg_chain or (end_chain and end_chain != beg_chain):
            continue
        rng = (
            sanitize_text(ss_sheet_id[i]) if i < len(ss_sheet_id) else f"sheet_{i + 1}",
            sanitize_text(ss_beg_seq[i]) if i < len(ss_beg_seq) else "",
            sanitize_text(ss_end_seq[i]) if i < len(ss_end_seq) else "",
        )
        beta_ranges_by_chain[beg_chain].add(rng)
    beta_count_by_chain = {chain: len(ranges) for chain, ranges in beta_ranges_by_chain.items()}

    # Assembly composition.
    assembly_ids = unique_preserve_order(listify(d, "_pdbx_struct_assembly.id"))
    assembly_gen_ids = listify(d, "_pdbx_struct_assembly_gen.assembly_id")
    assembly_gen_asym_lists = listify(d, "_pdbx_struct_assembly_gen.asym_id_list")
    assembly_gen_operations = listify(d, "_pdbx_struct_assembly_gen.oper_expression")

    assembly_chain_operations: dict[str, dict[str, set[tuple[str, ...]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    assembly_operation_errors: dict[str, list[str]] = defaultdict(list)
    for i, aid in enumerate(assembly_gen_ids):
        aid = sanitize_text(aid)
        asym_list = (
            sanitize_text(assembly_gen_asym_lists[i]) if i < len(assembly_gen_asym_lists) else ""
        )
        operation_expression = (
            sanitize_text(assembly_gen_operations[i]) if i < len(assembly_gen_operations) else "1"
        )
        try:
            operations = expand_operation_expression(operation_expression)
        except ValueError as error:
            assembly_operation_errors[aid].append(str(error))
            continue
        chains = []
        for token in asym_list.split(","):
            token = token.strip()
            if token:
                chains.append(token)
        for chain in chains:
            assembly_chain_operations[aid][chain].update(operations)

    assembly_chain_copy_counts = {
        assembly: {chain: len(operations) for chain, operations in chain_map.items()}
        for assembly, chain_map in assembly_chain_operations.items()
    }

    preferred_assembly_id = (
        "1"
        if "1" in assembly_ids or "1" in assembly_chain_copy_counts
        else (
            assembly_ids[0]
            if assembly_ids
            else (sorted(assembly_chain_copy_counts)[0] if assembly_chain_copy_counts else "1")
        )
    )
    preferred_chain_copy_counts = dict(assembly_chain_copy_counts.get(preferred_assembly_id, {}))
    if not preferred_chain_copy_counts:
        assembly_operation_errors[preferred_assembly_id].append(
            "preferred assembly has no valid assembly-generation definition"
        )
        # Keep asymmetric-unit counts for diagnostics. Classification fails
        # closed because the preferred biological assembly is unresolved.
        preferred_chain_copy_counts = dict.fromkeys(unique_preserve_order(struct_asym_ids), 1)
    preferred_assembly_chains = list(preferred_chain_copy_counts)

    # Build protein entity summary.
    entity_summary: dict[str, dict[str, Any]] = {}
    for eid in sorted(protein_entity_ids):
        chains = sorted([asym for asym, ent in asym_to_entity.items() if ent == eid])
        chain_infos: list[ChainInfo] = []
        for chain in chains:
            seq_len = entity_seq_len_map.get(eid, 0)
            observed_len = observed_len_by_chain.get(chain, 0)
            beta_n = beta_count_by_chain.get(chain, 0)
            modeled_frac = (observed_len / seq_len) if seq_len else 0.0
            chain_infos.append(
                {
                    "label_asym_id": chain,
                    "auth_asym_id": auth_chain_map.get(chain, ""),
                    "auth_asym_mapping_status": auth_mapping_status(chain),
                    "observed_residues": observed_len,
                    "beta_strands": beta_n,
                    "modeled_fraction": modeled_frac,
                }
            )
        entity_summary[eid] = {
            "entity_id": eid,
            "description": entity_desc_map.get(eid, ""),
            "entity_type": entity_type_map.get(eid, ""),
            "poly_type": entity_poly_type_map.get(eid, ""),
            "seq_len": entity_seq_len_map.get(eid, 0),
            "chains": chain_infos,
            "max_beta_strands": max([x["beta_strands"] for x in chain_infos], default=0),
            "max_observed_residues": max([x["observed_residues"] for x in chain_infos], default=0),
            "max_modeled_fraction": max([x["modeled_fraction"] for x in chain_infos], default=0.0),
            "chain_count_entry": len(chains),
            "chain_count_preferred_assembly": sum(
                copies
                for ch, copies in preferred_chain_copy_counts.items()
                if asym_to_entity.get(ch) == eid
            ),
        }

    protein_entities_in_preferred_assembly = sorted(
        {
            asym_to_entity.get(ch, "")
            for ch in preferred_assembly_chains
            if asym_to_entity.get(ch, "") in protein_entity_ids
        }
    )

    return {
        "entry_id": entry_id,
        "preferred_assembly_id": preferred_assembly_id,
        "preferred_assembly_chains": preferred_assembly_chains,
        "preferred_assembly_chain_copy_counts": preferred_chain_copy_counts,
        "assembly_operation_errors": assembly_operation_errors.get(preferred_assembly_id, []),
        "protein_entity_ids": sorted(protein_entity_ids),
        "protein_entities_in_preferred_assembly": protein_entities_in_preferred_assembly,
        "entity_summary": entity_summary,
        "available_assembly_ids": assembly_ids,
        "asym_to_entity": asym_to_entity,
        "auth_chain_map": auth_chain_map,
        "auth_chain_mapping_status": {
            chain: auth_mapping_status(chain) for chain in sorted(asym_to_entity)
        },
        "observed_len_by_chain": observed_len_by_chain,
        "beta_count_by_chain": beta_count_by_chain,
    }


def choose_candidate_entity(entity_summary: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not entity_summary:
        return None
    entities = list(entity_summary.values())
    entities.sort(
        key=lambda x: (
            int(x.get("max_beta_strands", 0)),
            float(x.get("max_modeled_fraction", 0.0)),
            int(x.get("seq_len", 0)),
            int(x.get("max_observed_residues", 0)),
        ),
        reverse=True,
    )
    return entities[0]


def classify_entry(
    rep: dict[str, str],
    cif_summary: dict[str, Any],
    overrides: dict[str, tuple[str, str]],
    parameters: ClassificationParameters = DEFAULT_CLASSIFICATION_PARAMETERS,
) -> dict[str, Any]:
    code = rep["pdb_code"]
    if code in overrides:
        forced_label, forced_note = overrides[code]
        out = {
            "class_label": forced_label,
            "class_reason": f"manual_override: {forced_note}".strip(),
            "automatic_candidate_stratum": str(
                forced_label
                in {
                    "SELF_CONTAINED_MONOMER",
                    "SELF_CONTAINED_HOMOOLIGOMER",
                    "SELF_CONTAINED_WITH_PARTNER_COMPLEX",
                }
            ),
        }
        return out

    subgroup = rep.get("subgroup_name", "")
    subgroup_l = subgroup.lower()
    combined_text = " ".join(
        [
            rep.get("mpstruc_name", ""),
            rep.get("description", ""),
            subgroup,
        ]
    ).lower()

    candidate = choose_candidate_entity(cif_summary.get("entity_summary", {}))
    if candidate is None:
        return {
            "class_label": "NEEDS_REVIEW",
            "class_reason": "no_protein_entity_detected",
            "automatic_candidate_stratum": "False",
        }

    operation_errors = cif_summary.get("assembly_operation_errors", [])
    if operation_errors:
        return {
            "class_label": "NEEDS_REVIEW",
            "class_reason": "unresolved_assembly_operation_expression: "
            + "; ".join(str(error) for error in operation_errors),
            "automatic_candidate_stratum": "False",
        }

    max_beta = int(candidate.get("max_beta_strands", 0))
    modeled_frac = float(candidate.get("max_modeled_fraction", 0.0))
    seq_len = int(candidate.get("seq_len", 0))
    preferred_entity_ids = cif_summary.get("protein_entities_in_preferred_assembly", [])
    n_protein_entities_assembly = len(preferred_entity_ids)
    n_candidate_chains_assembly = int(candidate.get("chain_count_preferred_assembly", 0))

    if "adventitious membrane proteins" in subgroup_l or "attack complexes" in subgroup_l:
        label = "ASSEMBLY_FORMED_OR_OUT_OF_SCOPE"
        reason = "mpstruc_subgroup_adventitious_pore_forming"
    elif "de novo designed" in subgroup_l:
        label = "DESIGNED_OR_OUT_OF_SCOPE"
        reason = "mpstruc_subgroup_de_novo_designed"
    elif any(k in combined_text for k in OUT_OF_SCOPE_KEYWORDS):
        label = "ASSEMBLY_FORMED_OR_OUT_OF_SCOPE"
        reason = "keyword_out_of_scope"
    elif modeled_frac < parameters.partial_modeled_fraction or (
        seq_len > 0
        and max_beta >= parameters.partial_keyword_min_beta_strands
        and modeled_frac < parameters.partial_keyword_modeled_fraction
        and any(k in combined_text for k in PARTIAL_KEYWORDS)
    ):
        label = "PARTIAL_OR_DOMAIN_ONLY"
        reason = f"low_modeled_fraction_or_partial_keyword(modeled_fraction={modeled_frac:.3f})"
    elif max_beta >= parameters.canonical_min_beta_strands:
        if n_protein_entities_assembly <= 1:
            if n_candidate_chains_assembly <= 1:
                label = "SELF_CONTAINED_MONOMER"
                reason = f"candidate_entity_has_{max_beta}_beta_strands_single_chain"
            else:
                label = "SELF_CONTAINED_HOMOOLIGOMER"
                reason = f"candidate_entity_has_{max_beta}_beta_strands_homooligomer"
        else:
            label = "SELF_CONTAINED_WITH_PARTNER_COMPLEX"
            reason = f"candidate_entity_has_{max_beta}_beta_strands_plus_partner_entities"
    elif (
        max_beta <= parameters.assembly_formed_max_beta_strands
        and n_candidate_chains_assembly >= parameters.assembly_formed_min_chain_copies
    ):
        label = "ASSEMBLY_FORMED_OR_OUT_OF_SCOPE"
        reason = f"low_per_chain_beta_strands({max_beta})_multichain_assembly"
    else:
        label = "NEEDS_REVIEW"
        reason = f"ambiguous_beta_strands={max_beta};modeled_fraction={modeled_frac:.3f};protein_entities_in_assembly={n_protein_entities_assembly}"

    return {
        "class_label": label,
        "class_reason": reason,
        "automatic_candidate_stratum": str(
            label
            in {
                "SELF_CONTAINED_MONOMER",
                "SELF_CONTAINED_HOMOOLIGOMER",
                "SELF_CONTAINED_WITH_PARTNER_COMPLEX",
            }
        ),
    }


def safe_symlink_or_copy(src: str, dst: str, link_mode: str = "symlink") -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    if link_mode == "symlink":
        try:
            rel_src = os.path.relpath(src, os.path.dirname(dst))
            os.symlink(rel_src, dst)
            return
        except Exception:
            pass
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    atomic_write_csv(rows, path, fieldnames)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python data/scripts/mpstruc_download_and_classify.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Parse an mpstruc snapshot, acquire entry-level mmCIF structures, summarize chain "
            "geometry, and organize beta-barrel candidates for curation."
        ),
        epilog=(
            "Output: parsed and classified metadata CSVs, entry and optional assembly files, "
            "curation tables, approved positive structures, summary.json, and run_manifest.json. "
            "The output directory must be new or empty. Argument errors exit with status 2; "
            "input, download, classification, or approval failures exit nonzero."
        ),
    )
    ap.add_argument(
        "input_xml",
        metavar="MPSTRUC_XML",
        help="Local mpstruc XML/TXT snapshot, such as Mpstrucis.txt.",
    )
    ap.add_argument(
        "--mpstruc-source-id",
        default=None,
        metavar="ID",
        help="mpstruc URL, DOI, or release identifier recorded in provenance.",
    )
    ap.add_argument(
        "--out",
        default="mpstruc_beta_barrels",
        metavar="DIRECTORY",
        help="New output directory for structures, classifications, and run metadata.",
    )
    ap.add_argument(
        "--mode",
        choices=["publication", "exploratory"],
        default="publication",
        help=(
            "publication uses a local structure archive and an approval manifest; exploratory "
            "permits downloads and produces candidate tables for curation."
        ),
    )
    ap.add_argument(
        "--threads", type=int, default=8, metavar="N", help="Concurrent structure downloads."
    )
    ap.add_argument(
        "--timeout", type=int, default=30, metavar="SECONDS", help="Timeout per HTTP request."
    )
    ap.add_argument(
        "--retries", type=int, default=2, metavar="N", help="Retries after each failed request."
    )
    ap.add_argument(
        "--backoff",
        type=float,
        default=0.8,
        metavar="SECONDS",
        help="Base delay for exponential HTTP retry backoff.",
    )
    ap.add_argument(
        "--download-related",
        action="store_true",
        help="Include relatedPdbEntries in addition to mpstruc master and member entries.",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Parse the snapshot and write metadata without acquiring structure files.",
    )
    ap.add_argument(
        "--skip-assemblies",
        action="store_true",
        help="Acquire entry-level mmCIF files without biological assembly files.",
    )
    ap.add_argument(
        "--override-csv",
        default=None,
        metavar="CSV",
        help="Manual classification table with columns pdb_code, class_label, and note.",
    )
    ap.add_argument(
        "--positive-approval-manifest",
        default=None,
        metavar="CSV",
        help=(
            "Frozen CSV truth manifest with filename,target_author_chain_id,group_id,curation_evidence; "
            "required in publication mode."
        ),
    )
    ap.add_argument(
        "--structure-source-dir",
        default=None,
        metavar="DIRECTORY",
        help=(
            "Local entry-level mmCIF archive; required in publication mode. Exploratory mode "
            "downloads missing entries when omitted."
        ),
    )
    ap.add_argument(
        "--structure-source-id",
        default=None,
        metavar="ID",
        help="Coordinate-archive URL, DOI, or release identifier recorded in provenance.",
    )
    ap.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="Filesystem operation used to place structures in classification directories.",
    )
    defaults = DEFAULT_CLASSIFICATION_PARAMETERS
    ap.add_argument(
        "--canonical-min-beta-strands",
        type=int,
        default=defaults.canonical_min_beta_strands,
        metavar="N",
        help="Minimum beta-strand count for the automatic canonical-barrel candidate stratum.",
    )
    ap.add_argument(
        "--assembly-formed-max-beta-strands",
        type=int,
        default=defaults.assembly_formed_max_beta_strands,
        metavar="N",
        help="Maximum per-chain beta-strand count for the assembly-formed candidate stratum.",
    )
    ap.add_argument(
        "--assembly-formed-min-chain-copies",
        type=int,
        default=defaults.assembly_formed_min_chain_copies,
        metavar="N",
        help="Minimum same-chain copies supporting an assembly-formed candidate.",
    )
    ap.add_argument(
        "--partial-modeled-fraction",
        type=float,
        default=defaults.partial_modeled_fraction,
        metavar="FRACTION",
        help="Maximum modeled-residue fraction for the partial-structure candidate stratum.",
    )
    ap.add_argument(
        "--partial-keyword-modeled-fraction",
        type=float,
        default=defaults.partial_keyword_modeled_fraction,
        metavar="FRACTION",
        help="Maximum modeled fraction for entries whose text indicates a partial structure.",
    )
    ap.add_argument(
        "--partial-keyword-min-beta-strands",
        type=int,
        default=defaults.partial_keyword_min_beta_strands,
        metavar="N",
        help="Minimum beta-strand count for keyword-supported partial candidates.",
    )
    return ap


def validate_arguments(args: argparse.Namespace) -> ClassificationParameters:
    require_exact_int("threads", args.threads, minimum=1)
    require_exact_int("timeout", args.timeout, minimum=1)
    require_exact_int("retries", args.retries, minimum=0)
    require_finite_real("backoff", args.backoff, minimum=0.0)
    input_file_record(args.input_xml, source="local mpstruc snapshot", pinned=True)
    if args.override_csv:
        input_file_record(args.override_csv, source="local manual overrides", pinned=True)
    if args.positive_approval_manifest:
        input_file_record(
            args.positive_approval_manifest,
            source="local frozen positive approvals",
            pinned=True,
        )
        load_positive_approvals(args.positive_approval_manifest)
    if args.structure_source_dir:
        input_directory_record(
            args.structure_source_dir, source="local frozen coordinate archive", pinned=True
        )
    if args.mode == "publication":
        if not args.positive_approval_manifest:
            raise ValueError(
                "publication mode requires --positive-approval-manifest; mpstruc/strand-range "
                "classification alone is not positive ground truth"
            )
        if not args.structure_source_dir:
            raise ValueError(
                "publication mode requires --structure-source-dir and never downloads live "
                "RCSB coordinates"
            )
        require_pinned_source_id("mpstruc_source_id", args.mpstruc_source_id)
        require_pinned_source_id("structure_source_id", args.structure_source_id)
    if args.mode == "publication" and args.skip_download:
        raise ValueError("--skip-download is exploratory-only because approvals cannot be verified")
    if args.mode == "publication" and args.skip_assemblies:
        raise ValueError("--skip-assemblies is exploratory-only in a publication build")
    for name in ("download_related", "skip_download", "skip_assemblies"):
        if not isinstance(getattr(args, name), bool):
            raise ValueError(f"{name} must be a boolean")

    parameters = ClassificationParameters(
        canonical_min_beta_strands=require_exact_int(
            "canonical_min_beta_strands", args.canonical_min_beta_strands, minimum=1
        ),
        assembly_formed_max_beta_strands=require_exact_int(
            "assembly_formed_max_beta_strands",
            args.assembly_formed_max_beta_strands,
            minimum=0,
        ),
        assembly_formed_min_chain_copies=require_exact_int(
            "assembly_formed_min_chain_copies",
            args.assembly_formed_min_chain_copies,
            minimum=2,
        ),
        partial_modeled_fraction=require_finite_real(
            "partial_modeled_fraction", args.partial_modeled_fraction, minimum=0.0, maximum=1.0
        ),
        partial_keyword_modeled_fraction=require_finite_real(
            "partial_keyword_modeled_fraction",
            args.partial_keyword_modeled_fraction,
            minimum=0.0,
            maximum=1.0,
        ),
        partial_keyword_min_beta_strands=require_exact_int(
            "partial_keyword_min_beta_strands",
            args.partial_keyword_min_beta_strands,
            minimum=1,
        ),
    )
    if parameters.assembly_formed_max_beta_strands >= parameters.canonical_min_beta_strands:
        raise ValueError(
            "assembly_formed_max_beta_strands must be below canonical_min_beta_strands"
        )
    if parameters.partial_modeled_fraction > parameters.partial_keyword_modeled_fraction:
        raise ValueError("partial_modeled_fraction cannot exceed partial_keyword_modeled_fraction")
    return parameters


def _run(
    args: argparse.Namespace,
    out_dir: Path,
    run_manifest: RunManifest,
    parameters: ClassificationParameters,
) -> dict[str, object]:
    inputs: dict[str, object] = {
        "mpstruc_xml": input_file_record(
            args.input_xml,
            source=args.mpstruc_source_id or "local mpstruc snapshot (exploratory)",
            pinned=True,
            release=args.mpstruc_source_id,
        )
    }
    if args.override_csv:
        inputs["manual_overrides"] = input_file_record(
            args.override_csv, source="local manual overrides", pinned=True
        )
    if args.positive_approval_manifest:
        inputs["positive_approvals"] = input_file_record(
            args.positive_approval_manifest,
            source="local frozen positive approvals",
            pinned=True,
        )
    if args.structure_source_dir:
        inputs["structure_archive"] = input_directory_record(
            args.structure_source_dir,
            source=args.structure_source_id or "local coordinate archive (exploratory)",
            pinned=True,
        )
    remote_sources: tuple[dict[str, object], ...] = ()
    if not args.structure_source_dir:
        remote_sources = (
            {
                "name": "RCSB PDB entry coordinates",
                "url_templates": [
                    "https://files.rcsb.org/download/{pdb}.cif",
                    "https://files.rcsb.org/download/{pdb}.cif.gz",
                ],
                "release": "current archive snapshot",
                "pinned": False,
                "reproducibility": "every downloaded coordinate file is SHA-256 inventoried",
            },
            {
                "name": "RCSB PDB biological assemblies",
                "url_templates": [
                    "https://files.rcsb.org/download/{pdb}-assembly{assembly}.cif",
                    "https://files.rcsb.org/download/{pdb}-assembly{assembly}.cif.gz",
                ],
                "release": "current archive snapshot",
                "pinned": False,
                "reproducibility": "every downloaded assembly file is SHA-256 inventoried",
            },
        )
    run_manifest.set_provenance(inputs=inputs, remote_sources=remote_sources)

    entries_dir = out_dir / "entries"
    assemblies_dir = out_dir / "assemblies"
    meta_dir = out_dir / "metadata"
    by_subgroup_dir = out_dir / "by_subgroup"
    by_class_dir = out_dir / "by_class"
    approved_dir = out_dir / "approved_positives"
    logs_dir = out_dir / "logs"
    for p in [
        entries_dir,
        assemblies_dir,
        meta_dir,
        by_subgroup_dir,
        by_class_dir,
        approved_dir,
        logs_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    overrides = load_overrides(args.override_csv)

    records = parse_mpstruc_xml(args.input_xml, include_related=args.download_related)
    if not records:
        raise SystemExit("No records parsed from mpstruc XML.")

    write_csv(str(meta_dir / "mpstruc_records.csv"), records)

    records_by_code: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in records:
        records_by_code[rec["pdb_code"]].append(rec)

    unique_entries = [
        choose_representative_record(records_by_code[code])
        for code in sorted(records_by_code.keys())
    ]
    write_csv(str(meta_dir / "mpstruc_unique_entries.csv"), unique_entries)

    print(f"[INFO] Parsed {len(records)} source records, {len(unique_entries)} unique PDB codes.")

    entry_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    download_fail_rows: list[dict[str, Any]] = []
    assembly_fail_rows: list[dict[str, Any]] = []
    structure_archive = (
        FrozenStructureArchive(args.structure_source_dir) if args.structure_source_dir else None
    )

    if args.skip_download:
        print("[INFO] --skip-download enabled. Wrote manifests only.")
        return {
            "n_source_records": len(records),
            "n_unique_pdb_codes": len(unique_entries),
            "download_skipped": True,
        }

    # Step 1: download entry mmCIF files.
    download_results: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        entry_futures: dict[cf.Future[tuple[str, bool, str, str]], str] = {}
        for rec in unique_entries:
            code = rec["pdb_code"]
            if structure_archive is None:
                entry_future = ex.submit(
                    download_entry_cif,
                    code,
                    str(entries_dir),
                    args.timeout,
                    args.retries,
                    args.backoff,
                )
            else:
                entry_future = ex.submit(stage_entry_cif, code, str(entries_dir), structure_archive)
            entry_futures[entry_future] = code
        for entry_completed in cf.as_completed(entry_futures):
            code = entry_futures[entry_completed]
            try:
                code, ok, msg, out_path = entry_completed.result()
            except Exception as e:
                ok = False
                msg = f"Unhandled exception: {e}"
                out_path = str(entries_dir / f"{code}.cif")
            if ok:
                download_results[code] = out_path
                print(f"[OK] entry {code}: {msg}")
            else:
                download_fail_rows.append({"pdb_code": code, "stage": "entry", "message": msg})
                print(f"[FAIL] entry {code}: {msg}")

    # Step 2: parse entry mmCIF, classify, and download assemblies.
    assembly_jobs: list[tuple[str, str]] = []
    for rep in unique_entries:
        code = rep["pdb_code"]
        cif_path = download_results.get(code)
        if not cif_path or not os.path.exists(cif_path):
            entry_rows.append(
                {
                    **rep,
                    "class_label": "DOWNLOAD_FAILED",
                    "class_reason": "entry_cif_download_failed",
                    "automatic_candidate_stratum": "False",
                }
            )
            continue

        try:
            cif_summary = parse_cif_summary(cif_path)
            class_info = classify_entry(rep, cif_summary, overrides, parameters)
            candidate = choose_candidate_entity(cif_summary.get("entity_summary", {}))
            candidate_entity_id = candidate.get("entity_id", "") if candidate else ""
            candidate_label_asym = ""
            representative_author_chain_id = ""
            candidate_label_asym_ids = ""
            candidate_auth_asym_ids = ""
            if candidate and candidate.get("chains"):
                candidate_label_asym_ids = ";".join(
                    sorted(
                        {
                            str(chain.get("label_asym_id", ""))
                            for chain in candidate["chains"]
                            if str(chain.get("label_asym_id", ""))
                        }
                    )
                )
                candidate_auth_asym_ids = ";".join(
                    sorted(
                        {
                            str(chain.get("auth_asym_id", ""))
                            for chain in candidate["chains"]
                            if str(chain.get("auth_asym_id", ""))
                        }
                    )
                )
                best_chain = sorted(
                    candidate["chains"],
                    key=lambda x: (
                        int(x.get("beta_strands", 0)),
                        float(x.get("modeled_fraction", 0.0)),
                        int(x.get("observed_residues", 0)),
                    ),
                    reverse=True,
                )[0]
                candidate_label_asym = best_chain.get("label_asym_id", "")
                representative_author_chain_id = best_chain.get("auth_asym_id", "")

            row = {
                **rep,
                **class_info,
                "entry_cif_path": cif_path,
                "preferred_assembly_id": cif_summary.get("preferred_assembly_id", ""),
                "available_assembly_ids": ";".join(cif_summary.get("available_assembly_ids", [])),
                "protein_entity_ids": ";".join(cif_summary.get("protein_entity_ids", [])),
                "protein_entities_in_preferred_assembly": ";".join(
                    cif_summary.get("protein_entities_in_preferred_assembly", [])
                ),
                "candidate_entity_id": candidate_entity_id,
                "candidate_label_asym_id": candidate_label_asym,
                "representative_author_chain_id": representative_author_chain_id,
                "candidate_label_asym_ids": candidate_label_asym_ids,
                "candidate_auth_asym_ids": candidate_auth_asym_ids,
                "candidate_entity_max_beta_strands": candidate.get("max_beta_strands", 0)
                if candidate
                else 0,
                "candidate_entity_seq_len": candidate.get("seq_len", 0) if candidate else 0,
                "candidate_entity_max_modeled_fraction": f"{candidate.get('max_modeled_fraction', 0.0):.3f}"
                if candidate
                else "0.000",
                "candidate_entity_chain_count_entry": candidate.get("chain_count_entry", 0)
                if candidate
                else 0,
                "candidate_entity_chain_count_preferred_assembly": candidate.get(
                    "chain_count_preferred_assembly", 0
                )
                if candidate
                else 0,
            }
            entry_rows.append(row)

            for eid, esum in cif_summary.get("entity_summary", {}).items():
                for ch in esum.get("chains", []):
                    chain_rows.append(
                        {
                            "pdb_code": code,
                            "entity_id": eid,
                            "entity_description": esum.get("description", ""),
                            "entity_seq_len": esum.get("seq_len", 0),
                            "label_asym_id": ch.get("label_asym_id", ""),
                            "auth_asym_id": ch.get("auth_asym_id", ""),
                            "observed_residues": ch.get("observed_residues", 0),
                            "beta_strands": ch.get("beta_strands", 0),
                            "modeled_fraction": f"{ch.get('modeled_fraction', 0.0):.3f}",
                            "candidate_entity": str(eid == candidate_entity_id),
                        }
                    )

            if not args.skip_assemblies:
                assembly_ids = cif_summary.get("available_assembly_ids", []) or [
                    cif_summary.get("preferred_assembly_id", "1")
                ]
                for aid in unique_preserve_order([str(x) for x in assembly_ids if str(x).strip()]):
                    assembly_jobs.append((code, aid))

        except Exception as e:
            tb = traceback.format_exc(limit=2)
            entry_rows.append(
                {
                    **rep,
                    "class_label": "PARSE_FAILED",
                    "class_reason": f"parse_error: {e}",
                    "automatic_candidate_stratum": "False",
                    "entry_cif_path": cif_path,
                }
            )
            download_fail_rows.append(
                {"pdb_code": code, "stage": "parse", "message": str(e), "traceback": tb}
            )
            print(f"[FAIL] parse {code}: {e}")

    # Step 3: download assembly files.
    if not args.skip_assemblies and assembly_jobs:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
            assembly_futures: dict[cf.Future[tuple[str, str, bool, str, str]], tuple[str, str]] = {}
            for code, aid in assembly_jobs:
                if structure_archive is None:
                    assembly_future = ex.submit(
                        download_assembly_cif,
                        code,
                        aid,
                        str(assemblies_dir),
                        args.timeout,
                        args.retries,
                        args.backoff,
                    )
                else:
                    assembly_future = ex.submit(
                        stage_assembly_cif,
                        code,
                        aid,
                        str(assemblies_dir),
                        structure_archive,
                    )
                assembly_futures[assembly_future] = (code, aid)
            for assembly_completed in cf.as_completed(assembly_futures):
                code, aid = assembly_futures[assembly_completed]
                try:
                    code, aid, ok, msg, out_path = assembly_completed.result()
                except Exception as e:
                    ok = False
                    msg = f"Unhandled exception: {e}"
                    out_path = str(assemblies_dir / f"{code}-assembly{aid}.cif")
                if ok:
                    print(f"[OK] assembly {code}-assembly{aid}: {msg}")
                else:
                    assembly_fail_rows.append(
                        {"pdb_code": code, "assembly_id": aid, "message": msg}
                    )
                    print(f"[FAIL] assembly {code}-assembly{aid}: {msg}")

    # Step 4: write metadata.
    write_csv(str(meta_dir / "entry_classification.csv"), entry_rows)
    write_csv(str(meta_dir / "chain_summary.csv"), chain_rows)
    write_csv(str(logs_dir / "download_failed.csv"), download_fail_rows)
    write_csv(str(logs_dir / "assembly_failed.csv"), assembly_fail_rows)
    if args.mode == "publication" and (download_fail_rows or assembly_fail_rows):
        raise RuntimeError(
            "Frozen publication structure archive did not provide a complete, parseable input "
            f"set (entry/parse failures={len(download_fail_rows)}, "
            f"assembly failures={len(assembly_fail_rows)})"
        )

    automatic_candidates = [
        row for row in entry_rows if row.get("automatic_candidate_stratum") == "True"
    ]
    approved_positives: list[dict[str, Any]] = []
    if args.positive_approval_manifest:
        approved_positives = match_positive_approvals(
            load_positive_approvals(args.positive_approval_manifest), entry_rows
        )
    exclusions = [
        r
        for r in entry_rows
        if r.get("class_label") in {"ASSEMBLY_FORMED_OR_OUT_OF_SCOPE", "DESIGNED_OR_OUT_OF_SCOPE"}
    ]
    review_cases = [
        r
        for r in entry_rows
        if r.get("class_label")
        in {"PARTIAL_OR_DOMAIN_ONLY", "NEEDS_REVIEW", "PARSE_FAILED", "DOWNLOAD_FAILED"}
    ]

    write_csv(str(meta_dir / "d1_automatic_candidates_for_review.csv"), automatic_candidates)
    write_csv(str(meta_dir / "d1_approved_positives.csv"), approved_positives)
    write_csv(str(meta_dir / "d1_exclusions.csv"), exclusions)
    write_csv(str(meta_dir / "d1_review_cases.csv"), review_cases)

    # Step 5: organize files.
    for row in entry_rows:
        code = row["pdb_code"]
        cif_path = row.get("entry_cif_path", "")
        if not cif_path or not os.path.exists(cif_path):
            continue
        subgroup_slug = slugify(row.get("subgroup_name", ""))
        class_slug = slugify(row.get("class_label", ""))
        subgroup_dst = by_subgroup_dir / subgroup_slug / f"{code}.cif"
        class_dst = by_class_dir / class_slug / f"{code}.cif"
        safe_symlink_or_copy(cif_path, str(subgroup_dst), args.link_mode)
        safe_symlink_or_copy(cif_path, str(class_dst), args.link_mode)

        # Link preferred assembly file if present.
        preferred_aid = str(row.get("preferred_assembly_id", "")).strip()
        if preferred_aid:
            assembly_path = assemblies_dir / f"{code}-assembly{preferred_aid}.cif"
            if assembly_path.exists():
                subgroup_assembly_dst = (
                    by_subgroup_dir / subgroup_slug / f"{code}-assembly{preferred_aid}.cif"
                )
                class_assembly_dst = (
                    by_class_dir / class_slug / f"{code}-assembly{preferred_aid}.cif"
                )
                safe_symlink_or_copy(str(assembly_path), str(subgroup_assembly_dst), args.link_mode)
                safe_symlink_or_copy(str(assembly_path), str(class_assembly_dst), args.link_mode)

    for approved in approved_positives:
        source = str(approved["entry_cif_path"])
        destination = approved_dir / approved["filename"]
        safe_symlink_or_copy(source, str(destination), args.link_mode)

    # Step 6: write a compact JSON summary.
    summary = {
        "n_source_records": len(records),
        "n_unique_pdb_codes": len(unique_entries),
        "n_download_failures": len(download_fail_rows),
        "n_assembly_failures": len(assembly_fail_rows),
        "n_automatic_candidates_for_review": len(automatic_candidates),
        "n_approved_positives": len(approved_positives),
        "n_exclusions": len(exclusions),
        "n_review_cases": len(review_cases),
        "class_counts": dict(
            sorted(
                {
                    k: sum(1 for r in entry_rows if r.get("class_label") == k)
                    for k in sorted({r.get("class_label", "") for r in entry_rows})
                }.items()
            )
        ),
    }
    atomic_write_json(summary, meta_dir / "summary.json")

    print("[DONE] Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[DONE] Output directory: {out_dir.resolve()}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    parameters = validate_arguments(args)
    out_dir = prepare_new_output_directory(args.out)
    run_manifest = RunManifest(
        out_dir,
        __file__,
        vars(args),
        mode=args.mode,
        random_algorithm=None,
        seed=None,
    )
    try:
        summary = _run(args, out_dir, run_manifest, parameters)
    except BaseException as error:
        run_manifest.fail(error)
        raise
    run_manifest.complete(summary)


if __name__ == "__main__":
    main()
