#!/usr/bin/env python3

"""
Build a chain-level easy-negative dataset from PISCES + CATH.

What this script does
---------------------
1. Downloads (or reuses) a precompiled non-redundant PISCES chain list.
2. Downloads (or reuses) the CATH domain list.
3. Maps PISCES chains to CATH class / architecture / topology at chain level.
4. Excludes mpstruc-derived positives and user-provided exclusions.
5. Selects a structurally diverse negative-candidate panel (default: 500 chains)
   across CATH classes 1/3/4; Mainly Beta is excluded.
6. Stages the selected full-entry mmCIF files from a frozen archive or RCSB.
7. Validates the target author chain and complete polymer declaration without
   reserializing a lossy chain-only coordinate file.
8. Writes complete CSV manifests for candidates, exclusions and final selections.

Notes
-----
- PISCES provides culled PDB chain lists with quality / sequence-identity filters.
- CATH provides domain-level classification; this script converts those domain
  annotations into chain-level dominant class / topology labels.
- The sampling unit here is chain-level, which matches a typical chain-level
  negative set better than directly using CATH domains.
- CATH assignments create candidate strata only. Publication mode requires an
  independent, frozen NON_BARREL approval for every final selected chain.

Typical usage
-------------
python build_easy_negatives_from_pisces_cath.py \
  --out easy_negatives_500 \
  --mode exploratory \
  --mpstruc-xml Mpstrucis.txt \
  --n-total 500 \
  --class-quotas 1:167,3:167,4:166 \
  --pc 20 --resolution-max 2.0 --rmax 0.25 --no-breaks \
  --threads 8

Dependencies
------------
Python >= 3.10
Biopython
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from _dataset_provenance import (
    FrozenStructureArchive,
    RunManifest,
    atomic_write_csv,
    atomic_write_json,
    infer_release,
    input_directory_record,
    input_file_record,
    prepare_new_output_directory,
    require_exact_int,
    require_finite_real,
    require_pinned_source_id,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cooper_beta.polymer_sequence import (  # noqa: E402
    declared_polymer_sequence_for_author_chain,
)

try:
    from Bio.PDB.MMCIFParser import MMCIFParser
except ImportError as e:  # pragma: no cover
    raise SystemExit("Biopython is required. Install it with: pip install biopython") from e


PISCES_DOWNLOAD_PAGE = "https://dunbrack.fccc.edu/pisces/download/"
PISCES_BASE = "https://dunbrack.fccc.edu/pisces/download/"
CATH_CANDIDATE_URLS = [
    "https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
    "https://download.cathdb.info/cath/releases/all-releases/v4_3_0/cath-classification-data/cath-domain-list-v4_3_0.txt",
]
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb}.cif"
RCSB_CIF_GZ_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"

PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")
DOMAIN_ID_RE = re.compile(r"^[0-9A-Za-z]{7}$")

CLASS_NAME_MAP = {
    1: "Mainly_Alpha",
    2: "Mainly_Beta",
    3: "Alpha_Beta",
    4: "Few_Secondary_Structures",
}
SAFE_NEGATIVE_CATH_CLASSES = (1, 3, 4)
DEFAULT_CLASS_QUOTAS = "1:167,3:167,4:166"
SELECTION_ALGORITHM = "python_random_mt19937_shuffle_sorted_identity_strict_caps"
NEGATIVE_APPROVAL_FIELDS = ("pdb_id", "target_author_chain_id", "group_id", "curation_evidence")
DEFAULT_PISCES_SOURCE_MIN_LENGTH = 40
DEFAULT_PISCES_SOURCE_MAX_LENGTH = 10_000
DEFAULT_MAX_RELATIVE_LENGTH_MISMATCH = 0.25
DEFAULT_MIN_LENGTH_MATCH_MARGIN = 0.15


@dataclass
class PiscesRecord:
    pdb_id: str
    chain_raw: str
    method: str
    length: int
    resolution: float | None
    r_value: float | None
    r_free: float | None
    source_line: str


@dataclass
class CathDomain:
    domain_id: str
    pdb_id: str
    chain_id: str
    class_num: int
    architecture: int
    topology: int
    superfamily: int
    domain_len: int
    resolution: float | None


@dataclass
class CathChainSummary:
    pdb_id: str
    chain_id: str
    total_domain_len: int
    domain_count: int
    dominant_class: int
    dominant_class_len: int
    dominant_class_fraction: float
    dominant_architecture: int
    dominant_topology: int
    dominant_superfamily: int
    dominant_cat_len: int
    classes_present: str
    topologies_present: str
    superfamilies_present: str
    domain_ids: str


@dataclass
class CandidateRecord:
    pdb_id: str
    pisces_chain: str
    resolved_chain: str
    map_status: str
    method: str
    chain_length: int
    resolution: float | None
    r_value: float | None
    r_free: float | None
    cath_total_domain_len: int
    cath_domain_count: int
    cath_dominant_class: int
    cath_dominant_class_name: str
    cath_dominant_class_fraction: float
    cath_dominant_architecture: int
    cath_dominant_topology: int
    cath_dominant_superfamily: int
    cath_classes_present: str
    cath_topologies_present: str
    cath_superfamilies_present: str
    cath_domain_ids: str
    group_id: str


def load_negative_approvals(
    path: os.PathLike[str] | str,
) -> dict[tuple[str, str], dict[str, str]]:
    """Load exact independent NON_BARREL approvals for a frozen selected panel."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != NEGATIVE_APPROVAL_FIELDS:
            raise ValueError(
                "Negative approval manifest must have exactly these columns in order: "
                + ",".join(NEGATIVE_APPROVAL_FIELDS)
            )
        approvals: dict[tuple[str, str], dict[str, str]] = {}
        seen_pdb_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            pdb_id = str(row.get("pdb_id", "")).strip().upper()
            target_author_chain_id = str(row.get("target_author_chain_id", "")).strip()
            group_id = str(row.get("group_id", "")).strip()
            evidence = str(row.get("curation_evidence", "")).strip()
            if not PDB_ID_RE.fullmatch(pdb_id):
                raise ValueError(
                    f"Negative approval row {line_number} has invalid PDB ID {pdb_id!r}"
                )
            if not target_author_chain_id or not group_id or not evidence:
                raise ValueError(
                    f"Negative approval row {line_number} has a blank chain, group, or evidence"
                )
            key = (pdb_id, target_author_chain_id)
            if key in approvals:
                raise ValueError(
                    f"Duplicate negative approval for {pdb_id}:{target_author_chain_id}"
                )
            if pdb_id in seen_pdb_ids:
                raise ValueError(
                    "Negative approval manifest must contain exactly one target chain per "
                    f"PDB structure; duplicate PDB ID {pdb_id!r}."
                )
            seen_pdb_ids.add(pdb_id)
            approvals[key] = {
                "pdb_id": pdb_id,
                "target_author_chain_id": target_author_chain_id,
                "group_id": group_id,
                "curation_evidence": evidence,
            }
    if not approvals:
        raise ValueError("Negative approval manifest contains no approved targets")
    return approvals


def match_negative_approvals(
    selected: Sequence[CandidateRecord],
    approvals: dict[tuple[str, str], dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """Require the approval set to equal the selected chain sampling units exactly."""

    selected_keys = {(record.pdb_id, record.resolved_chain) for record in selected}
    if len(selected_keys) != len(selected):
        raise ValueError("Selected negative panel contains duplicate PDB/chain identities")
    selected_pdb_ids = [record.pdb_id for record in selected]
    if len(set(selected_pdb_ids)) != len(selected_pdb_ids):
        raise ValueError("Selected negative panel must contain exactly one target chain per PDB")
    approval_keys = set(approvals)
    if selected_keys != approval_keys:
        missing = sorted(selected_keys - approval_keys)
        extra = sorted(approval_keys - selected_keys)
        raise ValueError(
            "Negative approvals must cover the final selected panel exactly; "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}"
        )
    return approvals


def apply_negative_approvals(
    rows: Sequence[dict[str, object]],
    approvals: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    """Attach independent truth evidence without overwriting the automatic stratum silently."""

    approved_rows: list[dict[str, object]] = []
    for original in rows:
        row = dict(original)
        key = (str(row["pdb_id"]).upper(), str(row["resolved_chain"]))
        approval = approvals[key]
        structure_path = str(row.get("structure_cif_path", ""))
        row.update(
            {
                "automatic_group_id": str(row.get("group_id", "")),
                "target_author_chain_id": approval["target_author_chain_id"],
                "group_id": approval["group_id"],
                "curation_evidence": approval["curation_evidence"],
                "truth_label": "NON_BARREL",
                "filename": Path(structure_path).name if structure_path else "",
            }
        )
        approved_rows.append(row)
    return approved_rows


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: object, path: os.PathLike[str] | str) -> None:
    atomic_write_json(obj, path)


def http_get(url: str, timeout: int = 60, retries: int = 3, backoff: float = 1.0) -> bytes:
    last_err: Exception | None = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "easy-negative-builder/1.0",
                    "Accept": "*/*",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload: object = resp.read()
                if not isinstance(payload, bytes):
                    raise TypeError(f"HTTP response for {url} did not return bytes")
                return payload
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
            if i < retries:
                time.sleep(backoff * (2**i))
            else:
                raise
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def download_to_file(
    url: str,
    out_path: os.PathLike[str] | str,
    timeout: int = 60,
    retries: int = 3,
    backoff: float = 1.0,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    data = http_get(url, timeout=timeout, retries=retries, backoff=backoff)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(out)
    return out


def maybe_decompress_gzip_bytes(data: bytes) -> bytes:
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def normalize_rvalue(x: float) -> str:
    s = f"{x:.2f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def discover_pisces_filename(
    page_html: str,
    pc: float,
    resolution_max: float,
    rmax: float,
    method: str,
    no_breaks: bool,
    min_len: int,
    max_len: int,
) -> str:
    pc_str = f"{pc:.1f}"
    res_str = f"{resolution_max:.1f}"
    r_str = normalize_rvalue(rmax)
    no_brks = "_noBrks" if no_breaks else ""
    method_pat = re.escape(method)
    pattern = (
        rf"(cullpdb_pc{re.escape(pc_str)}_res0\.0-{re.escape(res_str)}"
        rf"{re.escape(no_brks)}_len{min_len}-{max_len}_R{re.escape(r_str)}_{method_pat}"
        rf"_d\d{{4}}_\d{{2}}_\d{{2}}_chains\d+)"
    )
    filename_pattern: re.Pattern[str] = re.compile(pattern)
    matches = [str(match) for match in filename_pattern.findall(page_html)]
    if not matches:
        raise RuntimeError(
            "Could not find a matching precompiled PISCES list on the download page. "
            "Try a different combination, or pass --pisces-url manually."
        )

    # Keep the newest one if multiple dated files are present.
    def sort_key(name: str) -> tuple[str, int]:
        m = re.search(r"_d(\d{4}_\d{2}_\d{2})_chains(\d+)$", name)
        if not m:
            return ("", 0)
        return (m.group(1), int(m.group(2)))

    return sorted(set(matches), key=sort_key, reverse=True)[0]


def get_pisces_url(args: argparse.Namespace, work_dir: Path) -> tuple[str, Path]:
    if args.pisces_url:
        url = args.pisces_url
        local_name = Path(urllib.parse.urlparse(url).path).name or "pisces_list.txt"
        return url, work_dir / "downloads" / local_name

    html = http_get(
        PISCES_DOWNLOAD_PAGE, timeout=args.timeout, retries=args.retries, backoff=args.backoff
    ).decode("utf-8", errors="replace")
    fname = discover_pisces_filename(
        page_html=html,
        pc=args.pc,
        resolution_max=args.resolution_max,
        rmax=args.rmax,
        method=args.method,
        no_breaks=args.no_breaks,
        min_len=args.pisces_source_min_length,
        max_len=args.pisces_source_max_length,
    )
    return PISCES_BASE + fname, work_dir / "downloads" / fname


def resolve_pisces_file(args: argparse.Namespace, work_dir: Path) -> tuple[str, Path, bool, str]:
    if args.pisces_file:
        local = Path(args.pisces_file).expanduser().resolve()
        return str(local), local, True, infer_release(str(local), local.name)
    url, local = get_pisces_url(args, work_dir)
    download_to_file(url, local, timeout=args.timeout, retries=args.retries, backoff=args.backoff)
    return url, local, False, infer_release(url, local.name)


def get_cath_file(args: argparse.Namespace, work_dir: Path) -> tuple[Path, str, bool, str]:
    if args.cath_domain_list:
        src = Path(args.cath_domain_list).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"CATH domain list not found: {src}")
        return src, str(src), True, infer_release(str(src), src.name)

    out = work_dir / "downloads" / "cath-domain-list.txt"
    if out.exists() and out.stat().st_size > 0:
        raise RuntimeError("Unexpected pre-existing CATH download in a new output directory")

    last_err: Exception | None = None
    for url in CATH_CANDIDATE_URLS:
        try:
            data = http_get(url, timeout=args.timeout, retries=args.retries, backoff=args.backoff)
            data = maybe_decompress_gzip_bytes(data)
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_suffix(out.suffix + ".tmp")
            with open(temporary, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(out)
            return out, url, False, infer_release(url, out.name)
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
            continue
    raise RuntimeError(f"Failed to download CATH domain list: {last_err}")


def parse_float_or_none(s: str) -> float | None:
    s = s.strip()
    if s.upper() in {"NA", "N/A", "NONE", "NULL", "-", ""}:
        return None
    try:
        return require_finite_real("input numeric field", float(s))
    except ValueError as error:
        raise ValueError(f"Invalid numeric input field: {s!r}") from error


def parse_pisces_list(path: os.PathLike[str] | str) -> list[PiscesRecord]:
    records: list[PiscesRecord] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts and parts[0].lower() in {"pdbchain", "pdb_chain"}:
                continue
            if len(parts) >= 7 and PDB_ID_RE.match(parts[0].upper()):
                pdb_id = parts[0].upper()
                chain_raw = parts[1]
                method = parts[2]
                length_s = parts[3]
                resolution_s = parts[4]
                r_value_s = parts[5]
                r_free_s = parts[6]
            elif len(parts) >= 6 and len(parts[0]) >= 5 and PDB_ID_RE.match(parts[0][:4].upper()):
                pdb_id = parts[0][:4].upper()
                chain_raw = parts[0][4:]
                method = parts[2]
                length_s = parts[1]
                resolution_s = parts[3]
                r_value_s = parts[4]
                r_free_s = parts[5]
            else:
                continue
            try:
                length_value = require_finite_real("PISCES chain length", float(length_s))
                if not length_value.is_integer() or length_value < 1:
                    raise ValueError("chain length must be a positive integer")
                rec = PiscesRecord(
                    pdb_id=pdb_id,
                    chain_raw=chain_raw,
                    method=method,
                    length=int(length_value),
                    resolution=parse_float_or_none(resolution_s),
                    r_value=parse_float_or_none(r_value_s),
                    r_free=parse_float_or_none(r_free_s),
                    source_line=line,
                )
            except ValueError as error:
                raise ValueError(f"Invalid PISCES record at line {line_number}: {line}") from error
            records.append(rec)
    return records


def parse_cath_domain_list(path: os.PathLike[str] | str) -> list[CathDomain]:
    domains: list[CathDomain] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            domain_id = parts[0]
            if not DOMAIN_ID_RE.match(domain_id):
                continue
            try:
                class_num = int(parts[1])
                architecture = int(parts[2])
                topology = int(parts[3])
                superfamily = int(parts[4])
                domain_len = int(parts[10])
                resolution = parse_float_or_none(parts[11])
                if min(class_num, architecture, topology, superfamily, domain_len) < 1:
                    raise ValueError("CATH identifiers and domain length must be positive")
                if resolution is not None and resolution < 0:
                    raise ValueError("CATH resolution cannot be negative")
            except ValueError as error:
                raise ValueError(f"Invalid CATH record at line {line_number}: {line}") from error
            domains.append(
                CathDomain(
                    domain_id=domain_id,
                    pdb_id=domain_id[:4].upper(),
                    chain_id=domain_id[4],
                    class_num=class_num,
                    architecture=architecture,
                    topology=topology,
                    superfamily=superfamily,
                    domain_len=domain_len,
                    resolution=resolution,
                )
            )
    return domains


def build_cath_chain_summary(
    domains: Sequence[CathDomain],
) -> tuple[dict[tuple[str, str], CathChainSummary], dict[str, list[str]]]:
    by_chain: dict[tuple[str, str], list[CathDomain]] = defaultdict(list)
    entry_to_chains: dict[str, set[str]] = defaultdict(set)
    for d in domains:
        by_chain[(d.pdb_id, d.chain_id)].append(d)
        entry_to_chains[d.pdb_id].add(d.chain_id)

    summaries: dict[tuple[str, str], CathChainSummary] = {}
    for key, ds in by_chain.items():
        class_len: dict[int, int] = defaultdict(int)
        cat_len: dict[tuple[int, int, int, int], int] = defaultdict(int)
        total_len = 0
        domain_ids: list[str] = []
        classes_present: set[int] = set()
        topologies_present: set[str] = set()
        superfamilies_present: set[str] = set()
        for d in ds:
            total_len += d.domain_len
            class_len[d.class_num] += d.domain_len
            cat_key = (d.class_num, d.architecture, d.topology, d.superfamily)
            cat_len[cat_key] += d.domain_len
            domain_ids.append(d.domain_id)
            classes_present.add(d.class_num)
            topologies_present.add(f"{d.class_num}.{d.architecture}.{d.topology}")
            superfamilies_present.add(
                f"{d.class_num}.{d.architecture}.{d.topology}.{d.superfamily}"
            )

        dominant_class, dominant_class_len = max(class_len.items(), key=lambda kv: (kv[1], -kv[0]))
        dominant_cat, dominant_cat_len = max(cat_len.items(), key=lambda kv: (kv[1], kv[0]))
        dominant_fraction = dominant_class_len / total_len if total_len > 0 else 0.0
        summaries[key] = CathChainSummary(
            pdb_id=key[0],
            chain_id=key[1],
            total_domain_len=total_len,
            domain_count=len(ds),
            dominant_class=dominant_class,
            dominant_class_len=dominant_class_len,
            dominant_class_fraction=dominant_fraction,
            dominant_architecture=dominant_cat[1],
            dominant_topology=dominant_cat[2],
            dominant_superfamily=dominant_cat[3],
            dominant_cat_len=dominant_cat_len,
            classes_present=";".join(str(x) for x in sorted(classes_present)),
            topologies_present=";".join(sorted(topologies_present)),
            superfamilies_present=";".join(sorted(superfamilies_present)),
            domain_ids=";".join(sorted(domain_ids)),
        )

    entry_to_chains_sorted = {pdb: sorted(chains) for pdb, chains in entry_to_chains.items()}
    return summaries, entry_to_chains_sorted


def extract_pdb_codes_xml_iterparse(path: os.PathLike[str] | str) -> set[str]:
    codes: set[str] = set()
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag.endswith("pdbCode") and elem.text:
            code = elem.text.strip().upper()
            if PDB_ID_RE.match(code):
                codes.add(code)
        elem.clear()
    return codes


def extract_pdb_codes_loose_text(path: os.PathLike[str] | str) -> set[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {
        code.upper()
        for code in re.findall(r"<pdbCode>\s*([0-9A-Za-z]{4})\s*</pdbCode>", text)
        if PDB_ID_RE.match(code.upper())
    }


def load_mpstruc_exclusions(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"mpstruc file not found: {p}")
    try:
        return extract_pdb_codes_xml_iterparse(p)
    except ET.ParseError as e:
        codes = extract_pdb_codes_loose_text(p)
        if codes:
            return codes
        raise RuntimeError(f"Failed to parse mpstruc XML: {e}") from e


def load_generic_exclusions(path: str | None) -> tuple[set[str], set[tuple[str, str]]]:
    """
    Accepts txt/csv/tsv with any of the following:
    - one ID per line: 1ABC or 1ABC_A
    - csv header containing pdb, pdb_id, chain, chain_id
    """
    if not path:
        return set(), set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Exclude file not found: {p}")

    exclude_pdbs: set[str] = set()
    exclude_chains: set[tuple[str, str]] = set()

    def add_token(tok: str) -> None:
        tok = tok.strip().upper()
        if not tok:
            return
        if re.match(r"^[0-9A-Z]{4}$", tok):
            exclude_pdbs.add(tok)
        elif re.match(r"^[0-9A-Z]{4}[_:][^\s]+$", tok):
            pdb, chain = re.split(r"[_:]", tok, maxsplit=1)
            exclude_chains.add((pdb.upper(), chain))

    if p.suffix.lower() in {".csv", ".tsv"}:
        dialect = "excel-tab" if p.suffix.lower() == ".tsv" else "excel"
        with open(p, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            headers = [h.lower() for h in (reader.fieldnames or [])]
            has_fields = {"pdb", "pdb_id", "chain", "chain_id"} & set(headers)
            if has_fields:
                for row in reader:
                    pdb = (row.get("pdb") or row.get("pdb_id") or "").strip().upper()
                    chain = (row.get("chain") or row.get("chain_id") or "").strip()
                    if pdb and not chain:
                        exclude_pdbs.add(pdb)
                    elif pdb and chain:
                        exclude_chains.add((pdb, chain))
                return exclude_pdbs, exclude_chains

    with open(p, encoding="utf-8", errors="replace") as f:
        for raw in f:
            add_token(raw)
    return exclude_pdbs, exclude_chains


def resolve_pisces_to_cath_chain(
    rec: PiscesRecord,
    chain_summaries: dict[tuple[str, str], CathChainSummary],
    entry_to_chains: dict[str, list[str]],
    max_relative_length_mismatch: float = DEFAULT_MAX_RELATIVE_LENGTH_MISMATCH,
    min_length_match_margin: float = DEFAULT_MIN_LENGTH_MATCH_MARGIN,
    allow_heuristic_length_mapping: bool = False,
) -> tuple[CathChainSummary | None, str, str]:
    pdb = rec.pdb_id
    raw_chain = rec.chain_raw

    if (pdb, raw_chain) in chain_summaries:
        return chain_summaries[(pdb, raw_chain)], raw_chain, "exact"

    chains = entry_to_chains.get(pdb, [])
    if not chains:
        return None, "", "no_cath_assignment"

    if raw_chain == "0":
        if len(chains) == 1:
            ch = chains[0]
            return chain_summaries[(pdb, ch)], ch, "pisces_zero_unique_chain"

        if not allow_heuristic_length_mapping:
            return None, "", "ambiguous_zero_chain"

        # If PISCES says "0" but CATH has multiple chains, try the single best length match.
        candidates: list[tuple[float, str, CathChainSummary]] = []
        for ch in chains:
            sm = chain_summaries[(pdb, ch)]
            if rec.length <= 0:
                diff = float(abs(sm.total_domain_len))
            else:
                diff = abs(sm.total_domain_len - rec.length) / max(rec.length, 1)
            candidates.append((diff, ch, sm))
        candidates.sort(key=lambda x: (x[0], x[1]))
        if len(candidates) >= 1:
            best = candidates[0]
            second_diff = candidates[1][0] if len(candidates) > 1 else 999.0
            # Accept only if clearly better and reasonably close.
            if best[0] <= max_relative_length_mismatch and (
                second_diff - best[0] >= min_length_match_margin or len(candidates) == 1
            ):
                return best[2], best[1], "pisces_zero_best_length_match"
        return None, "", "ambiguous_zero_chain"

    # A concrete PISCES chain that is absent from CATH must never be silently
    # replaced by a different chain, even when CATH lists only one chain.
    return None, "", "pisces_chain_not_in_cath"


def make_candidate_rows(
    pisces_records: Sequence[PiscesRecord],
    cath_summaries: dict[tuple[str, str], CathChainSummary],
    entry_to_chains: dict[str, list[str]],
    exclude_pdbs: set[str],
    exclude_chains: set[tuple[str, str]],
    min_length: int,
    max_length: int,
    min_class_fraction: float,
    max_relative_length_mismatch: float = DEFAULT_MAX_RELATIVE_LENGTH_MISMATCH,
    min_length_match_margin: float = DEFAULT_MIN_LENGTH_MATCH_MARGIN,
    allow_heuristic_length_mapping: bool = False,
) -> tuple[list[CandidateRecord], list[dict[str, str]]]:
    candidates: list[CandidateRecord] = []
    excluded: list[dict[str, str]] = []

    for rec in pisces_records:
        if rec.pdb_id in exclude_pdbs:
            excluded.append(
                {"pdb_id": rec.pdb_id, "chain": rec.chain_raw, "reason": "excluded_pdb"}
            )
            continue
        if (rec.pdb_id, rec.chain_raw) in exclude_chains:
            excluded.append(
                {"pdb_id": rec.pdb_id, "chain": rec.chain_raw, "reason": "excluded_chain"}
            )
            continue
        if rec.length < min_length or rec.length > max_length:
            excluded.append(
                {
                    "pdb_id": rec.pdb_id,
                    "chain": rec.chain_raw,
                    "reason": f"length_out_of_range:{rec.length}",
                }
            )
            continue

        summary, resolved_chain, map_status = resolve_pisces_to_cath_chain(
            rec,
            cath_summaries,
            entry_to_chains,
            max_relative_length_mismatch,
            min_length_match_margin,
            allow_heuristic_length_mapping,
        )
        if summary is None:
            excluded.append({"pdb_id": rec.pdb_id, "chain": rec.chain_raw, "reason": map_status})
            continue

        cath_classes = {int(value) for value in summary.classes_present.split(";") if value.strip()}
        unsupported_classes = cath_classes.difference(SAFE_NEGATIVE_CATH_CLASSES)
        if unsupported_classes:
            excluded.append(
                {
                    "pdb_id": rec.pdb_id,
                    "chain": resolved_chain,
                    "reason": (
                        "contains_cath_class_not_safe_as_negative_without_independent_"
                        f"non_barrel_evidence:{';'.join(str(value) for value in sorted(unsupported_classes))}"
                    ),
                }
            )
            continue

        if summary.dominant_class not in SAFE_NEGATIVE_CATH_CLASSES:
            excluded.append(
                {
                    "pdb_id": rec.pdb_id,
                    "chain": resolved_chain,
                    "reason": f"unsupported_cath_class:{summary.dominant_class}",
                }
            )
            continue

        if summary.dominant_class_fraction < min_class_fraction:
            excluded.append(
                {
                    "pdb_id": rec.pdb_id,
                    "chain": resolved_chain,
                    "reason": f"dominant_class_fraction_too_low:{summary.dominant_class_fraction:.3f}",
                }
            )
            continue

        candidates.append(
            CandidateRecord(
                pdb_id=rec.pdb_id,
                pisces_chain=rec.chain_raw,
                resolved_chain=resolved_chain,
                map_status=map_status,
                method=rec.method,
                chain_length=rec.length,
                resolution=rec.resolution,
                r_value=rec.r_value,
                r_free=rec.r_free,
                cath_total_domain_len=summary.total_domain_len,
                cath_domain_count=summary.domain_count,
                cath_dominant_class=summary.dominant_class,
                cath_dominant_class_name=CLASS_NAME_MAP.get(
                    summary.dominant_class, f"Class_{summary.dominant_class}"
                ),
                cath_dominant_class_fraction=summary.dominant_class_fraction,
                cath_dominant_architecture=summary.dominant_architecture,
                cath_dominant_topology=summary.dominant_topology,
                cath_dominant_superfamily=summary.dominant_superfamily,
                cath_classes_present=summary.classes_present,
                cath_topologies_present=summary.topologies_present,
                cath_superfamilies_present=summary.superfamilies_present,
                cath_domain_ids=summary.domain_ids,
                group_id=(
                    "CATH:"
                    f"{summary.dominant_class}.{summary.dominant_architecture}."
                    f"{summary.dominant_topology}.{summary.dominant_superfamily}"
                ),
            )
        )
    return candidates, excluded


def parse_class_quotas(text: str, n_total: int) -> dict[int, int]:
    require_exact_int("n_total", n_total, minimum=1)
    if not text:
        base = n_total // len(SAFE_NEGATIVE_CATH_CLASSES)
        rem = n_total % len(SAFE_NEGATIVE_CATH_CLASSES)
        default_quotas = {cls: base for cls in SAFE_NEGATIVE_CATH_CLASSES}
        for cls in SAFE_NEGATIVE_CATH_CLASSES[:rem]:
            default_quotas[cls] += 1
        return default_quotas

    quotas: dict[int, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.count(":") != 1:
            raise ValueError(f"Invalid class quota item: {item!r}")
        cls_s, n_s = item.split(":", 1)
        try:
            cls_i = int(cls_s)
            quota = int(n_s)
        except ValueError as error:
            raise ValueError(f"Invalid class quota item: {item!r}") from error
        if cls_i not in SAFE_NEGATIVE_CATH_CLASSES:
            raise ValueError(
                f"CATH class {cls_i} is not an evidence-safe easy-negative class; "
                f"allowed classes are {SAFE_NEGATIVE_CATH_CLASSES}"
            )
        if cls_i in quotas:
            raise ValueError(f"Duplicate quota for CATH class {cls_i}")
        quotas[cls_i] = require_exact_int(f"quota for class {cls_i}", quota, minimum=0)
    for cls in SAFE_NEGATIVE_CATH_CLASSES:
        quotas.setdefault(cls, 0)
    if sum(quotas.values()) != n_total:
        raise ValueError(
            f"Class quotas must sum exactly to n_total={n_total}; got {sum(quotas.values())}"
        )
    return quotas


def select_diverse_easy_negatives(
    candidates: Sequence[CandidateRecord],
    n_total: int,
    class_quotas: dict[int, int],
    max_per_topology: int,
    max_per_pdb: int,
    seed: int,
) -> list[CandidateRecord]:
    require_exact_int("n_total", n_total, minimum=1)
    require_exact_int("max_per_topology", max_per_topology, minimum=1)
    require_exact_int("max_per_pdb", max_per_pdb, minimum=1)
    require_exact_int("seed", seed)
    if set(class_quotas) - set(SAFE_NEGATIVE_CATH_CLASSES):
        raise ValueError("class_quotas contains a class that is not safe for easy negatives")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in class_quotas.values()
    ):
        raise ValueError("Every class quota must be a non-negative integer")
    if sum(class_quotas.values()) != n_total:
        raise ValueError("class_quotas must sum exactly to n_total")

    rng = random.Random(seed)
    pools: dict[int, list[CandidateRecord]] = defaultdict(list)
    for rec in candidates:
        if rec.cath_dominant_class not in SAFE_NEGATIVE_CATH_CLASSES:
            raise ValueError(
                f"Unsafe CATH class {rec.cath_dominant_class} reached negative selection"
            )
        pools[rec.cath_dominant_class].append(rec)

    for cls in pools:
        pools[cls].sort(key=lambda rec: (rec.pdb_id, rec.resolved_chain))
        rng.shuffle(pools[cls])

    selected: list[CandidateRecord] = []
    selected_keys: set[tuple[str, str]] = set()
    per_topology: Counter[str] = Counter()
    per_pdb: Counter[str] = Counter()

    def topo_key(rec: CandidateRecord) -> str:
        return f"{rec.cath_dominant_class}.{rec.cath_dominant_architecture}.{rec.cath_dominant_topology}"

    def try_add(rec: CandidateRecord) -> bool:
        key = (rec.pdb_id, rec.resolved_chain)
        if key in selected_keys:
            return False
        if per_pdb[rec.pdb_id] >= max_per_pdb:
            return False
        tk = topo_key(rec)
        if per_topology[tk] >= max_per_topology:
            return False
        selected.append(rec)
        selected_keys.add(key)
        per_topology[tk] += 1
        per_pdb[rec.pdb_id] += 1
        return True

    # Fill exact class quotas without silently relaxing either diversity cap.
    for cls in SAFE_NEGATIVE_CATH_CLASSES:
        need = class_quotas.get(cls, 0)
        if need <= 0:
            continue
        for rec in pools.get(cls, []):
            if sum(item.cath_dominant_class == cls for item in selected) >= need:
                break
            try_add(rec)

    actual_counts = Counter(item.cath_dominant_class for item in selected)
    shortfalls = {
        cls: class_quotas[cls] - actual_counts[cls]
        for cls in SAFE_NEGATIVE_CATH_CLASSES
        if actual_counts[cls] != class_quotas[cls]
    }
    if shortfalls:
        raise RuntimeError(
            "Unable to satisfy exact class quotas under strict topology/PDB caps; "
            f"shortfalls={shortfalls}"
        )
    return selected


def candidate_to_dict(rec: CandidateRecord) -> dict[str, object]:
    return asdict(rec)


def write_csv(rows: Sequence[Mapping[str, object]], path: os.PathLike[str] | str) -> None:
    fieldnames = list(rows[0]) if rows else None
    atomic_write_csv(rows, path, fieldnames)


def download_one_cif(
    pdb_id: str, out_dir: Path, timeout: int, retries: int, backoff: float
) -> tuple[str, bool, str, Path | None]:
    pdb_id_l = pdb_id.lower()
    out_path = out_dir / f"{pdb_id_l}.cif"
    if out_path.exists() and out_path.stat().st_size > 0:
        return pdb_id, True, "exists", out_path

    urls = [
        (RCSB_CIF_URL.format(pdb=pdb_id_l), False),
        (RCSB_CIF_GZ_URL.format(pdb=pdb_id_l), True),
    ]
    last_msg = ""
    for url, gz in urls:
        try:
            data = http_get(url, timeout=timeout, retries=retries, backoff=backoff)
            if gz:
                data = maybe_decompress_gzip_bytes(data)
            text_head = data[:64].lstrip()
            if not (text_head.startswith(b"data_") or b"_entry.id" in data[:2048]):
                last_msg = f"unexpected_content_from:{url}"
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".cif.tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            tmp.replace(out_path)
            return pdb_id, True, "downloaded", out_path
        except urllib.error.HTTPError as e:  # pragma: no cover - network dependent
            last_msg = f"HTTPError {e.code}"
            continue
        except Exception as e:  # pragma: no cover - network dependent
            last_msg = f"Error {e}"
            continue
    return pdb_id, False, last_msg or "download_failed", None


def stage_one_cif(
    pdb_id: str, out_dir: Path, archive: FrozenStructureArchive
) -> tuple[str, bool, str, Path | None]:
    output = out_dir / f"{pdb_id.lower()}.cif"
    aliases = [
        f"{pdb_id}.cif",
        f"{pdb_id}.mmcif",
        f"{pdb_id}.cif.gz",
        f"{pdb_id}.mmcif.gz",
    ]
    source = archive.stage(aliases, output)
    return pdb_id, True, f"staged_local:{source.relative_to(archive.root)}", output


def validate_benchmark_structure(
    cif_path: Path,
    pdb_id: str,
    chain_id: str,
) -> tuple[bool, str, Path | None]:
    """Validate a full entry mmCIF without lossy single-chain reserialization."""

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(cif_path))
    except Exception as e:
        return False, f"parse_failed:{e}", None

    model = next(structure.get_models(), None)
    if model is None:
        return False, "no_model", None

    chain_ids = [str(ch.id) for ch in model]
    target_author_chain_id = chain_id
    if target_author_chain_id not in chain_ids:
        return False, f"chain_not_found:{chain_id};available={';'.join(chain_ids)}", None
    try:
        declaration = declared_polymer_sequence_for_author_chain(cif_path, target_author_chain_id)
    except (OSError, ValueError) as error:
        return False, f"complete_polymer_declaration_invalid:{error}", None
    if not declaration.sequence:
        return False, "complete_polymer_declaration_empty", None

    return True, "ok:full_entry_mmcif_preserved", cif_path


def run_structure_staging_and_validation(
    selected: Sequence[CandidateRecord],
    work_dir: Path,
    threads: int,
    timeout: int,
    retries: int,
    backoff: float,
    structure_archive: FrozenStructureArchive | None = None,
) -> list[dict[str, object]]:
    full_dir = ensure_dir(work_dir / "full_entries")
    unique_pdbs = sorted({rec.pdb_id for rec in selected})

    download_results: dict[str, tuple[bool, str, Path | None]] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        future_map = {}
        for pdb_id in unique_pdbs:
            if structure_archive is None:
                future = ex.submit(download_one_cif, pdb_id, full_dir, timeout, retries, backoff)
            else:
                future = ex.submit(stage_one_cif, pdb_id, full_dir, structure_archive)
            future_map[future] = pdb_id
        for fut in cf.as_completed(future_map):
            pdb_id = future_map[fut]
            try:
                pid, ok, msg, path = fut.result()
            except Exception as e:  # pragma: no cover
                pid, ok, msg, path = pdb_id, False, f"download_exception:{e}", None
            download_results[pid] = (ok, msg, path)
            print(f"[DOWNLOAD {'OK' if ok else 'FAIL'}] {pid}: {msg}", file=sys.stderr)

    structure_rows: list[dict[str, object]] = []
    for rec in selected:
        ok, dl_msg, cif_path = download_results.get(rec.pdb_id, (False, "not_downloaded", None))
        row = candidate_to_dict(rec)
        row.update(
            {
                "entry_cif_staged": ok,
                "entry_cif_message": dl_msg,
                "entry_cif_path": str(cif_path) if cif_path else "",
                "structure_cif_path": "",
                "structure_validation_ok": False,
                "structure_validation_message": "",
            }
        )
        if ok and cif_path is not None:
            validation_ok, validation_message, structure_cif = validate_benchmark_structure(
                cif_path=cif_path,
                pdb_id=rec.pdb_id,
                chain_id=rec.resolved_chain,
            )
            row["structure_validation_ok"] = validation_ok
            row["structure_validation_message"] = validation_message
            row["structure_cif_path"] = str(structure_cif) if structure_cif else ""
        structure_rows.append(row)
    return structure_rows


def summarize_selection(selected_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_class: Counter[int] = Counter()
    by_topology: Counter[str] = Counter()
    by_pdb: Counter[str] = Counter()
    for row in selected_rows:
        class_value = row["cath_dominant_class"]
        if isinstance(class_value, bool) or not isinstance(class_value, int):
            raise TypeError("cath_dominant_class must be an integer")
        cls = class_value
        by_class[cls] += 1
        tk = f"{row['cath_dominant_class']}.{row['cath_dominant_architecture']}.{row['cath_dominant_topology']}"
        by_topology[tk] += 1
        by_pdb[str(row["pdb_id"])] += 1
    return {
        "n_selected": len(selected_rows),
        "class_counts": {str(k): by_class[k] for k in sorted(by_class)},
        "n_unique_pdb": len(by_pdb),
        "n_unique_topologies": len(by_topology),
        "max_selected_per_pdb": max(by_pdb.values()) if by_pdb else 0,
        "max_selected_per_topology": max(by_topology.values()) if by_topology else 0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python data/scripts/build_easy_negatives_from_pisces_cath.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Build a chain-level negative candidate set from a PISCES sequence-reduced list "
            "and CATH structural classifications. Selection applies explicit class quotas, "
            "topology diversity limits, exclusions, and a reproducible random seed."
        ),
        epilog=(
            "Output: metadata CSVs for parsed inputs, candidates, exclusions, selections, and "
            "approved negatives; selected entry mmCIF files; selection_summary.json; and "
            "run_manifest.json. The output directory must be new or empty. Argument errors exit "
            "with status 2; input, download, or selection failures exit nonzero."
        ),
    )
    ap.add_argument(
        "--out",
        default="easy_negatives_from_pisces_cath",
        metavar="DIRECTORY",
        help="New output directory for the dataset and run metadata.",
    )
    ap.add_argument(
        "--mode",
        choices=["publication", "exploratory"],
        default="publication",
        help=(
            "publication uses explicit local source snapshots and an approval manifest; "
            "exploratory permits source discovery, downloads, and optional heuristic mapping."
        ),
    )

    # PISCES selection
    pisces_source = ap.add_mutually_exclusive_group()
    pisces_source.add_argument(
        "--pisces-file",
        default=None,
        metavar="FILE",
        help="Local PISCES chain list; required in publication mode.",
    )
    pisces_source.add_argument(
        "--pisces-url",
        default=None,
        metavar="URL",
        help="PISCES list URL for exploratory mode; omit to discover the current matching list.",
    )
    ap.add_argument(
        "--pisces-source-id",
        default=None,
        metavar="ID",
        help="PISCES URL, DOI, or release identifier recorded in provenance.",
    )
    ap.add_argument(
        "--pc",
        type=float,
        default=20.0,
        metavar="PERCENT",
        help="Maximum pairwise sequence identity used to select the PISCES list.",
    )
    ap.add_argument(
        "--resolution-max",
        type=float,
        default=2.0,
        metavar="ANGSTROMS",
        help="Maximum crystallographic resolution used to select the PISCES list.",
    )
    ap.add_argument(
        "--rmax",
        type=float,
        default=0.25,
        metavar="FRACTION",
        help="Maximum PISCES crystallographic R-factor in the closed interval [0, 1].",
    )
    ap.add_argument(
        "--method",
        default="Xray",
        choices=["Xray", "Xray+EM", "Xray+Nmr+EM"],
        help="Experimental-method set used to select the PISCES list.",
    )
    ap.add_argument(
        "--no-breaks",
        action="store_true",
        help="Select a PISCES source whose filename specifies no chain breaks.",
    )
    ap.add_argument(
        "--pisces-source-min-length",
        type=int,
        default=DEFAULT_PISCES_SOURCE_MIN_LENGTH,
        metavar="RESIDUES",
        help="Minimum chain length encoded in an auto-discovered PISCES filename.",
    )
    ap.add_argument(
        "--pisces-source-max-length",
        type=int,
        default=DEFAULT_PISCES_SOURCE_MAX_LENGTH,
        metavar="RESIDUES",
        help="Maximum chain length encoded in an auto-discovered PISCES filename.",
    )

    # Length / classification filtering
    ap.add_argument(
        "--min-length",
        type=int,
        default=80,
        metavar="RESIDUES",
        help="Minimum parsed PISCES chain length retained as a candidate.",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=1200,
        metavar="RESIDUES",
        help="Maximum parsed PISCES chain length retained as a candidate.",
    )
    ap.add_argument(
        "--min-class-fraction",
        type=float,
        default=0.70,
        metavar="FRACTION",
        help="Minimum fraction of classified domains belonging to the dominant CATH class.",
    )
    ap.add_argument(
        "--max-relative-length-mismatch",
        type=float,
        default=DEFAULT_MAX_RELATIVE_LENGTH_MISMATCH,
        metavar="FRACTION",
        help="Maximum relative length difference when mapping a PISCES chain labeled '0'.",
    )
    ap.add_argument(
        "--min-length-match-margin",
        type=float,
        default=DEFAULT_MIN_LENGTH_MATCH_MARGIN,
        metavar="FRACTION",
        help="Minimum relative-length advantage over the second-best chain-'0' match.",
    )
    ap.add_argument(
        "--allow-heuristic-length-mapping",
        action="store_true",
        help="In exploratory mode, resolve a PISCES chain labeled '0' by its length match.",
    )

    # Selection
    ap.add_argument(
        "--n-total",
        type=int,
        default=500,
        metavar="N",
        help="Total number of negative chains to select.",
    )
    ap.add_argument(
        "--class-quotas",
        default=DEFAULT_CLASS_QUOTAS,
        metavar="CLASS=N,...",
        help="Exact selection counts for CATH classes 1, 3, and 4; class 2 is omitted.",
    )
    ap.add_argument(
        "--max-per-topology",
        type=int,
        default=3,
        metavar="N",
        help="Maximum selected chains sharing one dominant CATH topology.",
    )
    ap.add_argument(
        "--max-per-pdb",
        type=int,
        default=1,
        metavar="N",
        help="Maximum selected chains from one PDB entry.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for sampling without replacement.",
    )

    # Exclusions
    ap.add_argument(
        "--mpstruc-xml",
        default=None,
        metavar="FILE",
        help="Local mpstruc XML/TXT snapshot whose PDB entries are excluded.",
    )
    ap.add_argument(
        "--mpstruc-source-id",
        default=None,
        metavar="ID",
        help="mpstruc URL, DOI, or release identifier recorded in provenance.",
    )
    ap.add_argument(
        "--exclude-file",
        default=None,
        metavar="FILE",
        help="Text, CSV, or TSV file containing additional PDB or PDB_chain identifiers to omit.",
    )
    ap.add_argument(
        "--negative-approval-manifest",
        default=None,
        metavar="CSV",
        help=(
            "Frozen CSV with exact pdb_id,target_author_chain_id,group_id,curation_evidence truth; "
            "required in publication mode."
        ),
    )

    # External files
    ap.add_argument(
        "--cath-domain-list",
        default=None,
        metavar="FILE",
        help="Local CATH domain-list snapshot; exploratory mode downloads it when omitted.",
    )
    ap.add_argument(
        "--cath-source-id",
        default=None,
        metavar="ID",
        help="CATH URL, DOI, or release identifier recorded in provenance.",
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

    # Download / extraction
    ap.add_argument(
        "--threads", type=int, default=8, metavar="N", help="Concurrent structure downloads."
    )
    ap.add_argument(
        "--timeout", type=int, default=60, metavar="SECONDS", help="Timeout per HTTP request."
    )
    ap.add_argument(
        "--retries", type=int, default=3, metavar="N", help="Retries after each failed request."
    )
    ap.add_argument(
        "--backoff",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Base delay for exponential HTTP retry backoff.",
    )
    ap.add_argument(
        "--no-download-structures",
        action="store_true",
        help="Write selection metadata without acquiring or extracting selected structures.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Write candidate and selection metadata, then stop before structure acquisition.",
    )

    return ap


def validate_arguments(args: argparse.Namespace) -> dict[int, int]:
    require_finite_real("pc", args.pc, minimum=0.0, maximum=100.0, minimum_inclusive=False)
    require_finite_real("resolution_max", args.resolution_max, minimum=0.0, minimum_inclusive=False)
    require_finite_real("rmax", args.rmax, minimum=0.0, maximum=1.0)
    require_exact_int("min_length", args.min_length, minimum=1)
    require_exact_int("max_length", args.max_length, minimum=1)
    if args.min_length > args.max_length:
        raise ValueError("min_length cannot exceed max_length")
    require_exact_int("pisces_source_min_length", args.pisces_source_min_length, minimum=1)
    require_exact_int("pisces_source_max_length", args.pisces_source_max_length, minimum=1)
    if args.pisces_source_min_length > args.pisces_source_max_length:
        raise ValueError("pisces_source_min_length cannot exceed pisces_source_max_length")
    require_finite_real(
        "min_class_fraction",
        args.min_class_fraction,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    require_finite_real(
        "max_relative_length_mismatch",
        args.max_relative_length_mismatch,
        minimum=0.0,
        maximum=1.0,
    )
    require_finite_real(
        "min_length_match_margin",
        args.min_length_match_margin,
        minimum=0.0,
        maximum=1.0,
    )
    require_exact_int("n_total", args.n_total, minimum=1)
    require_exact_int("max_per_topology", args.max_per_topology, minimum=1)
    require_exact_int("max_per_pdb", args.max_per_pdb, minimum=1)
    require_exact_int("seed", args.seed)
    require_exact_int("threads", args.threads, minimum=1)
    require_exact_int("timeout", args.timeout, minimum=1)
    require_exact_int("retries", args.retries, minimum=0)
    require_finite_real("backoff", args.backoff, minimum=0.0)
    for name in (
        "no_breaks",
        "no_download_structures",
        "dry_run",
        "allow_heuristic_length_mapping",
    ):
        if not isinstance(getattr(args, name), bool):
            raise ValueError(f"{name} must be a boolean")

    if args.mode == "publication" and (not args.pisces_file or not args.cath_domain_list):
        raise ValueError(
            "publication mode requires --pisces-file and --cath-domain-list; "
            "remote discovery is exploratory and unpinned"
        )
    if args.mode == "publication" and not args.mpstruc_xml:
        raise ValueError(
            "publication mode requires --mpstruc-xml so known positive entries cannot leak "
            "into the negative pool"
        )
    for name in (
        "pisces_file",
        "cath_domain_list",
        "mpstruc_xml",
        "exclude_file",
        "negative_approval_manifest",
    ):
        value = getattr(args, name)
        if value:
            input_file_record(value, source=f"local {name}", pinned=True)
    if args.negative_approval_manifest:
        load_negative_approvals(args.negative_approval_manifest)
    if args.structure_source_dir:
        input_directory_record(
            args.structure_source_dir, source="local frozen coordinate archive", pinned=True
        )
    if args.mode == "publication":
        if args.max_per_pdb != 1:
            raise ValueError("publication mode requires --max-per-pdb 1")
        if not args.negative_approval_manifest:
            raise ValueError(
                "publication mode requires --negative-approval-manifest; CATH classes are "
                "candidate strata, not independent NON_BARREL ground truth"
            )
        if not args.structure_source_dir:
            raise ValueError(
                "publication mode requires --structure-source-dir and never downloads live "
                "RCSB coordinates"
            )
        if args.no_download_structures or args.dry_run:
            raise ValueError(
                "--no-download-structures and --dry-run are exploratory-only; a publication "
                "build must verify its complete frozen coordinate archive"
            )
        if args.allow_heuristic_length_mapping:
            raise ValueError("heuristic PISCES-to-CATH chain mapping is exploratory-only")
        require_pinned_source_id("pisces_source_id", args.pisces_source_id)
        require_pinned_source_id("cath_source_id", args.cath_source_id)
        require_pinned_source_id("mpstruc_source_id", args.mpstruc_source_id)
        require_pinned_source_id("structure_source_id", args.structure_source_id)
    if args.pisces_url:
        parsed = urllib.parse.urlparse(args.pisces_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("pisces_url must be an absolute HTTP(S) URL")
    return parse_class_quotas(args.class_quotas, args.n_total)


def _run(
    args: argparse.Namespace,
    work_dir: Path,
    run_manifest: RunManifest,
    quotas: dict[int, int],
) -> dict[str, object]:
    meta_dir = ensure_dir(work_dir / "metadata")
    ensure_dir(work_dir / "downloads")

    write_json(vars(args), meta_dir / "run_config.json")

    print("[INFO] Resolving PISCES source...", file=sys.stderr)
    pisces_source, pisces_local, pisces_pinned, pisces_release = resolve_pisces_file(args, work_dir)
    print(f"[INFO] PISCES list: {pisces_source}", file=sys.stderr)

    print("[INFO] Resolving CATH domain list...", file=sys.stderr)
    cath_local, cath_source, cath_pinned, cath_release = get_cath_file(args, work_dir)
    print(f"[INFO] CATH domain list: {cath_local}", file=sys.stderr)

    input_records: dict[str, object] = {
        "pisces": input_file_record(
            pisces_local,
            source=args.pisces_source_id or pisces_source,
            pinned=pisces_pinned,
            release=args.pisces_source_id or pisces_release,
        ),
        "cath": input_file_record(
            cath_local,
            source=args.cath_source_id or cath_source,
            pinned=cath_pinned,
            release=args.cath_source_id or cath_release,
        ),
    }
    for key, value in (
        ("mpstruc", args.mpstruc_xml),
        ("exclusions", args.exclude_file),
        ("negative_approvals", args.negative_approval_manifest),
    ):
        if value:
            input_records[key] = input_file_record(
                value,
                source=(
                    args.mpstruc_source_id
                    if key == "mpstruc" and args.mpstruc_source_id
                    else (
                        "local frozen independent negative approvals"
                        if key == "negative_approvals"
                        else f"local {key} snapshot"
                    )
                ),
                pinned=True,
                release=args.mpstruc_source_id if key == "mpstruc" else None,
            )
    if args.structure_source_dir:
        input_records["structure_archive"] = input_directory_record(
            args.structure_source_dir,
            source=args.structure_source_id or "local coordinate archive (exploratory)",
            pinned=True,
        )
    remote_sources: list[dict[str, object]] = []
    if not pisces_pinned:
        remote_sources.append(
            {
                "name": "PISCES chain list",
                "url": pisces_source,
                "release": pisces_release,
                "pinned": False,
            }
        )
    if not cath_pinned:
        remote_sources.append(
            {
                "name": "CATH domain list",
                "url": cath_source,
                "release": cath_release,
                "pinned": False,
            }
        )
    if not args.structure_source_dir and not args.no_download_structures and not args.dry_run:
        remote_sources.append(
            {
                "name": "RCSB PDB coordinates",
                "url_templates": [RCSB_CIF_URL, RCSB_CIF_GZ_URL],
                "release": "current archive snapshot",
                "pinned": False,
                "reproducibility": "each coordinate output is SHA-256 inventoried",
            }
        )
    run_manifest.set_provenance(inputs=input_records, remote_sources=remote_sources)

    print("[INFO] Parsing PISCES list...", file=sys.stderr)
    pisces_records = parse_pisces_list(pisces_local)
    if not pisces_records:
        raise ValueError("PISCES input contains no parseable chain records")
    print(f"[INFO] Parsed {len(pisces_records)} PISCES chains.", file=sys.stderr)

    print("[INFO] Parsing CATH domain list...", file=sys.stderr)
    cath_domains = parse_cath_domain_list(cath_local)
    if not cath_domains:
        raise ValueError("CATH input contains no parseable domain records")
    cath_summaries, entry_to_chains = build_cath_chain_summary(cath_domains)
    print(
        f"[INFO] Parsed {len(cath_domains)} CATH domains and {len(cath_summaries)} CATH chains.",
        file=sys.stderr,
    )

    mpstruc_exclude = load_mpstruc_exclusions(args.mpstruc_xml)
    extra_exclude_pdbs, extra_exclude_chains = load_generic_exclusions(args.exclude_file)
    exclude_pdbs = set(mpstruc_exclude) | set(extra_exclude_pdbs)
    exclude_chains = set(extra_exclude_chains)
    print(
        f"[INFO] Exclusions: {len(exclude_pdbs)} PDBs, {len(exclude_chains)} chain IDs.",
        file=sys.stderr,
    )

    candidates, excluded = make_candidate_rows(
        pisces_records=pisces_records,
        cath_summaries=cath_summaries,
        entry_to_chains=entry_to_chains,
        exclude_pdbs=exclude_pdbs,
        exclude_chains=exclude_chains,
        min_length=args.min_length,
        max_length=args.max_length,
        min_class_fraction=args.min_class_fraction,
        max_relative_length_mismatch=args.max_relative_length_mismatch,
        min_length_match_margin=args.min_length_match_margin,
        allow_heuristic_length_mapping=args.allow_heuristic_length_mapping,
    )
    print(f"[INFO] Candidate easy negatives after filtering: {len(candidates)}", file=sys.stderr)
    print(f"[INFO] Excluded rows: {len(excluded)}", file=sys.stderr)

    # Write intermediate metadata.
    write_csv([asdict(x) for x in pisces_records], meta_dir / "pisces_records.csv")
    write_csv([asdict(x) for x in cath_summaries.values()], meta_dir / "cath_chain_summary.csv")
    write_csv([candidate_to_dict(x) for x in candidates], meta_dir / "easy_negative_candidates.csv")
    write_csv(excluded, meta_dir / "excluded_records.csv")

    selected = select_diverse_easy_negatives(
        candidates=candidates,
        n_total=args.n_total,
        class_quotas=quotas,
        max_per_topology=args.max_per_topology,
        max_per_pdb=args.max_per_pdb,
        seed=args.seed,
    )
    approvals: dict[tuple[str, str], dict[str, str]] = {}
    if args.negative_approval_manifest:
        approvals = match_negative_approvals(
            selected,
            load_negative_approvals(args.negative_approval_manifest),
        )
    print(f"[INFO] Selected {len(selected)} easy negatives.", file=sys.stderr)
    write_csv(
        [candidate_to_dict(x) for x in selected],
        meta_dir / "easy_negative_selected_pre_download.csv",
    )

    if args.dry_run or args.no_download_structures:
        selected_rows = [candidate_to_dict(x) for x in selected]
        if approvals:
            selected_rows = apply_negative_approvals(selected_rows, approvals)
        summary = summarize_selection(selected_rows)
        summary["selection_algorithm"] = SELECTION_ALGORITHM
        summary["seed"] = args.seed
        summary["class_quotas"] = {str(key): value for key, value in quotas.items()}
        write_json(summary, meta_dir / "selection_summary.json")
        write_csv(selected_rows, meta_dir / "easy_negative_selected.csv")
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return summary

    structure_rows = run_structure_staging_and_validation(
        selected=selected,
        work_dir=work_dir,
        threads=args.threads,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        structure_archive=(
            FrozenStructureArchive(args.structure_source_dir) if args.structure_source_dir else None
        ),
    )
    if approvals:
        structure_rows = apply_negative_approvals(structure_rows, approvals)
    write_csv(structure_rows, meta_dir / "easy_negative_selected.csv")
    if args.mode == "publication":
        failures = [
            row
            for row in structure_rows
            if row.get("entry_cif_staged") is not True
            or row.get("structure_validation_ok") is not True
        ]
        if failures:
            failed_ids = [f"{row['pdb_id']}:{row['resolved_chain']}" for row in failures]
            raise RuntimeError(
                "Frozen publication structure archive is incomplete or contains invalid "
                f"targets: {failed_ids}"
            )
        approval_rows = [
            {
                "filename": row["filename"],
                "target_author_chain_id": row["target_author_chain_id"],
                "group_id": row["group_id"],
                "curation_evidence": row["curation_evidence"],
                "truth_label": row["truth_label"],
            }
            for row in structure_rows
        ]
        write_csv(approval_rows, meta_dir / "d2_approved_negatives.csv")
    summary = summarize_selection(structure_rows)
    summary["selection_algorithm"] = SELECTION_ALGORITHM
    summary["seed"] = args.seed
    summary["class_quotas"] = {str(key): value for key, value in quotas.items()}
    write_json(summary, meta_dir / "selection_summary.json")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    quotas = validate_arguments(args)
    work_dir = prepare_new_output_directory(args.out)
    run_manifest = RunManifest(
        work_dir,
        __file__,
        vars(args),
        mode=args.mode,
        random_algorithm=SELECTION_ALGORITHM,
        seed=args.seed,
    )
    try:
        summary = _run(args, work_dir, run_manifest, quotas)
    except BaseException as error:
        run_manifest.fail(error)
        raise
    run_manifest.complete(summary)


if __name__ == "__main__":
    main()
