"""Strict extraction of declared complete protein-polymer sequences.

Coordinate observations are deliberately not a sequence source.  PDB inputs
must carry complete ``SEQRES`` declarations.  mmCIF inputs must carry mutually
consistent ``entity_poly``/``entity_poly_seq`` declarations and an exact,
one-to-one label-to-author chain mapping.
"""

from __future__ import annotations

import gzip
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

SUPPORTED_STRUCTURE_EXTENSIONS = frozenset({".pdb", ".ent", ".cif", ".mmcif"})
COMPLETE_SEQUENCE_POLICY = "declared_complete_polymer_sequence_strict_author_chain_mapping"
COMPLETE_SEQUENCE_SOURCES = frozenset(
    {
        "pdb_seqres",
        "mmcif_entity_poly_canonical",
        "mmcif_entity_poly_seq",
    }
)


class CompleteSequenceUnavailableError(ValueError):
    """The input contains no complete protein-polymer sequence declaration."""


@dataclass(frozen=True)
class CompletePolymerSequence:
    """One complete declared protein sequence with an exact author-chain identity."""

    author_chain_id: str
    label_asym_id: str
    entity_id: str
    sequence: str
    monomer_ids: tuple[str, ...]
    sequence_source: str

    def __post_init__(self) -> None:
        if not self.author_chain_id:
            raise ValueError("Complete polymer sequence has a blank author-chain identity.")
        if not self.sequence or re.fullmatch(r"[A-Z]+", self.sequence) is None:
            raise ValueError("Complete polymer sequence must contain uppercase residue codes.")
        if len(self.monomer_ids) != len(self.sequence):
            raise ValueError("Complete polymer sequence and monomer declaration lengths differ.")
        if self.sequence_source not in COMPLETE_SEQUENCE_SOURCES:
            raise ValueError(f"Unknown complete sequence source: {self.sequence_source!r}.")
        if self.sequence_source.startswith("mmcif_") and (
            not self.label_asym_id or not self.entity_id
        ):
            raise ValueError("mmCIF complete sequences require entity and label-chain IDs.")


def structure_extension(path: Path) -> str:
    """Return the true coordinate extension, preserving compound gzip suffixes."""

    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).suffix


@contextmanager
def _materialized_structure_path(path: Path) -> Iterator[Path]:
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    if not is_gzip:
        yield path
        return

    descriptor, temporary_name = tempfile.mkstemp(suffix=structure_extension(path) or ".pdb")
    temporary_path = Path(temporary_name)
    try:
        with gzip.open(path, "rb") as source, os.fdopen(descriptor, "wb") as target:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalized_chain_id(value: object) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(
            "Blank author chain identifiers cannot be represented unambiguously; "
            "complete-polymer sequence parsing fails closed."
        )
    return normalized


def _monomer_letter(monomer_id: str) -> str | None:
    normalized = monomer_id.strip().upper()
    if normalized in {"UNK", "XAA"}:
        return "X"
    value = protein_letters_3to1_extended.get(normalized)
    return str(value).upper() if value is not None else None


def _sequence_from_monomers(
    monomer_ids: Sequence[str],
    *,
    context: str,
    canonical_sequence: str | None = None,
) -> str | None:
    letters = [_monomer_letter(monomer) for monomer in monomer_ids]
    if any(letter is None for letter in letters) and canonical_sequence is not None:
        if len(canonical_sequence) != len(monomer_ids):
            raise ValueError(
                f"{context} canonical sequence has {len(canonical_sequence)} residues but "
                f"its entity_poly_seq declaration has {len(monomer_ids)} positions."
            )
        conflicts = [
            index
            for index, (letter, canonical_letter) in enumerate(
                zip(letters, canonical_sequence, strict=True), start=1
            )
            if letter is not None and canonical_letter != "X" and letter != canonical_letter
        ]
        if conflicts:
            raise ValueError(
                f"{context} canonical sequence conflicts with supported entity_poly_seq "
                f"monomers at positions {conflicts[:10]!r}."
            )
        return canonical_sequence
    if all(letter is None for letter in letters):
        return None
    if any(letter is None for letter in letters):
        unsupported = sorted(
            {
                monomer
                for monomer, letter in zip(monomer_ids, letters, strict=True)
                if letter is None
            }
        )
        raise ValueError(
            f"{context} mixes protein and unsupported polymer monomers: {unsupported!r}."
        )
    return "".join(str(letter) for letter in letters)


def _pdb_declared_sequences(path: Path) -> list[CompletePolymerSequence]:
    lines_by_chain: dict[str, list[tuple[int, int, list[str]]]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.startswith("SEQRES"):
                continue
            try:
                serial = int(line[7:10].strip())
                declared_count = int(line[13:17].strip())
            except ValueError as exc:
                raise ValueError(f"Malformed PDB SEQRES record at line {line_number}.") from exc
            chain_id = _normalized_chain_id(line[11:12])
            row_monomers = line[19:70].split()
            if serial < 1 or declared_count < 1 or not row_monomers:
                raise ValueError(f"Invalid PDB SEQRES record at line {line_number}.")
            lines_by_chain[chain_id].append((serial, declared_count, row_monomers))

    if not lines_by_chain:
        raise CompleteSequenceUnavailableError(
            f"PDB input {path} has no SEQRES records; a declared complete polymer "
            "sequence is required."
        )

    sequences: list[CompletePolymerSequence] = []
    for chain_id, rows in sorted(lines_by_chain.items()):
        rows.sort(key=lambda item: item[0])
        serials = [row[0] for row in rows]
        if serials != list(range(1, len(rows) + 1)):
            raise ValueError(f"PDB SEQRES serials are incomplete for chain {chain_id!r}.")
        declared_counts = {row[1] for row in rows}
        if len(declared_counts) != 1:
            raise ValueError(f"PDB SEQRES residue counts conflict for chain {chain_id!r}.")
        declared_monomers = tuple(
            monomer for _, _, row_monomers in rows for monomer in row_monomers
        )
        declared_count = next(iter(declared_counts))
        if len(declared_monomers) != declared_count:
            raise ValueError(
                f"PDB SEQRES chain {chain_id!r} declares {declared_count} residues but "
                f"contains {len(declared_monomers)} monomers."
            )
        sequence = _sequence_from_monomers(
            declared_monomers,
            context=f"PDB SEQRES chain {chain_id!r}",
        )
        if sequence is None:
            continue
        sequences.append(
            CompletePolymerSequence(
                author_chain_id=chain_id,
                label_asym_id="",
                entity_id="",
                sequence=sequence,
                monomer_ids=declared_monomers,
                sequence_source="pdb_seqres",
            )
        )
    if not sequences:
        raise CompleteSequenceUnavailableError(
            f"PDB input {path} declares no protein polymer sequence."
        )
    return sequences


def _mmcif_values(mmcif: Mapping[str, Any], column: str) -> list[str]:
    raw = mmcif.get(column, [])
    if isinstance(raw, str):
        return [raw]
    return [str(value) for value in raw]


def _require_equal_loop_lengths(
    mmcif: Mapping[str, Any],
    columns: Sequence[str],
) -> list[list[str]]:
    values = [_mmcif_values(mmcif, column) for column in columns]
    lengths = {len(column_values) for column_values in values}
    if len(lengths) != 1 or not values or not values[0]:
        details = ", ".join(
            f"{column}={len(column_values)}"
            for column, column_values in zip(columns, values, strict=True)
        )
        raise ValueError(f"Missing or inconsistent mmCIF loop columns: {details}.")
    return values


def _normalized_canonical_sequence(value: str, *, entity_id: str) -> str:
    sequence = re.sub(r"\s+", "", value).upper().replace("?", "X")
    if not sequence or re.fullmatch(r"[A-Z]+", sequence) is None:
        raise ValueError(f"mmCIF entity {entity_id!r} has an invalid canonical polymer sequence.")
    return sequence


def _mmcif_declared_sequences(path: Path) -> list[CompletePolymerSequence]:
    try:
        mmcif = MMCIF2Dict(str(path))
    except Exception as exc:
        raise ValueError(f"Could not parse mmCIF sequence declarations from {path}: {exc}") from exc

    entity_poly_columns = ("_entity_poly.entity_id", "_entity_poly.type")
    if not any(_mmcif_values(mmcif, column) for column in entity_poly_columns):
        raise CompleteSequenceUnavailableError(
            f"mmCIF input {path} has no entity_poly complete-sequence declaration."
        )
    entity_ids, polymer_types = _require_equal_loop_lengths(mmcif, entity_poly_columns)
    canonical_values = _mmcif_values(mmcif, "_entity_poly.pdbx_seq_one_letter_code_can")
    if canonical_values and len(canonical_values) != len(entity_ids):
        raise ValueError(
            "Missing or inconsistent mmCIF loop columns: "
            f"_entity_poly.entity_id={len(entity_ids)}, "
            f"_entity_poly.pdbx_seq_one_letter_code_can={len(canonical_values)}."
        )
    canonical_by_entity: dict[str, str] = {}
    polypeptide_entity_ids: set[str] = set()
    seen_entity_ids: set[str] = set()
    for row_index, (entity_id, polymer_type) in enumerate(
        zip(entity_ids, polymer_types, strict=True)
    ):
        if entity_id in seen_entity_ids:
            raise ValueError(f"Duplicate mmCIF entity_poly row for entity {entity_id!r}.")
        seen_entity_ids.add(entity_id)
        if polymer_type.lower().startswith("polypeptide"):
            polypeptide_entity_ids.add(entity_id)
            if canonical_values:
                canonical_by_entity[entity_id] = _normalized_canonical_sequence(
                    canonical_values[row_index],
                    entity_id=entity_id,
                )
    if not polypeptide_entity_ids:
        raise CompleteSequenceUnavailableError(
            f"mmCIF input {path} declares no polypeptide entity."
        )

    sequence_entity_ids, sequence_numbers, sequence_monomers = _require_equal_loop_lengths(
        mmcif,
        (
            "_entity_poly_seq.entity_id",
            "_entity_poly_seq.num",
            "_entity_poly_seq.mon_id",
        ),
    )
    sequence_heterogeneity = _mmcif_values(mmcif, "_entity_poly_seq.hetero")
    if sequence_heterogeneity and len(sequence_heterogeneity) != len(sequence_entity_ids):
        raise ValueError(
            "Missing or inconsistent mmCIF loop columns: "
            f"_entity_poly_seq.entity_id={len(sequence_entity_ids)}, "
            f"_entity_poly_seq.hetero={len(sequence_heterogeneity)}."
        )
    if not sequence_heterogeneity:
        sequence_heterogeneity = ["n"] * len(sequence_entity_ids)
    monomers_by_entity: dict[str, dict[int, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entity_id, raw_number, monomer, heterogeneous in zip(
        sequence_entity_ids,
        sequence_numbers,
        sequence_monomers,
        sequence_heterogeneity,
        strict=True,
    ):
        if entity_id not in polypeptide_entity_ids:
            continue
        try:
            number = int(raw_number)
        except ValueError as exc:
            raise ValueError(
                f"mmCIF entity {entity_id!r} has non-integer entity_poly_seq.num {raw_number!r}."
            ) from exc
        if number < 1:
            raise ValueError(f"mmCIF entity {entity_id!r} has invalid polymer position {number}.")
        alternatives = monomers_by_entity[entity_id][number]
        if alternatives and (
            heterogeneous.lower() != "y"
            or any(
                previous_heterogeneous.lower() != "y" for _, previous_heterogeneous in alternatives
            )
        ):
            raise ValueError(
                f"mmCIF entity {entity_id!r} has duplicate polymer position {number} "
                "without an explicit heterogeneous declaration."
            )
        alternatives.append((monomer, heterogeneous))

    ordered_monomers: dict[str, tuple[str, ...]] = {}
    polypeptide_sequences: dict[str, str] = {}
    sequence_sources: dict[str, str] = {}
    for entity_id in sorted(polypeptide_entity_ids):
        numbered = monomers_by_entity.get(entity_id, {})
        if not numbered:
            raise ValueError(
                f"mmCIF entity {entity_id!r} has no complete entity_poly_seq declaration."
            )
        expected_positions = list(range(1, max(numbered) + 1))
        if sorted(numbered) != expected_positions:
            raise ValueError(
                f"mmCIF entity {entity_id!r} entity_poly_seq positions do not exactly "
                "cover one contiguous sequence beginning at position 1."
            )
        canonical = canonical_by_entity.get(entity_id)
        if canonical is not None and len(canonical) != len(expected_positions):
            raise ValueError(
                f"mmCIF entity {entity_id!r} canonical sequence has {len(canonical)} residues "
                f"but its entity_poly_seq declaration has {len(expected_positions)} positions."
            )
        selected_monomers: list[str] = []
        for position in expected_positions:
            alternatives = numbered[position]
            monomer_ids = [monomer for monomer, _ in alternatives]
            if canonical is None:
                letters = {_monomer_letter(monomer) for monomer in monomer_ids}
                if None in letters or len(letters) != 1:
                    raise ValueError(
                        f"mmCIF entity {entity_id!r} has an ambiguous heterogeneous polymer "
                        f"position {position}: {monomer_ids!r}."
                    )
                selected_monomers.append(monomer_ids[0])
                continue
            canonical_letter = canonical[position - 1]
            matching = [
                monomer
                for monomer in monomer_ids
                if canonical_letter == "X" or _monomer_letter(monomer) == canonical_letter
            ]
            supported_letters = {
                letter
                for monomer in monomer_ids
                if (letter := _monomer_letter(monomer)) is not None
            }
            if matching:
                selected_monomers.append(matching[0])
            elif supported_letters:
                raise ValueError(
                    f"mmCIF entity {entity_id!r} canonical sequence conflicts with "
                    f"heterogeneous position {position}: {monomer_ids!r}."
                )
            else:
                selected_monomers.append(monomer_ids[0])
        monomers = tuple(selected_monomers)
        derived = _sequence_from_monomers(
            monomers,
            context=f"mmCIF entity {entity_id!r}",
            canonical_sequence=canonical_by_entity.get(entity_id),
        )
        if derived is None:
            raise ValueError(f"mmCIF entity {entity_id!r} contains an unsupported polymer monomer.")
        if canonical is not None and derived != canonical:
            canonical_matches = len(derived) == len(canonical) and all(
                canonical_letter == "X" or derived_letter == canonical_letter
                for derived_letter, canonical_letter in zip(derived, canonical, strict=True)
            )
            if not canonical_matches:
                raise ValueError(
                    f"mmCIF entity {entity_id!r} canonical sequence conflicts with its complete "
                    "entity_poly_seq monomer declaration."
                )
            derived = canonical
        ordered_monomers[entity_id] = monomers
        polypeptide_sequences[entity_id] = derived
        sequence_sources[entity_id] = (
            "mmcif_entity_poly_canonical" if canonical is not None else "mmcif_entity_poly_seq"
        )

    label_asym_ids, asym_entity_ids = _require_equal_loop_lengths(
        mmcif,
        ("_struct_asym.id", "_struct_asym.entity_id"),
    )
    entity_by_label: dict[str, str] = {}
    for label_asym_id, entity_id in zip(label_asym_ids, asym_entity_ids, strict=True):
        if label_asym_id in entity_by_label:
            raise ValueError(f"Duplicate mmCIF struct_asym ID {label_asym_id!r}.")
        entity_by_label[label_asym_id] = entity_id

    auth_candidates: dict[str, set[str]] = defaultdict(set)
    mapping_sources = (
        ("_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.pdb_strand_id"),
        ("_atom_site.label_asym_id", "_atom_site.auth_asym_id"),
    )
    for label_column, auth_column in mapping_sources:
        labels = _mmcif_values(mmcif, label_column)
        auths = _mmcif_values(mmcif, auth_column)
        if not labels and not auths:
            continue
        if len(labels) != len(auths):
            raise ValueError(
                f"Inconsistent mmCIF author-chain mapping columns {label_column!r} and "
                f"{auth_column!r}."
            )
        for label_asym_id, auth_asym_id in zip(labels, auths, strict=True):
            if auth_asym_id not in {"", ".", "?"}:
                auth_candidates[label_asym_id].add(_normalized_chain_id(auth_asym_id))

    declared: list[CompletePolymerSequence] = []
    used_author_chains: dict[str, str] = {}
    for label_asym_id, entity_id in sorted(entity_by_label.items()):
        if entity_id not in polypeptide_sequences:
            continue
        candidates = auth_candidates.get(label_asym_id, set())
        if len(candidates) != 1:
            raise ValueError(
                f"mmCIF polypeptide label chain {label_asym_id!r} must map to exactly one "
                f"author chain; observed {sorted(candidates)!r}."
            )
        author_chain_id = next(iter(candidates))
        previous_label = used_author_chains.get(author_chain_id)
        if previous_label is not None and previous_label != label_asym_id:
            raise ValueError(
                f"mmCIF author chain {author_chain_id!r} maps ambiguously from label chains "
                f"{previous_label!r} and {label_asym_id!r}."
            )
        used_author_chains[author_chain_id] = label_asym_id
        declared.append(
            CompletePolymerSequence(
                author_chain_id=author_chain_id,
                label_asym_id=label_asym_id,
                entity_id=entity_id,
                sequence=polypeptide_sequences[entity_id],
                monomer_ids=ordered_monomers[entity_id],
                sequence_source=sequence_sources[entity_id],
            )
        )
    if not declared:
        raise ValueError(
            f"mmCIF input {path} declares polypeptide entities but has no exactly mapped "
            "author chain."
        )
    return declared


def declared_polymer_sequences(path: str | Path) -> list[CompletePolymerSequence]:
    """Parse all strictly declared complete protein sequences from one structure."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with _materialized_structure_path(source) as materialized:
        extension = structure_extension(materialized)
        if extension in {".cif", ".mmcif"}:
            return _mmcif_declared_sequences(materialized)
        if extension in {".pdb", ".ent"}:
            return _pdb_declared_sequences(materialized)
    raise ValueError(f"Unsupported structure extension: {source}")


def observed_author_chain_ids(path: str | Path) -> frozenset[str]:
    """Return first-model author-chain IDs without treating coordinates as sequence truth."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with _materialized_structure_path(source) as materialized:
        extension = structure_extension(materialized)
        try:
            if extension in {".cif", ".mmcif"}:
                parser = MMCIFParser(
                    QUIET=True,
                    auth_chains=True,
                )
                structure = parser.get_structure("chain_identity", str(materialized))
            elif extension in {".pdb", ".ent"}:
                pdb_parser = PDBParser(
                    QUIET=True,
                    PERMISSIVE=False,
                    get_header=False,
                )
                structure = pdb_parser.get_structure("chain_identity", str(materialized))
            else:
                raise ValueError(f"Unsupported structure extension: {source}")
            model = next(structure.get_models())
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ValueError(f"Could not establish author-chain identities for {source}.") from exc
        chains = [str(chain.id).strip() for chain in model.get_chains()]
    if not chains:
        raise ValueError(f"Structure {source} has no observed author chains.")
    if any(not chain for chain in chains):
        raise ValueError(
            f"Structure {source} contains a blank author-chain identity; parsing fails closed."
        )
    if len(set(chains)) != len(chains):
        raise ValueError(f"Structure {source} has duplicate author-chain identities.")
    return frozenset(chains)


def declared_polymer_sequence_for_author_chain(
    path: str | Path,
    author_chain_id: str,
) -> CompletePolymerSequence:
    """Return one exact author-chain sequence or fail on missing/ambiguous identity."""

    normalized_chain = _normalized_chain_id(author_chain_id)
    matches = [
        declared
        for declared in declared_polymer_sequences(path)
        if declared.author_chain_id == normalized_chain
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Structure {Path(path)} must declare exactly one complete protein sequence for "
            f"author chain {normalized_chain!r}; observed {len(matches)}."
        )
    return matches[0]
