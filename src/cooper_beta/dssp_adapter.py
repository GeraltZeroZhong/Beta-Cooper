from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .constants import DSSP_RESIDUE_COVERAGE_POLICY

DsspResidueId = tuple[str, int, str]
DsspResidueKey = tuple[str, DsspResidueId]
DsspSourceStrandId = tuple[str, str]

# Changing the annotation mapping invalidates every prepared payload. The
# preparation cache includes this module in its producer state.
DSSP_ANNOTATION_ADAPTER_VERSION = 1
DSSP_COVERAGE_MISSING_EXAMPLE_LIMIT = 5
DSSP_SECONDARY_STRUCTURE_CODES = frozenset({"-", "B", "E", "G", "H", "I", "P", "S", "T"})
DSSP_STRAND_SECONDARY_STRUCTURE_CODE = "E"

_MMCIF_MISSING_VALUES = frozenset({"", ".", "?"})
_ATOM_MAPPING_COLUMNS = (
    "_atom_site.label_asym_id",
    "_atom_site.label_seq_id",
    "_atom_site.auth_asym_id",
    "_atom_site.auth_seq_id",
    "_atom_site.pdbx_PDB_ins_code",
    "_atom_site.group_PDB",
    "_atom_site.label_comp_id",
)
_DSSP_SUMMARY_COLUMNS = (
    "_dssp_struct_summary.label_asym_id",
    "_dssp_struct_summary.label_seq_id",
    "_dssp_struct_summary.secondary_structure",
)
_SHEET_RANGE_COLUMNS = (
    "_struct_sheet_range.sheet_id",
    "_struct_sheet_range.id",
    "_struct_sheet_range.beg_label_asym_id",
    "_struct_sheet_range.beg_label_seq_id",
    "_struct_sheet_range.end_label_asym_id",
    "_struct_sheet_range.end_label_seq_id",
)
_DSSP_LADDER_COLUMNS = (
    "_dssp_struct_ladder.beg_1_label_asym_id",
    "_dssp_struct_ladder.beg_1_label_seq_id",
    "_dssp_struct_ladder.end_1_label_asym_id",
    "_dssp_struct_ladder.end_1_label_seq_id",
    "_dssp_struct_ladder.beg_2_label_asym_id",
    "_dssp_struct_ladder.beg_2_label_seq_id",
    "_dssp_struct_ladder.end_2_label_asym_id",
    "_dssp_struct_ladder.end_2_label_seq_id",
)


@dataclass(frozen=True, order=True)
class DsspStrandRecord:
    """One DSSP E-strand with exact author-residue membership."""

    source_id: DsspSourceStrandId
    author_chain_id: str
    residue_keys: tuple[DsspResidueKey, ...]


@dataclass(frozen=True, order=True)
class DsspStrandEdge:
    """Undirected adjacency between two DSSP source strands."""

    first_source_id: DsspSourceStrandId
    second_source_id: DsspSourceStrandId

    def __post_init__(self) -> None:
        first = self.first_source_id
        second = self.second_source_id
        if first == second:
            raise ValueError("A DSSP strand adjacency cannot be a self-loop.")
        if second < first:
            object.__setattr__(self, "first_source_id", second)
            object.__setattr__(self, "second_source_id", first)


@dataclass(frozen=True)
class DsspAnnotation:
    """Fresh DSSP residue assignments, strand ranges, and adjacency edges."""

    residue_assignments: dict[DsspResidueKey, str]
    strand_records: tuple[DsspStrandRecord, ...]
    strand_edges: tuple[DsspStrandEdge, ...]


@dataclass(frozen=True)
class _MappedResidue:
    author_key: DsspResidueKey
    polymer_index: int


@dataclass(frozen=True)
class _ParsedStrand:
    record: DsspStrandRecord
    label_keys: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class _ParsedLadder:
    first_source_ids: tuple[DsspSourceStrandId, ...]
    second_source_ids: tuple[DsspSourceStrandId, ...]


class DsspAdapterError(RuntimeError):
    """Raised when fresh DSSP output cannot be mapped losslessly."""


def validate_dssp_coverage(
    expected_residue_keys: Collection[DsspResidueKey],
    assigned_residue_keys: Collection[DsspResidueKey],
    *,
    context: str,
) -> None:
    """Require an explicit DSSP state for every DSSP-eligible polypeptide residue."""

    expected = set(expected_residue_keys)
    missing = expected.difference(assigned_residue_keys)
    if not missing:
        return
    examples = sorted(missing)[:DSSP_COVERAGE_MISSING_EXAMPLE_LIMIT]
    assigned_count = len(expected) - len(missing)
    raise DsspAdapterError(
        f"DSSP residue coverage is incomplete for {context}: assigned "
        f"{assigned_count}/{len(expected)} required residues under policy "
        f"{DSSP_RESIDUE_COVERAGE_POLICY!r}; missing examples={examples!r}. "
        "Missing residues cannot be interpreted as coil."
    )


def validate_dssp_secondary_structure_code(code: str, *, context: str) -> None:
    """Reject values outside the DSSP 4 secondary-structure alphabet."""

    if code not in DSSP_SECONDARY_STRUCTURE_CODES:
        raise DsspAdapterError(
            f"Invalid DSSP secondary-structure code {code!r} for {context}; expected one of "
            f"{sorted(DSSP_SECONDARY_STRUCTURE_CODES)!r}."
        )


def _column_values(
    mmcif: Mapping[str, str | Sequence[str]],
    column_name: str,
) -> list[str]:
    try:
        values = mmcif[column_name]
    except KeyError as error:
        if column_name.startswith("_dssp_struct_summary."):
            raise DsspAdapterError(
                f"Missing required mmCIF column {column_name!r}. The annotated output "
                "lacks exact per-residue DSSP data."
            ) from error
        raise DsspAdapterError(f"Missing required mmCIF column {column_name!r}.") from error

    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _validated_columns(
    mmcif: Mapping[str, str | Sequence[str]],
    column_names: tuple[str, ...],
) -> list[list[str]]:
    columns = [_column_values(mmcif, name) for name in column_names]
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        details = ", ".join(
            f"{name}={len(column)}" for name, column in zip(column_names, columns, strict=True)
        )
        raise DsspAdapterError(f"Inconsistent mmCIF loop column lengths: {details}.")
    return columns


def _optional_validated_columns(
    mmcif: Mapping[str, str | Sequence[str]],
    column_names: tuple[str, ...],
) -> list[list[str]] | None:
    present = [name in mmcif for name in column_names]
    if not any(present):
        return None
    if not all(present):
        missing = [
            name for name, is_present in zip(column_names, present, strict=True) if not is_present
        ]
        raise DsspAdapterError(f"Incomplete DSSP mmCIF loop; missing columns {missing!r}.")
    return _validated_columns(mmcif, column_names)


def _parse_label_sequence_id(value: str, *, context: str) -> int:
    try:
        sequence_id = int(value)
    except ValueError as error:
        raise DsspAdapterError(f"{context} has non-integer label_seq_id {value!r}.") from error
    if sequence_id < 1:
        raise DsspAdapterError(f"{context} has non-positive label_seq_id {value!r}.")
    return sequence_id


def _parse_auth_sequence_id(value: str, *, label_key: tuple[str, str]) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise DsspAdapterError(
            f"DSSP residue {label_key!r} maps to non-integer `_atom_site.auth_seq_id` {value!r}."
        ) from error


def _biopython_hetero_field(group: str, component: str) -> str:
    """Reproduce the residue-id convention used by Biopython parsers."""

    if group.upper() != "HETATM":
        return " "
    if component in {"HOH", "WAT"}:
        return "W"
    return f"H_{component}"


def _atom_residue_mapping(
    mmcif: Mapping[str, str | Sequence[str]],
) -> dict[tuple[str, str], _MappedResidue]:
    atom_columns = _validated_columns(mmcif, _ATOM_MAPPING_COLUMNS)
    candidates: dict[tuple[str, str], set[_MappedResidue]] = {}
    for (
        label_chain,
        label_sequence,
        author_chain_id,
        auth_sequence,
        insertion_code,
        group,
        component,
    ) in zip(*atom_columns, strict=True):
        if label_sequence in _MMCIF_MISSING_VALUES:
            continue
        label_key = (label_chain, label_sequence)
        if label_chain in _MMCIF_MISSING_VALUES or author_chain_id in _MMCIF_MISSING_VALUES:
            raise DsspAdapterError(f"DSSP residue {label_key!r} has no exact chain mapping.")
        polymer_index = (
            _parse_label_sequence_id(
                label_sequence,
                context=f"DSSP residue {label_key!r}",
            )
            - 1
        )
        auth_residue_number = _parse_auth_sequence_id(auth_sequence, label_key=label_key)
        normalized_insertion = " " if insertion_code in _MMCIF_MISSING_VALUES else insertion_code
        author_key: DsspResidueKey = (
            author_chain_id,
            (
                _biopython_hetero_field(group, component),
                auth_residue_number,
                normalized_insertion,
            ),
        )
        candidates.setdefault(label_key, set()).add(
            _MappedResidue(author_key=author_key, polymer_index=polymer_index)
        )

    mapping: dict[tuple[str, str], _MappedResidue] = {}
    for label_key, mapped_values in candidates.items():
        if len(mapped_values) != 1:
            raise DsspAdapterError(
                f"DSSP residue {label_key!r} maps ambiguously to "
                f"{sorted(value.author_key for value in mapped_values)!r}."
            )
        mapping[label_key] = next(iter(mapped_values))
    return mapping


def _parse_summary(
    mmcif: Mapping[str, str | Sequence[str]],
    residue_mapping: Mapping[tuple[str, str], _MappedResidue],
) -> tuple[dict[DsspResidueKey, str], dict[tuple[str, str], str]]:
    summary_columns = _validated_columns(mmcif, _DSSP_SUMMARY_COLUMNS)
    assignments: dict[DsspResidueKey, str] = {}
    assignment_sources: dict[DsspResidueKey, tuple[str, str]] = {}
    label_assignments: dict[tuple[str, str], str] = {}
    for label_chain, label_sequence, secondary_structure in zip(*summary_columns, strict=True):
        label_key = (label_chain, label_sequence)
        mapped = residue_mapping.get(label_key)
        if mapped is None:
            raise DsspAdapterError(
                f"DSSP residue {label_key!r} has no matching `_atom_site` author residue."
            )
        normalized_structure = (
            "-" if secondary_structure in _MMCIF_MISSING_VALUES else secondary_structure
        )
        validate_dssp_secondary_structure_code(
            normalized_structure,
            context=f"DSSP residue {label_key!r}",
        )
        previous_label_structure = label_assignments.get(label_key)
        if previous_label_structure is not None:
            if previous_label_structure != normalized_structure:
                raise DsspAdapterError(
                    f"DSSP residue {label_key!r} has conflicting summary assignments "
                    f"{previous_label_structure!r} and {normalized_structure!r}."
                )
            raise DsspAdapterError(f"DSSP residue {label_key!r} has a duplicate summary row.")
        previous_source = assignment_sources.get(mapped.author_key)
        if previous_source is not None and previous_source != label_key:
            raise DsspAdapterError(
                f"DSSP residues {previous_source!r} and {label_key!r} both map to "
                f"author residue {mapped.author_key!r}."
            )
        label_assignments[label_key] = normalized_structure
        assignment_sources[mapped.author_key] = label_key
        assignments[mapped.author_key] = normalized_structure

    if not assignments:
        raise DsspAdapterError("DSSP annotated mmCIF contains no per-residue assignments.")
    return assignments, label_assignments


def _parse_strands(
    mmcif: Mapping[str, str | Sequence[str]],
    residue_mapping: Mapping[tuple[str, str], _MappedResidue],
    label_assignments: Mapping[tuple[str, str], str],
) -> tuple[_ParsedStrand, ...]:
    range_columns = _optional_validated_columns(mmcif, _SHEET_RANGE_COLUMNS)
    if range_columns is None:
        if DSSP_STRAND_SECONDARY_STRUCTURE_CODE in label_assignments.values():
            raise DsspAdapterError(
                "DSSP annotated mmCIF contains E residues but no `_struct_sheet_range` loop."
            )
        return ()

    parsed: list[_ParsedStrand] = []
    seen_source_ids: set[DsspSourceStrandId] = set()
    covered_e_labels: set[tuple[str, str]] = set()
    for sheet_id, range_id, begin_chain, begin_sequence, end_chain, end_sequence in zip(
        *range_columns,
        strict=True,
    ):
        source_id = (sheet_id, range_id)
        if sheet_id in _MMCIF_MISSING_VALUES or range_id in _MMCIF_MISSING_VALUES:
            raise DsspAdapterError("DSSP strand ranges require non-blank sheet and range IDs.")
        if source_id in seen_source_ids:
            raise DsspAdapterError(f"Duplicate DSSP strand range identifier {source_id!r}.")
        seen_source_ids.add(source_id)
        if begin_chain != end_chain:
            raise DsspAdapterError(
                f"DSSP strand range {source_id!r} spans label chains "
                f"{begin_chain!r} and {end_chain!r}."
            )
        begin_index = _parse_label_sequence_id(
            begin_sequence,
            context=f"DSSP strand range {source_id!r} begin",
        )
        end_index = _parse_label_sequence_id(
            end_sequence,
            context=f"DSSP strand range {source_id!r} end",
        )
        if end_index < begin_index:
            raise DsspAdapterError(f"DSSP strand range {source_id!r} has reversed endpoints.")
        begin_key = (begin_chain, begin_sequence)
        end_key = (end_chain, end_sequence)
        begin_mapping = residue_mapping.get(begin_key)
        end_mapping = residue_mapping.get(end_key)
        if begin_mapping is None or end_mapping is None:
            raise DsspAdapterError(
                f"DSSP strand range {source_id!r} has an endpoint absent from `_atom_site`."
            )
        if begin_mapping.author_key[0] != end_mapping.author_key[0]:
            raise DsspAdapterError(f"DSSP strand range {source_id!r} maps across author chains.")

        e_label_keys = tuple(
            sorted(
                (
                    label_key
                    for label_key, code in label_assignments.items()
                    if code == DSSP_STRAND_SECONDARY_STRUCTURE_CODE
                    and label_key[0] == begin_chain
                    and begin_index
                    <= _parse_label_sequence_id(label_key[1], context=f"DSSP residue {label_key!r}")
                    <= end_index
                ),
                key=lambda key: int(key[1]),
            )
        )
        # DSSP may emit standard ranges for isolated beta bridges. A physical
        # strand node requires at least one extended-strand (E) residue.
        if not e_label_keys:
            continue
        author_keys = tuple(residue_mapping[label_key].author_key for label_key in e_label_keys)
        if any(key[0] != begin_mapping.author_key[0] for key in author_keys):
            raise DsspAdapterError(
                f"DSSP strand range {source_id!r} maps to multiple author chains."
            )
        overlap = covered_e_labels.intersection(e_label_keys)
        if overlap:
            raise DsspAdapterError(
                f"DSSP E residues belong to multiple strand ranges: {sorted(overlap)!r}."
            )
        covered_e_labels.update(e_label_keys)
        parsed.append(
            _ParsedStrand(
                record=DsspStrandRecord(
                    source_id=source_id,
                    author_chain_id=begin_mapping.author_key[0],
                    residue_keys=author_keys,
                ),
                label_keys=frozenset(e_label_keys),
            )
        )

    all_e_labels = {
        label_key
        for label_key, code in label_assignments.items()
        if code == DSSP_STRAND_SECONDARY_STRUCTURE_CODE
    }
    missing_e_labels = all_e_labels.difference(covered_e_labels)
    if missing_e_labels:
        raise DsspAdapterError(
            "DSSP E residues are not covered by exactly one strand range: "
            f"{sorted(missing_e_labels)[:5]!r}."
        )
    return tuple(parsed)


def _ladder_side_source_ids(
    *,
    begin_key: tuple[str, str],
    end_key: tuple[str, str],
    parsed_strands: tuple[_ParsedStrand, ...],
    label_assignments: Mapping[tuple[str, str], str],
    residue_mapping: Mapping[tuple[str, str], _MappedResidue],
    context: str,
) -> tuple[DsspSourceStrandId, ...]:
    begin_code = label_assignments.get(begin_key)
    end_code = label_assignments.get(end_key)
    if begin_code is None or end_code is None:
        raise DsspAdapterError(f"{context} references a residue absent from the DSSP summary.")

    if begin_key[0] != end_key[0]:
        raise DsspAdapterError(f"{context} spans label chains {begin_key[0]!r} and {end_key[0]!r}.")
    begin_mapping = residue_mapping.get(begin_key)
    end_mapping = residue_mapping.get(end_key)
    if begin_mapping is None or end_mapping is None:
        raise DsspAdapterError(f"{context} has an endpoint absent from `_atom_site`.")
    if begin_mapping.author_key[0] != end_mapping.author_key[0]:
        raise DsspAdapterError(f"{context} maps across author chains.")

    begin_index = _parse_label_sequence_id(begin_key[1], context=f"{context} begin")
    end_index = _parse_label_sequence_id(end_key[1], context=f"{context} end")
    lower_index, upper_index = sorted((begin_index, end_index))
    label_chain_id = begin_key[0]
    candidates = tuple(
        sorted(
            (
                strand.record
                for strand in parsed_strands
                if any(
                    label_key[0] == label_chain_id
                    and lower_index <= int(label_key[1]) <= upper_index
                    for label_key in strand.label_keys
                )
            ),
            key=lambda record: record.source_id,
        )
    )
    if not candidates:
        # A ladder side supported only by isolated beta-bridge (B) residues is
        # not a strand node in the graph.
        if (
            begin_code != DSSP_STRAND_SECONDARY_STRUCTURE_CODE
            and end_code != DSSP_STRAND_SECONDARY_STRUCTURE_CODE
        ):
            return ()
        raise DsspAdapterError(f"{context} does not map to a DSSP E-strand.")

    expected_author_chain = begin_mapping.author_key[0]
    candidate_author_chains = {candidate.author_chain_id for candidate in candidates}
    if candidate_author_chains != {expected_author_chain}:
        raise DsspAdapterError(
            f"{context} has conflicting label-to-author chain mappings: "
            f"{sorted(candidate_author_chains | {expected_author_chain})!r}."
        )
    return tuple(candidate.source_id for candidate in candidates)


def _parse_ladders(
    mmcif: Mapping[str, str | Sequence[str]],
    parsed_strands: tuple[_ParsedStrand, ...],
    label_assignments: Mapping[tuple[str, str], str],
    residue_mapping: Mapping[tuple[str, str], _MappedResidue],
) -> tuple[_ParsedLadder, ...]:
    ladder_columns = _optional_validated_columns(mmcif, _DSSP_LADDER_COLUMNS)
    if ladder_columns is None:
        return ()
    ladders: list[_ParsedLadder] = []
    for row_number, values in enumerate(zip(*ladder_columns, strict=True), start=1):
        (
            begin_chain_1,
            begin_sequence_1,
            end_chain_1,
            end_sequence_1,
            begin_chain_2,
            begin_sequence_2,
            end_chain_2,
            end_sequence_2,
        ) = values
        first_source_ids = _ladder_side_source_ids(
            begin_key=(begin_chain_1, begin_sequence_1),
            end_key=(end_chain_1, end_sequence_1),
            parsed_strands=parsed_strands,
            label_assignments=label_assignments,
            residue_mapping=residue_mapping,
            context=f"DSSP ladder row {row_number} side 1",
        )
        second_source_ids = _ladder_side_source_ids(
            begin_key=(begin_chain_2, begin_sequence_2),
            end_key=(end_chain_2, end_sequence_2),
            parsed_strands=parsed_strands,
            label_assignments=label_assignments,
            residue_mapping=residue_mapping,
            context=f"DSSP ladder row {row_number} side 2",
        )
        ladders.append(
            _ParsedLadder(
                first_source_ids=first_source_ids,
                second_source_ids=second_source_ids,
            )
        )
    return tuple(ladders)


def _merge_ladder_side_strands(
    parsed_strands: tuple[_ParsedStrand, ...],
    ladders: tuple[_ParsedLadder, ...],
) -> tuple[tuple[DsspStrandRecord, ...], tuple[DsspStrandEdge, ...]]:
    """Merge split E ranges that one DSSP ladder side identifies as one strand."""

    parsed_by_source = {strand.record.source_id: strand for strand in parsed_strands}
    parent = {source_id: source_id for source_id in parsed_by_source}

    def find(source_id: DsspSourceStrandId) -> DsspSourceStrandId:
        try:
            root = parent[source_id]
        except KeyError as error:
            raise DsspAdapterError(
                f"DSSP ladder references unknown strand range {source_id!r}."
            ) from error
        while root != parent[root]:
            root = parent[root]
        while source_id != root:
            next_source_id = parent[source_id]
            parent[source_id] = root
            source_id = next_source_id
        return root

    def union(source_ids: tuple[DsspSourceStrandId, ...]) -> None:
        roots = {find(source_id) for source_id in source_ids}
        if not roots:
            return
        canonical_root = min(roots)
        for root in roots:
            parent[root] = canonical_root

    for ladder in ladders:
        union(ladder.first_source_ids)
        union(ladder.second_source_ids)

    grouped_sources: dict[DsspSourceStrandId, list[DsspSourceStrandId]] = {}
    for source_id in parsed_by_source:
        grouped_sources.setdefault(find(source_id), []).append(source_id)

    merged_records: list[DsspStrandRecord] = []
    merged_source_by_original: dict[DsspSourceStrandId, DsspSourceStrandId] = {}
    author_chain_by_merged_source: dict[DsspSourceStrandId, str] = {}
    for source_ids in grouped_sources.values():
        members = [parsed_by_source[source_id] for source_id in source_ids]
        author_chain_ids = {member.record.author_chain_id for member in members}
        if len(author_chain_ids) != 1:
            raise DsspAdapterError(
                "One DSSP ladder side merges strand ranges from multiple author chains: "
                f"{sorted(author_chain_ids)!r}."
            )
        members.sort(
            key=lambda member: (
                min(int(label_key[1]) for label_key in member.label_keys),
                member.record.source_id,
            )
        )
        merged_source_id = min(source_ids)
        residue_keys = tuple(
            residue_key for member in members for residue_key in member.record.residue_keys
        )
        if len(set(residue_keys)) != len(residue_keys):
            raise DsspAdapterError(
                f"Merged DSSP strand {merged_source_id!r} contains duplicate residues."
            )
        author_chain_id = next(iter(author_chain_ids))
        merged_records.append(
            DsspStrandRecord(
                source_id=merged_source_id,
                author_chain_id=author_chain_id,
                residue_keys=residue_keys,
            )
        )
        author_chain_by_merged_source[merged_source_id] = author_chain_id
        for source_id in source_ids:
            merged_source_by_original[source_id] = merged_source_id

    edges: set[DsspStrandEdge] = set()
    for row_number, ladder in enumerate(ladders, start=1):
        if not ladder.first_source_ids or not ladder.second_source_ids:
            continue
        first_sources = {
            merged_source_by_original[source_id] for source_id in ladder.first_source_ids
        }
        second_sources = {
            merged_source_by_original[source_id] for source_id in ladder.second_source_ids
        }
        if len(first_sources) != 1 or len(second_sources) != 1:
            raise DsspAdapterError(
                f"DSSP ladder row {row_number} has conflicting physical-strand mappings."
            )
        first_source_id = next(iter(first_sources))
        second_source_id = next(iter(second_sources))
        if (
            author_chain_by_merged_source[first_source_id]
            != author_chain_by_merged_source[second_source_id]
        ):
            # The detector is explicitly chain-level. Inter-chain sheet
            # contacts belong to an assembly-level scientific model.
            continue
        if first_source_id == second_source_id:
            # Bulge merging can collapse both DSSP ladder sides into one
            # physical strand. A simple strand-adjacency graph has no self-edge.
            continue
        edges.add(DsspStrandEdge(first_source_id, second_source_id))
    return tuple(sorted(merged_records)), tuple(sorted(edges))


def parse_dssp_annotated_mmcif(file_path: str | os.PathLike[str]) -> DsspAnnotation:
    """Parse one fresh DSSP annotated-mmCIF result with exact identities."""

    try:
        mmcif = MMCIF2Dict(os.fspath(file_path))
    except Exception as error:
        raise DsspAdapterError(f"Failed to parse DSSP-annotated mmCIF output: {error}") from error

    residue_mapping = _atom_residue_mapping(mmcif)
    assignments, label_assignments = _parse_summary(mmcif, residue_mapping)
    parsed_strands = _parse_strands(mmcif, residue_mapping, label_assignments)
    ladders = _parse_ladders(
        mmcif,
        parsed_strands,
        label_assignments,
        residue_mapping,
    )
    strand_records, edges = _merge_ladder_side_strands(parsed_strands, ladders)
    return DsspAnnotation(
        residue_assignments=assignments,
        strand_records=strand_records,
        strand_edges=edges,
    )


def _run_annotated_dssp(
    input_path: str | os.PathLike[str],
    *,
    dssp_executable: str,
) -> DsspAnnotation:
    file_descriptor, output_path = tempfile.mkstemp(suffix=".dssp.cif")
    os.close(file_descriptor)
    command = [
        dssp_executable,
        "--output-format=mmcif",
        os.fspath(input_path),
        output_path,
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
            raise DsspAdapterError(
                f"mkdssp annotated mmCIF output exited with status {completed.returncode}: {detail}"
            )
        output_file = Path(output_path)
        if not output_file.is_file() or output_file.stat().st_size == 0:
            detail = completed.stderr.strip() or "no diagnostic output"
            raise DsspAdapterError(f"mkdssp produced no annotated mmCIF output: {detail}")
        return parse_dssp_annotated_mmcif(output_file)
    except OSError as error:
        raise DsspAdapterError(f"Could not execute mkdssp: {error}") from error
    finally:
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass


def run_dssp_annotation(
    input_path: str | os.PathLike[str],
    *,
    dssp_executable: str,
) -> DsspAnnotation:
    """Run supported DSSP once and return its strict strand annotation."""

    return _run_annotated_dssp(
        input_path,
        dssp_executable=dssp_executable,
    )
