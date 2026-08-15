from __future__ import annotations

import gzip
import os
import re
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from Bio import BiopythonWarning
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa

from .config import InputConfig
from .constants import POLYMER_POSITION_POLICY
from .dssp_adapter import (
    DsspAdapterError,
    DsspAnnotation,
    DsspResidueKey,
    DsspStrandRecord,
    run_dssp_annotation,
    validate_dssp_coverage,
    validate_dssp_secondary_structure_code,
)
from .exceptions import ChainNotFoundError, DsspError, InputValidationError, StructureParseError
from .polymer_sequence import declared_polymer_sequence_for_author_chain
from .runtime import require_dssp_binary
from .strand_graph import StrandAdjacencyGraph, StrandEdge, StrandNode, StrandRange

_DSSP_DERIVED_MMCIF_PREFIXES = (
    "_dssp_",
    "_struct_conf.",
    "_struct_conf_type.",
    "_struct_sheet.",
    "_struct_sheet_range.",
    "_struct_sheet_order.",
    "_struct_sheet_hbond.",
    "_pdbx_struct_sheet_hbond.",
)
_DSSP_INPUT_MMCIF_PREFIXES = (
    "_atom_site.",
    "_cell.",
    "_chem_comp.",
    "_entity.",
    "_entity_poly.",
    "_entity_poly_seq.",
    "_entry.",
    "_pdbx_poly_seq_scheme.",
    "_space_group.",
    "_struct_asym.",
    "_symmetry.",
)
_MMCIF_MISSING_VALUES = frozenset({"", ".", "?"})
_MMCIF_ATOM_SITE_ONLY_FORBIDDEN_PREFIXES = (
    "_entity_poly_seq.",
    "_struct_asym.",
    "_pdbx_poly_seq_scheme.",
)


@dataclass(frozen=True)
class _MmcifPolymerMapping:
    positions: dict[DsspResidueKey, int]
    components: dict[DsspResidueKey, str]
    atom_site_only: bool


def _mmcif_values(
    mmcif: dict[str, str | list[str]],
    column: str,
) -> list[str]:
    raw_values = mmcif.get(column, [])
    if isinstance(raw_values, str):
        return [raw_values]
    return [str(value) for value in raw_values]


def _mmcif_uses_atom_site_only_polymer_identity(
    mmcif: dict[str, str | list[str]],
) -> bool:
    """Classify an mmCIF identity contract before DSSP is invoked."""

    entity_ids = _mmcif_values(mmcif, "_entity_poly.entity_id")
    polymer_types = _mmcif_values(mmcif, "_entity_poly.type")
    if bool(entity_ids) != bool(polymer_types):
        raise StructureParseError(
            "mmCIF polymer declarations require both `_entity_poly.entity_id` and "
            "`_entity_poly.type`."
        )
    if entity_ids:
        return False

    conflicting_categories = sorted(
        str(key) for key in mmcif if str(key).startswith(_MMCIF_ATOM_SITE_ONLY_FORBIDDEN_PREFIXES)
    )
    if conflicting_categories:
        raise StructureParseError(
            "mmCIF input has partial polymer metadata without an `_entity_poly` declaration: "
            f"{conflicting_categories[:5]!r}."
        )
    return True


# -------------------------
# Utilities: element / chain
# -------------------------
_TWO_LETTER_ELEMENTS = {
    "CL",
    "BR",
    "NA",
    "MG",
    "ZN",
    "FE",
    "CA",
    "CU",
    "NI",
    "CO",
    "MN",
    "SE",
    "SI",
    "AL",
    "CD",
    "HG",
    "PB",
    "SR",
    "CS",
    "LI",
    "AG",
    "AU",
    "PT",
    "IR",
    "KR",
    "XE",
    "AR",
    "NE",
    "HE",
}


def _infer_element_from_atom_name(atom_name: str) -> str:
    """
    Infer an element symbol from a PDB atom name.

    Returns Biopython-style capitalization such as ``C``, ``N``, ``O``, ``Cl``,
    or ``Zn``.
    """
    if not atom_name:
        return ""
    raw = str(atom_name)
    if len(raw) == 4 and raw[0].isdigit() and raw[1].isalpha():
        return raw[1].upper()
    if len(raw) == 4 and raw[0] == " ":
        # PDB right-justifies one-letter elements.  In particular, protein
        # ``" CA "`` is alpha carbon, whereas left-justified ``"CA  "`` is
        # calcium.  Preserve that distinction before stripping whitespace.
        for character in raw[1:]:
            if character.isalpha():
                return character.upper()

    s = raw.strip()
    if not s:
        return ""

    # Strip a leading digit from names such as "1HG1".
    if s[0].isdigit() and len(s) >= 2:
        s = s[1:]

    # Keep alphabetic characters only.
    s = re.sub(r"[^A-Za-z]", "", s)
    if not s:
        return ""
    s = s.upper()

    if len(s) >= 2 and s[:2] in _TWO_LETTER_ELEMENTS:
        return s[0] + s[1].lower()
    return s[0]


def _fill_missing_atom_elements(model: Any) -> int:
    """Fill empty or placeholder ``atom.element`` values and return the count."""
    fixed = 0
    for atom in model.get_atoms():
        elem = (getattr(atom, "element", "") or "").strip()
        if elem and elem != "X":
            continue
        inf = _infer_element_from_atom_name(atom.get_fullname())
        if inf:
            atom.element = inf
            fixed += 1
    return fixed


def _atom_occupancy(atom: Any) -> float:
    occupancy = atom.get_occupancy()
    return float(occupancy) if occupancy is not None else float("-inf")


def _select_atom(residue: Any, atom_name: str, policy: str) -> Any | None:
    """Select one atom deterministically from an optional alternate-location group."""
    if atom_name not in residue:
        return None
    atom = residue[atom_name]
    children = getattr(atom, "child_dict", None)
    if not children:
        return atom
    if policy == "error":
        raise StructureParseError(
            f"Disordered atom {atom_name!r} encountered at residue {residue.id!r}."
        )
    if policy == "biopython_selected":
        return getattr(atom, "selected_child", atom)
    candidates = list(children.values())
    return min(
        candidates,
        key=lambda candidate: (
            -_atom_occupancy(candidate),
            str(candidate.get_altloc()).strip(),
        ),
    )


def _select_residue_variant(residue: Any, config: InputConfig) -> Any:
    """Select one point-mutation/disordered residue variant deterministically."""
    children = getattr(residue, "child_dict", None)
    if not children or not all(hasattr(child, "resname") for child in children.values()):
        return residue
    policy = config.disordered_residue_policy
    if policy == "error":
        raise StructureParseError(f"Disordered residue encountered at {residue.id!r}.")
    if policy == "biopython_selected":
        return getattr(residue, "selected_child", residue)

    def ca_occupancy(candidate: Any) -> tuple[float, str]:
        atom = _select_atom(candidate, "CA", config.atom_altloc_policy)
        occupancy = _atom_occupancy(atom) if atom is not None else float("-inf")
        return -occupancy, str(candidate.get_resname())

    return min(children.values(), key=ca_occupancy)


def _require_public_chain_ids(model: Any, *, source_path: str) -> None:
    """Reject chain identities that the public result schema cannot represent."""

    blank_count = sum(not str(chain.id).strip() for chain in model.get_chains())
    if blank_count:
        raise StructureParseError(
            f"Structure {source_path} contains {blank_count} blank chain ID(s). "
            "Blank chains cannot be represented without colliding with the file-level "
            "preparation-error identity; the loader will not silently rename them."
        )


def _decompress_gzip_to_temp_if_needed(in_path: str) -> str | None:
    with open(in_path, "rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            return None

    lower_name = os.path.basename(in_path).lower()
    suffix = next(
        (
            coordinate_suffix
            for coordinate_suffix in (".mmcif", ".cif", ".pdb")
            if lower_name.endswith(f"{coordinate_suffix}.gz")
            or lower_name.endswith(coordinate_suffix)
        ),
        ".pdb",
    )
    fd, out_path = tempfile.mkstemp(suffix=suffix)
    with gzip.open(in_path, "rb") as source, os.fdopen(fd, "wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
    return out_path


def _mmcif_author_residue_key(
    *,
    group: str,
    component: str,
    author_chain_id: str,
    auth_sequence: str,
    insertion_code: str,
) -> DsspResidueKey:
    if group not in {"ATOM", "HETATM"}:
        raise StructureParseError(f"Invalid polymer `_atom_site.group_PDB` {group!r}.")
    if author_chain_id in _MMCIF_MISSING_VALUES:
        raise StructureParseError("mmCIF polymer residues require an author chain ID.")
    if component in _MMCIF_MISSING_VALUES:
        raise StructureParseError("mmCIF polymer residues require `label_comp_id`.")
    try:
        residue_number = int(auth_sequence)
    except ValueError as error:
        raise StructureParseError(
            f"Invalid polypeptide `_atom_site.auth_seq_id` {auth_sequence!r}."
        ) from error
    normalized_insertion = " " if insertion_code in _MMCIF_MISSING_VALUES else insertion_code
    if group == "ATOM":
        hetfield = " "
    elif component.upper() in {"HOH", "WAT"}:
        hetfield = "W"
    else:
        hetfield = f"H_{component}"
    return author_chain_id, (hetfield, residue_number, normalized_insertion)


def _declared_mmcif_polypeptide_mapping(
    mmcif: dict[str, str | list[str]],
    *,
    model_id: int,
    selected_model: Any | None = None,
    input_config: InputConfig | None = None,
) -> _MmcifPolymerMapping:
    entity_ids = _mmcif_values(mmcif, "_entity_poly.entity_id")
    polymer_types = _mmcif_values(mmcif, "_entity_poly.type")
    if len(entity_ids) != len(polymer_types):
        raise StructureParseError("Inconsistent `_entity_poly` column lengths in mmCIF input.")
    polypeptide_entities = {
        entity_id
        for entity_id, polymer_type in zip(entity_ids, polymer_types, strict=True)
        if polymer_type.lower().startswith("polypeptide")
    }
    if not polypeptide_entities:
        return _MmcifPolymerMapping({}, {}, False)

    atom_column_names = (
        "_atom_site.pdbx_PDB_model_num",
        "_atom_site.group_PDB",
        "_atom_site.label_entity_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
    )
    columns = [_mmcif_values(mmcif, name) for name in atom_column_names]
    if len({len(column) for column in columns}) != 1:
        raise StructureParseError("Inconsistent `_atom_site` column lengths in mmCIF input.")
    if not columns[0]:
        raise StructureParseError(
            "mmCIF polymer positions require complete non-empty `_atom_site` mapping columns."
        )

    model_numbers: list[int] = []
    for raw_model_number in columns[0]:
        try:
            model_number = int(raw_model_number)
        except ValueError as error:
            raise StructureParseError(
                f"Invalid `_atom_site.pdbx_PDB_model_num` {raw_model_number!r}."
            ) from error
        if model_number < 1:
            raise StructureParseError("mmCIF model numbers must be positive integers.")
        if model_number not in model_numbers:
            model_numbers.append(model_number)
    if model_id < 0 or model_id >= len(model_numbers):
        raise StructureParseError(
            f"mmCIF model index {model_id} has no matching `_atom_site` model mapping."
        )
    selected_model_number = model_numbers[model_id]
    selected_variants = (
        _selected_mmcif_residue_variants(selected_model, input_config)
        if selected_model is not None and input_config is not None
        else {}
    )

    positions: dict[DsspResidueKey, int] = {}
    components: dict[DsspResidueKey, str] = {}
    author_key_by_label_position: dict[tuple[str, str, int], DsspResidueKey] = {}
    label_chain_by_author_chain: dict[str, str] = {}
    for (
        raw_model_number,
        group,
        entity_id,
        label_chain,
        label_sequence,
        label_comp_id,
        author_chain_id,
        auth_sequence,
        insertion_code,
    ) in zip(*columns, strict=True):
        if int(raw_model_number) != selected_model_number:
            continue
        if entity_id not in polypeptide_entities or label_sequence in {"", ".", "?"}:
            continue
        if label_chain in _MMCIF_MISSING_VALUES:
            raise StructureParseError("mmCIF polymer residues require non-blank chain mappings.")
        key = _mmcif_author_residue_key(
            group=group,
            component=label_comp_id,
            author_chain_id=author_chain_id,
            auth_sequence=auth_sequence,
            insertion_code=insertion_code,
        )
        selected_variant = selected_variants.get((key[0], key[1][1], key[1][2]))
        if selected_variant is not None and selected_variant != (key, label_comp_id):
            continue
        try:
            polymer_position = int(label_sequence) - 1
        except ValueError as error:
            raise StructureParseError(
                f"Invalid polypeptide `_atom_site.label_seq_id` {label_sequence!r}."
            ) from error
        if polymer_position < 0:
            raise StructureParseError("Polypeptide label sequence positions must be positive.")
        previous = positions.get(key)
        if previous is not None and previous != polymer_position:
            raise StructureParseError(
                f"Author residue {key!r} maps to conflicting polymer positions."
            )
        label_position_key = (entity_id, label_chain, polymer_position)
        previous_author_key = author_key_by_label_position.get(label_position_key)
        if previous_author_key is not None and previous_author_key != key:
            raise StructureParseError(
                f"Declared polymer position {label_position_key!r} maps to multiple author "
                "residues."
            )
        previous_label_chain = label_chain_by_author_chain.get(author_chain_id)
        if previous_label_chain is not None and previous_label_chain != label_chain:
            raise StructureParseError(
                f"Author chain {author_chain_id!r} maps from multiple label chains in one model."
            )
        positions[key] = polymer_position
        previous_component = components.get(key)
        if previous_component is not None and previous_component != label_comp_id:
            raise StructureParseError(
                f"Author residue {key!r} maps to conflicting components "
                f"{previous_component!r} and {label_comp_id!r}."
            )
        components[key] = label_comp_id
        author_key_by_label_position[label_position_key] = key
        label_chain_by_author_chain[author_chain_id] = label_chain
    return _MmcifPolymerMapping(positions, components, False)


def _select_mmcif_residue_variant(
    variants: list[Any],
    *,
    author_chain_id: str,
    author_position: tuple[int, str],
    input_config: InputConfig,
) -> Any:
    if len(variants) == 1:
        return variants[0]
    backbone_variants = [
        variant
        for variant in variants
        if all(
            _select_atom(variant, atom_name, input_config.atom_altloc_policy) is not None
            for atom_name in ("N", "CA", "C", "O")
        )
    ]
    context = (author_chain_id, author_position)
    if len(backbone_variants) == 1:
        return backbone_variants[0]
    if not backbone_variants:
        raise StructureParseError(
            f"mmCIF author position {context!r} has multiple amino-acid "
            "representations and none has a canonical N/CA/C/O backbone."
        )
    variant_ca_atoms = [
        _select_atom(variant, "CA", input_config.atom_altloc_policy)
        for variant in backbone_variants
    ]
    variant_altlocs = [
        str(atom.get_altloc()).strip() if atom is not None else "" for atom in variant_ca_atoms
    ]
    if any(not altloc for altloc in variant_altlocs) or len(set(variant_altlocs)) != len(
        variant_altlocs
    ):
        raise StructureParseError(
            f"mmCIF author position {context!r} contains multiple amino-acid "
            "residues without unique alternate-location identities."
        )
    if input_config.disordered_residue_policy == "error":
        raise StructureParseError(f"Disordered mmCIF author position encountered at {context!r}.")
    if input_config.disordered_residue_policy == "biopython_selected":
        raise StructureParseError(
            f"Biopython did not expose a selected child for mmCIF author position {context!r}."
        )
    return min(
        backbone_variants,
        key=lambda variant: (
            -_atom_occupancy(_select_atom(variant, "CA", input_config.atom_altloc_policy)),
            str(variant.get_resname()),
        ),
    )


def _selected_mmcif_residue_variants(
    model: Any,
    input_config: InputConfig,
) -> dict[tuple[str, int, str], tuple[DsspResidueKey, str]]:
    selected: dict[tuple[str, int, str], tuple[DsspResidueKey, str]] = {}
    for chain in model.get_chains():
        groups: dict[tuple[int, str], list[Any]] = {}
        for raw_residue in chain:
            residue = _select_residue_variant(raw_residue, input_config)
            position = (int(residue.id[1]), str(residue.id[2]))
            groups.setdefault(position, []).append(residue)
        for position, variants in groups.items():
            residue = _select_mmcif_residue_variant(
                variants,
                author_chain_id=str(chain.id),
                author_position=position,
                input_config=input_config,
            )
            residue_id = residue.id
            key: DsspResidueKey = (
                str(chain.id),
                (str(residue_id[0]), int(residue_id[1]), str(residue_id[2])),
            )
            selected[(str(chain.id), position[0], position[1])] = (
                key,
                str(residue.get_resname()).strip(),
            )
    return selected


def _atom_site_only_mmcif_polypeptide_mapping(
    model: Any,
    input_config: InputConfig,
) -> _MmcifPolymerMapping:
    """Infer one coordinate-only protein chain from explicit residue chemistry."""

    chains = list(model.get_chains())
    if len(chains) != 1:
        raise StructureParseError(
            "Atom-site-only mmCIF input must contain exactly one author chain."
        )
    chain = chains[0]
    candidate_groups: dict[tuple[int, str], list[Any]] = {}
    ordered_author_positions: list[tuple[int, str]] = []
    atom_seed_author_positions: set[tuple[int, str]] = set()
    for raw_residue in chain:
        residue = _select_residue_variant(raw_residue, input_config)
        hetfield = str(residue.id[0])
        amino_acid = bool(is_aa(residue, standard=False))
        if hetfield == " " and not amino_acid:
            raise StructureParseError(
                "Atom-site-only mmCIF contains an ATOM residue that is not a recognized "
                f"amino acid: {residue.id!r}."
            )
        if not amino_acid:
            continue
        author_position = (int(residue.id[1]), str(residue.id[2]))
        if author_position not in candidate_groups:
            ordered_author_positions.append(author_position)
        candidate_groups.setdefault(author_position, []).append(residue)
        if hetfield == " ":
            atom_seed_author_positions.add(author_position)

    candidates: list[Any] = []
    atom_seed_indices: set[int] = set()
    for author_position in ordered_author_positions:
        selected = _select_mmcif_residue_variant(
            candidate_groups[author_position],
            author_chain_id=str(chain.id),
            author_position=author_position,
            input_config=input_config,
        )
        candidate_index = len(candidates)
        candidates.append(selected)
        if author_position in atom_seed_author_positions and str(selected.id[0]) == " ":
            atom_seed_indices.add(candidate_index)

    if not atom_seed_indices:
        raise StructureParseError(
            "Atom-site-only mmCIF contains no ATOM amino-acid residues from which to identify "
            "a protein polymer."
        )

    linked_neighbors: dict[int, set[int]] = {index: set() for index in range(len(candidates))}
    for previous_index, current_index in zip(
        range(len(candidates) - 1), range(1, len(candidates)), strict=True
    ):
        previous = candidates[previous_index]
        current = candidates[current_index]
        previous_c = _select_atom(previous, "C", input_config.atom_altloc_policy)
        current_n = _select_atom(current, "N", input_config.atom_altloc_policy)
        if previous_c is None or current_n is None:
            continue
        previous_coordinate = np.asarray(previous_c.get_coord(), dtype=float)
        current_coordinate = np.asarray(current_n.get_coord(), dtype=float)
        if (
            previous_coordinate.shape != (3,)
            or current_coordinate.shape != (3,)
            or not np.all(np.isfinite(previous_coordinate))
            or not np.all(np.isfinite(current_coordinate))
        ):
            continue
        distance = float(np.linalg.norm(current_coordinate - previous_coordinate))
        if distance <= input_config.atom_site_only_max_peptide_bond_distance_angstrom:
            linked_neighbors[previous_index].add(current_index)
            linked_neighbors[current_index].add(previous_index)

    included_indices = set(atom_seed_indices)
    pending = list(sorted(atom_seed_indices))
    while pending:
        current_index = pending.pop()
        for neighbor_index in linked_neighbors[current_index]:
            if neighbor_index in included_indices:
                continue
            included_indices.add(neighbor_index)
            pending.append(neighbor_index)

    positions: dict[DsspResidueKey, int] = {}
    components: dict[DsspResidueKey, str] = {}
    included_residues = [
        residue for index, residue in enumerate(candidates) if index in included_indices
    ]
    for polymer_position, residue in enumerate(included_residues):
        residue_id = residue.id
        key: DsspResidueKey = (
            str(chain.id),
            (str(residue_id[0]), int(residue_id[1]), str(residue_id[2])),
        )
        if key in positions:
            raise StructureParseError(
                f"Atom-site-only mmCIF maps multiple selected residues to {key!r}."
            )
        positions[key] = polymer_position
        components[key] = str(residue.get_resname()).strip()
    return _MmcifPolymerMapping(positions, components, True)


def _mmcif_polypeptide_mapping(
    file_path: str | os.PathLike[str],
    *,
    model_id: int,
    selected_model: Any | None = None,
    input_config: InputConfig | None = None,
) -> _MmcifPolymerMapping:
    mmcif = MMCIF2Dict(os.fspath(file_path))
    if not _mmcif_uses_atom_site_only_polymer_identity(mmcif):
        return _declared_mmcif_polypeptide_mapping(
            mmcif,
            model_id=model_id,
            selected_model=selected_model,
            input_config=input_config,
        )
    if selected_model is None or input_config is None:
        raise StructureParseError(
            "Atom-site-only mmCIF polymer identity requires the parsed selected model and "
            "input configuration."
        )
    return _atom_site_only_mmcif_polypeptide_mapping(selected_model, input_config)


def _mmcif_polypeptide_residue_positions(
    file_path: str | os.PathLike[str],
    *,
    model_id: int = 0,
) -> dict[DsspResidueKey, int]:
    """Map declared mmCIF polypeptide residues to polymer positions."""

    return _mmcif_polypeptide_mapping(file_path, model_id=model_id).positions


def _mmcif_polypeptide_residue_keys(
    file_path: str | os.PathLike[str],
    *,
    model_id: int = 0,
) -> set[tuple[str, tuple[str, int, str]]]:
    """Return author residue keys declared as polypeptide polymer residues."""

    return set(_mmcif_polypeptide_residue_positions(file_path, model_id=model_id))


def _retain_selected_polymer_atom_rows(
    mmcif: dict[str, str | list[str]],
    mapping: _MmcifPolymerMapping,
) -> dict[str, str | list[str]]:
    identity_columns = (
        "_atom_site.group_PDB",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
    )
    columns = [_mmcif_values(mmcif, column) for column in identity_columns]
    if not columns[0] or len({len(column) for column in columns}) != 1:
        raise StructureParseError("mmCIF requires complete atom identity columns for DSSP.")
    retained: list[int] = []
    covered: set[DsspResidueKey] = set()
    for index, (group, component, author_chain, sequence, insertion) in enumerate(
        zip(*columns, strict=True)
    ):
        try:
            key = _mmcif_author_residue_key(
                group=group,
                component=component,
                author_chain_id=author_chain,
                auth_sequence=sequence,
                insertion_code=insertion,
            )
        except StructureParseError:
            continue
        if mapping.components.get(key) != component:
            continue
        retained.append(index)
        covered.add(key)
    missing = set(mapping.positions).difference(covered)
    if missing:
        raise StructureParseError(
            f"Selected mmCIF polymer residues are missing atom rows: {sorted(missing)[:5]!r}."
        )
    filtered = dict(mmcif)
    row_count = len(columns[0])
    for atom_column, raw_values in tuple(filtered.items()):
        if not str(atom_column).startswith("_atom_site."):
            continue
        values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
        if len(values) != row_count:
            raise StructureParseError(
                f"mmCIF atom-site column {atom_column!r} has an inconsistent row count."
            )
        filtered[atom_column] = [values[index] for index in retained]
    return filtered


def _canonical_atom_site_only_dssp_input(
    mmcif: dict[str, str | list[str]],
    mapping: _MmcifPolymerMapping,
) -> dict[str, str | list[str]]:
    """Build a standard single-polymer mmCIF from one strict coordinate-only chain."""

    if not mapping.atom_site_only:
        raise TypeError("Atom-site-only DSSP materialization requires its matching mapping.")
    if not mapping.positions:
        raise StructureParseError("Atom-site-only mmCIF contains no mapped polymer residues.")
    expected_positions = set(range(len(mapping.positions)))
    if set(mapping.positions.values()) != expected_positions:
        raise StructureParseError(
            "Atom-site-only mmCIF polymer positions must be unique and contiguous."
        )

    identity_columns = (
        "_atom_site.group_PDB",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
    )
    columns = [_mmcif_values(mmcif, column) for column in identity_columns]
    if not columns[0] or len({len(column) for column in columns}) != 1:
        raise StructureParseError(
            "Atom-site-only mmCIF requires complete, equal-length atom identity columns."
        )
    if any(value not in _MMCIF_MISSING_VALUES for value in columns[2]):
        raise StructureParseError(
            "Atom-site-only mmCIF requires missing `_atom_site.label_entity_id` values; "
            "partially declared entities are ambiguous."
        )

    retained_row_indices: list[int] = []
    row_positions: list[int] = []
    original_label_chains_by_position: dict[int, set[str]] = {}
    covered_keys: set[DsspResidueKey] = set()
    for row_index, values in enumerate(zip(*columns, strict=True)):
        (
            group,
            label_chain,
            _label_entity,
            _label_sequence,
            component,
            author_chain,
            auth_sequence,
            insertion_code,
        ) = values
        try:
            author_key = _mmcif_author_residue_key(
                group=group,
                component=component,
                author_chain_id=author_chain,
                auth_sequence=auth_sequence,
                insertion_code=insertion_code,
            )
        except StructureParseError:
            continue
        polymer_position = mapping.positions.get(author_key)
        if polymer_position is None:
            continue
        expected_component = mapping.components[author_key]
        if component != expected_component:
            continue
        if label_chain in _MMCIF_MISSING_VALUES:
            raise StructureParseError(
                f"Mapped atom-site-only polymer residue {author_key!r} has no label chain."
            )
        retained_row_indices.append(row_index)
        row_positions.append(polymer_position)
        original_label_chains_by_position.setdefault(polymer_position, set()).add(label_chain)
        covered_keys.add(author_key)

    missing_keys = set(mapping.positions).difference(covered_keys)
    if missing_keys:
        raise StructureParseError(
            "Atom-site-only mmCIF rows do not exactly cover selected polymer residues: "
            f"{sorted(missing_keys)[:5]!r}."
        )
    if any(len(chains) != 1 for chains in original_label_chains_by_position.values()):
        raise StructureParseError(
            "One atom-site-only polymer residue maps to multiple label chain IDs."
        )

    first_position = min(expected_positions)
    canonical_label_chain = next(iter(original_label_chains_by_position[first_position]))
    atom_row_count = len(columns[0])
    materialized: dict[str, str | list[str]] = {
        key: raw_values
        for key, raw_values in mmcif.items()
        if not str(key).startswith(
            (
                "_entity.",
                "_entity_poly.",
                "_entity_poly_seq.",
                "_struct_asym.",
                "_pdbx_poly_seq_scheme.",
            )
        )
    }
    for key, raw_values in tuple(materialized.items()):
        if not str(key).startswith("_atom_site."):
            continue
        atom_column_values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
        if len(atom_column_values) != atom_row_count:
            raise StructureParseError(
                f"mmCIF atom-site column {key!r} has an inconsistent row count."
            )
        materialized[key] = [atom_column_values[index] for index in retained_row_indices]

    retained_count = len(retained_row_indices)
    materialized["_atom_site.label_asym_id"] = [canonical_label_chain] * retained_count
    materialized["_atom_site.label_entity_id"] = ["1"] * retained_count
    materialized["_atom_site.label_seq_id"] = [str(position + 1) for position in row_positions]

    keys_by_position = sorted(mapping.positions, key=mapping.positions.__getitem__)
    components = [mapping.components[key] for key in keys_by_position]
    author_chain_ids = {key[0] for key in keys_by_position}
    if len(author_chain_ids) != 1:
        raise StructureParseError(
            "Atom-site-only mmCIF polymer mapping spans multiple author chains."
        )
    author_chain_id = next(iter(author_chain_ids))
    sequence_ids = [str(position + 1) for position in range(len(keys_by_position))]
    author_sequence_ids = [str(key[1][1]) for key in keys_by_position]
    insertion_codes = ["?" if key[1][2] == " " else key[1][2] for key in keys_by_position]
    contains_modified_monomer = any(key[1][0] != " " for key in keys_by_position)

    materialized["_entity.id"] = ["1"]
    materialized["_entity.type"] = ["polymer"]
    materialized["_entity_poly.entity_id"] = ["1"]
    materialized["_entity_poly.type"] = ["polypeptide(L)"]
    materialized["_entity_poly.nstd_linkage"] = ["no"]
    materialized["_entity_poly.nstd_monomer"] = ["yes" if contains_modified_monomer else "no"]
    materialized["_entity_poly_seq.entity_id"] = ["1"] * len(keys_by_position)
    materialized["_entity_poly_seq.num"] = sequence_ids
    materialized["_entity_poly_seq.mon_id"] = components
    materialized["_entity_poly_seq.hetero"] = ["n"] * len(keys_by_position)
    materialized["_struct_asym.id"] = [canonical_label_chain]
    materialized["_struct_asym.entity_id"] = ["1"]
    materialized["_pdbx_poly_seq_scheme.asym_id"] = [canonical_label_chain] * len(keys_by_position)
    materialized["_pdbx_poly_seq_scheme.entity_id"] = ["1"] * len(keys_by_position)
    materialized["_pdbx_poly_seq_scheme.seq_id"] = sequence_ids
    materialized["_pdbx_poly_seq_scheme.mon_id"] = components
    materialized["_pdbx_poly_seq_scheme.hetero"] = ["n"] * len(keys_by_position)
    materialized["_pdbx_poly_seq_scheme.pdb_seq_num"] = author_sequence_ids
    materialized["_pdbx_poly_seq_scheme.auth_seq_num"] = author_sequence_ids
    materialized["_pdbx_poly_seq_scheme.pdb_mon_id"] = components
    materialized["_pdbx_poly_seq_scheme.auth_mon_id"] = components
    materialized["_pdbx_poly_seq_scheme.pdb_strand_id"] = [author_chain_id] * len(keys_by_position)
    materialized["_pdbx_poly_seq_scheme.pdb_ins_code"] = insertion_codes
    return materialized


@contextmanager
def _selected_model_mmcif_path(
    file_path: str | os.PathLike[str],
    *,
    model_id: int,
    polymer_mapping: _MmcifPolymerMapping,
) -> Iterator[str]:
    """Yield a clean mmCIF containing exactly one model renumbered as model 1.

    DSSP annotation categories are derived data. They are removed before every
    DSSP invocation so annotations shipped with an input cannot be mixed with
    the result produced for the selected model.
    """

    mmcif = MMCIF2Dict(os.fspath(file_path))
    atom_site_only = _mmcif_uses_atom_site_only_polymer_identity(mmcif)
    if atom_site_only and not polymer_mapping.atom_site_only:
        raise StructureParseError(
            "Atom-site-only mmCIF DSSP materialization requires its validated polymer mapping."
        )
    if not atom_site_only and polymer_mapping.atom_site_only:
        raise StructureParseError(
            "Declared mmCIF polymer metadata conflicts with an atom-site-only mapping."
        )
    raw_model_numbers = mmcif.get("_atom_site.pdbx_PDB_model_num", [])
    model_values = (
        [str(raw_model_numbers)]
        if isinstance(raw_model_numbers, str)
        else [str(value) for value in raw_model_numbers]
    )
    if not model_values:
        raise StructureParseError("mmCIF DSSP input requires `_atom_site.pdbx_PDB_model_num`.")
    ordered_models: list[str] = []
    for value in model_values:
        try:
            numeric = int(value)
        except ValueError as error:
            raise StructureParseError(f"Invalid mmCIF model number {value!r}.") from error
        if numeric < 1:
            raise StructureParseError("mmCIF model numbers must be positive integers.")
        normalized = str(numeric)
        if normalized not in ordered_models:
            ordered_models.append(normalized)
    if model_id < 0 or model_id >= len(ordered_models):
        raise StructureParseError(f"mmCIF model index {model_id} is not present.")
    selected_model = ordered_models[model_id]
    keep = [str(int(value)) == selected_model for value in model_values]
    filtered = {
        key: raw_values
        for key, raw_values in mmcif.items()
        if key == "data_" or str(key).startswith(_DSSP_INPUT_MMCIF_PREFIXES)
    }
    for key, raw_values in mmcif.items():
        if not str(key).startswith("_atom_site."):
            continue
        values = [raw_values] if isinstance(raw_values, str) else list(raw_values)
        if len(values) != len(keep):
            raise StructureParseError(
                f"mmCIF atom-site column {key!r} has an inconsistent row count."
            )
        retained_values = [value for value, selected in zip(values, keep, strict=True) if selected]
        if key == "_atom_site.pdbx_PDB_model_num":
            retained_values = ["1"] * len(retained_values)
        filtered[key] = retained_values
    filtered = _retain_selected_polymer_atom_rows(filtered, polymer_mapping)
    if atom_site_only:
        filtered = _canonical_atom_site_only_dssp_input(filtered, polymer_mapping)

    descriptor, temporary_name = tempfile.mkstemp(suffix=".cif")
    os.close(descriptor)
    try:
        writer = MMCIFIO()
        writer.set_dict(filtered)
        writer.save(temporary_name)
        yield temporary_name
    except StructureParseError:
        raise
    except Exception as error:
        raise StructureParseError(
            f"Could not materialize selected mmCIF model {model_id}: {error}"
        ) from error
    finally:
        try:
            os.remove(temporary_name)
        except FileNotFoundError:
            pass


def _pdb_observed_residue_letter(residue: Any) -> str:
    residue_name = str(residue.get_resname()).strip().upper()
    if residue_name in {"UNK", "XAA"}:
        return "X"
    letter = protein_letters_3to1_extended.get(residue_name)
    if letter is None:
        raise StructureParseError(
            f"PDB residue {residue.id!r} has no unambiguous one-letter sequence mapping."
        )
    return str(letter).upper()


def _unique_subsequence_positions(observed: str, declared: str) -> tuple[int, ...]:
    """Map a deletion-only observed sequence into a declaration, rejecting ambiguity."""

    leftmost: list[int] = []
    cursor = 0
    for letter in observed:
        position = declared.find(letter, cursor)
        if position < 0:
            raise StructureParseError(
                "Observed PDB polymer sequence is not a subsequence of its SEQRES declaration."
            )
        leftmost.append(position)
        cursor = position + 1

    rightmost_reversed: list[int] = []
    cursor = len(declared)
    for letter in reversed(observed):
        position = declared.rfind(letter, 0, cursor)
        if position < 0:
            raise StructureParseError(
                "Observed PDB polymer sequence is not a subsequence of its SEQRES declaration."
            )
        rightmost_reversed.append(position)
        cursor = position
    rightmost = list(reversed(rightmost_reversed))
    if leftmost != rightmost:
        raise StructureParseError(
            "Observed PDB residues have more than one valid mapping to the SEQRES declaration; "
            "polymer positions are ambiguous. Use an mmCIF input with label_seq_id mapping."
        )
    return tuple(leftmost)


# -------------------------
# DSSP: export protein only
# -------------------------
class _ProteinOnlySelect(Select):
    """Export only the amino-acid definition selected by the input config."""

    def __init__(self, *, include_nonstandard_amino_acids: bool):
        self.include_nonstandard_amino_acids = bool(include_nonstandard_amino_acids)

    def accept_residue(self, residue: Any) -> int:
        standard_only = not self.include_nonstandard_amino_acids
        return 1 if is_aa(residue, standard=standard_only) else 0

    def accept_atom(self, atom: Any) -> int:
        return 1


@dataclass(frozen=True)
class ChainPreparationResult:
    """Complete data or one explicit preparation error for an author chain."""

    author_chain_id: str
    residues: tuple[dict[str, object], ...]
    strand_graph: StrandAdjacencyGraph
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.author_chain_id, str) or not self.author_chain_id.strip():
            raise ValueError("`author_chain_id` must be a non-empty string.")
        if self.strand_graph.author_chain_id != self.author_chain_id:
            raise ValueError("Chain preparation result and strand graph IDs must match.")
        if (self.error_code is None) != (self.error_message is None):
            raise ValueError("Chain preparation error code and message must be set together.")

    @property
    def failed(self) -> bool:
        """Whether this chain could not be prepared for scientific analysis."""

        return self.error_code is not None


class ProteinLoader:
    """
    Load PDB/mmCIF structures, run DSSP, and extract per-chain CA data.

    Polymer positions follow ``POLYMER_POSITION_POLICY``.
    """

    polymer_position_policy = POLYMER_POSITION_POLICY

    def __init__(
        self,
        file_path: str | os.PathLike[str],
        input_config: InputConfig,
        *,
        dssp_bin: str | None,
    ):
        if not isinstance(input_config, InputConfig):
            raise TypeError("`input_config` must be an InputConfig instance.")

        self.file_path = os.fspath(file_path)
        self.input_config = input_config
        self.model_id = int(input_config.model_id)
        self.dssp_bin = dssp_bin
        self.dssp_failure_policy = input_config.dssp_failure_policy
        self._dssp_sheet_codes = frozenset(str(code) for code in input_config.dssp_sheet_codes)

        self.structure: Any = None
        self.model: Any = None
        self.secondary_structure: dict[DsspResidueKey, str] | None = None
        self.strand_graphs: dict[str, StrandAdjacencyGraph] | None = None
        self._strand_node_by_residue: dict[DsspResidueKey, str] = {}
        self.secondary_structure_error: str | None = None
        self._chain_preparation_results: dict[str, ChainPreparationResult] = {}
        self._structure_file_type = ""
        self._mmcif_polypeptide_positions: dict[DsspResidueKey, int] = {}
        self._mmcif_polymer_mapping: _MmcifPolymerMapping | None = None
        self._pdb_polypeptide_positions_by_chain: dict[str, dict[tuple[str, int, str], int]] = {}

        self._load_structure()

    def _load_structure(self) -> None:
        if not os.path.exists(self.file_path):
            raise InputValidationError(f"Structure file not found: {self.file_path}")

        input_tmp = _decompress_gzip_to_temp_if_needed(self.file_path)
        parse_path = input_tmp or self.file_path
        ext = os.path.splitext(parse_path)[1].lower()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BiopythonWarning)
                if ext in [".cif", ".mmcif"]:
                    mmcif_parser = MMCIFParser(QUIET=True)
                    self.structure = mmcif_parser.get_structure("struct", parse_path)
                    self._structure_file_type = "MMCIF"
                    self.model = self.structure[self.model_id]
                    self._mmcif_polymer_mapping = _mmcif_polypeptide_mapping(
                        parse_path,
                        model_id=self.model_id,
                        selected_model=self.model,
                        input_config=self.input_config,
                    )
                    self._mmcif_polypeptide_positions = dict(self._mmcif_polymer_mapping.positions)
                else:
                    # Important: disable header parsing.
                    pdb_parser = PDBParser(
                        QUIET=True,
                        PERMISSIVE=self.input_config.pdb_parser_permissive,
                        get_header=False,
                    )
                    self.structure = pdb_parser.get_structure("struct", parse_path)
                    self._structure_file_type = "PDB"

            self.model = self.structure[self.model_id]
            _require_public_chain_ids(self.model, source_path=self.file_path)
            return

        except Exception as e:
            raise StructureParseError(f"Failed to parse structure {self.file_path}: {e}") from None
        finally:
            if input_tmp and os.path.exists(input_tmp):
                try:
                    os.remove(input_tmp)
                except OSError:
                    pass

    def _export_protein_only_pdb(self) -> str:
        _fill_missing_atom_elements(self.model)

        fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
        with os.fdopen(fd, "w") as handle:
            handle.write("HEADER    GENERATED BY LOADER                         \n")
            handle.write(self.input_config.dssp_pdb_export_cryst1_record.rstrip("\n") + "\n")
            io = PDBIO()
            io.set_structure(self.model)
            io.save(
                handle,
                select=_ProteinOnlySelect(
                    include_nonstandard_amino_acids=(
                        self.input_config.include_nonstandard_amino_acids
                    )
                ),
            )
        return tmp_path

    def _run_dssp(self, input_path: str) -> DsspAnnotation:
        dssp_bin = require_dssp_binary(self.dssp_bin)
        return run_dssp_annotation(input_path, dssp_executable=dssp_bin)

    def _polymer_index_for_annotation_residue(self, residue_key: DsspResidueKey) -> int:
        chain_id, residue_id = residue_key
        if self._structure_file_type == "MMCIF":
            mmcif_positions = self._mmcif_polypeptide_positions
            exact_key = residue_key
            if exact_key in mmcif_positions:
                return mmcif_positions[exact_key]
            hetfield, resseq, insertion_code = residue_id
            blank_key: DsspResidueKey = (chain_id, (" ", resseq, insertion_code))
            if hetfield != " " and blank_key in mmcif_positions:
                return mmcif_positions[blank_key]
            raise DsspAdapterError(
                f"DSSP strand residue {residue_key!r} has no declared polymer position."
            )

        if chain_id not in self.model:
            raise DsspAdapterError(f"DSSP annotation references unknown author chain {chain_id!r}.")
        chain = self.model[chain_id]
        pdb_positions = self._pdb_polymer_positions(chain)
        if residue_id in pdb_positions:
            return pdb_positions[residue_id]
        hetfield, resseq, insertion_code = residue_id
        matches = [
            position
            for (
                candidate_hetfield,
                candidate_resseq,
                candidate_insertion,
            ), position in pdb_positions.items()
            if candidate_resseq == resseq and candidate_insertion == insertion_code
        ]
        if len(matches) != 1:
            raise DsspAdapterError(
                f"DSSP strand residue {residue_key!r} has no unique SEQRES position."
            )
        return matches[0]

    def _build_strand_graphs(self, annotation: DsspAnnotation) -> None:
        graph_nodes: dict[str, list[StrandNode]] = {
            str(chain.id): [] for chain in self.model.get_chains()
        }
        source_node: dict[tuple[str, str], tuple[str, str]] = {}
        node_by_residue: dict[DsspResidueKey, str] = {}

        records_by_chain: dict[str, list[tuple[int, int, DsspStrandRecord]]] = {}
        for record in annotation.strand_records:
            positions = tuple(
                self._polymer_index_for_annotation_residue(residue_key)
                for residue_key in record.residue_keys
            )
            if not positions:
                raise DsspAdapterError(f"DSSP strand {record.source_id!r} contains no E residues.")
            if len(set(positions)) != len(positions):
                raise DsspAdapterError(
                    f"DSSP strand {record.source_id!r} maps multiple residues to one "
                    "polymer position."
                )
            records_by_chain.setdefault(record.author_chain_id, []).append(
                (min(positions), max(positions), record)
            )

        for chain_id, records in records_by_chain.items():
            if chain_id not in graph_nodes:
                raise DsspAdapterError(
                    f"DSSP strand data references unknown author chain {chain_id!r}."
                )
            records.sort(key=lambda item: (item[0], item[1], item[2].source_id))
            previous_end: int | None = None
            for ordinal, (start, end, record) in enumerate(records):
                if previous_end is not None and start <= previous_end:
                    raise DsspAdapterError(
                        f"DSSP strands in chain {chain_id!r} overlap in polymer positions."
                    )
                node_id = f"strand_{ordinal}"
                graph_nodes[chain_id].append(
                    StrandNode(
                        node_id=node_id,
                        residue_range=StrandRange(
                            start_polymer_index=start,
                            end_polymer_index=end,
                        ),
                    )
                )
                source_node[record.source_id] = (chain_id, node_id)
                for residue_key in record.residue_keys:
                    previous_node = node_by_residue.get(residue_key)
                    if previous_node is not None:
                        raise DsspAdapterError(
                            f"DSSP residue {residue_key!r} belongs to multiple strands."
                        )
                    node_by_residue[residue_key] = node_id
                previous_end = end

        graph_edges: dict[str, set[StrandEdge]] = {chain_id: set() for chain_id in graph_nodes}
        for edge in annotation.strand_edges:
            first = source_node.get(edge.first_source_id)
            second = source_node.get(edge.second_source_id)
            if first is None or second is None:
                raise DsspAdapterError(
                    "DSSP ladder edge references an unknown E-strand: "
                    f"{edge.first_source_id!r}, {edge.second_source_id!r}."
                )
            if first[0] != second[0]:
                raise DsspAdapterError(
                    "A chain-level strand graph cannot contain cross-chain edges."
                )
            graph_edges[first[0]].add(StrandEdge(first[1], second[1]))

        self.strand_graphs = {
            chain_id: StrandAdjacencyGraph(
                author_chain_id=chain_id,
                nodes=tuple(nodes),
                edges=tuple(sorted(graph_edges[chain_id])),
            )
            for chain_id, nodes in graph_nodes.items()
        }
        self._strand_node_by_residue = node_by_residue

    def _install_dssp_annotation(self, annotation: DsspAnnotation) -> None:
        if not isinstance(annotation, DsspAnnotation):
            raise TypeError("DSSP adapter must return a DsspAnnotation.")
        self._build_strand_graphs(annotation)
        self.secondary_structure = dict(annotation.residue_assignments)
        self._chain_preparation_results.clear()

    def _run_secondary_structure(self) -> None:
        if self.secondary_structure is not None:
            return

        tmp_path = None
        try:
            if self._structure_file_type == "MMCIF":
                # The classic DSSP text format has a one-character chain field.
                # Parse modern annotated mmCIF output instead, preserving the
                # exact label-to-author residue mapping for every chain.
                tmp_path = _decompress_gzip_to_temp_if_needed(self.file_path)
                dssp_input = tmp_path or self.file_path
                dssp_bin = require_dssp_binary(self.dssp_bin)
                assert self._mmcif_polymer_mapping is not None
                with _selected_model_mmcif_path(
                    dssp_input,
                    model_id=self.model_id,
                    polymer_mapping=self._mmcif_polymer_mapping,
                ) as selected_model_path:
                    self._install_dssp_annotation(
                        run_dssp_annotation(
                            selected_model_path,
                            dssp_executable=dssp_bin,
                        )
                    )
            else:
                # PDB and mmCIF inputs share the same annotated-mmCIF parser so
                # strand boundaries and ladder edges have one definition.
                tmp_path = self._export_protein_only_pdb()
                self._install_dssp_annotation(self._run_dssp(tmp_path))

        except Exception as error:
            self._record_dssp_failure(error)
            self.secondary_structure = {}
            self.strand_graphs = {
                str(chain.id): StrandAdjacencyGraph(str(chain.id), (), ())
                for chain in self.model.get_chains()
            }
            self._strand_node_by_residue = {}
            self._chain_preparation_results.clear()

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _record_dssp_failure(self, error: Exception) -> None:
        self.secondary_structure_error = (
            f"DSSP failed for {os.path.basename(self.file_path)}: {error}"
        )
        if self.dssp_failure_policy == "error":
            raise DsspError(self.secondary_structure_error) from error

    def _secondary_structure_code(
        self,
        *,
        chain_id: str,
        residue_id: tuple[str, int, str],
        chain_residue_ids: set[tuple[str, int, str]],
    ) -> str | None:
        if self.secondary_structure is None:
            return None
        dssp_key = (chain_id, residue_id)
        if dssp_key in self.secondary_structure:
            return self.secondary_structure[dssp_key]

        hetfield, resseq, insertion_code = residue_id
        blank_residue_id = (" ", resseq, insertion_code)
        blank_key = (chain_id, blank_residue_id)
        if (
            hetfield != " "
            and blank_residue_id not in chain_residue_ids
            and blank_key in self.secondary_structure
        ):
            return self.secondary_structure[blank_key]
        return None

    def _strand_node_id(
        self,
        *,
        chain_id: str,
        residue_id: tuple[str, int, str],
        chain_residue_ids: set[tuple[str, int, str]],
    ) -> str | None:
        dssp_key: DsspResidueKey = (chain_id, residue_id)
        if dssp_key in self._strand_node_by_residue:
            return self._strand_node_by_residue[dssp_key]

        hetfield, resseq, insertion_code = residue_id
        blank_residue_id = (" ", resseq, insertion_code)
        blank_key: DsspResidueKey = (chain_id, blank_residue_id)
        if (
            hetfield != " "
            and blank_residue_id not in chain_residue_ids
            and blank_key in self._strand_node_by_residue
        ):
            return self._strand_node_by_residue[blank_key]
        return None

    def available_chains(self) -> list[str]:
        """Return chain IDs available in the selected model."""
        return [str(chain.id) for chain in self.model.get_chains()]

    def _resolve_chain(self, chain_id: str) -> tuple[str, Any]:
        chain = self.model[chain_id] if chain_id in self.model else None
        if chain is None:
            chains = list(self.model.get_chains())
            if not self.input_config.strict_chain and len(chains) == 1:
                chain = chains[0]
            else:
                available = ", ".join(self.available_chains()) or "none"
                raise ChainNotFoundError(
                    f"Chain {chain_id!r} not found in {os.path.basename(self.file_path)}. "
                    f"Available chains: {available}."
                )
        return str(chain.id), chain

    def _loaded_strand_graph(self, chain_id: str) -> StrandAdjacencyGraph:
        if self.strand_graphs is None or chain_id not in self.strand_graphs:
            raise DsspError(
                f"DSSP produced no strand graph for chain {chain_id!r} in "
                f"{os.path.basename(self.file_path)}."
            )
        return self.strand_graphs[chain_id]

    def _has_complete_finite_backbone(self, residue: Any) -> bool:
        for atom_name in ("N", "CA", "C", "O"):
            atom = _select_atom(residue, atom_name, self.input_config.atom_altloc_policy)
            if atom is None:
                return False
            coordinate = np.asarray(atom.get_coord(), dtype=float)
            if coordinate.shape != (3,) or not np.all(np.isfinite(coordinate)):
                return False
        return True

    def _pdb_polymer_positions(self, chain: Any) -> dict[tuple[str, int, str], int]:
        """Return unique SEQRES positions for every observed PDB polymer residue."""

        chain_id = str(chain.id)
        cached = self._pdb_polypeptide_positions_by_chain.get(chain_id)
        if cached is not None:
            return cached

        selected_residues = []
        letters: list[str] = []
        for raw_residue in chain:
            residue = _select_residue_variant(raw_residue, self.input_config)
            if not is_aa(residue, standard=False):
                continue
            selected_residues.append(residue)
            letters.append(_pdb_observed_residue_letter(residue))
        if not selected_residues:
            return {}

        try:
            declaration = declared_polymer_sequence_for_author_chain(
                self.file_path,
                str(chain.id),
            )
        except (OSError, TypeError, ValueError) as error:
            raise StructureParseError(
                f"PDB chain {str(chain.id)!r} requires a complete, uniquely mappable "
                f"SEQRES declaration: {error}"
            ) from error
        positions = _unique_subsequence_positions("".join(letters), declaration.sequence)
        result = {
            residue.id: position
            for residue, position in zip(selected_residues, positions, strict=True)
        }
        self._pdb_polypeptide_positions_by_chain[chain_id] = result
        return result

    def _extract_ca_data(self, chain: Any) -> list[dict[str, object]]:
        data: list[dict[str, object]] = []
        chain_residue_ids = {res.id for res in chain}
        expected_dssp_keys: set[tuple[str, tuple[str, int, str]]] = set()
        assigned_dssp_keys: set[tuple[str, tuple[str, int, str]]] = set()
        pdb_polymer_positions = (
            self._pdb_polymer_positions(chain) if self._structure_file_type == "PDB" else {}
        )
        positioned_residues: list[tuple[int, Any]] = []
        for raw_residue in chain:
            res = _select_residue_variant(raw_residue, self.input_config)
            polymer_key = (str(chain.id), res.id)
            if self._structure_file_type == "MMCIF":
                declared_position = self._mmcif_polypeptide_positions.get(polymer_key)
                if declared_position is None:
                    continue
                polymer_index = declared_position
            else:
                if not is_aa(res, standard=False):
                    continue
                try:
                    polymer_index = pdb_polymer_positions[res.id]
                except KeyError as error:
                    raise StructureParseError(
                        f"Included PDB residue {res.id!r} lacks a unique SEQRES position."
                    ) from error
            positioned_residues.append((polymer_index, res))

        positioned_residues.sort(key=lambda item: item[0])
        polymer_indices = [position for position, _ in positioned_residues]
        if len(polymer_indices) != len(set(polymer_indices)):
            raise StructureParseError(
                f"Chain {str(chain.id)!r} maps multiple residues to one polymer position."
            )

        previous_included_residue: Any | None = None
        for polymer_index, res in positioned_residues:
            if not self.input_config.include_nonstandard_amino_acids and not is_aa(
                res, standard=True
            ):
                continue
            ca_atom = _select_atom(res, "CA", self.input_config.atom_altloc_policy)
            if ca_atom is None:
                continue

            coordinate = np.asarray(ca_atom.get_coord(), dtype=float)
            if coordinate.shape != (3,) or not np.all(np.isfinite(coordinate)):
                raise StructureParseError(
                    f"Residue {res.id!r} in chain {chain.id!r} has invalid CA coordinates."
                )

            peptide_bond_distance = None
            if previous_included_residue is not None:
                previous_c = _select_atom(
                    previous_included_residue,
                    "C",
                    self.input_config.atom_altloc_policy,
                )
                current_n = _select_atom(res, "N", self.input_config.atom_altloc_policy)
                if previous_c is not None and current_n is not None:
                    previous_c_coord = np.asarray(previous_c.get_coord(), dtype=float)
                    current_n_coord = np.asarray(current_n.get_coord(), dtype=float)
                    if np.all(np.isfinite(previous_c_coord)) and np.all(
                        np.isfinite(current_n_coord)
                    ):
                        peptide_bond_distance = float(
                            np.linalg.norm(current_n_coord - previous_c_coord)
                        )

            hetfield, resseq, icode = res.id
            dssp_key = (str(chain.id), res.id)
            dssp_eligible = self._has_complete_finite_backbone(res)
            mapped_code = None
            if dssp_eligible:
                expected_dssp_keys.add(dssp_key)
                mapped_code = self._secondary_structure_code(
                    chain_id=str(chain.id),
                    residue_id=res.id,
                    chain_residue_ids=chain_residue_ids,
                )
                if mapped_code is not None:
                    validate_dssp_secondary_structure_code(
                        mapped_code,
                        context=f"{os.path.basename(self.file_path)} residue {dssp_key!r}",
                    )
                    assigned_dssp_keys.add(dssp_key)
            dssp_assignment_available = mapped_code is not None
            ss_code = mapped_code if mapped_code is not None else "-"
            is_sheet = ss_code in self._dssp_sheet_codes
            strand_node_id = (
                self._strand_node_id(
                    chain_id=str(chain.id),
                    residue_id=res.id,
                    chain_residue_ids=chain_residue_ids,
                )
                if is_sheet and ss_code == "E"
                else None
            )
            if is_sheet and ss_code == "E" and strand_node_id is None:
                raise DsspAdapterError(
                    f"DSSP E residue {dssp_key!r} is not assigned to a strand node."
                )

            data.append(
                {
                    "res_id": resseq,
                    "polymer_index": polymer_index,
                    "resseq": resseq,
                    "icode": str(icode).strip(),
                    "hetfield": str(hetfield).strip(),
                    "res_uid": {
                        "chain": chain.id,
                        "hetfield": str(hetfield).strip(),
                        "resseq": resseq,
                        "icode": str(icode).strip(),
                    },
                    "chain": chain.id,
                    "coord": coordinate.tolist(),
                    "peptide_bond_distance_to_previous_angstrom": peptide_bond_distance,
                    "dssp_assignment_available": dssp_assignment_available,
                    "is_sheet": is_sheet,
                    "strand_node_id": strand_node_id,
                }
            )
            previous_included_residue = res

        validate_dssp_coverage(
            expected_dssp_keys,
            assigned_dssp_keys,
            context=(f"{os.path.basename(self.file_path)} chain {str(chain.id)!r}"),
        )
        return data

    @staticmethod
    def _chain_error_code(error: Exception) -> str:
        if isinstance(error, (DsspAdapterError, DsspError)):
            return DsspError.error_code
        return str(getattr(error, "error_code", "UNEXPECTED_CHAIN_PREPARATION_FAILURE"))

    def prepare_chain(self, chain_id: str) -> ChainPreparationResult:
        """Prepare one chain without allowing its failure to discard sibling chains."""

        resolved_chain_id, chain = self._resolve_chain(chain_id)
        cached = self._chain_preparation_results.get(resolved_chain_id)
        if cached is not None:
            return cached

        if self.secondary_structure is None:
            self._run_secondary_structure()

        graph = self._loaded_strand_graph(resolved_chain_id)
        if self.secondary_structure_error is not None:
            result = ChainPreparationResult(
                author_chain_id=resolved_chain_id,
                residues=(),
                strand_graph=graph,
                error_code=DsspError.error_code,
                error_message=self.secondary_structure_error,
            )
        else:
            try:
                residues = self._extract_ca_data(chain)
            except Exception as error:
                if isinstance(error, DsspAdapterError):
                    message = (
                        f"DSSP annotation is invalid for {os.path.basename(self.file_path)} "
                        f"chain {resolved_chain_id!r}: {error}"
                    )
                else:
                    message = str(error)
                result = ChainPreparationResult(
                    author_chain_id=resolved_chain_id,
                    residues=(),
                    strand_graph=graph,
                    error_code=self._chain_error_code(error),
                    error_message=message,
                )
            else:
                result = ChainPreparationResult(
                    author_chain_id=resolved_chain_id,
                    residues=tuple(residues),
                    strand_graph=graph,
                )

        self._chain_preparation_results[resolved_chain_id] = result
        return result

    def _raise_chain_preparation_error(self, result: ChainPreparationResult) -> None:
        if not result.failed:
            return
        error_code = result.error_code or "UNEXPECTED_CHAIN_PREPARATION_FAILURE"
        message = result.error_message or "Chain preparation failed."
        if error_code == DsspError.error_code:
            if self.dssp_failure_policy == "error":
                raise DsspError(message)
            return
        if error_code == StructureParseError.error_code:
            raise StructureParseError(message)
        raise InputValidationError(message)

    def get_strand_graph(self, chain_id: str) -> StrandAdjacencyGraph:
        """Return the validated DSSP strand-adjacency graph for one author chain."""

        result = self.prepare_chain(chain_id)
        self._raise_chain_preparation_error(result)
        return result.strand_graph

    def get_ca_data(self, chain_id: str) -> list[dict[str, object]]:
        """Return CA records when the requested chain has valid DSSP preparation."""

        result = self.prepare_chain(chain_id)
        self._raise_chain_preparation_error(result)
        return [dict(residue) for residue in result.residues]

    def get_chain_data(self, chain_id: str) -> list[dict[str, object]]:
        return self.get_ca_data(chain_id)
