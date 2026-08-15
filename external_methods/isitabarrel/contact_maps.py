from __future__ import annotations

import argparse
import csv
import gzip
import os
import pickle
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Model import Model
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from numpy.typing import NDArray

DEFAULT_CA_CUTOFF = 8.0
DEFAULT_LOCAL_EXCLUSION = 2
DEFAULT_MIN_RESIDUES = 15
SUPPORTED_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}


@dataclass(frozen=True)
class GeneratedContactMap:
    sample_id: str
    source_path: str
    chain_id: str
    n_residues: int
    n_contacts: int
    map_path: str


@dataclass(frozen=True)
class GeneratedContactMapSet:
    output_dir: str
    map_dir: str
    protid_list_path: str
    residue_mapping_path: str
    records: list[GeneratedContactMap]


class ResidueMappingRow(TypedDict):
    sample_id: str
    matrix_index: int
    source_file: str
    chain_id: str
    residue_name: str
    residue_number: int
    insertion_code: str


class _StructureParser(Protocol):
    def get_structure(self, structure_id: str, filename: str) -> Structure: ...


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "blank"


def _structure_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).suffix


def _structure_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    while Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
        name = Path(name).stem
    return _safe_id(name)


def _is_structure_path(path: Path) -> bool:
    return path.is_file() and _structure_extension(path) in SUPPORTED_EXTENSIONS


def discover_structure_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Structure input does not exist: {path}")
    if path.is_file():
        if not _is_structure_path(path):
            raise ValueError(f"Unsupported structure extension: {path}")
        return [path]

    files = [candidate for candidate in sorted(path.rglob("*")) if _is_structure_path(candidate)]
    if not files:
        raise ValueError(f"No supported structure files found in {path}")
    return files


def _decompress_gzip_to_temp_if_needed(path: Path) -> Path | None:
    with path.open("rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            return None

    suffix = _structure_extension(path) or ".pdb"
    fd, temp_name = tempfile.mkstemp(suffix=suffix)
    with gzip.open(path, "rb") as source, os.fdopen(fd, "wb") as target:
        shutil.copyfileobj(source, target)
    return Path(temp_name)


def _parse_structure(path: Path) -> Structure:
    temp_path = _decompress_gzip_to_temp_if_needed(path)
    parse_path = temp_path or path
    extension = _structure_extension(parse_path)
    parser: _StructureParser = (
        MMCIFParser(QUIET=True)
        if extension in {".cif", ".mmcif"}
        else PDBParser(QUIET=True, PERMISSIVE=True, get_header=False)
    )
    try:
        structure: Structure = parser.get_structure(_structure_stem(path), str(parse_path))
        return structure
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _chain_residues(chain: Chain) -> list[Residue]:
    residues: list[Residue] = []
    unpacked_residues: Iterable[Residue] = chain.get_unpacked_list()
    for residue in unpacked_residues:
        # Predicted structures can encode polymer residues as ATOM/UNK.  Their
        # alpha-carbon trace is a valid input to this structure-only baseline.
        # The blank hetero flag identifies ATOM polymer records; recognized
        # modified amino acids remain accepted through Biopython's classifier.
        is_polymer_atom = residue.id[0] == " "
        if (is_polymer_atom or is_aa(residue, standard=False)) and "CA" in residue:
            residues.append(residue)
    return residues


def _ca_contact_map(
    residues: Sequence[Residue],
    *,
    cutoff: float,
    local_exclusion: int,
) -> NDArray[np.float32]:
    ca_coordinates: list[NDArray[np.float32]] = []
    for residue in residues:
        atom: Atom = residue["CA"]
        ca_coordinates.append(atom.coord)
    coords: NDArray[np.float32] = np.asarray(ca_coordinates, dtype=np.float32)
    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    contact_map: NDArray[np.float32] = (distances <= cutoff).astype(np.float32)

    if local_exclusion >= 0:
        for offset in range(-local_exclusion, local_exclusion + 1):
            contact_map[np.eye(len(contact_map), k=offset, dtype=bool)] = 0.0

    return contact_map


def _write_residue_mapping(
    mapping_rows: Iterable[ResidueMappingRow],
    residue_mapping_path: Path,
) -> None:
    fieldnames = [
        "sample_id",
        "matrix_index",
        "source_file",
        "chain_id",
        "residue_name",
        "residue_number",
        "insertion_code",
    ]
    with residue_mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)


def generate_structure_contact_maps(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    cutoff: float = DEFAULT_CA_CUTOFF,
    local_exclusion: int = DEFAULT_LOCAL_EXCLUSION,
    min_residues: int = DEFAULT_MIN_RESIDUES,
) -> GeneratedContactMapSet:
    output = Path(output_dir).expanduser().resolve()
    map_dir = output / "maps"
    protid_list_path = output / "protid_list.tsv"
    residue_mapping_path = output / "residue_mapping.csv"
    map_dir.mkdir(parents=True, exist_ok=True)

    records: list[GeneratedContactMap] = []
    mapping_rows: list[ResidueMappingRow] = []
    seen_ids: Counter[str] = Counter()

    for structure_path in discover_structure_files(structure_input):
        structure = _parse_structure(structure_path)
        model: Model = structure[0]
        chains: Iterable[Chain] = model.get_chains()
        for chain in chains:
            residues = _chain_residues(chain)
            if len(residues) < min_residues:
                continue

            chain_id = str(chain.id).strip()
            if not chain_id:
                raise ValueError(
                    "Blank author chain identifiers cannot be represented unambiguously; "
                    "IsItABarrel contact-map generation fails closed."
                )
            base_id = f"{_structure_stem(structure_path)}_{_safe_id(chain_id)}"
            seen_ids[base_id] += 1
            sample_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"

            contact_map = _ca_contact_map(
                residues,
                cutoff=cutoff,
                local_exclusion=local_exclusion,
            )
            map_path = map_dir / f"{sample_id}.pkl"
            with map_path.open("wb") as handle:
                pickle.dump(contact_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

            n_contacts = int(np.triu(contact_map, k=1).sum())
            records.append(
                GeneratedContactMap(
                    sample_id=sample_id,
                    source_path=str(structure_path),
                    chain_id=chain_id,
                    n_residues=len(residues),
                    n_contacts=n_contacts,
                    map_path=str(map_path),
                )
            )

            for index, residue in enumerate(residues):
                residue_id: tuple[str, int, str] = residue.get_id()
                insertion_code = str(residue_id[2]).strip()
                mapping_rows.append(
                    {
                        "sample_id": sample_id,
                        "matrix_index": index,
                        "source_file": str(structure_path),
                        "chain_id": chain_id,
                        "residue_name": str(residue.get_resname()),
                        "residue_number": residue_id[1],
                        "insertion_code": insertion_code,
                    }
                )

    if not records:
        raise ValueError("No protein chains with enough CA residues were found.")

    with protid_list_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record.sample_id}\n")

    _write_residue_mapping(mapping_rows, residue_mapping_path)

    return GeneratedContactMapSet(
        output_dir=str(output),
        map_dir=str(map_dir),
        protid_list_path=str(protid_list_path),
        residue_mapping_path=str(residue_mapping_path),
        records=records,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/isitabarrel/contact_maps.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate one alpha-carbon contact map per eligible protein chain for the "
            "IsItABarrel adapter. PDB or mmCIF directories are searched recursively."
        ),
        epilog=(
            "Output: <OUT_DIR>/maps/*.pkl, protid_list.tsv, and residue_mapping.csv. Chains "
            "shorter than --min-residues alpha-carbon observations are omitted. Invalid arguments "
            "exit with status 2; structure parsing or output failures exit nonzero."
        ),
    )
    parser.add_argument(
        "structure_input",
        metavar="STRUCTURE_OR_DIRECTORY",
        help="PDB, CIF, or mmCIF file, or a directory searched recursively.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="DIRECTORY",
        help="Directory for contact-map pickle files and mapping metadata.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CA_CUTOFF,
        metavar="ANGSTROMS",
        help="Inclusive alpha-carbon distance cutoff used to define a contact.",
    )
    parser.add_argument(
        "--local-exclusion",
        type=int,
        default=DEFAULT_LOCAL_EXCLUSION,
        metavar="RESIDUES",
        help="Set contacts to zero when sequence-index distance is at most this value.",
    )
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        metavar="RESIDUES",
        help="Minimum alpha-carbon residue count required to export a chain contact map.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    generated = generate_structure_contact_maps(
        args.structure_input,
        args.out_dir,
        cutoff=args.cutoff,
        local_exclusion=args.local_exclusion,
        min_residues=args.min_residues,
    )

    print(f"Generated maps: {len(generated.records)}")
    print(f"Map directory: {generated.map_dir}")
    print(f"Protein id list: {generated.protid_list_path}")
    print(f"Residue mapping: {generated.residue_mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
