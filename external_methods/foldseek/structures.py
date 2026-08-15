from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

DEFAULT_MIN_RESIDUES = 15
SUPPORTED_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}


@dataclass(frozen=True)
class GeneratedStructureChain:
    sample_id: str
    source_path: str
    chain_id: str
    n_residues: int
    chain_path: str


@dataclass(frozen=True)
class GeneratedStructureSet:
    output_dir: str
    chain_dir: str
    manifest_path: str
    residue_mapping_path: str
    records: list[GeneratedStructureChain]


class ResidueMappingRow(TypedDict):
    sample_id: str
    chain_file_index: int
    source_file: str
    source_chain_id: str
    exported_chain_id: str
    residue_name: str
    residue_number: int
    insertion_code: str


class _StructureParser(Protocol):
    def get_structure(self, structure_id: str, filename: str) -> Structure: ...


class _ProteinResidueSelect(Select):
    def accept_residue(self, residue: Residue) -> int:
        return 1 if _is_protein_backbone_residue(residue) else 0

    def accept_atom(self, atom: Atom) -> int:
        return 1


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
        if _is_protein_backbone_residue(residue):
            residues.append(residue)
    return residues


def _is_protein_backbone_residue(residue: Residue) -> bool:
    """Accept named amino acids and declared unknown residues with complete backbone."""

    return is_aa(residue, standard=False) or all(
        atom_name in residue for atom_name in ("N", "CA", "C")
    )


def _write_chain_pdb(chain: Chain, chain_path: Path) -> None:
    chain_copy: Chain = chain.copy()
    chain_copy.id = "A"
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(chain_copy)
    io.save(str(chain_path), select=_ProteinResidueSelect())


def _write_manifest(records: Iterable[GeneratedStructureChain], manifest_path: Path) -> None:
    fieldnames = ["sample_id", "source_file", "chain_id", "n_residues", "chain_path"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "source_file": record.source_path,
                    "chain_id": record.chain_id,
                    "n_residues": record.n_residues,
                    "chain_path": record.chain_path,
                }
            )


def _write_residue_mapping(
    mapping_rows: Iterable[ResidueMappingRow],
    residue_mapping_path: Path,
) -> None:
    fieldnames = [
        "sample_id",
        "chain_file_index",
        "source_file",
        "source_chain_id",
        "exported_chain_id",
        "residue_name",
        "residue_number",
        "insertion_code",
    ]
    with residue_mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)


def foldseek_query_aliases(records: Sequence[GeneratedStructureChain]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for record in records:
        path = Path(record.chain_path)
        candidates = {
            record.sample_id,
            path.stem,
            path.name,
            f"{path.stem}_A",
            f"{path.name}_A",
        }
        for candidate in candidates:
            aliases[candidate] = record.sample_id
    return aliases


def generate_structure_chains(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    min_residues: int = DEFAULT_MIN_RESIDUES,
) -> GeneratedStructureSet:
    output = Path(output_dir).expanduser().resolve()
    chain_dir = output / "chains"
    manifest_path = output / "chain_manifest.csv"
    residue_mapping_path = output / "residue_mapping.csv"
    chain_dir.mkdir(parents=True, exist_ok=True)

    records: list[GeneratedStructureChain] = []
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
                    "Foldseek chain generation fails closed."
                )
            base_id = f"{_structure_stem(structure_path)}_{_safe_id(chain_id)}"
            seen_ids[base_id] += 1
            sample_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"
            chain_path = chain_dir / f"{sample_id}.pdb"
            _write_chain_pdb(chain, chain_path)

            records.append(
                GeneratedStructureChain(
                    sample_id=sample_id,
                    source_path=str(structure_path),
                    chain_id=chain_id,
                    n_residues=len(residues),
                    chain_path=str(chain_path),
                )
            )

            for index, residue in enumerate(residues):
                residue_id: tuple[str, int, str] = residue.get_id()
                insertion_code = str(residue_id[2]).strip()
                mapping_rows.append(
                    {
                        "sample_id": sample_id,
                        "chain_file_index": index,
                        "source_file": str(structure_path),
                        "source_chain_id": chain_id,
                        "exported_chain_id": "A",
                        "residue_name": str(residue.get_resname()),
                        "residue_number": residue_id[1],
                        "insertion_code": insertion_code,
                    }
                )

    if not records:
        raise ValueError("No protein chains with enough CA residues were found.")

    _write_manifest(records, manifest_path)
    _write_residue_mapping(mapping_rows, residue_mapping_path)

    return GeneratedStructureSet(
        output_dir=str(output),
        chain_dir=str(chain_dir),
        manifest_path=str(manifest_path),
        residue_mapping_path=str(residue_mapping_path),
        records=records,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/foldseek/structures.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Extract every eligible protein chain from PDB or mmCIF input as a separate PDB "
            "query for the Foldseek adapter. Directory inputs are searched recursively."
        ),
        epilog=(
            "Output: <OUT_DIR>/chains/*.pdb, chain_manifest.csv, and residue_mapping.csv. "
            "Chains shorter than --min-residues alpha-carbon observations are omitted. Invalid "
            "arguments exit with status 2; structure parsing or output failures exit nonzero."
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
        help="Directory for generated chain PDB files and mapping metadata.",
    )
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        metavar="RESIDUES",
        help="Minimum alpha-carbon residue count required to export a chain.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    generated = generate_structure_chains(
        args.structure_input,
        args.out_dir,
        min_residues=args.min_residues,
    )

    print(f"Generated chains: {len(generated.records)}")
    print(f"Chain directory: {generated.chain_dir}")
    print(f"Manifest: {generated.manifest_path}")
    print(f"Residue mapping: {generated.residue_mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
