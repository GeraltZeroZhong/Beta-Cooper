from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cooper_beta.polymer_sequence import (
    COMPLETE_SEQUENCE_SOURCES,
    SUPPORTED_STRUCTURE_EXTENSIONS,
    declared_polymer_sequences,
    structure_extension,
)

DEFAULT_MIN_RESIDUES = 15
SUPPORTED_EXTENSIONS = SUPPORTED_STRUCTURE_EXTENSIONS


@dataclass(frozen=True)
class GeneratedSequence:
    sample_id: str
    source_path: str
    author_chain_id: str
    n_residues: int
    sequence: str
    sequence_sha256: str
    sequence_source: str
    polymer_entity_id: str
    label_asym_id: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.source_path or not self.author_chain_id:
            raise ValueError("Generated sequence identity fields must not be blank.")
        if self.n_residues != len(self.sequence) or self.n_residues <= 0:
            raise ValueError("Generated sequence length is inconsistent with its sequence.")
        if re.fullmatch(r"[A-Z]+", self.sequence) is None:
            raise ValueError("Generated sequence must contain uppercase amino-acid codes only.")
        expected_digest = hashlib.sha256(self.sequence.encode("ascii")).hexdigest()
        if self.sequence_sha256 != expected_digest:
            raise ValueError("Generated sequence SHA-256 is inconsistent with its sequence.")
        if self.sequence_source not in COMPLETE_SEQUENCE_SOURCES:
            raise ValueError(f"Unknown complete sequence source: {self.sequence_source!r}.")
        if self.sequence_source.startswith("mmcif_") and (
            not self.polymer_entity_id or not self.label_asym_id
        ):
            raise ValueError("mmCIF sequence identity requires entity and label-chain IDs.")


@dataclass(frozen=True)
class GeneratedFastaSet:
    output_dir: str
    fasta_path: str
    residue_mapping_path: str
    records: list[GeneratedSequence]


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "blank"


def _structure_extension(path: Path) -> str:
    return structure_extension(path)


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


def _wrap_fasta(sequence: str, width: int = 60) -> str:
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


def _write_residue_mapping(
    mapping_rows: Iterable[dict[str, object]],
    residue_mapping_path: Path,
) -> None:
    fieldnames = [
        "sample_id",
        "sequence_index",
        "source_file",
        "author_chain_id",
        "label_asym_id",
        "polymer_entity_id",
        "sequence_source",
        "sequence_sha256",
        "monomer_id",
        "one_letter_code",
    ]
    with residue_mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)


def generate_structure_fasta(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    min_residues: int = DEFAULT_MIN_RESIDUES,
) -> GeneratedFastaSet:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fasta_path = output / "sequences.fasta"
    residue_mapping_path = output / "residue_mapping.csv"

    records: list[GeneratedSequence] = []
    mapping_rows: list[dict[str, object]] = []
    seen_ids: Counter[str] = Counter()

    for structure_path in discover_structure_files(structure_input):
        declared_sequences = declared_polymer_sequences(structure_path)
        for declared in declared_sequences:
            sequence = declared.sequence
            if len(sequence) < min_residues:
                continue

            chain_id = declared.author_chain_id
            base_id = f"{_structure_stem(structure_path)}_{_safe_id(chain_id)}"
            seen_ids[base_id] += 1
            sample_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"
            sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()

            records.append(
                GeneratedSequence(
                    sample_id=sample_id,
                    source_path=str(structure_path),
                    author_chain_id=chain_id,
                    n_residues=len(sequence),
                    sequence=sequence,
                    sequence_sha256=sequence_sha256,
                    sequence_source=declared.sequence_source,
                    polymer_entity_id=declared.entity_id,
                    label_asym_id=declared.label_asym_id,
                )
            )

            for index, (monomer_id, one_letter_code) in enumerate(
                zip(declared.monomer_ids, sequence, strict=True),
            ):
                mapping_rows.append(
                    {
                        "sample_id": sample_id,
                        "sequence_index": index,
                        "source_file": str(structure_path),
                        "author_chain_id": chain_id,
                        "label_asym_id": declared.label_asym_id,
                        "polymer_entity_id": declared.entity_id,
                        "sequence_source": declared.sequence_source,
                        "sequence_sha256": sequence_sha256,
                        "monomer_id": monomer_id,
                        "one_letter_code": one_letter_code,
                    }
                )

    if not records:
        raise ValueError("No declared protein polymer sequences meet the minimum length.")

    with fasta_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.sample_id}\n{_wrap_fasta(record.sequence)}\n")

    _write_residue_mapping(mapping_rows, residue_mapping_path)

    return GeneratedFastaSet(
        output_dir=str(output),
        fasta_path=str(fasta_path),
        residue_mapping_path=str(residue_mapping_path),
        records=records,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/pred_tmbb2/sequences.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Extract declared complete polymer sequences from PDB or mmCIF structures and write "
            "chain-level FASTA input for the PRED-TMBB2 adapter. Directories are searched recursively."
        ),
        epilog=(
            "Output: <OUT_DIR>/sequences.fasta and residue_mapping.csv. Chains whose complete "
            "declared sequence is shorter than --min-residues are omitted. Invalid arguments exit "
            "with status 2; sequence parsing or output failures exit nonzero."
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
        help="Directory for the generated FASTA and residue mapping.",
    )
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        metavar="RESIDUES",
        help="Minimum declared complete polymer-sequence length required to export a chain.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    generated = generate_structure_fasta(
        args.structure_input,
        args.out_dir,
        min_residues=args.min_residues,
    )

    print(f"Generated sequences: {len(generated.records)}")
    print(f"FASTA: {generated.fasta_path}")
    print(f"Residue mapping: {generated.residue_mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
