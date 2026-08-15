from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.pred_tmbb2.runner import (
    DEFAULT_MIN_TM_STRANDS,
    DEFAULT_PREDICTION_FIELD,
    PredTmbb2Result,
    run_baseline,
)
from external_methods.pred_tmbb2.sequences import (
    DEFAULT_MIN_RESIDUES,
    GeneratedFastaSet,
    generate_structure_fasta,
)


@dataclass(frozen=True)
class StructureSequenceBaselineRun:
    generated_fasta: GeneratedFastaSet
    results: list[PredTmbb2Result]
    output_path: str | None = None


def run_structure_sequence_baseline(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    juchmme_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    min_residues: int = DEFAULT_MIN_RESIDUES,
    prediction_field: str = DEFAULT_PREDICTION_FIELD,
    min_tm_strands: int = DEFAULT_MIN_TM_STRANDS,
    java_executable: str | None = None,
    command_prefix: Sequence[str] | None = None,
    timeout: float | None = None,
) -> StructureSequenceBaselineRun:
    output = Path(output_dir).expanduser().resolve()
    generated = generate_structure_fasta(
        structure_input,
        output,
        min_residues=min_residues,
    )
    results = run_baseline(
        generated.fasta_path,
        juchmme_dir=juchmme_dir,
        work_dir=output / "juchmme_work",
        output_path=output_path,
        prediction_field=prediction_field,
        min_tm_strands=min_tm_strands,
        java_executable=java_executable,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    return StructureSequenceBaselineRun(
        generated_fasta=generated,
        results=results,
        output_path=str(Path(output_path).expanduser().resolve()) if output_path else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/pred_tmbb2/structure_sequence.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Extract complete chain sequences from PDB or mmCIF structures, run the PRED-TMBB2 "
            "single-sequence JUCHMME baseline, and normalize one decision per chain."
        ),
        epilog=(
            "Output: sequences.fasta, residue_mapping.csv, and upstream working files under "
            "--out-dir; --out additionally writes normalized results CSV. Input or upstream "
            "execution failures exit with status 2."
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
        help="Working directory for generated FASTA, metadata, and upstream output.",
    )
    parser.add_argument(
        "--juchmme-dir",
        metavar="DIRECTORY",
        help="JUCHMME release/checkout directory. Defaults to PRED_TMBB2_JUCHMME_DIR.",
    )
    parser.add_argument(
        "--out",
        metavar="CSV",
        help="Normalized result CSV; omit to retain working files and print counts only.",
    )
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        metavar="RESIDUES",
        help="Minimum declared complete polymer-sequence length required to evaluate a chain.",
    )
    parser.add_argument(
        "--prediction-field",
        default=DEFAULT_PREDICTION_FIELD,
        choices=["LP", "VP", "lp", "vp"],
        help="JUCHMME topology field used to count membrane beta-strand segments.",
    )
    parser.add_argument(
        "--min-tm-strands",
        type=int,
        default=DEFAULT_MIN_TM_STRANDS,
        metavar="N",
        help="Inclusive minimum predicted membrane beta-strand count for BARREL.",
    )
    parser.add_argument(
        "--java",
        default="java",
        metavar="COMMAND",
        help="Java executable used to run JUCHMME.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum elapsed time for the JUCHMME subprocess; omit for no timeout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run = run_structure_sequence_baseline(
        args.structure_input,
        args.out_dir,
        juchmme_dir=args.juchmme_dir,
        output_path=args.out,
        min_residues=args.min_residues,
        prediction_field=args.prediction_field.upper(),
        min_tm_strands=args.min_tm_strands,
        java_executable=args.java,
        timeout=args.timeout,
    )

    barrel_count = sum(result.result == "BARREL" for result in run.results)
    print(f"Generated sequences: {len(run.generated_fasta.records)}")
    print(f"Rows: {len(run.results)}")
    print(f"BARREL: {barrel_count}")
    print(f"NON_BARREL: {len(run.results) - barrel_count}")
    if run.output_path:
        print(f"Output: {run.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
