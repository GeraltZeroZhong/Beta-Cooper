from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.foldseek.runner import (
    DEFAULT_ALIGNMENT_TYPE,
    DEFAULT_EVALUE,
    DEFAULT_MAX_SEQS,
    DEFAULT_MIN_QUERY_COVERAGE,
    DEFAULT_MIN_TARGET_COVERAGE,
    DEFAULT_SCORE_MODE,
    DEFAULT_SCORE_THRESHOLD,
    SUPPORTED_SCORE_MODES,
    FoldseekResult,
    run_baseline,
)
from external_methods.foldseek.structures import (
    DEFAULT_MIN_RESIDUES,
    GeneratedStructureSet,
    foldseek_query_aliases,
    generate_structure_chains,
)


@dataclass(frozen=True)
class StructureSearchBaselineRun:
    generated_chains: GeneratedStructureSet
    results: list[FoldseekResult]
    output_path: str | None = None


def run_structure_search_baseline(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    target_db: str | Path | None = None,
    reference_structures: str | Path | None = None,
    foldseek_executable: str | Path | None = None,
    output_path: str | Path | None = None,
    min_residues: int = DEFAULT_MIN_RESIDUES,
    create_index: bool = False,
    alignment_type: int = DEFAULT_ALIGNMENT_TYPE,
    score_mode: str = DEFAULT_SCORE_MODE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_query_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
    min_target_coverage: float = DEFAULT_MIN_TARGET_COVERAGE,
    evalue: float = DEFAULT_EVALUE,
    max_seqs: int = DEFAULT_MAX_SEQS,
    extra_args: Sequence[str] | None = None,
    command_prefix: Sequence[str] | None = None,
    timeout: float | None = None,
) -> StructureSearchBaselineRun:
    if (target_db is None) == (reference_structures is None):
        raise ValueError("Provide exactly one of target_db or reference_structures.")

    output = Path(output_dir).expanduser().resolve()
    generated = generate_structure_chains(
        structure_input,
        output / "query_chains",
        min_residues=min_residues,
    )
    target = target_db if target_db is not None else reference_structures
    assert target is not None

    results = run_baseline(
        generated.chain_dir,
        target,
        foldseek_executable=foldseek_executable,
        work_dir=output / "foldseek_work",
        output_path=output_path,
        query_ids=[record.sample_id for record in generated.records],
        query_aliases=foldseek_query_aliases(generated.records),
        build_target_db=reference_structures is not None,
        create_index=create_index,
        alignment_type=alignment_type,
        score_mode=score_mode,
        score_threshold=score_threshold,
        min_query_coverage=min_query_coverage,
        min_target_coverage=min_target_coverage,
        evalue=evalue,
        max_seqs=max_seqs,
        extra_args=extra_args,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    return StructureSearchBaselineRun(
        generated_chains=generated,
        results=results,
        output_path=str(Path(output_path).expanduser().resolve()) if output_path else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/foldseek/structure_search.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Extract eligible chains from PDB or mmCIF structures, search them with Foldseek "
            "global TMalign, and normalize one baseline decision per chain."
        ),
        epilog=(
            "Output: generated query chains, residue mapping, and raw Foldseek files under "
            "--out-dir; --out additionally writes normalized results CSV. Arguments after -- are "
            "passed to Foldseek easy-search. Input or subprocess failures exit with status 2."
        ),
    )
    parser.add_argument(
        "structure_input",
        metavar="STRUCTURE_OR_DIRECTORY",
        help="PDB, CIF, or mmCIF query file, or a directory searched recursively.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="DIRECTORY",
        help="Working directory for generated chains, metadata, and Foldseek output.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-db", metavar="PREFIX", help="Existing Foldseek target database prefix."
    )
    target_group.add_argument(
        "--reference-structures",
        metavar="STRUCTURE_OR_DIRECTORY",
        help="Reference PDB/CIF/mmCIF file or directory to convert with foldseek createdb.",
    )
    parser.add_argument(
        "--foldseek",
        metavar="COMMAND",
        help="Foldseek executable; omit to use FOLDSEEK_BIN, then foldseek on PATH.",
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
        help="Minimum alpha-carbon residue count required to search a chain.",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Run foldseek createindex for --reference-structures before searching.",
    )
    parser.add_argument(
        "--alignment-type",
        type=int,
        default=DEFAULT_ALIGNMENT_TYPE,
        metavar="N",
        help="Foldseek --alignment-type value; 1 selects global TMalign.",
    )
    parser.add_argument(
        "--score-mode",
        default=DEFAULT_SCORE_MODE,
        choices=sorted(SUPPORTED_SCORE_MODES),
        help="Per-hit score used for the threshold decision.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        metavar="FRACTION",
        help="Inclusive minimum selected hit score for a BARREL decision.",
    )
    parser.add_argument(
        "--min-query-coverage",
        type=float,
        default=DEFAULT_MIN_QUERY_COVERAGE,
        metavar="FRACTION",
        help="Inclusive minimum query coverage for an eligible hit, from 0 to 1.",
    )
    parser.add_argument(
        "--min-target-coverage",
        type=float,
        default=DEFAULT_MIN_TARGET_COVERAGE,
        metavar="FRACTION",
        help="Inclusive minimum target coverage for an eligible hit, from 0 to 1.",
    )
    parser.add_argument(
        "--evalue",
        type=float,
        default=DEFAULT_EVALUE,
        metavar="EVALUE",
        help="Maximum E-value passed to Foldseek easy-search.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=DEFAULT_MAX_SEQS,
        metavar="N",
        help="Maximum candidate sequences passed to Foldseek --max-seqs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum elapsed time for each Foldseek subprocess; omit for no timeout.",
    )
    return parser


def _parse_args_and_passthrough(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw_args:
        return parser.parse_args(raw_args), []
    passthrough_index = raw_args.index("--")
    args = parser.parse_args(raw_args[:passthrough_index])
    return args, raw_args[passthrough_index + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, extra_args = _parse_args_and_passthrough(parser, argv)

    run = run_structure_search_baseline(
        args.structure_input,
        args.out_dir,
        target_db=args.target_db,
        reference_structures=args.reference_structures,
        foldseek_executable=args.foldseek,
        output_path=args.out,
        min_residues=args.min_residues,
        create_index=args.create_index,
        alignment_type=args.alignment_type,
        score_mode=args.score_mode,
        score_threshold=args.score_threshold,
        min_query_coverage=args.min_query_coverage,
        min_target_coverage=args.min_target_coverage,
        evalue=args.evalue,
        max_seqs=args.max_seqs,
        extra_args=extra_args,
        timeout=args.timeout,
    )

    barrel_count = sum(result.result == "BARREL" for result in run.results)
    print(f"Generated chains: {len(run.generated_chains.records)}")
    print(f"Rows: {len(run.results)}")
    print(f"BARREL: {barrel_count}")
    print(f"NON_BARREL: {len(run.results) - barrel_count}")
    if run.output_path:
        print(f"Output: {run.output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
