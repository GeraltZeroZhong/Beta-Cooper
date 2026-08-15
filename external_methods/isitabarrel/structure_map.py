from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.isitabarrel.contact_maps import (
    DEFAULT_CA_CUTOFF,
    DEFAULT_LOCAL_EXCLUSION,
    DEFAULT_MIN_RESIDUES,
    GeneratedContactMapSet,
    generate_structure_contact_maps,
)
from external_methods.isitabarrel.runner import (
    DEFAULT_DECISION_COLUMN,
    IsItABarrelResult,
    run_baseline,
)


@dataclass(frozen=True)
class StructureMapBaselineRun:
    generated_maps: GeneratedContactMapSet
    results: list[IsItABarrelResult]
    output_path: str | None = None


def run_structure_map_baseline(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    script_path: str | Path | None = None,
    output_path: str | Path | None = None,
    cutoff: float = DEFAULT_CA_CUTOFF,
    local_exclusion: int = DEFAULT_LOCAL_EXCLUSION,
    min_residues: int = DEFAULT_MIN_RESIDUES,
    decision_column: str = DEFAULT_DECISION_COLUMN,
    python_executable: str | None = None,
    extra_args: Sequence[str] | None = None,
    timeout: float | None = None,
) -> StructureMapBaselineRun:
    output = Path(output_dir).expanduser().resolve()
    generated = generate_structure_contact_maps(
        structure_input,
        output,
        cutoff=cutoff,
        local_exclusion=local_exclusion,
        min_residues=min_residues,
    )
    results = run_baseline(
        generated.protid_list_path,
        generated.map_dir,
        script_path=script_path,
        work_dir=output / "isitabarrel_work",
        output_path=output_path,
        decision_column=decision_column,
        python_executable=python_executable,
        extra_args=extra_args,
        timeout=timeout,
    )
    return StructureMapBaselineRun(
        generated_maps=generated,
        results=results,
        output_path=str(Path(output_path).expanduser().resolve()) if output_path else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/isitabarrel/structure_map.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate chain-level alpha-carbon contact maps from PDB or mmCIF structures, run "
            "IsItABarrel, and normalize one baseline decision per chain."
        ),
        epilog=(
            "Output: contact-map pickle files, protein identifiers, residue mapping, and upstream "
            "results.tsv under --out-dir; --out additionally writes normalized results CSV. "
            "Arguments after -- are passed to isitabarrel.py. Runtime failures exit nonzero."
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
        help="Working directory for generated maps, metadata, and upstream output.",
    )
    parser.add_argument(
        "--script",
        metavar="FILE",
        help="Path to the official isitabarrel.py script. "
        "Defaults to the ISITABARREL_SCRIPT environment variable.",
    )
    parser.add_argument(
        "--out",
        metavar="CSV",
        help="Normalized result CSV; omit to retain working files and print counts only.",
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
        help="Minimum alpha-carbon residue count required to evaluate a chain.",
    )
    parser.add_argument(
        "--decision-column",
        default=DEFAULT_DECISION_COLUMN,
        metavar="COLUMN",
        help="IsItABarrel results.tsv score column; values greater than zero produce BARREL.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        metavar="COMMAND",
        help="Python executable used to run the upstream script.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum elapsed time for isitabarrel.py; omit for no timeout.",
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

    run = run_structure_map_baseline(
        args.structure_input,
        args.out_dir,
        script_path=args.script,
        output_path=args.out,
        cutoff=args.cutoff,
        local_exclusion=args.local_exclusion,
        min_residues=args.min_residues,
        decision_column=args.decision_column,
        python_executable=args.python,
        extra_args=extra_args,
        timeout=args.timeout,
    )

    barrel_count = sum(result.result == "BARREL" for result in run.results)
    print(f"Generated maps: {len(run.generated_maps.records)}")
    print(f"Rows: {len(run.results)}")
    print(f"BARREL: {barrel_count}")
    print(f"NON_BARREL: {len(run.results) - barrel_count}")
    if run.output_path:
        print(f"Output: {run.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
