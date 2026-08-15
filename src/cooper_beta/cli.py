from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException

if __package__ in {None, ""}:  # pragma: no cover - path execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cooper_beta._version import __version__
    from cooper_beta.bootstrap import configure_thread_environment
    from cooper_beta.config import build_config
    from cooper_beta.exceptions import CooperBetaError
    from cooper_beta.runtime import runtime_summary
else:
    from ._version import __version__
    from .bootstrap import configure_thread_environment
    from .config import build_config
    from .exceptions import CooperBetaError
    from .runtime import runtime_summary


def _looks_like_hydra_override(token: str) -> bool:
    return token.startswith(("+", "~")) or ("=" in token)


def _package_version() -> str:
    return __version__


def _has_override(overrides: list[str], key: str) -> bool:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    return any(override.startswith(prefixes) for override in overrides)


def _has_true_override(overrides: list[str], key: str) -> bool:
    for override in overrides:
        if not override.startswith((f"{key}=", f"+{key}=", f"++{key}=")):
            continue
        value = override.split("=", 1)[1].strip().lower()
        return value in {"1", "true", "yes", "on"}
    return False


def _reject_unknown_options(parser: argparse.ArgumentParser, tokens: list[str]) -> None:
    for token in tokens:
        if token.startswith("-"):
            parser.error(f"unrecognized argument: {token}")


def _recover_positional_path(
    parser: argparse.ArgumentParser,
    path: str | None,
    tokens: list[str],
) -> tuple[str | None, list[str]]:
    remaining: list[str] = []
    recovered_path = path
    for token in tokens:
        if not _looks_like_hydra_override(token) and not token.startswith("-"):
            if recovered_path is None:
                recovered_path = token
                continue
            parser.error(f"unexpected extra input path: {token}")
        remaining.append(token)
    return recovered_path, remaining


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cooper-beta",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Detect beta-barrel-like protein chains in PDB or mmCIF structures. A directory "
            "input is searched recursively. Options that use KEY=VALUE syntax override the "
            "default configuration. A coordinate-supported strand adjacency "
            "requires at least two C-alpha pairs within 6.8 Angstrom, with two distinct "
            "contacting residues on each strand. BARREL requires all three rule groups: "
            "strand_adjacency_count >= 8; cycle_strand_count >= 4 and "
            "cycle_strand_fraction >= 0.05; cycle_rank >= 1. The cycle-strand fields describe "
            "the largest closed component. Coordinate-only mmCIF "
            "inputs require one author chain; linked modified amino acids use a maximum C-N "
            "distance of 1.8 Angstrom "
            "(input.atom_site_only_max_peptide_bond_distance_angstrom=1.8)."
        ),
        epilog=(
            "Output: a chain-level results CSV and, by default, <CSV>.manifest.json with the "
            "resolved configuration and run provenance. Existing output files cause an error "
            "unless output.existing_artifact_policy=replace is supplied. Invalid configuration, "
            "unreadable input, DSSP failure, or analysis failure exits with status 2. Example: "
            "cooper-beta structures/ --out results.csv "
            "rules.cycle_strand_count_fraction.minimum_fraction=0.05"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="STRUCTURE_OR_DIRECTORY",
        help=(
            "Input .pdb, .cif, .mmcif, or gzip-compressed structure file, or a directory "
            "searched recursively. Required unless input.path=PATH is supplied."
        ),
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        metavar="N",
        help="Analysis worker processes; omit to use the configured CPU-based selection.",
    )
    parser.add_argument(
        "--prepare-workers",
        "--prep",
        type=int,
        default=None,
        metavar="N",
        help="Structure-preparation worker processes; omit to follow the resolved analysis count.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        metavar="CSV",
        help="Results CSV path; omit to use the configured output.csv_path.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Print the Python and DSSP versions without analyzing structures, then exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    args, hydra_overrides = parser.parse_known_args(argv)
    if args.path and _looks_like_hydra_override(args.path):
        hydra_overrides = [args.path, *hydra_overrides]
        args.path = None
    args.path, hydra_overrides = _recover_positional_path(
        parser,
        args.path,
        hydra_overrides,
    )
    _reject_unknown_options(parser, hydra_overrides)

    if (
        args.path is None
        and not args.check_env
        and not _has_override(hydra_overrides, "input.path")
        and not _has_true_override(hydra_overrides, "runtime.check_env")
    ):
        parser.error("the input path is required (or pass input.path=...)")

    try:
        cfg = build_config(hydra_overrides)
        if args.check_env or cfg.runtime.check_env:
            summary = runtime_summary(
                cfg.runtime.dssp_bin_path,
                require_dssp=bool(cfg.runtime.dssp_bin_path),
            )
            print(f"Python: {summary['python']} ({summary['python_executable']})")
            print(f"DSSP: {summary['dssp']}")
            if summary["dssp"] == "not found":
                raise SystemExit(2)
            return

        configure_thread_environment(cfg.runtime.native_threads_per_process)
        if __package__ in {None, ""}:  # pragma: no cover - path execution convenience
            from cooper_beta.pipeline import apply_runtime_overrides, run_pipeline_result
        else:
            from .pipeline import apply_runtime_overrides, run_pipeline_result

        cfg = apply_runtime_overrides(
            cfg,
            input_path=args.path,
            workers=args.workers,
            prepare_workers=args.prepare_workers,
            out_csv=args.out,
        )
        run_pipeline_result(cfg, write_csv=True, print_summary=True, strict_input=True)
    except (
        CooperBetaError,
        FileNotFoundError,
        HydraException,
        OmegaConfBaseException,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
