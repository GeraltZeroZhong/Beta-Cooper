from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - path execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cooper_beta.evaluation.runner import evaluate
else:
    from .runner import evaluate


def _run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cooper-beta-eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate Cooper-Beta on positive and negative structure directories. "
            "The command writes chain predictions, any-chain file predictions, metrics, "
            "and run metadata."
        ),
        epilog=(
            "Output: detector CSVs and manifests plus chain/file evaluation CSVs under "
            "--save-dir. Chain metrics require both target manifests. ERROR observations "
            "stop evaluation unless --metric-error-policy exclude is selected. Invalid "
            "inputs or processing failures exit with status 2."
        ),
    )
    parser.add_argument(
        "--positives",
        "--true",
        dest="true",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing structures labeled positive.",
    )
    parser.add_argument(
        "--negatives",
        "--false",
        dest="false",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing structures labeled negative.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Analysis worker processes; omit to use the detector configuration.",
    )
    parser.add_argument(
        "--prepare",
        "--prepare-workers",
        "--prep",
        type=int,
        default=None,
        metavar="N",
        help="Preparation worker processes; omit to follow the analysis count.",
    )
    parser.add_argument(
        "--save-dir",
        default="evaluation-results",
        metavar="DIRECTORY",
        help="Output directory for predictions, metrics, and manifests.",
    )
    parser.add_argument(
        "--metric-level",
        choices=["chain", "file", "both"],
        default="file",
        help="Metric granularity to compute.",
    )
    parser.add_argument(
        "--positive-manifest",
        default=None,
        metavar="CSV",
        help=(
            "Positive target-chain CSV with exact columns "
            "filename,source_path,structure_sha256,target_author_chain_id. Required with "
            "--negative-manifest for chain metrics."
        ),
    )
    parser.add_argument(
        "--negative-manifest",
        default=None,
        metavar="CSV",
        help=(
            "Negative target-chain CSV with exact columns "
            "filename,source_path,structure_sha256,target_author_chain_id. Required with "
            "--positive-manifest for chain metrics."
        ),
    )
    parser.add_argument(
        "--metric-error-policy",
        choices=["strict", "exclude"],
        default="strict",
        help="Treatment of detector ERROR observations.",
    )
    args = parser.parse_args(argv)

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    row = evaluate(
        true_dir=Path(args.true),
        false_dir=Path(args.false),
        workers=args.workers,
        prepare_workers=args.prepare,
        save_dir=Path(args.save_dir),
        metric_level=args.metric_level,
        tag=tag,
        print_metric_tables=True,
        metric_error_policy=args.metric_error_policy,
        positive_manifest=(
            Path(args.positive_manifest) if args.positive_manifest is not None else None
        ),
        negative_manifest=(
            Path(args.negative_manifest) if args.negative_manifest is not None else None
        ),
    )
    print("\nSaved evaluation artifacts:")
    print(f"  chain-level: {row['chain_csv']}")
    print(f"  file-level:  {row['file_csv']}")


def main(argv: list[str] | None = None) -> None:
    try:
        _run(argv)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
