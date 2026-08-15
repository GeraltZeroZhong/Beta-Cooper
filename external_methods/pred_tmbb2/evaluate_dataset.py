from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.evaluation_common import (
    FILE_FIELDS,
    SUMMARY_FIELDS,
    apply_target_chain_labels,
    atomic_write_csv,
    atomic_write_text,
    checkout_identity,
    complete_manifest,
    executable_identity,
    fail_manifest,
    file_rows_from_chain_rows,
    file_state,
    files_inventory,
    fresh_run_directory,
    initialize_manifest,
    read_target_manifest,
    summary_markdown,
    summary_rows,
    update_running_manifest,
    validate_disjoint_labeled_inventories,
    validate_generated_source_coverage,
    validate_metric_contract,
    validate_prediction_rows,
)
from external_methods.pred_tmbb2.runner import (
    DEFAULT_MIN_TM_STRANDS,
    DEFAULT_PREDICTION_FIELD,
    PredTmbb2Result,
    run_baseline,
)
from external_methods.pred_tmbb2.sequences import (
    DEFAULT_MIN_RESIDUES,
    GeneratedFastaSet,
    GeneratedSequence,
    discover_structure_files,
    generate_structure_fasta,
)

BASELINE_NAME = "pred_tmbb2_single_juchmme"
DEFAULT_METRIC_LEVEL = "file"
NORMALIZED_FIELDS = [
    "baseline",
    "sample_id",
    "result",
    "score",
    "tm_strands",
    "decision_rule",
    "prediction_field",
    "reliability",
    "algorithm_score",
    "length",
    "logodds",
    "max_prob",
    "neg_logprob_per_length",
    "topology",
]
CHAIN_FIELDS = [
    "filename",
    "relative_path",
    "author_chain_id",
    "source_file",
    "sample_id",
    "baseline",
    "result",
    "pred_barrel",
    "decision_score",
    "decision_threshold",
    "chain_residue_count",
    "sequence_sha256",
    "sequence_source",
    "polymer_entity_id",
    "label_asym_id",
    "is_error",
    "is_filtered_out",
    "is_skip",
    "is_target_author_chain",
    "use_for_metrics",
    "y_true",
    "split",
    "pdb_id",
    "tm_strands",
    "prediction_field",
    "reliability",
    "algorithm_score",
    "length",
    "logodds",
    "max_prob",
    "neg_logprob_per_length",
    "topology",
]


@dataclass(frozen=True)
class SplitRun:
    split_name: str
    generated: GeneratedFastaSet
    results: list[PredTmbb2Result]


def _pdb_id_from_filename(filename: str) -> str:
    return Path(filename).stem.split("_", 1)[0].upper()


def _metadata_by_sample(generated: GeneratedFastaSet) -> dict[str, GeneratedSequence]:
    metadata = {record.sample_id: record for record in generated.records}
    if len(metadata) != len(generated.records):
        raise ValueError("Generated PRED-TMBB2 sequences contain duplicate sample IDs.")
    return metadata


def _chain_rows_for_split(
    run: SplitRun,
    *,
    min_tm_strands: int = DEFAULT_MIN_TM_STRANDS,
) -> list[dict[str, object]]:
    metadata = _metadata_by_sample(run.generated)
    result_by_sample = {result.sample_id: result for result in run.results}
    if len(result_by_sample) != len(run.results):
        raise ValueError("PRED-TMBB2 returned duplicate sample IDs.")
    missing_samples = sorted(set(metadata) - set(result_by_sample))
    unexpected_samples = sorted(set(result_by_sample) - set(metadata))
    if missing_samples or unexpected_samples:
        raise ValueError(
            "PRED-TMBB2 result identity does not match generated queries; "
            f"missing={missing_samples[:10]!r}, unexpected={unexpected_samples[:10]!r}."
        )

    rows: list[dict[str, object]] = []
    for sample_id, record in metadata.items():
        result = result_by_sample[sample_id]
        if result.result in {"BARREL", "NON_BARREL"}:
            if result.length is None or result.length != record.n_residues:
                raise ValueError(
                    f"PRED-TMBB2 output length for {sample_id!r} must equal the declared "
                    f"complete polymer-sequence length {record.n_residues}; observed "
                    f"{result.length!r}."
                )
            if len(result.topology) != record.n_residues:
                raise ValueError(
                    f"PRED-TMBB2 topology length for {sample_id!r} must equal the declared "
                    f"complete polymer-sequence length {record.n_residues}; observed "
                    f"{len(result.topology)}."
                )
            counted_strands = len(re.findall(r"M+", result.topology.upper()))
            if result.tm_strands != counted_strands or result.score != float(counted_strands):
                raise ValueError(
                    f"PRED-TMBB2 normalized strand count/score for {sample_id!r} is "
                    "inconsistent with its topology string."
                )
            expected_result = "BARREL" if counted_strands >= min_tm_strands else "NON_BARREL"
            if result.result != expected_result:
                raise ValueError(
                    f"PRED-TMBB2 result for {sample_id!r} is inconsistent with the frozen "
                    f"minimum-strand rule ({min_tm_strands})."
                )
        filename = Path(record.source_path).name
        rows.append(
            {
                "filename": filename,
                "relative_path": "",
                "author_chain_id": record.author_chain_id,
                "result": result.result,
                "pred_barrel": result.result == "BARREL",
                "decision_score": result.score,
                "decision_threshold": min_tm_strands,
                "chain_residue_count": record.n_residues,
                "sequence_sha256": record.sequence_sha256,
                "sequence_source": record.sequence_source,
                "polymer_entity_id": record.polymer_entity_id,
                "label_asym_id": record.label_asym_id,
                "y_true": "",
                "split": "",
                "is_error": False,
                "is_filtered_out": False,
                "is_skip": False,
                "use_for_metrics": False,
                "is_target_author_chain": False,
                "sample_id": result.sample_id,
                "baseline": BASELINE_NAME,
                "source_file": str(Path(record.source_path).resolve()),
                "pdb_id": _pdb_id_from_filename(filename),
                "tm_strands": result.tm_strands,
                "prediction_field": result.prediction_field,
                "reliability": result.reliability,
                "algorithm_score": result.algorithm_score,
                "length": result.length,
                "logodds": result.logodds,
                "max_prob": result.max_prob,
                "neg_logprob_per_length": result.neg_logprob_per_length,
                "topology": result.topology,
            }
        )
    validate_prediction_rows(rows)
    return rows


def _validate_parameters(
    *,
    min_residues: int,
    prediction_field: str,
    min_tm_strands: int,
    timeout: float | None,
) -> None:
    if isinstance(min_residues, bool) or not isinstance(min_residues, int) or min_residues <= 0:
        raise ValueError("min_residues must be a positive integer.")
    if prediction_field not in {"LP", "VP"}:
        raise ValueError("prediction_field must be LP or VP.")
    if (
        isinstance(min_tm_strands, bool)
        or not isinstance(min_tm_strands, int)
        or min_tm_strands <= 0
    ):
        raise ValueError("min_tm_strands must be a positive integer.")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("timeout must be finite and > 0 when provided.")


def run_dataset(
    positive_dir: Path,
    negative_dir: Path,
    output_root: Path,
    *,
    juchmme_dir: Path,
    positive_target_manifest: Path | None,
    negative_target_manifest: Path | None,
    metric_level: str,
    min_residues: int,
    prediction_field: str,
    min_tm_strands: int,
    java_executable: str,
    timeout: float | None,
    tag: str,
) -> Path:
    supplied_parameters = {
        "positive_dir": str(positive_dir),
        "negative_dir": str(negative_dir),
        "output_root": str(output_root),
        "juchmme_dir": str(juchmme_dir),
        "positive_target_manifest": (
            str(positive_target_manifest) if positive_target_manifest is not None else None
        ),
        "negative_target_manifest": (
            str(negative_target_manifest) if negative_target_manifest is not None else None
        ),
        "metric_level": metric_level,
        "min_residues": min_residues,
        "prediction_field": prediction_field,
        "min_tm_strands": min_tm_strands,
        "java_executable": java_executable,
        "timeout": timeout,
        "tag": tag,
    }
    run_dir = fresh_run_directory(output_root, baseline=BASELINE_NAME, tag=tag)
    manifest_path = run_dir / "evaluation_manifest.json"
    manifest = initialize_manifest(
        manifest_path,
        baseline=BASELINE_NAME,
        script_path=Path(__file__),
        supplied_parameters=supplied_parameters,
    )
    phase = "validation"
    try:
        update_running_manifest(manifest_path, manifest, phase=phase)
        validate_metric_contract(metric_level, positive_target_manifest, negative_target_manifest)
        normalized_prediction_field = str(prediction_field).upper()
        _validate_parameters(
            min_residues=min_residues,
            prediction_field=normalized_prediction_field,
            min_tm_strands=min_tm_strands,
            timeout=timeout,
        )
        roots = {
            "positive": positive_dir.expanduser().resolve(),
            "negative": negative_dir.expanduser().resolve(),
        }
        for label, root in roots.items():
            if not root.is_dir():
                raise NotADirectoryError(f"{label} input must be a directory: {root}")
        structure_files = {split: discover_structure_files(root) for split, root in roots.items()}
        inventories = {
            split: files_inventory(files, root=roots[split])
            for split, files in structure_files.items()
        }
        validate_disjoint_labeled_inventories(inventories["positive"], inventories["negative"])
        target_states = {
            "positive": (
                file_state(positive_target_manifest)
                if positive_target_manifest is not None
                else None
            ),
            "negative": (
                file_state(negative_target_manifest)
                if negative_target_manifest is not None
                else None
            ),
        }
        targets = {
            "positive": (
                read_target_manifest(
                    positive_target_manifest,
                    split_root=roots["positive"],
                    structure_files=structure_files["positive"],
                )
                if positive_target_manifest is not None
                else None
            ),
            "negative": (
                read_target_manifest(
                    negative_target_manifest,
                    split_root=roots["negative"],
                    structure_files=structure_files["negative"],
                )
                if negative_target_manifest is not None
                else None
            ),
        }
        java_identity = executable_identity(java_executable)
        checkout_state = checkout_identity(juchmme_dir)
        manifest["parameters"] = {
            **supplied_parameters,
            "positive_dir": str(roots["positive"]),
            "negative_dir": str(roots["negative"]),
            "juchmme_dir": checkout_state["path"],
            "java_executable": java_identity["path"],
            "prediction_field": normalized_prediction_field,
            "filtered_out_policy": "strict",
            "sequence_identity_schema": (
                "declared_complete_polymer_sequence_exact_author_chain_mapping"
            ),
        }
        code_dependencies = {
            "evaluation_common": file_state(
                Path(__file__).resolve().parents[1] / "evaluation_common.py"
            ),
            "runner": file_state(Path(__file__).with_name("runner.py")),
            "sequences": file_state(Path(__file__).with_name("sequences.py")),
        }
        manifest["code_dependencies"] = code_dependencies
        manifest["inputs"] = {
            "structure_inventories": inventories,
            "target_chain_manifests": target_states,
        }
        manifest["external_software"] = {
            "java": java_identity,
            "juchmme_checkout": checkout_state,
        }
        manifest["metric_sampling"] = {
            "file": "one_directory_labeled_structure_file_any_chain_prediction",
            "chain": (
                "one_manifest_target_chain_per_positive_and_negative_file"
                if targets["positive"] is not None
                else None
            ),
            "partner_chain_labels": "unlabeled",
            "error_policy": "fail_closed",
            "filtered_out_policy": "strict",
            "sequence_input": "pdb_seqres_or_complete_mmcif_entity_poly_seq",
        }
        phase = "input_preparation"
        update_running_manifest(manifest_path, manifest, phase=phase)

        split_runs: list[SplitRun] = []
        for split_name in ("positive", "negative"):
            phase = f"external_run_{split_name}"
            update_running_manifest(manifest_path, manifest, phase=phase)
            split_output = run_dir / split_name
            generated = generate_structure_fasta(
                roots[split_name], split_output, min_residues=min_residues
            )
            validate_generated_source_coverage(
                [record.source_path for record in generated.records],
                structure_files=structure_files[split_name],
                context=split_name,
            )
            results = run_baseline(
                generated.fasta_path,
                juchmme_dir=Path(str(checkout_state["path"])),
                work_dir=split_output / "juchmme_work",
                output_path=None,
                prediction_field=normalized_prediction_field,
                min_tm_strands=min_tm_strands,
                java_executable=str(java_identity["path"]),
                timeout=timeout,
            )
            atomic_write_csv(
                split_output / "normalized.csv",
                NORMALIZED_FIELDS,
                [result.as_row() for result in results],
            )
            split_runs.append(SplitRun(split_name, generated, results))

        phase = "metric_construction"
        update_running_manifest(manifest_path, manifest, phase=phase)
        all_predictions: list[dict[str, object]] = []
        all_target_rows: list[dict[str, object]] = []
        all_file_rows: list[dict[str, object]] = []
        for split_run in split_runs:
            raw_rows = _chain_rows_for_split(split_run, min_tm_strands=min_tm_strands)
            predictions, target_rows = apply_target_chain_labels(
                raw_rows,
                split=split_run.split_name,
                split_root=roots[split_run.split_name],
                targets=targets[split_run.split_name],
            )
            all_predictions.extend(predictions)
            all_target_rows.extend(target_rows)
            all_file_rows.extend(
                file_rows_from_chain_rows(
                    raw_rows,
                    split=split_run.split_name,
                    split_root=roots[split_run.split_name],
                    structure_files=structure_files[split_run.split_name],
                )
            )
        metric_rows = summary_rows(
            metric_level=metric_level,
            file_rows=all_file_rows,
            target_chain_rows=all_target_rows,
        )

        phase = "output_writing"
        update_running_manifest(manifest_path, manifest, phase=phase)
        atomic_write_csv(run_dir / "chain_predictions.csv", CHAIN_FIELDS, all_predictions)
        atomic_write_csv(run_dir / "file_results.csv", FILE_FIELDS, all_file_rows)
        if metric_level in {"chain", "both"}:
            atomic_write_csv(run_dir / "target_chain_results.csv", CHAIN_FIELDS, all_target_rows)
        atomic_write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, metric_rows)
        atomic_write_text(run_dir / "summary.md", summary_markdown(metric_rows))

        phase = "provenance_verification"
        update_running_manifest(manifest_path, manifest, phase=phase)
        current_inventories = {
            split: files_inventory(files, root=roots[split])
            for split, files in structure_files.items()
        }
        if current_inventories != inventories:
            raise RuntimeError("A structure input changed during PRED-TMBB2 evaluation.")
        for split in ("positive", "negative"):
            state = target_states[split]
            if state is not None and file_state(Path(state["path"])) != state:
                raise RuntimeError(f"The {split} target-chain manifest changed during evaluation.")
        if executable_identity(Path(str(java_identity["path"]))) != java_identity:
            raise RuntimeError("The Java executable changed during evaluation.")
        if checkout_identity(Path(str(checkout_state["path"]))) != checkout_state:
            raise RuntimeError("The JUCHMME checkout changed during evaluation.")
        if file_state(Path(__file__)) != manifest["script"]:
            raise RuntimeError("The PRED-TMBB2 evaluator script changed during evaluation.")
        current_code_dependencies = {
            "evaluation_common": file_state(
                Path(__file__).resolve().parents[1] / "evaluation_common.py"
            ),
            "runner": file_state(Path(__file__).with_name("runner.py")),
            "sequences": file_state(Path(__file__).with_name("sequences.py")),
        }
        if current_code_dependencies != code_dependencies:
            raise RuntimeError("A PRED-TMBB2 evaluator code dependency changed during evaluation.")
        complete_manifest(manifest_path, manifest)
    except Exception as exc:
        fail_manifest(manifest_path, manifest, phase=phase, error=exc)
        raise

    print(f"Chain predictions: {len(all_predictions)}")
    print(f"File observations: {len(all_file_rows)}")
    print(f"Target-chain observations: {len(all_target_rows)}")
    print(f"Run directory: {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/pred_tmbb2/evaluate_dataset.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate the PRED-TMBB2 single-sequence JUCHMME baseline on labeled Cooper-Beta "
            "structure directories."
        ),
        epilog=(
            "Output: a new timestamped directory under --out-dir with extracted FASTA, upstream "
            "predictions, normalized file and optional target-chain metrics, and run_manifest.json. "
            "Input or upstream execution failures exit with status 2."
        ),
    )
    parser.add_argument(
        "--positive-dir",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing structures labeled positive.",
    )
    parser.add_argument(
        "--negative-dir",
        required=True,
        metavar="DIRECTORY",
        help="Directory containing structures labeled negative.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="DIRECTORY",
        help="Output root; a fresh timestamped run directory is created.",
    )
    parser.add_argument(
        "--juchmme-dir",
        required=True,
        metavar="DIRECTORY",
        help="PRED-TMBB2 JUCHMME release or checkout directory.",
    )
    parser.add_argument(
        "--metric-level",
        choices=["file", "chain", "both"],
        default=DEFAULT_METRIC_LEVEL,
        help="Metric granularity; file metrics use directory labels and any-positive-chain output.",
    )
    parser.add_argument(
        "--positive-target-manifest",
        metavar="CSV",
        help=(
            "Positive target-selection CSV with exactly relative_path,author_chain_id; required "
            "with the negative target manifest for chain metrics."
        ),
    )
    parser.add_argument(
        "--negative-target-manifest",
        metavar="CSV",
        help=(
            "Negative target-selection CSV with exactly relative_path,author_chain_id; required "
            "with the positive target manifest for chain metrics."
        ),
    )
    parser.add_argument(
        "--tag",
        required=True,
        metavar="NAME",
        help="Run label included in the output directory name.",
    )
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        metavar="RESIDUES",
        help="Minimum declared complete sequence length required to evaluate a chain.",
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
    run_dataset(
        Path(args.positive_dir),
        Path(args.negative_dir),
        Path(args.out_dir),
        juchmme_dir=Path(args.juchmme_dir),
        positive_target_manifest=(
            Path(args.positive_target_manifest) if args.positive_target_manifest else None
        ),
        negative_target_manifest=(
            Path(args.negative_target_manifest) if args.negative_target_manifest else None
        ),
        metric_level=args.metric_level,
        min_residues=args.min_residues,
        prediction_field=args.prediction_field,
        min_tm_strands=args.min_tm_strands,
        java_executable=args.java,
        timeout=args.timeout,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
