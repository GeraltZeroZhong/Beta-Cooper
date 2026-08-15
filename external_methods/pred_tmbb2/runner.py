from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

BASELINE_NAME = "pred_tmbb2_single_juchmme"
DEFAULT_MIN_TM_STRANDS = 3
DEFAULT_PREDICTION_FIELD = "LP"

DEFAULT_CLASSPATH = "bin"
DEFAULT_TRANSITIONS = "tables/A_TMBB2_TRAINED"
DEFAULT_EMISSIONS = "tables/E_TMBB2_TRAINED"
DEFAULT_MODEL = "models/tmbb2.mdel"
DEFAULT_CONFIG = "conf/conf.tmbb"

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


@dataclass(frozen=True)
class PredTmbb2Result:
    sample_id: str
    result: str
    score: float
    tm_strands: int
    decision_rule: str
    prediction_field: str
    topology: str
    reliability: float | None = None
    algorithm_score: float | None = None
    length: int | None = None
    logodds: float | None = None
    max_prob: float | None = None
    neg_logprob_per_length: float | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "baseline": BASELINE_NAME,
            "sample_id": self.sample_id,
            "result": self.result,
            "score": self.score,
            "tm_strands": self.tm_strands,
            "decision_rule": self.decision_rule,
            "prediction_field": self.prediction_field,
            "reliability": self.reliability,
            "algorithm_score": self.algorithm_score,
            "length": self.length,
            "logodds": self.logodds,
            "max_prob": self.max_prob,
            "neg_logprob_per_length": self.neg_logprob_per_length,
            "topology": self.topology,
        }


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_int_float(value: str | None) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    if not math.isfinite(parsed) or not parsed.is_integer() or parsed <= 0.0:
        raise ValueError("PRED-TMBB2 sequence length must be a positive integer.")
    return int(parsed)


def _parse_metric(line: str, name: str) -> str | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*({_FLOAT_PATTERN})", line)
    return match.group(1) if match else None


def _count_tm_strands(topology: str) -> int:
    return len(re.findall(r"M+", topology.upper()))


def _score_to_result(tm_strands: int, min_tm_strands: int) -> str:
    return "BARREL" if tm_strands >= min_tm_strands else "NON_BARREL"


def _clean_sample_id(header: str) -> str:
    cleaned = header.strip()
    if cleaned.startswith(">"):
        cleaned = cleaned[1:].strip()
    return cleaned.split(None, 1)[0]


def _record_to_result(
    record: Mapping[str, str | None],
    *,
    prediction_field: str,
    min_tm_strands: int,
) -> PredTmbb2Result:
    sample_id = str(record.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("PRED-TMBB2 output record is missing an ID line.")

    topology = str(record.get(prediction_field) or "").strip()
    if not topology:
        raise ValueError(f"PRED-TMBB2 output for {sample_id!r} is missing {prediction_field}.")

    tm_strands = _count_tm_strands(topology)
    reliability = _parse_float(record.get(f"{prediction_field[0]}R"))
    algorithm_score = _parse_float(record.get(f"{prediction_field[0]}S"))
    decision_rule = f"{prediction_field}_tm_strands>={min_tm_strands}"

    return PredTmbb2Result(
        sample_id=sample_id,
        result=_score_to_result(tm_strands, min_tm_strands),
        score=float(tm_strands),
        tm_strands=tm_strands,
        decision_rule=decision_rule,
        prediction_field=prediction_field,
        topology=topology,
        reliability=reliability,
        algorithm_score=algorithm_score,
        length=_parse_int_float(record.get("length")),
        logodds=_parse_float(record.get("logodds")),
        max_prob=_parse_float(record.get("max_prob")),
        neg_logprob_per_length=_parse_float(record.get("neg_logprob_per_length")),
    )


def parse_juchmme_stdout(
    stdout: str,
    *,
    prediction_field: str = DEFAULT_PREDICTION_FIELD,
    min_tm_strands: int = DEFAULT_MIN_TM_STRANDS,
) -> list[PredTmbb2Result]:
    field = prediction_field.upper()
    if field not in {"LP", "VP"}:
        raise ValueError("prediction_field must be one of: LP, VP")
    if min_tm_strands < 1:
        raise ValueError("min_tm_strands must be >= 1")

    records: list[dict[str, str | None]] = []
    current: dict[str, str | None] | None = None

    def finish_current() -> None:
        if current is not None:
            records.append(current.copy())

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("ID:"):
            finish_current()
            current = {"sample_id": _clean_sample_id(line.partition(":")[2])}
            continue
        if current is None:
            continue
        if line.startswith("CC:"):
            current["length"] = _parse_metric(line, "len")
            current["logodds"] = _parse_metric(line, "logodds")
            current["max_prob"] = _parse_metric(line, "maxProb")
            current["neg_logprob_per_length"] = _parse_metric(line, "(-logprob/lng)")
            continue

        key, separator, value = line.partition(":")
        if separator and key in {"VS", "VR", "VP", "LS", "LR", "LP"}:
            current[key] = value.strip()

    finish_current()
    if not records:
        raise ValueError("No PRED-TMBB2 records were found in JUCHMME output.")

    return [
        _record_to_result(
            record,
            prediction_field=field,
            min_tm_strands=min_tm_strands,
        )
        for record in records
    ]


def write_results_csv(results: Iterable[PredTmbb2Result], output_path: str | Path) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_row())

    return output_file


def _require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_juchmme_dir(juchmme_dir: str | Path | None) -> Path:
    if juchmme_dir is None:
        juchmme_dir = os.environ.get("PRED_TMBB2_JUCHMME_DIR")
    if juchmme_dir is None:
        raise ValueError(
            "Provide a JUCHMME checkout/release directory with --juchmme-dir "
            "or PRED_TMBB2_JUCHMME_DIR."
        )
    root = _require_path(juchmme_dir, "JUCHMME directory")
    if not root.is_dir():
        raise NotADirectoryError(f"JUCHMME path is not a directory: {root}")
    return root


def _resolve_child(root: Path, path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return _require_path(candidate, label)


def _build_juchmme_args(
    fasta_path: Path,
    *,
    transitions_path: Path,
    emissions_path: Path,
    model_path: Path,
    config_path: Path,
) -> list[str]:
    return [
        "-f",
        str(fasta_path),
        "-a",
        str(transitions_path),
        "-e",
        str(emissions_path),
        "-m",
        str(model_path),
        "-c",
        str(config_path),
    ]


def _invoke_juchmme(
    fasta_path: Path,
    *,
    juchmme_dir: str | Path | None,
    java_executable: str,
    classpath: str | Path,
    transitions: str | Path,
    emissions: str | Path,
    model: str | Path,
    config: str | Path,
    work_dir: Path,
    command_prefix: Sequence[str] | None,
    timeout: float | None,
) -> str:
    if command_prefix is None:
        root = _resolve_juchmme_dir(juchmme_dir)
        classpath_path = _resolve_child(root, classpath, "JUCHMME classpath")
        command_start = [java_executable, "-cp", str(classpath_path), "hmm.Juchmme"]
        transitions_path = _resolve_child(root, transitions, "PRED-TMBB2 transition table")
        emissions_path = _resolve_child(root, emissions, "PRED-TMBB2 emission table")
        model_path = _resolve_child(root, model, "PRED-TMBB2 model")
        config_path = _resolve_child(root, config, "PRED-TMBB2 config")
    else:
        command_start = list(command_prefix)
        if juchmme_dir is None:
            transitions_path = Path(transitions).expanduser()
            emissions_path = Path(emissions).expanduser()
            model_path = Path(model).expanduser()
            config_path = Path(config).expanduser()
        else:
            root = _resolve_juchmme_dir(juchmme_dir)
            transitions_path = _resolve_child(root, transitions, "PRED-TMBB2 transition table")
            emissions_path = _resolve_child(root, emissions, "PRED-TMBB2 emission table")
            model_path = _resolve_child(root, model, "PRED-TMBB2 model")
            config_path = _resolve_child(root, config, "PRED-TMBB2 config")

    command = [
        *command_start,
        *_build_juchmme_args(
            fasta_path,
            transitions_path=transitions_path,
            emissions_path=emissions_path,
            model_path=model_path,
            config_path=config_path,
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=work_dir,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "PRED-TMBB2/JUCHMME baseline failed with exit code "
            f"{completed.returncode}.\nstdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}"
        )
    return completed.stdout


def run_baseline(
    fasta_path: str | Path,
    *,
    juchmme_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    java_executable: str | None = None,
    classpath: str | Path = DEFAULT_CLASSPATH,
    transitions: str | Path = DEFAULT_TRANSITIONS,
    emissions: str | Path = DEFAULT_EMISSIONS,
    model: str | Path = DEFAULT_MODEL,
    config: str | Path = DEFAULT_CONFIG,
    prediction_field: str = DEFAULT_PREDICTION_FIELD,
    min_tm_strands: int = DEFAULT_MIN_TM_STRANDS,
    command_prefix: Sequence[str] | None = None,
    timeout: float | None = None,
) -> list[PredTmbb2Result]:
    fasta_file = _require_path(fasta_path, "FASTA input")
    py_exec = java_executable or "java"

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="cooper_beta_pred_tmbb2_") as temp_dir:
            run_dir = Path(temp_dir)
            stdout = _invoke_juchmme(
                fasta_file,
                juchmme_dir=juchmme_dir,
                java_executable=py_exec,
                classpath=classpath,
                transitions=transitions,
                emissions=emissions,
                model=model,
                config=config,
                work_dir=run_dir,
                command_prefix=command_prefix,
                timeout=timeout,
            )
    else:
        run_dir = Path(work_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout = _invoke_juchmme(
            fasta_file,
            juchmme_dir=juchmme_dir,
            java_executable=py_exec,
            classpath=classpath,
            transitions=transitions,
            emissions=emissions,
            model=model,
            config=config,
            work_dir=run_dir,
            command_prefix=command_prefix,
            timeout=timeout,
        )

    results = parse_juchmme_stdout(
        stdout,
        prediction_field=prediction_field,
        min_tm_strands=min_tm_strands,
    )
    if output_path is not None:
        write_results_csv(results, output_path)
    return results


def _print_summary(results: Sequence[PredTmbb2Result], output_path: str | Path | None) -> None:
    counts = Counter(result.result for result in results)
    print(f"Rows: {len(results)}")
    print(f"BARREL: {counts.get('BARREL', 0)}")
    print(f"NON_BARREL: {counts.get('NON_BARREL', 0)}")
    if output_path is not None:
        print(f"Output: {Path(output_path).expanduser().resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/pred_tmbb2/runner.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run the PRED-TMBB2 single-sequence JUCHMME baseline and normalize one BARREL or "
            "NON_BARREL decision per FASTA record."
        ),
        epilog=(
            "Output: class counts on stdout and normalized rows when --out is set. A sequence is "
            "BARREL when the selected topology field contains at least --min-tm-strands complete "
            "membrane beta-strand segments. Input or upstream execution failures exit with status 2."
        ),
    )
    parser.add_argument(
        "fasta", metavar="FASTA", help="FASTA file containing one or more protein sequences."
    )
    parser.add_argument(
        "--juchmme-dir",
        metavar="DIRECTORY",
        help="JUCHMME release/checkout directory. Defaults to PRED_TMBB2_JUCHMME_DIR.",
    )
    parser.add_argument(
        "--work-dir",
        metavar="DIRECTORY",
        help="Persistent working directory; omit to use and remove a temporary directory.",
    )
    parser.add_argument(
        "--out",
        metavar="CSV",
        help="Normalized result CSV; omit to print counts without retaining normalized rows.",
    )
    parser.add_argument(
        "--java",
        default="java",
        metavar="COMMAND",
        help="Java executable used to run JUCHMME.",
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
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Maximum elapsed time for the JUCHMME subprocess; omit for no timeout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    results = run_baseline(
        args.fasta,
        juchmme_dir=args.juchmme_dir,
        work_dir=args.work_dir,
        output_path=args.out,
        java_executable=args.java,
        prediction_field=args.prediction_field.upper(),
        min_tm_strands=args.min_tm_strands,
        timeout=args.timeout,
    )
    _print_summary(results, args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
