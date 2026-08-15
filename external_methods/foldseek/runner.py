from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import is_aa

if __package__ in {None, ""}:  # pragma: no cover - exercised by subprocess CLI test
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.foldseek.structures import _parse_structure, discover_structure_files

BASELINE_NAME = "foldseek_tmalign_structure_search"
DEFAULT_ALIGNMENT_TYPE = 1
DEFAULT_SCORE_MODE = "min_qtmscore_ttmscore"
DEFAULT_SCORE_THRESHOLD = 0.50
DEFAULT_MIN_QUERY_COVERAGE = 0.60
DEFAULT_MIN_TARGET_COVERAGE = 0.60
DEFAULT_EVALUE = 10.0
DEFAULT_MAX_SEQS = 1000
DEFAULT_FORMAT_FIELDS = [
    "query",
    "target",
    "qlen",
    "tlen",
    "alnlen",
    "qcov",
    "tcov",
    "qtmscore",
    "ttmscore",
    "alntmscore",
    "evalue",
    "bits",
]
SUPPORTED_SCORE_MODES = {
    "min_qtmscore_ttmscore",
    "qtmscore",
    "ttmscore",
    "alntmscore",
}
STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
FOLDSEEK_DB_SIDECARE_SUFFIXES = (
    ".dbtype",
    ".index",
    ".lookup",
    ".source",
    ".seq",
    ".ca",
    "_h",
)


@dataclass(frozen=True)
class FoldseekHit:
    query: str
    target: str
    qlen: int | None
    tlen: int | None
    alnlen: int | None
    qcov: float | None
    tcov: float | None
    qtmscore: float | None
    ttmscore: float | None
    alntmscore: float | None
    evalue: float | None
    bits: float | None


@dataclass(frozen=True)
class FoldseekResult:
    sample_id: str
    result: str
    score: float
    decision_rule: str
    score_mode: str
    score_threshold: float
    min_query_coverage: float
    min_target_coverage: float
    hit_count: int
    eligible_hit_count: int
    ignored_target_hit_count: int = 0
    best_target: str | None = None
    qlen: int | None = None
    tlen: int | None = None
    alnlen: int | None = None
    qcov: float | None = None
    tcov: float | None = None
    qtmscore: float | None = None
    ttmscore: float | None = None
    alntmscore: float | None = None
    evalue: float | None = None
    bits: float | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "baseline": BASELINE_NAME,
            "sample_id": self.sample_id,
            "result": self.result,
            "score": self.score,
            "decision_rule": self.decision_rule,
            "score_mode": self.score_mode,
            "score_threshold": self.score_threshold,
            "min_query_coverage": self.min_query_coverage,
            "min_target_coverage": self.min_target_coverage,
            "hit_count": self.hit_count,
            "eligible_hit_count": self.eligible_hit_count,
            "ignored_target_hit_count": self.ignored_target_hit_count,
            "best_target": self.best_target,
            "qlen": self.qlen,
            "tlen": self.tlen,
            "alnlen": self.alnlen,
            "qcov": self.qcov,
            "tcov": self.tcov,
            "qtmscore": self.qtmscore,
            "ttmscore": self.ttmscore,
            "alntmscore": self.alntmscore,
            "evalue": self.evalue,
            "bits": self.bits,
        }


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _foldseek_db_prefix_exists(path: Path) -> bool:
    if path.exists():
        return True
    path_string = str(path)
    return any(Path(f"{path_string}{suffix}").exists() for suffix in FOLDSEEK_DB_SIDECARE_SUFFIXES)


def _require_foldseek_input(path: str | Path, label: str) -> Path:
    expanded = Path(path).expanduser()
    resolved = expanded.resolve()
    if not _foldseek_db_prefix_exists(expanded) and not _foldseek_db_prefix_exists(resolved):
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _infer_query_ids(query_structures: Path) -> list[str] | None:
    try:
        structure_paths = discover_structure_files(query_structures)
    except (FileNotFoundError, ValueError):
        return None

    query_ids: list[str] = []
    for structure_path in structure_paths:
        try:
            structure = _parse_structure(structure_path)
            model: Model = structure[0]
        except Exception:
            continue
        chains: Iterable[Chain] = model.get_chains()
        for chain in chains:
            residues = [
                residue
                for residue in chain.get_unpacked_list()
                if is_aa(residue, standard=False) and "CA" in residue
            ]
            if residues:
                query_ids.append(f"{structure_path.name}_{chain.id}")
    return query_ids or None


def _read_query_ids(path: str | Path) -> list[str]:
    id_path = _require_path(path, "Foldseek query id list")
    ids = [
        line.strip()
        for line in id_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not ids:
        raise ValueError(f"Foldseek query id list is empty: {id_path}")
    return ids


def _foldseek_prefix(
    foldseek_executable: str | Path | None,
    command_prefix: Sequence[str] | None,
) -> list[str]:
    if command_prefix is not None:
        return list(command_prefix)
    if foldseek_executable is None:
        foldseek_executable = os.environ.get("FOLDSEEK_BIN", "foldseek")
    executable_path = Path(foldseek_executable).expanduser()
    if executable_path.exists():
        return [str(executable_path.resolve())]
    if shutil.which(str(foldseek_executable)) is None:
        raise FileNotFoundError(
            f"Foldseek executable was not found: {foldseek_executable}. "
            "Install Foldseek, set FOLDSEEK_BIN, or pass --foldseek."
        )
    return [str(foldseek_executable)]


def _run_foldseek_command(
    args: Sequence[str],
    *,
    work_dir: Path,
    foldseek_executable: str | Path | None,
    command_prefix: Sequence[str] | None,
    timeout: float | None,
) -> None:
    command = [
        *_foldseek_prefix(foldseek_executable, command_prefix),
        *[str(arg) for arg in args],
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
            "Foldseek baseline failed with exit code "
            f"{completed.returncode} while running: {' '.join(command)}\n"
            f"stdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}"
        )


def _row_to_hit(row: Mapping[str, str]) -> FoldseekHit:
    return FoldseekHit(
        query=(row.get("query") or "").strip(),
        target=(row.get("target") or "").strip(),
        qlen=_parse_int(row.get("qlen")),
        tlen=_parse_int(row.get("tlen")),
        alnlen=_parse_int(row.get("alnlen")),
        qcov=_parse_float(row.get("qcov")),
        tcov=_parse_float(row.get("tcov")),
        qtmscore=_parse_float(row.get("qtmscore")),
        ttmscore=_parse_float(row.get("ttmscore")),
        alntmscore=_parse_float(row.get("alntmscore")),
        evalue=_parse_float(row.get("evalue")),
        bits=_parse_float(row.get("bits")),
    )


def load_hits_tsv(
    hits_path: str | Path,
    *,
    fields: Sequence[str] = DEFAULT_FORMAT_FIELDS,
) -> list[FoldseekHit]:
    hits_file = _require_path(hits_path, "Foldseek hits TSV")
    hits: list[FoldseekHit] = []

    with hits_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, values in enumerate(reader, start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != len(fields):
                raise ValueError(
                    f"Foldseek hits row {line_number} has {len(values)} columns; "
                    f"expected {len(fields)} for fields {','.join(fields)!r}."
                )
            row = dict(zip(fields, values, strict=True))
            hit = _row_to_hit(row)
            if not hit.query:
                raise ValueError(f"Foldseek hits row {line_number} is missing query id.")
            hits.append(hit)

    return hits


def _score_hit(hit: FoldseekHit, score_mode: str) -> float:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError("score_mode must be one of: " + ", ".join(sorted(SUPPORTED_SCORE_MODES)))
    if score_mode == "qtmscore":
        return float(hit.qtmscore or 0.0)
    if score_mode == "ttmscore":
        return float(hit.ttmscore or 0.0)
    if score_mode == "alntmscore":
        return float(hit.alntmscore or 0.0)
    return min(float(hit.qtmscore or 0.0), float(hit.ttmscore or 0.0))


def _passes_coverage(
    hit: FoldseekHit,
    *,
    min_query_coverage: float,
    min_target_coverage: float,
) -> bool:
    return (
        float(hit.qcov or 0.0) >= min_query_coverage
        and float(hit.tcov or 0.0) >= min_target_coverage
    )


def _decision_rule(
    *,
    score_mode: str,
    score_threshold: float,
    min_query_coverage: float,
    min_target_coverage: float,
) -> str:
    return (
        f"{score_mode}>={score_threshold:g};"
        f"qcov>={min_query_coverage:g};"
        f"tcov>={min_target_coverage:g}"
    )


def _result_from_best_hit(
    sample_id: str,
    hits: Sequence[FoldseekHit],
    *,
    score_mode: str,
    score_threshold: float,
    min_query_coverage: float,
    min_target_coverage: float,
    ignored_target_hit_count: int = 0,
) -> FoldseekResult:
    rule = _decision_rule(
        score_mode=score_mode,
        score_threshold=score_threshold,
        min_query_coverage=min_query_coverage,
        min_target_coverage=min_target_coverage,
    )
    eligible_hits = [
        hit
        for hit in hits
        if _passes_coverage(
            hit,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
        )
    ]
    best_pool = eligible_hits or list(hits)
    best_hit = max(best_pool, key=lambda hit: _score_hit(hit, score_mode), default=None)
    if best_hit is None:
        return FoldseekResult(
            sample_id=sample_id,
            result="NON_BARREL",
            score=0.0,
            decision_rule=rule,
            score_mode=score_mode,
            score_threshold=score_threshold,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
            hit_count=0,
            eligible_hit_count=0,
            ignored_target_hit_count=ignored_target_hit_count,
        )

    score = _score_hit(best_hit, score_mode)
    result = "BARREL" if best_hit in eligible_hits and score >= score_threshold else "NON_BARREL"
    return FoldseekResult(
        sample_id=sample_id,
        result=result,
        score=score,
        decision_rule=rule,
        score_mode=score_mode,
        score_threshold=score_threshold,
        min_query_coverage=min_query_coverage,
        min_target_coverage=min_target_coverage,
        hit_count=len(hits),
        eligible_hit_count=len(eligible_hits),
        ignored_target_hit_count=ignored_target_hit_count,
        best_target=best_hit.target,
        qlen=best_hit.qlen,
        tlen=best_hit.tlen,
        alnlen=best_hit.alnlen,
        qcov=best_hit.qcov,
        tcov=best_hit.tcov,
        qtmscore=best_hit.qtmscore,
        ttmscore=best_hit.ttmscore,
        alntmscore=best_hit.alntmscore,
        evalue=best_hit.evalue,
        bits=best_hit.bits,
    )


def summarize_hits(
    hits: Sequence[FoldseekHit],
    *,
    query_ids: Sequence[str] | None = None,
    query_aliases: Mapping[str, str] | None = None,
    target_aliases: Mapping[str, str] | None = None,
    ignore_target_ids_by_query: Mapping[str, set[str]] | None = None,
    score_mode: str = DEFAULT_SCORE_MODE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_query_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
    min_target_coverage: float = DEFAULT_MIN_TARGET_COVERAGE,
) -> list[FoldseekResult]:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError("score_mode must be one of: " + ", ".join(sorted(SUPPORTED_SCORE_MODES)))

    aliases = dict(query_aliases or {})
    target_alias_map = dict(target_aliases or {})
    ignored_targets = {
        sample_id: set(target_ids)
        for sample_id, target_ids in (ignore_target_ids_by_query or {}).items()
    }
    grouped: dict[str, list[FoldseekHit]] = defaultdict(list)
    ignored_counts: Counter[str] = Counter()
    for hit in hits:
        sample_id = aliases.get(hit.query, hit.query)
        target_id = target_alias_map.get(hit.target, hit.target)
        if target_id in ignored_targets.get(sample_id, set()):
            ignored_counts[sample_id] += 1
            continue
        grouped[sample_id].append(
            hit if target_id == hit.target else replace(hit, target=target_id)
        )

    if query_ids is None:
        ordered_ids = sorted(grouped)
    else:
        ordered_ids = list(query_ids)

    return [
        _result_from_best_hit(
            sample_id,
            grouped.get(sample_id, []),
            score_mode=score_mode,
            score_threshold=score_threshold,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
            ignored_target_hit_count=ignored_counts[sample_id],
        )
        for sample_id in ordered_ids
    ]


def write_results_csv(results: Iterable[FoldseekResult], output_path: str | Path) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "baseline",
        "sample_id",
        "result",
        "score",
        "decision_rule",
        "score_mode",
        "score_threshold",
        "min_query_coverage",
        "min_target_coverage",
        "hit_count",
        "eligible_hit_count",
        "ignored_target_hit_count",
        "best_target",
        "qlen",
        "tlen",
        "alnlen",
        "qcov",
        "tcov",
        "qtmscore",
        "ttmscore",
        "alntmscore",
        "evalue",
        "bits",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_row())

    return output_file


def _prepare_target(
    target: str | Path,
    *,
    run_dir: Path,
    build_target_db: bool,
    create_index: bool,
    foldseek_executable: str | Path | None,
    command_prefix: Sequence[str] | None,
    timeout: float | None,
) -> Path:
    if not build_target_db:
        return _require_foldseek_input(target, "Foldseek target DB or structures")

    target_path = _require_path(target, "Foldseek reference structures")
    target_db = run_dir / "targetDB"
    _run_foldseek_command(
        ["createdb", str(target_path), str(target_db)],
        work_dir=run_dir,
        foldseek_executable=foldseek_executable,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    if create_index:
        index_tmp = run_dir / "target_index_tmp"
        _run_foldseek_command(
            ["createindex", str(target_db), str(index_tmp)],
            work_dir=run_dir,
            foldseek_executable=foldseek_executable,
            command_prefix=command_prefix,
            timeout=timeout,
        )
    return target_db


def _run_in_directory(
    query_structures: Path,
    target: Path,
    *,
    run_dir: Path,
    foldseek_executable: str | Path | None,
    command_prefix: Sequence[str] | None,
    alignment_type: int,
    evalue: float,
    max_seqs: int,
    extra_args: Sequence[str] | None,
    timeout: float | None,
) -> Path:
    hits_path = run_dir / "foldseek_hits.tsv"
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "easy-search",
        str(query_structures),
        str(target),
        str(hits_path),
        str(tmp_dir),
        "--alignment-type",
        str(alignment_type),
        "--format-output",
        ",".join(DEFAULT_FORMAT_FIELDS),
        "-e",
        str(evalue),
        "--max-seqs",
        str(max_seqs),
        *(extra_args or []),
    ]
    _run_foldseek_command(
        command,
        work_dir=run_dir,
        foldseek_executable=foldseek_executable,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    return hits_path


def run_baseline(
    query_structures: str | Path,
    target: str | Path,
    *,
    foldseek_executable: str | Path | None = None,
    work_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    query_ids: Sequence[str] | None = None,
    query_aliases: Mapping[str, str] | None = None,
    target_aliases: Mapping[str, str] | None = None,
    ignore_target_ids_by_query: Mapping[str, set[str]] | None = None,
    build_target_db: bool = False,
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
) -> list[FoldseekResult]:
    query_path = _require_foldseek_input(query_structures, "Foldseek query structures")
    resolved_query_ids = list(query_ids) if query_ids is not None else _infer_query_ids(query_path)

    def run_in(run_dir: Path) -> list[FoldseekResult]:
        target_input = _prepare_target(
            target,
            run_dir=run_dir,
            build_target_db=build_target_db,
            create_index=create_index,
            foldseek_executable=foldseek_executable,
            command_prefix=command_prefix,
            timeout=timeout,
        )
        hits_path = _run_in_directory(
            query_path,
            target_input,
            run_dir=run_dir,
            foldseek_executable=foldseek_executable,
            command_prefix=command_prefix,
            alignment_type=alignment_type,
            evalue=evalue,
            max_seqs=max_seqs,
            extra_args=extra_args,
            timeout=timeout,
        )
        hits = load_hits_tsv(hits_path)
        return summarize_hits(
            hits,
            query_ids=resolved_query_ids,
            query_aliases=query_aliases,
            target_aliases=target_aliases,
            ignore_target_ids_by_query=ignore_target_ids_by_query,
            score_mode=score_mode,
            score_threshold=score_threshold,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
        )

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="cooper_beta_foldseek_") as temp_dir:
            results = run_in(Path(temp_dir))
    else:
        run_dir = Path(work_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        results = run_in(run_dir)

    if output_path is not None:
        write_results_csv(results, output_path)
    return results


def _print_summary(results: Sequence[FoldseekResult], output_path: str | Path | None) -> None:
    counts = Counter(result.result for result in results)
    print(f"Rows: {len(results)}")
    print(f"BARREL: {counts.get('BARREL', 0)}")
    print(f"NON_BARREL: {counts.get('NON_BARREL', 0)}")
    if output_path is not None:
        print(f"Output: {Path(output_path).expanduser().resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python external_methods/foldseek/runner.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run the Foldseek global-TMalign baseline and normalize one BARREL or NON_BARREL "
            "decision per query. A query is positive when an eligible hit reaches the selected "
            "score threshold and both coverage thresholds."
        ),
        epilog=(
            "Output: a row count and class counts on stdout, plus a normalized CSV when --out is "
            "set. The temporary work directory is removed unless --work-dir is set. Arguments "
            "after -- are passed to Foldseek easy-search. Input, executable, subprocess, and "
            "result-format errors exit with status 2."
        ),
    )
    parser.add_argument(
        "query_structures",
        metavar="QUERY",
        help="PDB, CIF, or mmCIF query file or directory, or a Foldseek query database prefix.",
    )
    parser.add_argument(
        "target",
        metavar="TARGET",
        help="Foldseek target database prefix or reference structure path.",
    )
    parser.add_argument(
        "--foldseek",
        metavar="COMMAND",
        help="Foldseek executable; omit to use FOLDSEEK_BIN, then foldseek on PATH.",
    )
    parser.add_argument(
        "--work-dir",
        metavar="DIRECTORY",
        help="Persistent working directory for the Foldseek database and raw hit TSV.",
    )
    parser.add_argument(
        "--out",
        metavar="CSV",
        help="Normalized result CSV; omit to print counts without retaining normalized rows.",
    )
    parser.add_argument(
        "--query-id-list",
        metavar="FILE",
        help=(
            "Optional text file with one expected query id per line. "
            "Used to emit NON_BARREL rows for queries with no Foldseek hits."
        ),
    )
    parser.add_argument(
        "--build-target-db",
        action="store_true",
        help="Run foldseek createdb on the target path before searching.",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Run foldseek createindex after createdb. Used only with --build-target-db.",
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

    results = run_baseline(
        args.query_structures,
        args.target,
        foldseek_executable=args.foldseek,
        work_dir=args.work_dir,
        output_path=args.out,
        query_ids=_read_query_ids(args.query_id_list) if args.query_id_list else None,
        build_target_db=args.build_target_db,
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
    _print_summary(results, args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
