from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bootstrap import configure_thread_environment
from .config import AppConfig, build_config, validate_config
from .constants import RESULT_ERROR, RESULT_STAGE_PREPARATION
from .exceptions import InputValidationError
from .integrity import (
    FrozenInputIdentity,
    freeze_input_identities,
    verify_input_identities,
)
from .logging_config import configure_logging
from .models import PipelineRunResult
from .output_artifacts import OutputArtifactTransaction, resolved_output_artifact_paths
from .provenance import write_run_manifest
from .runtime import require_dssp_binary

if TYPE_CHECKING:
    from .preparation import PrepareFailure

LOGGER = logging.getLogger(__name__)


def iter_prepared_payload_batches(*args: Any, **kwargs: Any) -> Iterator[list[dict[str, object]]]:
    """Import numerical worker code only after the run environment is configured."""
    from .execution import iter_prepared_payload_batches as implementation

    return implementation(*args, **kwargs)


def run_analysis_stream(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
    """Import numerical worker code only after the run environment is configured."""
    from .execution import run_analysis_stream as implementation

    return implementation(*args, **kwargs)


def write_results_csv(rows: list[dict[str, object]], output_path: str) -> None:
    from .results import write_results_csv as implementation

    implementation(rows, output_path)


def print_results_summary(
    results: list[dict[str, object]],
    output_path: str,
    **kwargs: Any,
) -> None:
    from .results import print_results_summary as implementation

    implementation(results, output_path, **kwargs)


def discover_input_files(
    input_path: str,
    allowed_suffixes: Sequence[str],
    *,
    strict: bool = False,
) -> list[str]:
    """Resolve a directory or single structure file into an explicit file list."""
    if not str(input_path).strip():
        raise InputValidationError("Input path is required.")
    path = Path(input_path).expanduser()
    normalized_suffixes = tuple(str(suffix).lower() for suffix in allowed_suffixes)

    def has_allowed_suffix(candidate: Path) -> bool:
        normalized_name = candidate.name.lower()
        return any(normalized_name.endswith(suffix) for suffix in normalized_suffixes)

    if not path.exists():
        if strict:
            raise InputValidationError(f"Input path does not exist: {path}")
        return [str(path)]

    if path.is_dir():
        files = sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and has_allowed_suffix(file_path)
        )
        if strict and not files:
            allowed = ", ".join(allowed_suffixes)
            raise InputValidationError(f"No structure files ({allowed}) were found in: {path}")
        return [str(file_path) for file_path in sorted(files)]

    if strict and normalized_suffixes and not has_allowed_suffix(path):
        allowed = ", ".join(allowed_suffixes)
        raise InputValidationError(
            f"Input file has an unsupported filename suffix: {path.name!r}. "
            f"Expected one of: {allowed}."
        )
    return [str(path)]


def os_cpu_count() -> int:
    import os

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return os.cpu_count() or 1


def resolve_analysis_worker_count(configured_workers: int | None, cpu_reserve: int) -> int:
    """Choose a sensible default analysis worker count from available CPUs."""
    if configured_workers is not None:
        return max(1, int(configured_workers))

    available_cpus = os_cpu_count()
    return max(1, available_cpus - max(0, cpu_reserve))


def resolve_prepare_worker_count(configured_workers: int | None, analysis_workers: int) -> int:
    if configured_workers is not None:
        return max(1, int(configured_workers))
    return max(1, analysis_workers)


def apply_runtime_overrides(
    cfg: AppConfig,
    *,
    input_path: str | None = None,
    workers: int | None = None,
    prepare_workers: int | None = None,
    out_csv: str | None = None,
) -> AppConfig:
    runtime = cfg.runtime
    input_config = cfg.input
    output_config = cfg.output
    if input_path is not None:
        input_config = replace(input_config, path=str(input_path))
    if workers is not None:
        runtime = replace(runtime, workers=int(workers))
    if prepare_workers is not None:
        runtime = replace(runtime, prepare_workers=int(prepare_workers))
    if out_csv is not None:
        output_config = replace(output_config, csv_path=str(out_csv))
    return replace(
        cfg,
        runtime=runtime,
        input=input_config,
        output=output_config,
    )


def _write_manifest_if_enabled(
    cfg: AppConfig,
    *,
    input_files: list[str],
    output_path: str,
    analysis_workers: int,
    prepare_workers: int,
    started_at_utc: str,
    run_id: str,
    input_identities: list[FrozenInputIdentity] | None = None,
) -> None:
    if not cfg.output.write_manifest:
        return
    write_run_manifest(
        config=cfg,
        input_files=input_files,
        input_identities=input_identities,
        input_identities_verified=input_identities is not None,
        output_path=output_path,
        hash_input_files=cfg.output.hash_input_files,
        resolved_analysis_workers=analysis_workers,
        resolved_prepare_workers=prepare_workers,
        started_at_utc=started_at_utc,
        run_id=run_id,
    )


def _validate_artifact_path_disjointness(
    cfg: AppConfig,
    input_identities: list[FrozenInputIdentity],
    *,
    write_csv: bool,
) -> None:
    """Reject every run artifact path that could overwrite an input or another artifact."""

    artifact_paths: dict[str, Path] = {}
    if write_csv:
        output_path, manifest_path, lock_path = resolved_output_artifact_paths(cfg.output.csv_path)
        artifact_paths.update(
            {
                "output CSV": output_path,
                "output manifest": manifest_path,
                "output lock": lock_path,
            }
        )
    if cfg.runtime.log_jsonl_path is not None:
        artifact_paths["JSONL log"] = Path(cfg.runtime.log_jsonl_path).expanduser().resolve()

    paths_to_labels: dict[Path, list[str]] = {}
    for label, path in artifact_paths.items():
        paths_to_labels.setdefault(path, []).append(label)
    duplicate_artifacts = {
        path: labels for path, labels in paths_to_labels.items() if len(labels) > 1
    }
    if duplicate_artifacts:
        details = "; ".join(
            f"{path}: {', '.join(labels)}"
            for path, labels in sorted(duplicate_artifacts.items(), key=lambda item: str(item[0]))
        )
        raise InputValidationError(f"Run artifact paths collide with each other: {details}")

    input_paths = {Path(identity.path) for identity in input_identities}
    collisions = [(label, path) for label, path in artifact_paths.items() if path in input_paths]
    if collisions:
        details = ", ".join(f"{label}={path}" for label, path in collisions)
        raise InputValidationError(
            "Run artifact paths must be disjoint from every frozen input structure; "
            f"refusing destructive collision(s): {details}"
        )


def _publish_results(
    rows: list[dict[str, object]],
    cfg: AppConfig,
    *,
    input_files: list[str],
    analysis_workers: int,
    prepare_workers: int,
    started_at_utc: str,
    input_identities: list[FrozenInputIdentity],
) -> None:
    """Publish a CSV and its provenance sidecar as one authenticated transaction."""
    _validate_artifact_path_disjointness(cfg, input_identities, write_csv=True)
    with OutputArtifactTransaction(
        cfg.output.csv_path,
        write_manifest=cfg.output.write_manifest,
        existing_artifact_policy=cfg.output.existing_artifact_policy,
        started_at_utc=started_at_utc,
    ) as transaction:
        write_results_csv(rows, str(transaction.output_path))
        transaction.record_csv_commit()
        _write_manifest_if_enabled(
            cfg,
            input_files=input_files,
            output_path=str(transaction.output_path),
            analysis_workers=analysis_workers,
            prepare_workers=prepare_workers,
            started_at_utc=started_at_utc,
            run_id=transaction.run_id,
            input_identities=input_identities,
        )
        transaction.mark_complete()


def _prepare_error_rows(
    failures: list[PrepareFailure],
) -> list[dict[str, object]]:
    """Convert file-level preparation failures into complete public result rows."""
    rows: list[dict[str, object]] = []
    for failure in failures:
        rows.append(
            {
                "filename": Path(failure.source_path).name,
                "source_path": failure.source_path,
                "author_chain_id": "",
                "result": RESULT_ERROR,
                "result_stage": RESULT_STAGE_PREPARATION,
                "dssp_unassigned_residue_count": 0,
                "strand_count": 0,
                "strand_adjacency_count": 0,
                "cycle_strand_count": 0,
                "cycle_strand_fraction": 0.0,
                "cycle_rank": 0,
                "reason": failure.message,
                "error_code": failure.error_code,
                "degraded": False,
            }
        )
    return rows


def _ordered_result_rows(
    rows: list[dict[str, object]],
    input_files: list[str],
) -> list[dict[str, object]]:
    """Return rows in input file order, then chain order, for reproducible output."""
    file_order: dict[str, int] = {}
    for index, file_path in enumerate(input_files):
        basename = Path(file_path).name
        file_order.setdefault(str(file_path), index)
        file_order.setdefault(str(Path(file_path).expanduser().resolve()), index)
        file_order.setdefault(basename, index)

    def sort_key(row: dict[str, object]) -> tuple[int, str, str, str, str]:
        filename = str(row.get("filename", ""))
        source_path = str(row.get("source_path", ""))
        primary_id = source_path or filename
        return (
            file_order.get(primary_id, file_order.get(filename, len(file_order))),
            source_path,
            filename,
            str(row.get("author_chain_id", "")),
            str(row.get("result_stage", "")),
        )

    return sorted(rows, key=sort_key)


def run_pipeline_result(
    cfg: AppConfig,
    *,
    write_csv: bool = True,
    print_summary: bool = True,
    strict_input: bool = True,
    show_progress: bool = True,
) -> PipelineRunResult:
    """Run the full beta-barrel detection pipeline and return structured results."""
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    validate_config(cfg)
    files = discover_input_files(cfg.input.path, cfg.input.allowed_suffixes, strict=strict_input)
    input_identities = freeze_input_identities(files)
    _validate_artifact_path_disjointness(cfg, input_identities, write_csv=write_csv)
    resolved_log_path = configure_logging(
        level=cfg.runtime.log_level,
        console=cfg.runtime.log_console,
        jsonl_path=cfg.runtime.log_jsonl_path,
    )
    configure_thread_environment(cfg.runtime.native_threads_per_process)

    analysis_workers = resolve_analysis_worker_count(cfg.runtime.workers, cfg.runtime.cpu_reserve)
    prepare_workers = resolve_prepare_worker_count(cfg.runtime.prepare_workers, analysis_workers)
    LOGGER.info(
        "Starting detection run",
        extra={
            "stage": "initialization",
            "source_path": cfg.input.path,
        },
    )
    if resolved_log_path is not None:
        LOGGER.info(
            "Structured JSONL log enabled",
            extra={"stage": "initialization", "source_path": str(resolved_log_path)},
        )
    if write_csv and Path(cfg.output.csv_path).expanduser().is_dir():
        raise InputValidationError(f"Output CSV path points to a directory: {cfg.output.csv_path}")

    if Path(cfg.input.path).expanduser().is_dir() and not files:
        allowed = "/".join(cfg.input.allowed_suffixes)
        if print_summary:
            print(f"No {allowed} files found in: {cfg.input.path}")
        if write_csv:
            _publish_results(
                [],
                cfg,
                input_files=files,
                analysis_workers=analysis_workers,
                prepare_workers=prepare_workers,
                started_at_utc=started_at_utc,
                input_identities=input_identities,
            )
            if print_summary:
                print(f"\nResults written to: {cfg.output.csv_path}")
        output_path = cfg.output.csv_path if write_csv else None
        return PipelineRunResult.from_rows(
            [], input_files=files, output_path=output_path, config=cfg
        )

    resolved_dssp_path = require_dssp_binary(cfg.runtime.dssp_bin_path)
    cfg = replace(
        cfg,
        runtime=replace(
            cfg.runtime,
            dssp_bin_path=resolved_dssp_path,
        ),
    )

    if print_summary:
        print(
            f"\nRunning streaming pipeline with {prepare_workers} prepare worker(s) "
            f"and {analysis_workers} analysis worker(s)..."
        )
    prepare_failures: list[PrepareFailure] = []

    def record_prepare_failures(failures: list[PrepareFailure]) -> None:
        prepare_failures.extend(failures)

    payload_batches = iter_prepared_payload_batches(
        files,
        cfg,
        prepare_workers,
        input_identities=input_identities,
        on_failures=record_prepare_failures,
        show_progress=show_progress,
    )
    results = run_analysis_stream(
        payload_batches,
        cfg,
        analysis_workers,
        show_progress=show_progress,
    )
    prepare_rows = _prepare_error_rows(prepare_failures)

    all_results = _ordered_result_rows([*results, *prepare_rows], files)
    verify_input_identities(input_identities)
    LOGGER.info(
        "Detection analysis completed",
        extra={"stage": "complete"},
    )
    if prepare_rows and not results:
        if print_summary:
            print_results_summary(
                all_results,
                cfg.output.csv_path,
                summary_limit=cfg.output.summary_limit,
                write_csv=False,
            )
        if write_csv:
            _publish_results(
                all_results,
                cfg,
                input_files=files,
                analysis_workers=analysis_workers,
                prepare_workers=prepare_workers,
                started_at_utc=started_at_utc,
                input_identities=input_identities,
            )
            if print_summary:
                print(f"\nResults written to: {cfg.output.csv_path}")
        raise InputValidationError(f"All {len(files)} input file(s) failed during preparation.")

    if not all_results:
        if print_summary:
            print("No prepared chain payloads were produced.")
        if write_csv:
            _publish_results(
                [],
                cfg,
                input_files=files,
                analysis_workers=analysis_workers,
                prepare_workers=prepare_workers,
                started_at_utc=started_at_utc,
                input_identities=input_identities,
            )
            if print_summary:
                print(f"\nResults written to: {cfg.output.csv_path}")
        output_path = cfg.output.csv_path if write_csv else None
        return PipelineRunResult.from_rows(
            [], input_files=files, output_path=output_path, config=cfg
        )

    if print_summary:
        print_results_summary(
            all_results,
            cfg.output.csv_path,
            summary_limit=cfg.output.summary_limit,
            write_csv=False,
        )
    if write_csv:
        _publish_results(
            all_results,
            cfg,
            input_files=files,
            analysis_workers=analysis_workers,
            prepare_workers=prepare_workers,
            started_at_utc=started_at_utc,
            input_identities=input_identities,
        )
        if print_summary:
            print(f"\nResults written to: {cfg.output.csv_path}")
    output_path = cfg.output.csv_path if write_csv else None
    return PipelineRunResult.from_rows(
        all_results,
        input_files=files,
        output_path=output_path,
        config=cfg,
    )


def run_pipeline(cfg: AppConfig) -> list[dict[str, object]]:
    """Run the full beta-barrel detection pipeline from a resolved config."""
    return run_pipeline_result(cfg).to_rows()


def detect(
    input_path: str,
    *,
    config: AppConfig | None = None,
    overrides: dict[str, object] | list[str] | None = None,
    workers: int | None = None,
    prepare_workers: int | None = None,
    output: str | None = None,
    write_csv: bool | None = None,
    print_summary: bool = False,
    show_progress: bool | None = None,
    strict_input: bool = True,
) -> PipelineRunResult:
    """
    Public Python API for running detection with structured results.

    CSV output is written only when ``output`` is provided or ``write_csv=True``.
    """
    if overrides is not None and config is not None:
        raise TypeError("Pass `overrides` only when Cooper-Beta builds the config for you.")
    resolved_cfg = config or build_config(overrides)
    resolved_cfg = apply_runtime_overrides(
        resolved_cfg,
        input_path=input_path,
        workers=workers,
        prepare_workers=prepare_workers,
        out_csv=output,
    )
    should_write_csv = bool(output) if write_csv is None else bool(write_csv)
    should_show_progress = bool(print_summary) if show_progress is None else bool(show_progress)
    return run_pipeline_result(
        resolved_cfg,
        write_csv=should_write_csv,
        print_summary=print_summary,
        show_progress=should_show_progress,
        strict_input=strict_input,
    )
