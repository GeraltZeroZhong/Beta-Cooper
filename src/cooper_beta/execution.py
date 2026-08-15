from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from types import TracebackType
from typing import Any, Generic, Protocol, TypeVar

from .bootstrap import configure_thread_environment
from .chain_analysis import analyze_chain_payload, unhandled_analysis_failure_row
from .config import AppConfig
from .constants import RESULT_ERROR
from .integrity import FrozenInputIdentity
from .logging_config import configure_logging
from .preparation import PreparationBatchResult, PrepareFailure, prepare_file_batch

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class ProgressReporter(Protocol):
    def update(self, value: int = 1) -> None: ...


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only by minimal installations

    class _NullProgress(Generic[T]):
        def __init__(self, iterable: Iterable[T] | None = None, **kwargs: Any) -> None:
            del kwargs
            self.iterable = iterable

        def __iter__(self) -> Iterator[T]:
            return iter(self.iterable or [])

        def __enter__(self) -> _NullProgress[T]:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exc_type, exc, traceback

        def update(self, value: int = 1) -> None:
            del value

    def tqdm(iterable: Iterable[T] | None = None, **kwargs: Any) -> _NullProgress[T]:
        return _NullProgress(iterable, **kwargs)


def _initialize_worker(
    native_threads: int,
    log_level: str,
    log_console: bool,
) -> None:
    configure_thread_environment(native_threads)
    # A shared JSONL FileHandler would permit interleaved multi-process writes.
    # Workers therefore emit only to stderr; the parent records structured result
    # failures in its configured JSONL stream.
    configure_logging(level=log_level, console=log_console, jsonl_path=None)


def _batched(items: list[T], batch_size: int) -> Iterator[list[T]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _report_failures(failures: list[PrepareFailure]) -> None:
    for failure in failures:
        LOGGER.error(
            "Structure preparation failed for %s: %s",
            Path(failure.source_path).name,
            failure.message,
            extra={
                "source_path": failure.source_path,
                "error_code": failure.error_code,
            },
        )


def iter_prepared_payload_batches(
    files: Iterable[str],
    config: AppConfig,
    prepare_workers: int,
    *,
    input_identities: list[FrozenInputIdentity] | None = None,
    on_failures: Callable[[list[PrepareFailure]], None] | None = None,
    show_progress: bool = True,
) -> Iterator[list[dict[str, object]]]:
    """Prepare files with bounded work submission and structured failure reporting."""
    file_list = list(files)
    batches = list(_batched(file_list, config.runtime.prepare_batch_size))
    identity_map: dict[str, FrozenInputIdentity] | None = None
    if input_identities is not None:
        resolved_files = [str(Path(path).expanduser().resolve()) for path in file_list]
        identity_paths = [identity.path for identity in input_identities]
        if resolved_files != identity_paths:
            raise ValueError(
                "Frozen input identities must correspond one-to-one with the ordered file list."
            )
        identity_map = {identity.path: identity for identity in input_identities}

    def identities_for_batch(batch: list[str]) -> dict[str, FrozenInputIdentity] | None:
        if identity_map is None:
            return None
        return {
            str(Path(path).expanduser().resolve()): identity_map[
                str(Path(path).expanduser().resolve())
            ]
            for path in batch
        }

    def handle_result(
        result: PreparationBatchResult, progress_bar: ProgressReporter
    ) -> list[dict[str, object]]:
        if result.failures:
            _report_failures(result.failures)
            if on_failures is not None:
                on_failures(result.failures)
        progress_bar.update(result.processed_files)
        return result.payloads

    if prepare_workers <= 1:
        with tqdm(
            total=len(file_list),
            desc="Preparing",
            unit="file",
            disable=not show_progress,
        ) as progress_bar:
            for batch in batches:
                batch_identities = identities_for_batch(batch)
                payloads = handle_result(
                    (
                        prepare_file_batch(batch, config)
                        if batch_identities is None
                        else prepare_file_batch(batch, config, batch_identities)
                    ),
                    progress_bar,
                )
                if payloads:
                    yield payloads
        return

    max_in_flight = prepare_workers * config.runtime.prepare_in_flight_multiplier
    batch_iterator = iter(batches)
    pending: set[Future[PreparationBatchResult]] = set()
    executor = ProcessPoolExecutor(
        max_workers=prepare_workers,
        initializer=_initialize_worker,
        initargs=(
            config.runtime.native_threads_per_process,
            config.runtime.log_level,
            config.runtime.log_console,
        ),
    )
    try:

        def submit_next() -> bool:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                return False
            batch_identities = identities_for_batch(batch)
            if batch_identities is None:
                pending.add(executor.submit(prepare_file_batch, batch, config))
            else:
                pending.add(
                    executor.submit(
                        prepare_file_batch,
                        batch,
                        config,
                        batch_identities,
                    )
                )
            return True

        for _ in range(max_in_flight):
            if not submit_next():
                break

        with tqdm(
            total=len(file_list),
            desc="Preparing",
            unit="file",
            disable=not show_progress,
        ) as progress_bar:
            while pending:
                completed, still_pending = wait(pending, return_when=FIRST_COMPLETED)
                pending = set(still_pending)
                for future in completed:
                    payloads = handle_result(future.result(), progress_bar)
                    if payloads:
                        yield payloads
                    while len(pending) < max_in_flight and submit_next():
                        pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def analyze_payload_batch(
    payloads: list[dict[str, object]],
    config: AppConfig,
) -> list[dict[str, object]]:
    """Contain unexpected failures at the process boundary, one row per chain."""
    rows: list[dict[str, object]] = []
    for payload in payloads:
        try:
            rows.append(analyze_chain_payload(payload, config))
        except Exception as exc:
            LOGGER.exception(
                "Unhandled analysis worker failure",
                extra={
                    "source_path": str(payload.get("source_path", "")),
                    "chain": str(payload.get("chain", "")),
                    "error_code": "ANALYSIS_WORKER_FAILED",
                },
            )
            rows.append(unhandled_analysis_failure_row(payload, config, exc))
    return rows


def run_analysis_stream(
    payload_batches: Iterable[list[dict[str, object]]],
    config: AppConfig,
    workers: int,
    *,
    on_results: Callable[[list[dict[str, object]]], None] | None = None,
    show_progress: bool = True,
) -> list[dict[str, object]]:
    """Analyze chain payloads with deterministic batching and bounded concurrency."""
    results: list[dict[str, object]] = []

    def handle_rows(rows: list[dict[str, object]], progress_bar: ProgressReporter) -> None:
        for row in rows:
            if row.get("result") == RESULT_ERROR:
                LOGGER.error(
                    "Chain analysis produced an ERROR result: %s",
                    row.get("reason", ""),
                    extra={
                        "stage": row.get("result_stage", "analysis"),
                        "structure_filename": row.get("filename", ""),
                        "source_path": row.get("source_path", ""),
                        "chain": row.get("author_chain_id", ""),
                        "error_code": row.get("error_code", "ANALYSIS_FAILED"),
                    },
                )
        if on_results is not None:
            on_results(rows)
        results.extend(rows)
        progress_bar.update(len(rows))

    def payload_groups() -> Iterator[list[dict[str, object]]]:
        for payload_group in payload_batches:
            yield from _batched(payload_group, config.runtime.analysis_batch_size)

    if workers <= 1:
        with tqdm(desc="Analyzing", unit="chain", disable=not show_progress) as progress_bar:
            for payload_batch in payload_groups():
                handle_rows(analyze_payload_batch(payload_batch, config), progress_bar)
        return results

    max_in_flight = workers * config.runtime.analysis_in_flight_multiplier
    pending: set[Future[list[dict[str, object]]]] = set()
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(
            config.runtime.native_threads_per_process,
            config.runtime.log_level,
            config.runtime.log_console,
        ),
    )
    try:
        with tqdm(desc="Analyzing", unit="chain", disable=not show_progress) as progress_bar:
            for payload_batch in payload_groups():
                pending.add(executor.submit(analyze_payload_batch, payload_batch, config))
                while len(pending) >= max_in_flight:
                    completed, still_pending = wait(pending, return_when=FIRST_COMPLETED)
                    pending = set(still_pending)
                    for future in completed:
                        handle_rows(future.result(), progress_bar)

            while pending:
                completed, still_pending = wait(pending, return_when=FIRST_COMPLETED)
                pending = set(still_pending)
                for future in completed:
                    handle_rows(future.result(), progress_bar)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results
