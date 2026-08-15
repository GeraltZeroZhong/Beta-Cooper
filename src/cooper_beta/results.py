from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from .constants import DEFAULT_RESULT_COLUMNS, DEFAULT_SUMMARY_COLUMNS, SUMMARY_COLUMN_WIDTHS
from .models import DetectionResult

SUMMARY_DISPLAY_NAMES = {
    "filename": "Filename",
    "author_chain_id": "Author chain",
    "result": "Result",
    "strand_adjacency_count": "Adjacencies",
    "cycle_strand_count": "Cycle strands",
    "cycle_strand_fraction": "Cycle fraction",
    "cycle_rank": "Cycle rank",
    "reason": "Reason",
}


def _row_for_fieldnames(row: dict[str, object], fieldnames: list[str]) -> dict[str, object]:
    expected = set(fieldnames)
    actual = set(row)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "Result row does not match the fixed public CSV schema: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    # Validate scientific values as well as column names before committing a CSV.
    validated = DetectionResult.from_row(row).to_dict()
    return {key: validated[key] for key in fieldnames}


def _ensure_output_parent(output_path: str) -> None:
    parent = Path(output_path).expanduser().parent
    if str(parent) and parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def _optional_pandas() -> Any | None:
    """Load pandas only for console presentation, without hiding broken installs."""
    try:
        import pandas
    except ImportError:  # pragma: no cover - exercised in minimal installations
        return None
    return pandas


class ResultCsvWriter:
    """Incrementally write result rows using the stable Cooper-Beta schema."""

    def __init__(self, output_path: str, fieldnames: Iterable[str] | None = None):
        self.output_path = output_path
        requested_fieldnames = tuple(DEFAULT_RESULT_COLUMNS if fieldnames is None else fieldnames)
        if requested_fieldnames != tuple(DEFAULT_RESULT_COLUMNS):
            raise ValueError("Result CSV fieldnames must match DEFAULT_RESULT_COLUMNS exactly.")
        self.fieldnames = list(DEFAULT_RESULT_COLUMNS)
        self._handle: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._temporary_path: str | None = None

    def __enter__(self) -> ResultCsvWriter:
        if self._handle is not None:
            raise RuntimeError("ResultCsvWriter is already open.")
        _ensure_output_parent(self.output_path)
        output_path = Path(self.output_path).expanduser()
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        self._temporary_path = temporary_path
        try:
            self._handle = os.fdopen(descriptor, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
            self._writer.writeheader()
        except Exception:
            if self._handle is None:
                os.close(descriptor)
            self.close(commit=False)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        self.close(commit=exc_type is None)

    def write_rows(self, rows: Iterable[dict[str, object]]) -> None:
        if self._writer is None:
            raise RuntimeError("ResultCsvWriter must be opened before writing rows.")
        for row in rows:
            self._writer.writerow(_row_for_fieldnames(row, self.fieldnames))

    def close(self, *, commit: bool = True) -> None:
        handle = self._handle
        temporary_path = self._temporary_path
        try:
            if handle is not None:
                if commit:
                    handle.flush()
                    os.fsync(handle.fileno())
                handle.close()
                handle = None
            if commit and temporary_path is not None:
                os.replace(temporary_path, Path(self.output_path).expanduser())
                temporary_path = None
        finally:
            self._handle = None
            self._writer = None
            self._temporary_path = None
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass


def write_results_csv(rows: list[dict[str, object]], output_path: str) -> None:
    """Atomically write result rows using the fixed public schema."""
    with ResultCsvWriter(output_path) as writer:
        writer.write_rows(rows)


def _summary_rows(results: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in results:
        validated = DetectionResult.from_row(row)
        rows.append(
            {
                "filename": validated.filename,
                "author_chain_id": validated.author_chain_id,
                "result": validated.result,
                "strand_adjacency_count": validated.strand_adjacency_count,
                "cycle_strand_count": validated.cycle_strand_count,
                "cycle_strand_fraction": validated.cycle_strand_fraction,
                "cycle_rank": validated.cycle_rank,
                "reason": validated.reason,
            }
        )
    return rows


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common())


def _summary_limit_value(summary_limit: int) -> int | None:
    if isinstance(summary_limit, bool) or not isinstance(summary_limit, int):
        raise TypeError("`summary_limit` must be an integer; use -1 to print all rows.")
    value = summary_limit
    if value < 0:
        return None
    return value


def _limited_summary_rows(
    results: list[dict[str, object]],
    summary_limit: int,
) -> tuple[list[dict[str, object]], int | None]:
    limit = _summary_limit_value(summary_limit)
    if limit is None:
        return _summary_rows(results), None
    return _summary_rows(results[:limit]), limit


def print_results_summary(
    results: list[dict[str, object]],
    output_path: str,
    *,
    summary_limit: int,
    write_csv: bool = True,
) -> None:
    """Print a human-readable summary and persist the CSV."""
    summary_rows, resolved_limit = _limited_summary_rows(results, summary_limit)
    result_counts = Counter(str(row.get("result", "") or "<blank>") for row in results)
    print("\n=== Summary ===")
    print(f"Rows: {len(results)}")
    print(f"Results: {_format_counter(result_counts)}")

    pandas = _optional_pandas()
    if pandas is not None:
        dataframe = pandas.DataFrame(summary_rows)
        if not dataframe.empty:
            display_frame = dataframe[list(DEFAULT_SUMMARY_COLUMNS)].rename(
                columns=SUMMARY_DISPLAY_NAMES
            )
            print()
            print(display_frame.to_string(index=False))
        if resolved_limit is not None and len(results) > resolved_limit:
            omitted = len(results) - resolved_limit
            print(f"\n... omitted {omitted} row(s) from console summary.")
        if write_csv:
            write_results_csv(results, output_path)
        if write_csv:
            print(f"\nResults written to: {output_path}")
        return

    filename_width = SUMMARY_COLUMN_WIDTHS["filename"]
    author_chain_width = SUMMARY_COLUMN_WIDTHS["author_chain_id"]
    result_width = SUMMARY_COLUMN_WIDTHS["result"]
    adjacency_width = SUMMARY_COLUMN_WIDTHS["strand_adjacency_count"]
    cycle_count_width = SUMMARY_COLUMN_WIDTHS["cycle_strand_count"]
    cycle_fraction_width = SUMMARY_COLUMN_WIDTHS["cycle_strand_fraction"]
    rank_width = SUMMARY_COLUMN_WIDTHS["cycle_rank"]
    reason_width = SUMMARY_COLUMN_WIDTHS["reason"]
    header = (
        f"{'Filename':<{filename_width}} | "
        f"{'Author chain':<{author_chain_width}} | "
        f"{'Result':<{result_width}} | "
        f"{'Adjacencies':<{adjacency_width}} | "
        f"{'Cycle strands':<{cycle_count_width}} | "
        f"{'Cycle fraction':<{cycle_fraction_width}} | "
        f"{'Cycle rank':<{rank_width}} | "
        f"{'Reason':<{reason_width}}"
    )
    if summary_rows:
        print()
        print(header)
        print("-" * len(header))
    for row in summary_rows:
        print(
            f"{str(row.get('filename', '')):<{filename_width}} | "
            f"{str(row.get('author_chain_id', '')):<{author_chain_width}} | "
            f"{str(row.get('result', '')):<{result_width}} | "
            f"{str(row.get('strand_adjacency_count', '')):<{adjacency_width}} | "
            f"{str(row.get('cycle_strand_count', '')):<{cycle_count_width}} | "
            f"{str(row.get('cycle_strand_fraction', '')):<{cycle_fraction_width}} | "
            f"{str(row.get('cycle_rank', '')):<{rank_width}} | "
            f"{str(row.get('reason', '')):<{reason_width}}"
        )

    if resolved_limit is not None and len(results) > resolved_limit:
        omitted = len(results) - resolved_limit
        print(f"\n... omitted {omitted} row(s) from console summary.")
    if write_csv:
        write_results_csv(results, output_path)
    if write_csv:
        print(f"\nResults written to: {output_path}")
