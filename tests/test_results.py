from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cooper_beta.constants import DEFAULT_RESULT_COLUMNS
from cooper_beta.results import ResultCsvWriter, write_results_csv


def _row(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "filename": "toy.pdb",
        "source_path": "/input/toy.pdb",
        "author_chain_id": "A",
        "result": "BARREL",
        "result_stage": "decision",
        "dssp_unassigned_residue_count": 2,
        "strand_count": 8,
        "strand_adjacency_count": 8,
        "cycle_strand_count": 8,
        "cycle_strand_fraction": 1.0,
        "cycle_rank": 1,
        "reason": "All three strand-graph rule groups passed",
        "error_code": "",
        "degraded": False,
    }
    value.update(updates)
    return value


def test_result_csv_uses_exact_schema(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    write_results_csv([_row()], str(output))
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == DEFAULT_RESULT_COLUMNS
    assert rows[0]["strand_adjacency_count"] == "8"
    assert rows[0]["cycle_strand_fraction"] == "1.0"


def test_result_csv_rejects_schema_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed public CSV schema"):
        write_results_csv([_row(unexpected=True)], str(tmp_path / "results.csv"))


def test_result_csv_write_is_atomic_on_failure(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with ResultCsvWriter(str(output)) as writer:
            writer.write_rows([_row()])
            raise RuntimeError("abort")
    assert output.read_text(encoding="utf-8") == "existing\n"
