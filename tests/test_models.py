from __future__ import annotations

import pytest

from cooper_beta.constants import DEFAULT_RESULT_COLUMNS
from cooper_beta.models import (
    DetectionResult,
    PipelineRunResult,
    PreparedChainPayload,
    strand_graph_from_mapping,
    strand_graph_to_mapping,
)


def _graph() -> dict[str, object]:
    return {
        "author_chain_id": "A",
        "nodes": [
            {"node_id": "s0", "start_polymer_index": 0, "end_polymer_index": 1},
            {"node_id": "s1", "start_polymer_index": 2, "end_polymer_index": 3},
        ],
        "edges": [{"first_node_id": "s0", "second_node_id": "s1"}],
    }


def _residue() -> dict[str, object]:
    return {
        "res_id": 1,
        "coord": [1.0, 2.0, 3.0],
        "dssp_assignment_available": True,
        "is_sheet": True,
        "strand_node_id": "s0",
        "polymer_index": 0,
        "peptide_bond_distance_to_previous_angstrom": None,
        "chain": "A",
        "resseq": 1,
        "icode": "",
        "hetfield": "",
        "res_uid": {"chain": "A", "hetfield": "", "resseq": 1, "icode": ""},
    }


def _result(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "filename": "toy.pdb",
        "source_path": "/input/toy.pdb",
        "author_chain_id": "A",
        "result": "BARREL",
        "result_stage": "decision",
        "dssp_unassigned_residue_count": 0,
        "strand_count": 8,
        "strand_adjacency_count": 8,
        "cycle_strand_count": 8,
        "cycle_strand_fraction": 1.0,
        "cycle_rank": 1,
        "reason": "All three strand-graph rule groups passed",
        "error_code": "",
        "degraded": False,
    }
    row.update(updates)
    return row


def test_prepared_payload_and_graph_round_trip() -> None:
    payload = PreparedChainPayload.from_mapping(
        {
            "filename": "toy.pdb",
            "source_path": "/input/toy.pdb",
            "chain": "A",
            "residues_data": [_residue()],
            "strand_graph": _graph(),
        }
    )

    assert payload.residues_data[0]["coord"] == [1.0, 2.0, 3.0]
    assert strand_graph_to_mapping(strand_graph_from_mapping(_graph())) == _graph()


@pytest.mark.parametrize(
    "updates",
    [
        {"strand_graph": {"author_chain_id": "B", "nodes": [], "edges": []}},
        {"residues_data": [dict(_residue(), is_sheet="false")]},
        {"residues_data": [dict(_residue(), coord=[0.0, float("nan"), 1.0])]},
    ],
)
def test_prepared_payload_rejects_invalid_process_data(updates: dict[str, object]) -> None:
    value = {
        "filename": "toy.pdb",
        "source_path": "/input/toy.pdb",
        "chain": "A",
        "residues_data": [_residue()],
        "strand_graph": _graph(),
    }
    value.update(updates)
    with pytest.raises((TypeError, ValueError)):
        PreparedChainPayload.from_mapping(value)


def test_detection_result_round_trip_uses_exact_lean_schema() -> None:
    parsed = DetectionResult.from_row(_result())

    assert tuple(parsed.to_dict()) == DEFAULT_RESULT_COLUMNS
    assert PipelineRunResult.from_rows([_result()]).result_counts == {"BARREL": 1}


@pytest.mark.parametrize(
    "updates",
    [
        {"cycle_strand_count": 7, "cycle_strand_fraction": 1.0},
        {"cycle_strand_count": 0, "cycle_strand_fraction": 0.0, "cycle_rank": 1},
        {"result": "ERROR", "error_code": ""},
        {"result": "NON_BARREL", "result_stage": "worker"},
        {"unexpected": 1},
    ],
)
def test_detection_result_rejects_inconsistent_rows(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DetectionResult.from_row(_result(**updates))
