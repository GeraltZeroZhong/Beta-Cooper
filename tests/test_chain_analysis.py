from __future__ import annotations

import numpy as np

from cooper_beta.chain_analysis import analyze_chain_payload
from cooper_beta.config import build_config
from cooper_beta.constants import DEFAULT_RESULT_COLUMNS, RESULT_ERROR, RESULT_NON_BARREL


def _graph(*, closed: bool = True, strand_count: int = 8) -> dict[str, object]:
    ids = [f"s{index}" for index in range(strand_count)]
    edges = [
        {"first_node_id": ids[index], "second_node_id": ids[index + 1]}
        for index in range(strand_count - 1)
    ]
    if closed:
        edges.append({"first_node_id": ids[0], "second_node_id": ids[-1]})
    return {
        "author_chain_id": "A",
        "nodes": [
            {
                "node_id": node_id,
                "start_polymer_index": index * 2,
                "end_polymer_index": index * 2 + 1,
            }
            for index, node_id in enumerate(ids)
        ],
        "edges": edges,
    }


def _residues(*, contact_closure: bool = False) -> list[dict[str, object]]:
    rows = []
    for index in range(16):
        strand = index // 2
        angle = 2.0 * np.pi * strand / 8.0
        coordinate = (float(20 * np.cos(angle)), float(20 * np.sin(angle)), float(index % 2))
        if contact_closure:
            coordinate = (
                6.7 if strand == 7 else float(strand * 30),
                0.0,
                float(index % 2),
            )
        rows.append(
            {
                "res_id": index + 1,
                "coord": coordinate,
                "dssp_assignment_available": True,
                "is_sheet": True,
                "strand_node_id": f"s{strand}",
                "polymer_index": index,
                "peptide_bond_distance_to_previous_angstrom": None if index == 0 else 1.33,
                "chain": "A",
                "resseq": index + 1,
                "icode": "",
                "hetfield": "",
                "res_uid": {
                    "chain": "A",
                    "hetfield": "",
                    "resseq": index + 1,
                    "icode": "",
                },
            }
        )
    return rows


def _payload(*, closed: bool = True, contact_closure: bool = False) -> dict[str, object]:
    return {
        "filename": "toy.pdb",
        "source_path": "/data/toy.pdb",
        "chain": "A",
        "residues_data": _residues(contact_closure=contact_closure),
        "strand_graph": _graph(closed=closed),
        "degraded": False,
        "degradation_code": "",
        "degradation_reason": "",
    }


def test_closed_eight_strand_graph_passes_all_rules() -> None:
    row = analyze_chain_payload(_payload(), build_config())

    assert tuple(row) == DEFAULT_RESULT_COLUMNS
    assert row["result"] == "BARREL"
    assert row["strand_adjacency_count"] == 8
    assert row["cycle_strand_count"] == 8
    assert row["cycle_strand_fraction"] == 1.0
    assert row["cycle_rank"] == 1


def test_open_graph_is_a_normal_negative_decision() -> None:
    row = analyze_chain_payload(_payload(closed=False), build_config())

    assert row["result"] == RESULT_NON_BARREL
    assert row["result_stage"] == "decision"
    assert row["cycle_strand_count"] == 0
    assert row["cycle_rank"] == 0


def test_contact_supported_adjacency_can_close_an_open_graph() -> None:
    row = analyze_chain_payload(_payload(closed=False, contact_closure=True), build_config())

    assert row["result"] == "BARREL"
    assert row["strand_adjacency_count"] == 8
    assert row["cycle_strand_count"] == 8


def test_degraded_preparation_remains_an_error() -> None:
    payload = _payload()
    payload.update(
        degraded=True,
        degradation_code="DSSP_FAILED",
        degradation_reason="DSSP did not produce annotations",
    )
    row = analyze_chain_payload(payload, build_config())

    assert row["result"] == RESULT_ERROR
    assert row["error_code"] == "DSSP_FAILED"
    assert row["degraded"] is True
