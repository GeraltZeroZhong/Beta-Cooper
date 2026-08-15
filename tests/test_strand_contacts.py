from __future__ import annotations

import numpy as np
import pytest

from cooper_beta.strand_contacts import add_contact_supported_adjacencies
from cooper_beta.strand_graph import (
    StrandAdjacencyGraph,
    StrandEdge,
    StrandNode,
    StrandRange,
    measure_strand_graph,
)

MAXIMUM_CA_DISTANCE_ANGSTROM = 6.8
MINIMUM_CONTACT_PAIR_COUNT = 2
MINIMUM_CONTACT_RESIDUE_COUNT_PER_STRAND = 2


def _graph(*edges: tuple[str, str]) -> StrandAdjacencyGraph:
    nodes = tuple(
        StrandNode(node_id, StrandRange(index * 10, index * 10 + 2))
        for index, node_id in enumerate(("A", "B", "C"))
    )
    return StrandAdjacencyGraph(
        author_chain_id="X",
        nodes=nodes,
        edges=tuple(StrandEdge(*edge) for edge in edges),
    )


def _residue(node_id: str | None, coordinate: tuple[float, float, float]) -> dict[str, object]:
    return {"strand_node_id": node_id, "coord": coordinate}


def test_adds_adjacency_with_multiple_residues_on_both_strands() -> None:
    graph = _graph(("A", "B"), ("B", "C"))
    residues = [
        _residue("A", (0.0, 0.0, 0.0)),
        _residue("A", (0.0, 0.0, 3.5)),
        _residue("B", (20.0, 0.0, 0.0)),
        _residue("B", (20.0, 0.0, 3.5)),
        _residue("C", (6.7, 0.0, 0.0)),
        _residue("C", (6.7, 0.0, 3.5)),
    ]

    result = add_contact_supported_adjacencies(
        graph,
        residues,
        maximum_ca_distance_angstrom=MAXIMUM_CA_DISTANCE_ANGSTROM,
        minimum_contact_pair_count=MINIMUM_CONTACT_PAIR_COUNT,
        minimum_contact_residue_count_per_strand=(MINIMUM_CONTACT_RESIDUE_COUNT_PER_STRAND),
    )

    assert {edge.endpoints for edge in result.edges} == {
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    }
    features = measure_strand_graph(result)
    assert features.cycle_strand_count == 3
    assert features.cycle_rank == 1


def test_single_close_residue_cannot_create_adjacency() -> None:
    graph = _graph(("A", "B"), ("B", "C"))
    residues = [
        _residue("A", (0.0, 0.0, 0.0)),
        _residue("A", (0.0, 0.0, 20.0)),
        _residue("C", (6.7, 0.0, 0.0)),
        _residue("C", (20.0, 0.0, 20.0)),
    ]

    result = add_contact_supported_adjacencies(
        graph,
        residues,
        maximum_ca_distance_angstrom=MAXIMUM_CA_DISTANCE_ANGSTROM,
        minimum_contact_pair_count=MINIMUM_CONTACT_PAIR_COUNT,
        minimum_contact_residue_count_per_strand=(MINIMUM_CONTACT_RESIDUE_COUNT_PER_STRAND),
    )

    assert result == graph


def test_existing_ladder_edge_does_not_require_contact_support() -> None:
    graph = _graph(("A", "B"))
    result = add_contact_supported_adjacencies(
        graph,
        [],
        maximum_ca_distance_angstrom=MAXIMUM_CA_DISTANCE_ANGSTROM,
        minimum_contact_pair_count=MINIMUM_CONTACT_PAIR_COUNT,
        minimum_contact_residue_count_per_strand=(MINIMUM_CONTACT_RESIDUE_COUNT_PER_STRAND),
    )
    assert result == graph


def test_accepts_loader_numpy_coordinates() -> None:
    graph = _graph(("A", "B"), ("B", "C"))
    residues = [
        {"strand_node_id": "A", "coord": np.asarray((0.0, 0.0, 0.0))},
        {"strand_node_id": "A", "coord": np.asarray((0.0, 0.0, 3.5))},
        {"strand_node_id": "C", "coord": np.asarray((6.7, 0.0, 0.0))},
        {"strand_node_id": "C", "coord": np.asarray((6.7, 0.0, 3.5))},
    ]

    result = add_contact_supported_adjacencies(
        graph,
        residues,
        maximum_ca_distance_angstrom=MAXIMUM_CA_DISTANCE_ANGSTROM,
        minimum_contact_pair_count=MINIMUM_CONTACT_PAIR_COUNT,
        minimum_contact_residue_count_per_strand=(MINIMUM_CONTACT_RESIDUE_COUNT_PER_STRAND),
    )

    assert ("A", "C") in {edge.endpoints for edge in result.edges}


@pytest.mark.parametrize(
    ("field", "value", "expected_exception"),
    [
        ("maximum_ca_distance_angstrom", 0.0, ValueError),
        ("maximum_ca_distance_angstrom", float("nan"), ValueError),
        ("minimum_contact_pair_count", 1.5, TypeError),
        ("minimum_contact_residue_count_per_strand", 0, ValueError),
    ],
)
def test_rejects_invalid_contact_parameters(
    field: str,
    value: object,
    expected_exception: type[Exception],
) -> None:
    parameters: dict[str, object] = {
        "maximum_ca_distance_angstrom": MAXIMUM_CA_DISTANCE_ANGSTROM,
        "minimum_contact_pair_count": MINIMUM_CONTACT_PAIR_COUNT,
        "minimum_contact_residue_count_per_strand": (MINIMUM_CONTACT_RESIDUE_COUNT_PER_STRAND),
    }
    parameters[field] = value
    with pytest.raises(expected_exception):
        add_contact_supported_adjacencies(_graph(), [], **parameters)  # type: ignore[arg-type]


def test_rejects_unknown_strand_node_reference() -> None:
    with pytest.raises(ValueError, match="unknown strand node"):
        add_contact_supported_adjacencies(
            _graph(),
            [_residue("missing", (0.0, 0.0, 0.0))],
            maximum_ca_distance_angstrom=MAXIMUM_CA_DISTANCE_ANGSTROM,
            minimum_contact_pair_count=MINIMUM_CONTACT_PAIR_COUNT,
            minimum_contact_residue_count_per_strand=(MINIMUM_CONTACT_RESIDUE_COUNT_PER_STRAND),
        )
