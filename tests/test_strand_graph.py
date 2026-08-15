from __future__ import annotations

from collections.abc import Iterable

import pytest

from cooper_beta.strand_graph import (
    StrandAdjacencyGraph,
    StrandEdge,
    StrandGraphMeasurements,
    StrandNode,
    StrandRange,
    measure_strand_graph,
)


def _graph(
    node_ids: Iterable[str],
    edge_pairs: Iterable[tuple[str, str]],
    *,
    residue_counts: dict[str, int] | None = None,
) -> StrandAdjacencyGraph:
    nodes = []
    next_start = 0
    for node_id in node_ids:
        residue_count = 1 if residue_counts is None else residue_counts[node_id]
        nodes.append(
            StrandNode(
                node_id,
                StrandRange(next_start, next_start + residue_count - 1),
            )
        )
        next_start += residue_count
    return StrandAdjacencyGraph(
        author_chain_id="A",
        nodes=tuple(nodes),
        edges=tuple(StrandEdge(*pair) for pair in edge_pairs),
    )


def test_cycle_marks_every_ring_strand_and_edge() -> None:
    graph = _graph("ABCD", (("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")))

    features = measure_strand_graph(graph)

    assert isinstance(features, StrandGraphMeasurements)
    assert features.author_chain_id == "A"
    assert features.strand_count == 4
    assert features.strand_adjacency_count == 4
    assert features.cycle_rank == 1
    assert features.cycle_strand_count == 4
    assert features.cycle_strand_fraction == 1.0


def test_tree_has_no_cycle_coverage() -> None:
    graph = _graph("ABCDE", (("A", "B"), ("A", "C"), ("C", "D"), ("C", "E")))

    features = measure_strand_graph(graph)

    assert features.cycle_rank == 0
    assert features.cycle_strand_count == 0
    assert features.cycle_strand_fraction == 0.0


def test_ring_with_leaf_reports_continuous_partial_coverage() -> None:
    graph = _graph(
        "ABCD",
        (("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")),
        residue_counts={"A": 2, "B": 2, "C": 2, "D": 6},
    )

    features = measure_strand_graph(graph)

    assert features.cycle_strand_count == 3
    assert features.cycle_strand_fraction == pytest.approx(3 / 4)


def test_bridge_between_two_rings_is_not_counted_as_cycle_edge() -> None:
    graph = _graph(
        "ABCDEF",
        (
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("C", "D"),
            ("D", "E"),
            ("E", "F"),
            ("F", "D"),
        ),
    )

    features = measure_strand_graph(graph)

    assert features.cycle_rank == 2
    assert features.cycle_strand_count == 3
    assert features.cycle_strand_fraction == pytest.approx(1 / 2)


def test_multiple_components_include_tree_and_isolated_strands_in_denominator() -> None:
    graph = _graph(
        "ABCDEF",
        (("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")),
    )

    features = measure_strand_graph(graph)

    assert features.cycle_rank == 1
    assert features.cycle_strand_count == 3
    assert features.cycle_strand_fraction == pytest.approx(1 / 2)


def test_empty_graph_has_explicit_zero_features() -> None:
    graph = StrandAdjacencyGraph(author_chain_id="A", nodes=(), edges=())

    features = measure_strand_graph(graph)

    assert features.strand_count == 0
    assert features.strand_adjacency_count == 0
    assert features.cycle_rank == 0
    assert features.cycle_strand_count == 0
    assert features.cycle_strand_fraction == 0.0


@pytest.mark.parametrize(
    "range_value",
    [
        pytest.param(StrandRange, id="range-type"),
    ],
)
def test_node_requires_a_strand_range(range_value: object) -> None:
    with pytest.raises(TypeError, match="StrandRange"):
        StrandNode("A", range_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "error_type", "message"),
    [
        ((-1, 2), ValueError, "non-negative"),
        ((True, 2), TypeError, "integer"),
        ((3, 2), ValueError, "at least"),
    ],
)
def test_strand_range_rejects_invalid_boundaries(
    args: tuple[object, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        StrandRange(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize("node_id", ["", "   ", 1])
def test_node_rejects_invalid_identifier(node_id: object) -> None:
    with pytest.raises((TypeError, ValueError), match="node_id"):
        StrandNode(node_id, StrandRange(0, 1))  # type: ignore[arg-type]


def test_undirected_edge_is_canonical_and_rejects_self_loop() -> None:
    assert StrandEdge("B", "A").endpoints == ("A", "B")
    with pytest.raises(ValueError, match="self-loop"):
        StrandEdge("A", " A ")


@pytest.mark.parametrize(
    ("graph", "error_type", "message"),
    [
        (
            lambda: StrandAdjacencyGraph("", (), ()),
            ValueError,
            "author_chain_id",
        ),
        (
            lambda: StrandAdjacencyGraph("A", [], ()),  # type: ignore[arg-type]
            TypeError,
            "nodes",
        ),
        (
            lambda: StrandAdjacencyGraph(
                "A",
                (
                    StrandNode("A", StrandRange(0, 1)),
                    StrandNode(" A ", StrandRange(2, 3)),
                ),
                (),
            ),
            ValueError,
            "unique",
        ),
        (
            lambda: StrandAdjacencyGraph(
                "A",
                (
                    StrandNode("A", StrandRange(0, 1)),
                    StrandNode("B", StrandRange(2, 3)),
                ),
                (StrandEdge("A", "B"), StrandEdge("B", "A")),
            ),
            ValueError,
            "edges must be unique",
        ),
        (
            lambda: StrandAdjacencyGraph(
                "A",
                (StrandNode("A", StrandRange(0, 1)),),
                (StrandEdge("A", "B"),),
            ),
            ValueError,
            "unknown node",
        ),
    ],
)
def test_graph_rejects_invalid_or_ambiguous_input(
    graph: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        graph()  # type: ignore[operator]


def test_analysis_rejects_unvalidated_graph_like_input() -> None:
    with pytest.raises(TypeError, match="StrandAdjacencyGraph"):
        measure_strand_graph({})  # type: ignore[arg-type]
