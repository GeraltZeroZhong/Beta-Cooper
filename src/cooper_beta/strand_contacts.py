from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .strand_graph import StrandAdjacencyGraph, StrandEdge

FloatArray = NDArray[np.float64]


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{field_name}` must be an integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"`{field_name}` must be positive.")
    return normalized


def _positive_finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{field_name}` must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"`{field_name}` must be finite and positive.")
    return normalized


def _strand_coordinates(
    graph: StrandAdjacencyGraph,
    residues: Sequence[Mapping[str, object]],
) -> dict[str, FloatArray]:
    known_node_ids = {node.node_id for node in graph.nodes}
    coordinates: dict[str, list[tuple[float, float, float]]] = {
        node_id: [] for node_id in known_node_ids
    }
    for index, residue in enumerate(residues):
        if not isinstance(residue, Mapping):
            raise TypeError(f"Residue record {index} must be a mapping.")
        node_id = residue.get("strand_node_id")
        if node_id is None:
            continue
        if not isinstance(node_id, str) or node_id not in known_node_ids:
            raise ValueError(f"Residue record {index} references unknown strand node {node_id!r}.")
        raw_coordinate = residue.get("coord")
        if isinstance(raw_coordinate, np.ndarray):
            raw_coordinate = raw_coordinate.tolist()
        if (
            not isinstance(raw_coordinate, (tuple, list))
            or len(raw_coordinate) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, Real) for value in raw_coordinate
            )
        ):
            raise TypeError(f"Residue record {index} requires a three-dimensional coordinate.")
        coordinate = (
            float(raw_coordinate[0]),
            float(raw_coordinate[1]),
            float(raw_coordinate[2]),
        )
        if not all(math.isfinite(value) for value in coordinate):
            raise ValueError(f"Residue record {index} coordinate must be finite.")
        coordinates[node_id].append(coordinate)

    return {
        node_id: np.asarray(values, dtype=np.float64).reshape((-1, 3))
        for node_id, values in coordinates.items()
    }


def _has_multiresidue_contact(
    first: FloatArray,
    second: FloatArray,
    *,
    maximum_distance_squared: float,
    minimum_contact_pair_count: int,
    minimum_contact_residue_count_per_strand: int,
) -> bool:
    if (
        first.shape[0] < minimum_contact_residue_count_per_strand
        or second.shape[0] < minimum_contact_residue_count_per_strand
    ):
        return False
    displacement = first[:, np.newaxis, :] - second[np.newaxis, :, :]
    contact_mask = np.einsum("ijk,ijk->ij", displacement, displacement) <= (
        maximum_distance_squared
    )
    if int(np.count_nonzero(contact_mask)) < minimum_contact_pair_count:
        return False
    return bool(
        int(np.count_nonzero(np.any(contact_mask, axis=1)))
        >= minimum_contact_residue_count_per_strand
        and int(np.count_nonzero(np.any(contact_mask, axis=0)))
        >= minimum_contact_residue_count_per_strand
    )


def add_contact_supported_adjacencies(
    graph: StrandAdjacencyGraph,
    residues: Sequence[Mapping[str, object]],
    *,
    maximum_ca_distance_angstrom: float,
    minimum_contact_pair_count: int,
    minimum_contact_residue_count_per_strand: int,
) -> StrandAdjacencyGraph:
    """Return one strand graph containing ladder and multi-residue contact edges.

    DSSP ladder edges remain the direct hydrogen-bond evidence. An additional
    adjacency is added when several C-alpha contacts support spatial pairing of
    two DSSP E-strand nodes. Requiring contacts from distinct residues on both
    strands prevents a single turn or uncertain coordinate from creating an
    adjacency by itself.
    """

    if not isinstance(graph, StrandAdjacencyGraph):
        raise TypeError("`graph` must be a StrandAdjacencyGraph.")
    if not isinstance(residues, Sequence):
        raise TypeError("`residues` must be a sequence of residue mappings.")
    distance = _positive_finite_float(
        maximum_ca_distance_angstrom,
        "maximum_ca_distance_angstrom",
    )
    minimum_pairs = _positive_integer(
        minimum_contact_pair_count,
        "minimum_contact_pair_count",
    )
    minimum_residues = _positive_integer(
        minimum_contact_residue_count_per_strand,
        "minimum_contact_residue_count_per_strand",
    )
    coordinates = _strand_coordinates(graph, residues)
    existing_pairs = {edge.endpoints for edge in graph.edges}
    contact_edges: list[StrandEdge] = []
    node_ids = sorted(coordinates)
    for first_node_id, second_node_id in combinations(node_ids, 2):
        candidate = StrandEdge(first_node_id, second_node_id)
        if candidate.endpoints in existing_pairs:
            continue
        if _has_multiresidue_contact(
            coordinates[first_node_id],
            coordinates[second_node_id],
            maximum_distance_squared=distance * distance,
            minimum_contact_pair_count=minimum_pairs,
            minimum_contact_residue_count_per_strand=minimum_residues,
        ):
            contact_edges.append(candidate)

    return StrandAdjacencyGraph(
        author_chain_id=graph.author_chain_id,
        nodes=graph.nodes,
        edges=tuple(sorted((*graph.edges, *contact_edges))),
    )
