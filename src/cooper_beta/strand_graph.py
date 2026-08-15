from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _normalized_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"`{field_name}` cannot be empty.")
    return normalized


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{field_name}` must be an integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"`{field_name}` must be non-negative.")
    return normalized


@dataclass(frozen=True, order=True)
class StrandRange:
    """Inclusive polymer-index range assigned to one DSSP beta strand."""

    start_polymer_index: int
    end_polymer_index: int

    def __post_init__(self) -> None:
        start = _nonnegative_integer(self.start_polymer_index, "start_polymer_index")
        end = _nonnegative_integer(self.end_polymer_index, "end_polymer_index")
        if end < start:
            raise ValueError("`end_polymer_index` must be at least `start_polymer_index`.")
        object.__setattr__(self, "start_polymer_index", start)
        object.__setattr__(self, "end_polymer_index", end)

    @property
    def residue_count(self) -> int:
        """Number of polymer positions in the inclusive strand range."""

        return self.end_polymer_index - self.start_polymer_index + 1


@dataclass(frozen=True, order=True)
class StrandNode:
    """One physical beta-strand range in a chain adjacency graph."""

    node_id: str
    residue_range: StrandRange

    def __post_init__(self) -> None:
        node_id = _normalized_identifier(self.node_id, "node_id")
        if not isinstance(self.residue_range, StrandRange):
            raise TypeError("`residue_range` must be a StrandRange.")
        object.__setattr__(self, "node_id", node_id)


@dataclass(frozen=True, order=True)
class StrandEdge:
    """Undirected adjacency between two distinct physical-strand nodes."""

    first_node_id: str
    second_node_id: str

    def __post_init__(self) -> None:
        first = _normalized_identifier(self.first_node_id, "first_node_id")
        second = _normalized_identifier(self.second_node_id, "second_node_id")
        if first == second:
            raise ValueError("A strand adjacency cannot be a self-loop.")
        if second < first:
            first, second = second, first
        object.__setattr__(self, "first_node_id", first)
        object.__setattr__(self, "second_node_id", second)

    @property
    def endpoints(self) -> tuple[str, str]:
        """Canonical endpoint pair for this undirected edge."""

        return (self.first_node_id, self.second_node_id)


@dataclass(frozen=True)
class StrandAdjacencyGraph:
    """Validated, immutable strand adjacency graph for one author chain."""

    author_chain_id: str
    nodes: tuple[StrandNode, ...]
    edges: tuple[StrandEdge, ...]

    def __post_init__(self) -> None:
        author_chain_id = _normalized_identifier(self.author_chain_id, "author_chain_id")
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(node, StrandNode) for node in self.nodes
        ):
            raise TypeError("`nodes` must be a tuple of StrandNode values.")
        if not isinstance(self.edges, tuple) or any(
            not isinstance(edge, StrandEdge) for edge in self.edges
        ):
            raise TypeError("`edges` must be a tuple of StrandEdge values.")

        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Strand node identifiers must be unique within a chain graph.")
        edge_pairs = [edge.endpoints for edge in self.edges]
        if len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError("Strand adjacency edges must be unique within a chain graph.")

        known_ids = set(node_ids)
        unknown_ids = sorted(
            endpoint
            for edge in self.edges
            for endpoint in edge.endpoints
            if endpoint not in known_ids
        )
        if unknown_ids:
            raise ValueError(
                "Strand adjacency edges reference unknown node identifiers: "
                f"{sorted(set(unknown_ids))!r}."
            )
        object.__setattr__(self, "author_chain_id", author_chain_id)


@dataclass(frozen=True)
class StrandGraphMeasurements:
    """The graph values reported by Cooper-Beta and consumed by its rules."""

    author_chain_id: str
    strand_count: int
    strand_adjacency_count: int
    cycle_strand_count: int
    cycle_strand_fraction: float
    cycle_rank: int


def _adjacency(graph: StrandAdjacencyGraph) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.first_node_id].append(edge.second_node_id)
        adjacency[edge.second_node_id].append(edge.first_node_id)
    for neighbors in adjacency.values():
        neighbors.sort()
    return adjacency


def _find_bridges(adjacency: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Find every bridge using Tarjan low-link values in O(V + E)."""

    discovery_time: dict[str, int] = {}
    low_link: dict[str, int] = {}
    bridges: set[tuple[str, str]] = set()
    next_time = 0

    def visit(node_id: str, parent_id: str | None) -> None:
        nonlocal next_time
        discovery_time[node_id] = next_time
        low_link[node_id] = next_time
        next_time += 1

        for neighbor_id in adjacency[node_id]:
            if neighbor_id == parent_id:
                continue
            if neighbor_id not in discovery_time:
                visit(neighbor_id, node_id)
                low_link[node_id] = min(low_link[node_id], low_link[neighbor_id])
                if low_link[neighbor_id] > discovery_time[node_id]:
                    bridges.add(
                        (node_id, neighbor_id) if node_id < neighbor_id else (neighbor_id, node_id)
                    )
            else:
                low_link[node_id] = min(low_link[node_id], discovery_time[neighbor_id])

    for node_id in sorted(adjacency):
        if node_id not in discovery_time:
            visit(node_id, None)
    return bridges


def _component_sizes(
    node_ids: set[str],
    edge_pairs: set[tuple[str, str]],
) -> tuple[int, ...]:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for first, second in edge_pairs:
        adjacency[first].append(second)
        adjacency[second].append(first)

    sizes: list[int] = []
    unseen = set(node_ids)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in component:
                continue
            component.add(node_id)
            unseen.discard(node_id)
            stack.extend(adjacency[node_id])
        sizes.append(len(component))
    return tuple(sorted(sizes, reverse=True))


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def measure_strand_graph(graph: StrandAdjacencyGraph) -> StrandGraphMeasurements:
    """Measure the four values used by the three classification rule groups.

    In an undirected graph, an edge belongs to at least one cycle exactly when
    it is not a bridge. This definition excludes tree branches and the bridge
    between two otherwise cyclic components without requiring a Hamiltonian
    cycle or forcing every detected strand into one ring.
    """

    if not isinstance(graph, StrandAdjacencyGraph):
        raise TypeError("`graph` must be a StrandAdjacencyGraph.")

    adjacency = _adjacency(graph)
    all_node_ids = set(adjacency)
    all_edge_pairs = {edge.endpoints for edge in graph.edges}
    bridge_pairs = _find_bridges(adjacency)
    cyclic_edge_pairs = all_edge_pairs.difference(bridge_pairs)
    cyclic_node_ids = {endpoint for edge_pair in cyclic_edge_pairs for endpoint in edge_pair}

    component_count = len(_component_sizes(all_node_ids, all_edge_pairs))
    cycle_strand_count = max(
        _component_sizes(cyclic_node_ids, cyclic_edge_pairs),
        default=0,
    )
    return StrandGraphMeasurements(
        author_chain_id=graph.author_chain_id,
        strand_count=len(graph.nodes),
        strand_adjacency_count=len(graph.edges),
        cycle_strand_count=cycle_strand_count,
        cycle_strand_fraction=_fraction(cycle_strand_count, len(graph.nodes)),
        cycle_rank=len(graph.edges) - len(graph.nodes) + component_count,
    )
