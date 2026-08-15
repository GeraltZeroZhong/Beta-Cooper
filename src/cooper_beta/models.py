from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from numbers import Integral, Real
from typing import Any, cast

from .constants import (
    DEFAULT_RESULT_COLUMNS,
    RESULT_BARREL,
    RESULT_ERROR,
    RESULT_NON_BARREL,
    RESULT_STAGE_DECISION,
    RESULT_STAGE_PREPARATION,
    RESULT_STAGES,
    SERIALIZED_FRACTION_ABSOLUTE_TOLERANCE,
)
from .strand_graph import StrandAdjacencyGraph, StrandEdge, StrandNode, StrandRange


def _require_fields(mapping: Mapping[str, object], names: Iterable[str], context: str) -> None:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing!r}.")


def _string(value: object, name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"`{name}` must be a string.")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ValueError(f"`{name}` cannot be empty.")
    return normalized


def _boolean(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"`{name}` must be a boolean.")


def _strict_boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"`{name}` must be a boolean.")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be numeric.")
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be numeric.") from exc
    if not isfinite(numeric):
        raise ValueError(f"`{name}` must be finite.")
    return numeric


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be an integer.")
    if isinstance(value, Integral):
        numeric = int(value)
    elif isinstance(value, Real) and isfinite(float(value)) and float(value).is_integer():
        numeric = int(float(value))
    elif isinstance(value, str):
        try:
            numeric = int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"`{name}` must be an integer.") from exc
    else:
        raise ValueError(f"`{name}` must be an integer.")
    return numeric


def _nonnegative_int(value: object, name: str) -> int:
    numeric = _integer(value, name)
    if numeric < 0:
        raise ValueError(f"`{name}` must be non-negative.")
    return numeric


@dataclass(frozen=True)
class ResidueRecord:
    """C-alpha residue record passed from preparation to graph analysis."""

    res_id: int
    coord: tuple[float, float, float]
    dssp_assignment_available: bool
    is_sheet: bool
    strand_node_id: str | None
    polymer_index: int
    peptide_bond_distance_to_previous_angstrom: float | None
    chain: str
    resseq: int
    icode: str
    hetfield: str
    res_uid: dict[str, object]

    @classmethod
    def from_mapping(cls, residue: Mapping[str, object]) -> ResidueRecord:
        required = (
            "res_id",
            "coord",
            "dssp_assignment_available",
            "is_sheet",
            "strand_node_id",
            "polymer_index",
            "peptide_bond_distance_to_previous_angstrom",
            "chain",
            "resseq",
            "icode",
            "hetfield",
            "res_uid",
        )
        _require_fields(residue, required, "Residue record")
        coord_value = residue["coord"]
        if not isinstance(coord_value, (list, tuple)) or len(coord_value) != 3:
            raise ValueError("Residue `coord` must contain three values.")
        coord = cast(
            tuple[float, float, float],
            tuple(_finite_float(value, "coord") for value in coord_value),
        )
        assignment_available = _strict_boolean(
            residue["dssp_assignment_available"], "dssp_assignment_available"
        )
        is_sheet = _strict_boolean(residue["is_sheet"], "is_sheet")
        raw_node_id = residue["strand_node_id"]
        node_id = (
            None
            if raw_node_id is None
            else _string(raw_node_id, "strand_node_id", allow_empty=False)
        )
        if not assignment_available and (is_sheet or node_id is not None):
            raise ValueError("A residue without a DSSP assignment cannot belong to a strand.")
        if node_id is not None and not is_sheet:
            raise ValueError("A residue with a strand node must be a sheet residue.")

        chain = _string(residue["chain"], "chain", allow_empty=False)
        resseq = _integer(residue["resseq"], "resseq")
        res_id = _integer(residue["res_id"], "res_id")
        polymer_index = _nonnegative_int(residue["polymer_index"], "polymer_index")
        icode = _string(residue["icode"], "icode")
        hetfield = _string(residue["hetfield"], "hetfield")
        if res_id != resseq:
            raise ValueError("`res_id` and `resseq` must agree.")

        raw_distance = residue["peptide_bond_distance_to_previous_angstrom"]
        peptide_distance = (
            None
            if raw_distance is None
            else _finite_float(raw_distance, "peptide_bond_distance_to_previous_angstrom")
        )
        if peptide_distance is not None and peptide_distance < 0.0:
            raise ValueError("Peptide-bond distance must be non-negative.")

        raw_uid = residue["res_uid"]
        if not isinstance(raw_uid, Mapping):
            raise ValueError("Residue `res_uid` must be a mapping.")
        _require_fields(raw_uid, ("chain", "hetfield", "resseq", "icode"), "Residue UID")
        uid = {
            "chain": _string(raw_uid["chain"], "res_uid.chain", allow_empty=False),
            "hetfield": _string(raw_uid["hetfield"], "res_uid.hetfield"),
            "resseq": _integer(raw_uid["resseq"], "res_uid.resseq"),
            "icode": _string(raw_uid["icode"], "res_uid.icode"),
        }
        if uid != {"chain": chain, "hetfield": hetfield, "resseq": resseq, "icode": icode}:
            raise ValueError("Residue UID must match the residue identity.")

        return cls(
            res_id=res_id,
            coord=coord,
            dssp_assignment_available=assignment_available,
            is_sheet=is_sheet,
            strand_node_id=node_id,
            polymer_index=polymer_index,
            peptide_bond_distance_to_previous_angstrom=peptide_distance,
            chain=chain,
            resseq=resseq,
            icode=icode,
            hetfield=hetfield,
            res_uid=uid,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def strand_graph_from_mapping(
    graph: Mapping[str, object], *, context: str = "Strand graph"
) -> StrandAdjacencyGraph:
    """Parse the JSON-compatible strand graph carried across process boundaries."""

    _require_fields(graph, ("author_chain_id", "nodes", "edges"), context)
    chain = _string(graph["author_chain_id"], f"{context}.author_chain_id", allow_empty=False)
    raw_nodes = graph["nodes"]
    raw_edges = graph["edges"]
    if not isinstance(raw_nodes, list) or not all(isinstance(item, Mapping) for item in raw_nodes):
        raise TypeError(f"`{context}.nodes` must be a list of mappings.")
    if not isinstance(raw_edges, list) or not all(isinstance(item, Mapping) for item in raw_edges):
        raise TypeError(f"`{context}.edges` must be a list of mappings.")

    nodes: list[StrandNode] = []
    for index, value in enumerate(raw_nodes):
        item = cast(Mapping[str, object], value)
        _require_fields(
            item, ("node_id", "start_polymer_index", "end_polymer_index"), f"node {index}"
        )
        nodes.append(
            StrandNode(
                node_id=_string(item["node_id"], "node_id", allow_empty=False),
                residue_range=StrandRange(
                    _nonnegative_int(item["start_polymer_index"], "start_polymer_index"),
                    _nonnegative_int(item["end_polymer_index"], "end_polymer_index"),
                ),
            )
        )
    nodes.sort(key=lambda node: (node.residue_range.start_polymer_index, node.node_id))
    if any(
        current.residue_range.start_polymer_index <= previous.residue_range.end_polymer_index
        for previous, current in zip(nodes, nodes[1:], strict=False)
    ):
        raise ValueError("Strand ranges cannot overlap.")

    edges: list[StrandEdge] = []
    for index, value in enumerate(raw_edges):
        item = cast(Mapping[str, object], value)
        _require_fields(item, ("first_node_id", "second_node_id"), f"edge {index}")
        edges.append(
            StrandEdge(
                _string(item["first_node_id"], "first_node_id", allow_empty=False),
                _string(item["second_node_id"], "second_node_id", allow_empty=False),
            )
        )
    return StrandAdjacencyGraph(chain, tuple(nodes), tuple(sorted(edges)))


def strand_graph_to_mapping(graph: StrandAdjacencyGraph) -> dict[str, object]:
    return {
        "author_chain_id": graph.author_chain_id,
        "nodes": [
            {
                "node_id": node.node_id,
                "start_polymer_index": node.residue_range.start_polymer_index,
                "end_polymer_index": node.residue_range.end_polymer_index,
            }
            for node in graph.nodes
        ],
        "edges": [
            {"first_node_id": edge.first_node_id, "second_node_id": edge.second_node_id}
            for edge in graph.edges
        ],
    }


@dataclass(frozen=True)
class PreparedChainPayload:
    """Prepared per-chain data passed to the analysis workers."""

    filename: str
    source_path: str
    chain: str
    residues_data: list[dict[str, object]]
    strand_graph: StrandAdjacencyGraph
    degraded: bool = False
    degradation_code: str = ""
    degradation_reason: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PreparedChainPayload:
        _require_fields(
            payload,
            ("filename", "source_path", "chain", "residues_data", "strand_graph"),
            "Prepared chain payload",
        )
        residues = payload["residues_data"]
        graph_value = payload["strand_graph"]
        if not isinstance(residues, list) or not all(
            isinstance(item, Mapping) for item in residues
        ):
            raise TypeError("`residues_data` must be a list of mappings.")
        if not isinstance(graph_value, Mapping):
            raise TypeError("`strand_graph` must be a mapping.")
        chain = _string(payload["chain"], "chain", allow_empty=False)
        graph = strand_graph_from_mapping(graph_value)
        if graph.author_chain_id != chain:
            raise ValueError("Strand graph author chain must match the payload chain.")

        normalized: list[dict[str, object]] = []
        previous_index = -1
        node_by_id = {node.node_id: node for node in graph.nodes}
        for value in residues:
            record = ResidueRecord.from_mapping(cast(Mapping[str, object], value))
            if record.chain != chain or record.polymer_index <= previous_index:
                raise ValueError("Residues must match the chain and increase by polymer index.")
            if record.strand_node_id is not None:
                node = node_by_id.get(record.strand_node_id)
                if node is None or not (
                    node.residue_range.start_polymer_index
                    <= record.polymer_index
                    <= node.residue_range.end_polymer_index
                ):
                    raise ValueError("Residue strand membership is inconsistent with the graph.")
            previous_index = record.polymer_index
            item = record.to_dict()
            item["coord"] = list(record.coord)
            normalized.append(item)

        degraded = _strict_boolean(payload.get("degraded", False), "degraded")
        code = _string(payload.get("degradation_code", ""), "degradation_code")
        reason = _string(payload.get("degradation_reason", ""), "degradation_reason")
        if degraded != bool(code and reason):
            raise ValueError(
                "Degraded payloads require one code and reason; normal payloads require neither."
            )
        return cls(
            filename=_string(payload["filename"], "filename", allow_empty=False),
            source_path=_string(payload["source_path"], "source_path", allow_empty=False),
            chain=chain,
            residues_data=normalized,
            strand_graph=graph,
            degraded=degraded,
            degradation_code=code,
            degradation_reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "source_path": self.source_path,
            "chain": self.chain,
            "residues_data": [dict(item) for item in self.residues_data],
            "strand_graph": strand_graph_to_mapping(self.strand_graph),
            "degraded": self.degraded,
            "degradation_code": self.degradation_code,
            "degradation_reason": self.degradation_reason,
        }


@dataclass(frozen=True)
class DetectionResult:
    """Public result for one structure chain."""

    filename: str
    source_path: str
    author_chain_id: str
    result: str
    result_stage: str
    dssp_unassigned_residue_count: int
    strand_count: int
    strand_adjacency_count: int
    cycle_strand_count: int
    cycle_strand_fraction: float
    cycle_rank: int
    reason: str
    error_code: str
    degraded: bool

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> DetectionResult:
        expected = set(DEFAULT_RESULT_COLUMNS)
        actual = set(row)
        if actual != expected:
            raise ValueError(
                "Detection row does not match the public schema: "
                f"missing={sorted(expected - actual)!r}, unexpected={sorted(actual - expected)!r}."
            )
        result = _string(row["result"], "result", allow_empty=False)
        if result not in {RESULT_BARREL, RESULT_NON_BARREL, RESULT_ERROR}:
            raise ValueError(f"Unknown detection result {result!r}.")
        stage = _string(row["result_stage"], "result_stage", allow_empty=False)
        if stage not in RESULT_STAGES:
            raise ValueError(f"Unknown result stage {stage!r}.")
        chain = _string(row["author_chain_id"], "author_chain_id")
        if not chain and not (result == RESULT_ERROR and stage == RESULT_STAGE_PREPARATION):
            raise ValueError("Only file-level preparation errors may omit the author chain.")

        counts = {
            name: _nonnegative_int(row[name], name)
            for name in (
                "dssp_unassigned_residue_count",
                "strand_count",
                "strand_adjacency_count",
                "cycle_strand_count",
                "cycle_rank",
            )
        }
        fraction = _finite_float(row["cycle_strand_fraction"], "cycle_strand_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("`cycle_strand_fraction` must be within [0, 1].")
        if counts["cycle_strand_count"] > counts["strand_count"]:
            raise ValueError("`cycle_strand_count` cannot exceed `strand_count`.")
        expected_fraction = (
            counts["cycle_strand_count"] / counts["strand_count"] if counts["strand_count"] else 0.0
        )
        if abs(fraction - expected_fraction) > SERIALIZED_FRACTION_ABSOLUTE_TOLERANCE:
            raise ValueError("`cycle_strand_fraction` is inconsistent with strand counts.")
        if (counts["cycle_strand_count"] == 0) != (counts["cycle_rank"] == 0):
            raise ValueError("Cycle strand count and cycle rank must agree on cycle presence.")
        if counts["cycle_rank"] > counts["strand_adjacency_count"]:
            raise ValueError("`cycle_rank` cannot exceed `strand_adjacency_count`.")
        if not chain and any(counts.values()):
            raise ValueError("A file-level preparation error cannot carry chain evidence.")

        error_code = _string(row["error_code"], "error_code")
        degraded = _boolean(row["degraded"], "degraded")
        if (result == RESULT_ERROR) != bool(error_code):
            raise ValueError("Exactly ERROR rows require an error code.")
        if degraded and result != RESULT_ERROR:
            raise ValueError("Degraded rows must be errors.")
        if result in {RESULT_BARREL, RESULT_NON_BARREL} and stage != RESULT_STAGE_DECISION:
            raise ValueError("Classified rows must use the decision stage.")
        if result == RESULT_ERROR and stage == RESULT_STAGE_DECISION:
            raise ValueError("Decision evaluation does not produce ERROR rows.")

        return cls(
            filename=_string(row["filename"], "filename", allow_empty=False),
            source_path=_string(row["source_path"], "source_path", allow_empty=False),
            author_chain_id=chain,
            result=result,
            result_stage=stage,
            dssp_unassigned_residue_count=counts["dssp_unassigned_residue_count"],
            strand_count=counts["strand_count"],
            strand_adjacency_count=counts["strand_adjacency_count"],
            cycle_strand_count=counts["cycle_strand_count"],
            cycle_strand_fraction=fraction,
            cycle_rank=counts["cycle_rank"],
            reason=_string(row["reason"], "reason", allow_empty=False),
            error_code=error_code,
            degraded=degraded,
        )

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        return {name: values[name] for name in DEFAULT_RESULT_COLUMNS}


@dataclass(frozen=True)
class PipelineRunResult:
    """Structured result for a complete Cooper-Beta run."""

    rows: list[DetectionResult]
    input_files: list[str] = field(default_factory=list)
    output_path: str | None = None
    config: object | None = None

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        input_files: Iterable[str] | None = None,
        output_path: str | None = None,
        config: object | None = None,
    ) -> PipelineRunResult:
        return cls(
            rows=[DetectionResult.from_row(row) for row in rows],
            input_files=list(input_files or []),
            output_path=output_path,
            config=config,
        )

    @property
    def result_counts(self) -> dict[str, int]:
        return dict(Counter(row.result for row in self.rows))

    def to_rows(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]
