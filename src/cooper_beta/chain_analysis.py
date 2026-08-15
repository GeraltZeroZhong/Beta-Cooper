from __future__ import annotations

from typing import TypedDict, cast

from .config import AppConfig
from .constants import (
    RESULT_ERROR,
    RESULT_STAGE_DECISION,
    RESULT_STAGE_PREPARATION,
    RESULT_STAGE_WORKER,
)
from .decision import DecisionInput, evaluate_decision
from .models import PreparedChainPayload
from .strand_contacts import add_contact_supported_adjacencies
from .strand_graph import StrandAdjacencyGraph, StrandGraphMeasurements, measure_strand_graph


class ResidueRecord(TypedDict, total=False):
    coord: tuple[float, float, float]
    dssp_assignment_available: bool
    is_sheet: bool
    strand_node_id: str | None
    polymer_index: int
    peptide_bond_distance_to_previous_angstrom: float | None


class PreparedChain(TypedDict):
    filename: str
    source_path: str
    chain: str
    residues_data: list[ResidueRecord]
    strand_graph: StrandAdjacencyGraph
    degraded: bool
    degradation_code: str
    degradation_reason: str


def _analysis_payload(prepared: PreparedChainPayload) -> PreparedChain:
    return {
        "filename": prepared.filename,
        "source_path": prepared.source_path,
        "chain": prepared.chain,
        "residues_data": [cast(ResidueRecord, dict(item)) for item in prepared.residues_data],
        "strand_graph": prepared.strand_graph,
        "degraded": prepared.degraded,
        "degradation_code": prepared.degradation_code,
        "degradation_reason": prepared.degradation_reason,
    }


def _complete_strand_graph(
    graph: StrandAdjacencyGraph,
    residues: list[ResidueRecord],
    config: AppConfig,
) -> StrandAdjacencyGraph:
    adjacency = config.strand_adjacency
    return add_contact_supported_adjacencies(
        graph,
        residues,
        maximum_ca_distance_angstrom=adjacency.maximum_ca_distance_angstrom,
        minimum_contact_pair_count=adjacency.minimum_contact_pair_count,
        minimum_contact_residue_count_per_strand=(
            adjacency.minimum_contact_residue_count_per_strand
        ),
    )


def _dssp_unassigned_residue_count(residues: list[ResidueRecord]) -> int:
    return sum(residue["dssp_assignment_available"] is False for residue in residues)


def _result_row(
    payload: PreparedChain,
    measurements: StrandGraphMeasurements,
    *,
    result: str,
    result_stage: str,
    reason: str,
    error_code: str = "",
    degraded: bool = False,
) -> dict[str, object]:
    return {
        "filename": payload["filename"],
        "source_path": payload["source_path"],
        "author_chain_id": payload["chain"],
        "result": result,
        "result_stage": result_stage,
        "dssp_unassigned_residue_count": _dssp_unassigned_residue_count(payload["residues_data"]),
        "strand_count": measurements.strand_count,
        "strand_adjacency_count": measurements.strand_adjacency_count,
        "cycle_strand_count": measurements.cycle_strand_count,
        "cycle_strand_fraction": measurements.cycle_strand_fraction,
        "cycle_rank": measurements.cycle_rank,
        "reason": reason,
        "error_code": error_code,
        "degraded": degraded,
    }


def analyze_chain_payload(
    payload: PreparedChain | dict[str, object],
    config: AppConfig,
) -> dict[str, object]:
    """Classify one prepared chain from the three strand-graph rule groups."""

    normalized = _analysis_payload(PreparedChainPayload.from_mapping(payload))
    measurements = measure_strand_graph(
        _complete_strand_graph(normalized["strand_graph"], normalized["residues_data"], config)
    )
    if normalized["degraded"]:
        return _result_row(
            normalized,
            measurements,
            result=RESULT_ERROR,
            result_stage=RESULT_STAGE_PREPARATION,
            reason=normalized["degradation_reason"],
            error_code=normalized["degradation_code"],
            degraded=True,
        )

    outcome = evaluate_decision(
        DecisionInput(
            strand_adjacency_count=measurements.strand_adjacency_count,
            cycle_strand_count=measurements.cycle_strand_count,
            cycle_strand_fraction=measurements.cycle_strand_fraction,
            cycle_rank=measurements.cycle_rank,
        ),
        config.rules,
    )
    return _result_row(
        normalized,
        measurements,
        result=outcome.result,
        result_stage=RESULT_STAGE_DECISION,
        reason=outcome.reason,
    )


def unhandled_analysis_failure_row(
    payload: dict[str, object],
    config: AppConfig,
    exception: Exception,
) -> dict[str, object]:
    """Represent an unexpected worker failure without inventing graph evidence."""

    normalized = _analysis_payload(PreparedChainPayload.from_mapping(payload))
    measurements = measure_strand_graph(
        _complete_strand_graph(normalized["strand_graph"], normalized["residues_data"], config)
    )
    return _result_row(
        normalized,
        measurements,
        result=RESULT_ERROR,
        result_stage=RESULT_STAGE_WORKER,
        reason=f"Analysis worker failed: {type(exception).__name__}",
        error_code="ANALYSIS_WORKER_FAILED",
    )
