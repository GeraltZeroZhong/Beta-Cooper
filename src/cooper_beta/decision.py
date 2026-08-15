"""Direct classification from three strand-graph rule groups."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .config import RuleConfig
from .constants import RESULT_BARREL, RESULT_NON_BARREL


@dataclass(frozen=True)
class DecisionInput:
    """The four observed values consumed by the three rule groups."""

    strand_adjacency_count: int
    cycle_strand_count: int
    cycle_strand_fraction: float
    cycle_rank: int

    def __post_init__(self) -> None:
        for name in ("strand_adjacency_count", "cycle_strand_count", "cycle_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        if not isfinite(self.cycle_strand_fraction) or not 0.0 <= self.cycle_strand_fraction <= 1.0:
            raise ValueError("`cycle_strand_fraction` must be finite and within [0, 1].")


@dataclass(frozen=True)
class DecisionOutcome:
    """Final classification and its rule-level explanation."""

    result: str
    reason: str


def evaluate_decision(data: DecisionInput, rules: RuleConfig) -> DecisionOutcome:
    """Classify a chain by requiring all three graph rule groups to pass."""

    failures: list[str] = []
    if data.strand_adjacency_count < rules.strand_adjacency_count.minimum:
        failures.append(
            "strand_adjacency_count "
            f"{data.strand_adjacency_count} < {rules.strand_adjacency_count.minimum}"
        )
    if data.cycle_strand_count < rules.cycle_strand_count_fraction.minimum_count:
        failures.append(
            f"cycle_strand_count {data.cycle_strand_count} < "
            f"{rules.cycle_strand_count_fraction.minimum_count}"
        )
    if data.cycle_strand_fraction < rules.cycle_strand_count_fraction.minimum_fraction:
        failures.append(
            f"cycle_strand_fraction {data.cycle_strand_fraction:.4f} < "
            f"{rules.cycle_strand_count_fraction.minimum_fraction:.4f}"
        )
    if data.cycle_rank < rules.cycle_rank.minimum:
        failures.append(f"cycle_rank {data.cycle_rank} < {rules.cycle_rank.minimum}")

    return DecisionOutcome(
        result=RESULT_NON_BARREL if failures else RESULT_BARREL,
        reason="; ".join(failures) if failures else "All three strand-graph rule groups passed",
    )
