from __future__ import annotations

import pytest

from cooper_beta.config import build_config
from cooper_beta.constants import RESULT_BARREL, RESULT_NON_BARREL
from cooper_beta.decision import DecisionInput, evaluate_decision


def _input(**updates: object) -> DecisionInput:
    values: dict[str, object] = {
        "strand_adjacency_count": 8,
        "cycle_strand_count": 4,
        "cycle_strand_fraction": 0.05,
        "cycle_rank": 1,
    }
    values.update(updates)
    return DecisionInput(**values)


def test_all_three_rule_groups_pass_at_inclusive_boundaries() -> None:
    outcome = evaluate_decision(_input(), build_config().rules)

    assert outcome.result == RESULT_BARREL
    assert outcome.reason == "All three strand-graph rule groups passed"


@pytest.mark.parametrize(
    ("updates", "expected_text"),
    [
        ({"strand_adjacency_count": 7}, "strand_adjacency_count 7 < 8"),
        ({"cycle_strand_count": 3}, "cycle_strand_count 3 < 4"),
        ({"cycle_strand_fraction": 0.04}, "cycle_strand_fraction 0.0400 < 0.0500"),
        ({"cycle_rank": 0, "cycle_strand_count": 0}, "cycle_rank 0 < 1"),
    ],
)
def test_each_rule_group_can_reject_a_chain(updates: dict[str, object], expected_text: str) -> None:
    outcome = evaluate_decision(_input(**updates), build_config().rules)

    assert outcome.result == RESULT_NON_BARREL
    assert expected_text in outcome.reason


def test_multiple_failed_rules_are_reported_without_aggregate_score() -> None:
    outcome = evaluate_decision(
        _input(
            strand_adjacency_count=3,
            cycle_strand_count=0,
            cycle_strand_fraction=0.0,
            cycle_rank=0,
        ),
        build_config().rules,
    )

    assert outcome.result == RESULT_NON_BARREL
    assert outcome.reason.count(";") == 3


@pytest.mark.parametrize(
    "updates",
    [
        {"strand_adjacency_count": -1},
        {"cycle_strand_count": True},
        {"cycle_strand_fraction": float("nan")},
        {"cycle_strand_fraction": 1.1},
        {"cycle_rank": 0.5},
    ],
)
def test_decision_input_rejects_invalid_graph_values(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _input(**updates)
