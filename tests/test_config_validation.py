from __future__ import annotations

from dataclasses import fields, replace

import pytest

from cooper_beta.config import build_config, validate_config
from cooper_beta.exceptions import ConfigValidationError


@pytest.mark.parametrize(
    "overrides",
    [
        {"strand_adjacency.maximum_ca_distance_angstrom": 0.0},
        {"strand_adjacency.minimum_contact_pair_count": 0},
        {"strand_adjacency.minimum_contact_residue_count_per_strand": 0},
        {"rules.strand_adjacency_count.minimum": 0},
        {"rules.cycle_strand_count_fraction.minimum_count": 0},
        {"rules.cycle_strand_count_fraction.minimum_fraction": 0.0},
        {"rules.cycle_strand_count_fraction.minimum_fraction": 1.1},
        {"rules.cycle_rank.minimum": 0},
        {"input.dssp_sheet_codes": ["E", "H"]},
        {"output.existing_artifact_policy": "overwrite"},
    ],
)
def test_invalid_current_parameters_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigValidationError):
        build_config(overrides)


def test_configuration_exposes_exactly_three_rule_groups() -> None:
    config = build_config()

    assert [item.name for item in fields(config.rules)] == [
        "strand_adjacency_count",
        "cycle_strand_count_fraction",
        "cycle_rank",
    ]
    assert config.rules.strand_adjacency_count.minimum == 8
    assert config.rules.cycle_strand_count_fraction.minimum_count == 4
    assert config.rules.cycle_strand_count_fraction.minimum_fraction == 0.05
    assert config.rules.cycle_rank.minimum == 1


def test_contact_definition_is_one_required_configuration_section() -> None:
    config = build_config()
    assert [item.name for item in fields(config.strand_adjacency)] == [
        "maximum_ca_distance_angstrom",
        "minimum_contact_pair_count",
        "minimum_contact_residue_count_per_strand",
    ]
    with pytest.raises(ConfigValidationError):
        build_config({"strand_adjacency.enabled": False})


def test_direct_dataclass_construction_still_checks_exact_types() -> None:
    config = build_config()
    malformed = replace(
        config,
        rules=replace(
            config.rules,
            cycle_rank=replace(config.rules.cycle_rank, minimum=1.5),
        ),
    )
    with pytest.raises(ConfigValidationError, match="integer"):
        validate_config(malformed)


def test_sequence_configuration_values_are_immutable_tuples() -> None:
    config = build_config()
    assert isinstance(config.input.allowed_suffixes, tuple)
    assert isinstance(config.input.dssp_sheet_codes, tuple)
