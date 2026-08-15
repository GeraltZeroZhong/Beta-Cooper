from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cooper_beta import detect
from cooper_beta.config import build_config
from cooper_beta.integrity import file_sha256
from cooper_beta.polymer_sequence import declared_polymer_sequence_for_author_chain

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPOSITORY_ROOT / "examples"


def test_curated_positive_examples_match_their_manifest() -> None:
    document = json.loads((EXAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert document["license"] == "CC-BY-4.0"
    assert {entry["uniprot_accession"] for entry in document["examples"]} == {
        "M4QT10",
        "A0A2R4ALS6",
    }
    for entry in document["examples"]:
        structure_path = EXAMPLES_DIR / entry["file"]
        assert structure_path.is_file()
        assert entry["source_record_id"]
        assert file_sha256(structure_path) == entry["sha256"]
        assert entry["scientific_label"] == "positive"
        assert entry["expected_result"] == "BARREL"
        sequence = declared_polymer_sequence_for_author_chain(
            structure_path, entry["target_author_chain_id"]
        )
        assert sequence.author_chain_id == "A"
        assert sequence.sequence_source == "mmcif_entity_poly_seq"
        assert len(sequence.sequence) == entry["sequence_length"]


@pytest.mark.skipif(
    shutil.which("mkdssp") is None and shutil.which("dssp") is None,
    reason="DSSP is required for the real-structure regression test.",
)
def test_curated_positives_match_strand_graph_and_decision_contract() -> None:
    config = build_config(
        {
            "runtime.prepare_cache_enabled": False,
            "runtime.log_console": False,
        }
    )
    run = detect(
        str(EXAMPLES_DIR),
        config=config,
        workers=1,
        prepare_workers=1,
        write_csv=False,
        print_summary=False,
        show_progress=False,
    )
    rows = {row.filename: row for row in run.rows}
    expected_graph_counts = {
        "M4QT10.cif": (8, 8, 8),
        "A0A2R4ALS6.cif": (9, 9, 8),
    }

    assert set(rows) == set(expected_graph_counts)
    for filename, (
        strand_count,
        adjacency_count,
        cycle_strand_count,
    ) in expected_graph_counts.items():
        row = rows[filename]
        assert row.author_chain_id == "A"
        assert row.strand_count == strand_count
        assert row.strand_adjacency_count == adjacency_count
        assert row.cycle_strand_count == cycle_strand_count
        assert row.cycle_strand_fraction == pytest.approx(cycle_strand_count / strand_count)
        assert row.cycle_rank == 1
        assert row.result == "BARREL"
        assert row.strand_adjacency_count >= config.rules.strand_adjacency_count.minimum
        assert row.cycle_strand_count >= config.rules.cycle_strand_count_fraction.minimum_count
        assert (
            row.cycle_strand_fraction >= config.rules.cycle_strand_count_fraction.minimum_fraction
        )
        assert row.cycle_rank >= config.rules.cycle_rank.minimum
