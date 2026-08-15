from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CLI_HELP_CASES = (
    (("-m", "cooper_beta"), "chain-level results CSV"),
    (("-m", "cooper_beta.evaluation"), "structure_sha256,target_author_chain_id"),
    (("scripts/annotate_bfvd_candidates_blastp.py",), "relative_path/author_chain_id"),
    (("scripts/nested_grouped_decision_experiment.py",), "target_author_chain_id"),
    (("scripts/perturbation_eval.py",), "structure_sha256,target_author_chain_id"),
    (
        ("data/scripts/build_easy_negatives_from_pisces_cath.py",),
        "pdb_id,target_author_chain_id",
    ),
    (
        ("data/scripts/mpstruc_download_and_classify.py",),
        "filename,target_author_chain_id",
    ),
    (("external_methods/foldseek/evaluate_dataset.py",), "author_chain_id,group_id"),
    (("external_methods/foldseek/runner.py",), "coverage thresholds"),
    (("external_methods/foldseek/structure_search.py",), "global TMalign"),
    (("external_methods/foldseek/structures.py",), "chain_manifest.csv"),
    (("external_methods/isitabarrel/contact_maps.py",), "protid_list.tsv"),
    (("external_methods/isitabarrel/runner.py",), "score greater than zero"),
    (("external_methods/isitabarrel/structure_map.py",), "upstream results.tsv"),
    (("external_methods/pred_tmbb2/evaluate_dataset.py",), "relative_path,author_chain_id"),
    (("external_methods/pred_tmbb2/runner.py",), "membrane beta-strand segments"),
    (("external_methods/pred_tmbb2/sequences.py",), "residue_mapping.csv"),
    (("external_methods/pred_tmbb2/structure_sequence.py",), "sequences.fasta"),
)


@pytest.mark.parametrize(("arguments", "behavior_text"), CLI_HELP_CASES)
def test_public_cli_help_is_executable_and_explains_behavior(
    arguments: tuple[str, ...],
    behavior_text: str,
) -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
    }
    completed = subprocess.run(
        [sys.executable, *arguments, "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.isascii()
    compact_help = "".join(completed.stdout.split()).lower()
    assert "usage:" in completed.stdout
    assert "Output:" in completed.stdout
    assert "default:" in completed.stdout
    assert "exit" in completed.stdout
    assert "".join(behavior_text.split()).lower() in compact_help


def test_detector_help_explains_contact_and_three_rule_defaults() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "cooper_beta", "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    compact_help = " ".join(completed.stdout.split())
    assert "at least two C-alpha pairs within 6.8 Angstrom" in compact_help
    assert "two distinct contacting residues on each strand" in compact_help
    assert "strand_adjacency_count >= 8" in compact_help
    assert "cycle_strand_count >= 4" in compact_help
    assert "cycle_strand_fraction >= 0.05" in compact_help
    assert "cycle_rank >= 1" in compact_help
    assert "Coordinate-only mmCIF inputs require one author chain" in compact_help
    assert "input.atom_site_only_max_peptide_bond_distance_angstrom=1.8" in compact_help
