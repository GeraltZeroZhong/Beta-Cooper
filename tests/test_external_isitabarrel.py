from __future__ import annotations

import csv
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runner = importlib.import_module("external_methods.isitabarrel.runner")
contact_maps = importlib.import_module("external_methods.isitabarrel.contact_maps")
structure_map = importlib.import_module("external_methods.isitabarrel.structure_map")
load_results_tsv = runner.load_results_tsv
run_baseline = runner.run_baseline
generate_structure_contact_maps = contact_maps.generate_structure_contact_maps
run_structure_map_baseline = structure_map.run_structure_map_baseline
SMOKE_DATA = ROOT / "data" / "external_methods" / "isitabarrel_smoke"
STRUCTURE_SMOKE_DATA = ROOT / "data" / "external_methods" / "isitabarrel_structure_smoke"


def test_load_isitabarrel_results_from_data_fixture():
    results = load_results_tsv(SMOKE_DATA / "results.tsv")

    assert [result.sample_id for result in results] == ["toy_barrel", "toy_nonbarrel"]
    assert [result.result for result in results] == ["BARREL", "NON_BARREL"]
    assert results[0].score == 0.75
    assert results[0].decision_column == "CC2_TO_H4"


def test_run_isitabarrel_adapter_reads_data_and_writes_normalized_results(tmp_path: Path):
    output_csv = tmp_path / "normalized.csv"

    results = run_baseline(
        SMOKE_DATA / "protid_list.tsv",
        SMOKE_DATA / "maps",
        script_path=SMOKE_DATA / "fake_isitabarrel.py",
        work_dir=tmp_path / "work",
        output_path=output_csv,
        python_executable=sys.executable,
    )

    assert [result.result for result in results] == ["BARREL", "NON_BARREL"]
    assert output_csv.exists()

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "isitabarrel_structure_map"
    assert rows[0]["sample_id"] == "toy_barrel"
    assert rows[0]["result"] == "BARREL"
    assert rows[1]["sample_id"] == "toy_nonbarrel"
    assert float(rows[1]["score"]) == 0.0


def test_generate_structure_contact_maps_from_pdb_fixture(tmp_path: Path):
    generated = generate_structure_contact_maps(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "generated",
    )

    assert [record.sample_id for record in generated.records] == ["toy_barrel_A"]
    assert Path(generated.protid_list_path).read_text(encoding="utf-8") == "toy_barrel_A\n"
    assert Path(generated.residue_mapping_path).exists()

    with Path(generated.records[0].map_path).open("rb") as handle:
        contact_map = pickle.load(handle)

    assert contact_map.shape == (16, 16)
    assert contact_map.dtype == np.float32
    assert contact_map[0, 8] == 1.0
    assert contact_map[0, 1] == 0.0
    assert np.allclose(contact_map, contact_map.T)


def test_generate_structure_contact_maps_rejects_blank_author_chain(tmp_path: Path):
    structure = tmp_path / "blank-chain.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA     1       1.000   0.000   0.000  1.00 80.00           C\nEND\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Blank author chain identifiers"):
        generate_structure_contact_maps(structure, tmp_path / "generated", min_residues=1)


def test_generate_structure_contact_maps_accepts_atom_unk_polymer(tmp_path: Path):
    structure = tmp_path / "unknown-polymer.pdb"
    structure.write_text(
        "ATOM      1  CA  UNK A   1       1.000   0.000   0.000  1.00 80.00           C\n"
        "HETATM    2 CA    CA A   2       2.000   0.000   0.000  1.00 80.00          CA\n"
        "END\n",
        encoding="utf-8",
    )

    generated = generate_structure_contact_maps(
        structure,
        tmp_path / "generated",
        min_residues=1,
    )

    assert len(generated.records) == 1
    assert generated.records[0].n_residues == 1
    with Path(generated.residue_mapping_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["residue_name"], row["residue_number"]) for row in rows] == [("UNK", "1")]


def test_run_structure_map_baseline_smoke(tmp_path: Path):
    output_csv = tmp_path / "structure_baseline.csv"

    run = run_structure_map_baseline(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "structure_work",
        script_path=SMOKE_DATA / "fake_isitabarrel.py",
        output_path=output_csv,
        python_executable=sys.executable,
    )

    assert [record.sample_id for record in run.generated_maps.records] == ["toy_barrel_A"]
    assert [result.result for result in run.results] == ["BARREL"]

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "isitabarrel_structure_map"
    assert rows[0]["sample_id"] == "toy_barrel_A"
    assert rows[0]["result"] == "BARREL"


def test_isitabarrel_cli_passthrough_requires_explicit_separator():
    parser = runner.build_arg_parser()

    args, extra_args = runner._parse_args_and_passthrough(
        parser,
        ["ids.tsv", "maps", "--", "--upstream-flag", "1"],
    )
    assert args.protid_list == "ids.tsv"
    assert extra_args == ["--upstream-flag", "1"]

    with pytest.raises(SystemExit):
        runner._parse_args_and_passthrough(
            parser,
            ["ids.tsv", "maps", "--upstream-flag", "1"],
        )
