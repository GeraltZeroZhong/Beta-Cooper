from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runner = importlib.import_module("external_methods.pred_tmbb2.runner")
sequences = importlib.import_module("external_methods.pred_tmbb2.sequences")
structure_sequence = importlib.import_module("external_methods.pred_tmbb2.structure_sequence")
evaluate_dataset = importlib.import_module("external_methods.pred_tmbb2.evaluate_dataset")

parse_juchmme_stdout = runner.parse_juchmme_stdout
run_baseline = runner.run_baseline
generate_structure_fasta = sequences.generate_structure_fasta
run_structure_sequence_baseline = structure_sequence.run_structure_sequence_baseline

SMOKE_DATA = ROOT / "data" / "external_methods" / "pred_tmbb2_smoke"
STRUCTURE_SMOKE_DATA = ROOT / "data" / "external_methods" / "isitabarrel_structure_smoke"


def _synthetic_sequence(sample_id: str, source_path: Path, author_chain_id: str, sequence: str):
    return sequences.GeneratedSequence(
        sample_id=sample_id,
        source_path=str(source_path),
        author_chain_id=author_chain_id,
        n_residues=len(sequence),
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        sequence_source="pdb_seqres",
        polymer_entity_id="",
        label_asym_id="",
    )


def test_parse_juchmme_stdout_from_fixture():
    output = (SMOKE_DATA / "juchmme_stdout.txt").read_text(encoding="utf-8")

    results = parse_juchmme_stdout(output)

    assert [result.sample_id for result in results] == ["toy_barrel", "toy_nonbarrel"]
    assert [result.result for result in results] == ["BARREL", "NON_BARREL"]
    assert results[0].tm_strands == 3
    assert results[0].score == 3.0
    assert results[0].reliability == 0.93
    assert results[1].logodds == -5.0


def test_parse_juchmme_stdout_rejects_fractional_sequence_length():
    output = """\
ID: >fractional
CC: len = 3.5 logodds = 1 maxProb = 0.9 (-logprob/lng) = 1
LP: MMM
"""

    with pytest.raises(ValueError, match="length must be a positive integer"):
        parse_juchmme_stdout(output)


def test_run_pred_tmbb2_adapter_reads_fasta_and_writes_normalized_results(tmp_path: Path):
    output_csv = tmp_path / "normalized.csv"

    results = run_baseline(
        SMOKE_DATA / "input.fasta",
        work_dir=tmp_path / "work",
        output_path=output_csv,
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_juchmme.py")],
    )

    assert [result.result for result in results] == ["BARREL", "NON_BARREL"]
    assert output_csv.exists()

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "pred_tmbb2_single_juchmme"
    assert rows[0]["sample_id"] == "toy_barrel"
    assert rows[0]["result"] == "BARREL"
    assert rows[0]["decision_rule"] == "LP_tm_strands>=3"
    assert rows[1]["sample_id"] == "toy_nonbarrel"
    assert float(rows[1]["score"]) == 0.0


def test_generate_structure_fasta_from_pdb_fixture(tmp_path: Path):
    generated = generate_structure_fasta(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "generated",
    )

    assert [record.sample_id for record in generated.records] == ["toy_barrel_A"]
    assert generated.records[0].sequence == "AVGSTLIFAVGSTLIF"
    assert Path(generated.fasta_path).read_text(encoding="utf-8") == (
        ">toy_barrel_A\nAVGSTLIFAVGSTLIF\n"
    )
    assert Path(generated.residue_mapping_path).exists()


def test_pdb_fasta_uses_complete_seqres_when_coordinate_residue_is_missing(
    tmp_path: Path,
):
    structure = tmp_path / "missing-coordinate.pdb"
    structure.write_text(
        """\
SEQRES   1 A    3  ALA GLY SER
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  SER A   3       2.000   0.000   0.000  1.00 20.00           C
TER
END
""",
        encoding="utf-8",
    )

    generated = generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)

    assert len(generated.records) == 1
    record = generated.records[0]
    assert record.sequence == "AGS"
    assert record.n_residues == 3
    assert record.sequence_source == "pdb_seqres"
    with Path(generated.residue_mapping_path).open(newline="", encoding="utf-8") as handle:
        mapping = list(csv.DictReader(handle))
    assert [row["monomer_id"] for row in mapping] == ["ALA", "GLY", "SER"]
    assert [row["one_letter_code"] for row in mapping] == ["A", "G", "S"]
    assert {row["author_chain_id"] for row in mapping} == {"A"}
    assert len({row["sequence_sha256"] for row in mapping}) == 1


def test_pdb_fasta_fails_when_declared_complete_sequence_is_missing(tmp_path: Path):
    structure = tmp_path / "atom-only.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no SEQRES records"):
        generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)


def test_pdb_fasta_rejects_blank_author_chain_identity(tmp_path: Path):
    structure = tmp_path / "blank-chain.pdb"
    structure.write_text(
        "SEQRES   1      1  ALA\nEND\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Blank author chain identifiers"):
        generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)


def _write_polymer_mmcif(
    path: Path,
    *,
    mapping_rows: str,
) -> None:
    path.write_text(
        f"""\
data_polymer
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 'polypeptide(L)' AGS
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 GLY
1 3 SER
#
loop_
_struct_asym.id
_struct_asym.entity_id
LABEL_LONG 1
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.pdb_strand_id
{mapping_rows}
#
""",
        encoding="utf-8",
    )


def test_mmcif_fasta_maps_complete_entity_sequence_to_exact_author_chain(
    tmp_path: Path,
):
    structure = tmp_path / "polymer.cif"
    _write_polymer_mmcif(structure, mapping_rows="LABEL_LONG AUTH-LONG")

    generated = generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)

    assert len(generated.records) == 1
    record = generated.records[0]
    assert record.author_chain_id == "AUTH-LONG"
    assert record.label_asym_id == "LABEL_LONG"
    assert record.polymer_entity_id == "1"
    assert record.sequence == "AGS"
    assert record.sequence_source == "mmcif_entity_poly_canonical"


@pytest.mark.parametrize(
    "mapping_rows",
    ["LABEL_LONG ?", "LABEL_LONG A\nLABEL_LONG B"],
    ids=["missing", "ambiguous"],
)
def test_mmcif_fasta_fails_on_missing_or_ambiguous_author_chain_mapping(
    tmp_path: Path,
    mapping_rows: str,
):
    structure = tmp_path / "polymer.cif"
    _write_polymer_mmcif(structure, mapping_rows=mapping_rows)

    with pytest.raises(ValueError, match="must map to exactly one author chain"):
        generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)


def test_mmcif_fasta_rejects_conflicting_canonical_and_monomer_sequences(
    tmp_path: Path,
):
    structure = tmp_path / "conflicting.cif"
    _write_polymer_mmcif(structure, mapping_rows="LABEL_LONG AUTH-LONG")
    structure.write_text(
        structure.read_text(encoding="utf-8").replace("1 2 GLY", "1 2 TYR"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical sequence conflicts"):
        generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)


def test_mmcif_fasta_uses_canonical_sequence_for_declared_modified_monomer(
    tmp_path: Path,
):
    structure = tmp_path / "modified.cif"
    _write_polymer_mmcif(structure, mapping_rows="LABEL_LONG AUTH-LONG")
    structure.write_text(
        structure.read_text(encoding="utf-8").replace("1 2 GLY", "1 2 DSN"),
        encoding="utf-8",
    )

    generated = generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)

    assert generated.records[0].sequence == "AGS"
    assert generated.records[0].sequence_source == "mmcif_entity_poly_canonical"


def test_mmcif_fasta_preserves_canonical_unknown_for_modified_monomer(
    tmp_path: Path,
):
    structure = tmp_path / "canonical-unknown.cif"
    _write_polymer_mmcif(structure, mapping_rows="LABEL_LONG AUTH-LONG")
    structure.write_text(
        structure.read_text(encoding="utf-8")
        .replace("1 'polypeptide(L)' AGS", "1 'polypeptide(L)' AXS")
        .replace("1 2 GLY", "1 2 TYR"),
        encoding="utf-8",
    )

    generated = generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)

    assert generated.records[0].sequence == "AXS"


def test_mmcif_fasta_accepts_explicit_heterogeneous_polymer_position(tmp_path: Path):
    structure = tmp_path / "heterogeneous.cif"
    structure.write_text(
        """\
data_heterogeneous
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 'polypeptide(L)' AKG
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 ALA n
1 2 MLZ y
1 2 LYS y
1 3 GLY n
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.pdb_strand_id
A A
#
""",
        encoding="utf-8",
    )

    generated = generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)

    assert generated.records[0].sequence == "AKG"


def test_mmcif_fasta_rejects_duplicate_position_without_heterogeneity(tmp_path: Path):
    structure = tmp_path / "duplicate-position.cif"
    structure.write_text(
        """\
data_duplicate
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 'polypeptide(L)' AKG
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 ALA n
1 2 MLZ n
1 2 LYS n
1 3 GLY n
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.pdb_strand_id
A A
#
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="without an explicit heterogeneous declaration"):
        generate_structure_fasta(structure, tmp_path / "generated", min_residues=1)


def test_run_structure_sequence_baseline_smoke(tmp_path: Path):
    output_csv = tmp_path / "structure_baseline.csv"

    run = run_structure_sequence_baseline(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "structure_work",
        output_path=output_csv,
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_juchmme.py")],
    )

    assert [record.sample_id for record in run.generated_fasta.records] == ["toy_barrel_A"]
    assert [result.result for result in run.results] == ["BARREL"]

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "pred_tmbb2_single_juchmme"
    assert rows[0]["sample_id"] == "toy_barrel_A"
    assert rows[0]["result"] == "BARREL"


def test_pred_tmbb2_target_manifest_is_strict_and_one_target_per_file(tmp_path: Path):
    structure_root = tmp_path / "structures"
    structure_root.mkdir()
    structure = structure_root / "toy.pdb"
    structure.write_text("MODEL\nEND\n", encoding="utf-8")
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("relative_path,author_chain_id\ntoy.pdb,A\ntoy.pdb,B\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one target"):
        importlib.import_module("external_methods.evaluation_common").read_target_manifest(
            duplicate,
            split_root=structure_root,
            structure_files=[structure],
        )


def test_pred_tmbb2_dataset_reports_missing_upstream_results(tmp_path: Path):
    record = _synthetic_sequence(
        "missing_A",
        tmp_path / "missing.pdb",
        "A",
        "A" * 20,
    )
    run = evaluate_dataset.SplitRun(
        split_name="positive",
        generated=sequences.GeneratedFastaSet(
            output_dir=str(tmp_path),
            fasta_path=str(tmp_path / "input.fasta"),
            residue_mapping_path=str(tmp_path / "mapping.csv"),
            records=[record],
        ),
        results=[],
    )

    with pytest.raises(ValueError, match="result identity"):
        evaluate_dataset._chain_rows_for_split(run)


def test_pred_tmbb2_dataset_fails_closed_on_error_result(tmp_path: Path):
    record = _synthetic_sequence(
        "bad_A",
        tmp_path / "bad.pdb",
        "A",
        "A" * 20,
    )
    result = runner.PredTmbb2Result(
        sample_id="bad_A",
        result="ERROR",
        score=0.0,
        tm_strands=0,
        decision_rule="failed",
        prediction_field="LP",
        topology="",
    )
    run = evaluate_dataset.SplitRun(
        "positive",
        sequences.GeneratedFastaSet(
            str(tmp_path), str(tmp_path / "input.fasta"), str(tmp_path / "map.csv"), [record]
        ),
        [result],
    )

    with pytest.raises(ValueError, match="invalid/ERROR"):
        evaluate_dataset._chain_rows_for_split(run)


def test_pred_tmbb2_dataset_rejects_truncated_topology_output(tmp_path: Path):
    record = _synthetic_sequence(
        "truncated_A",
        tmp_path / "truncated.pdb",
        "A",
        "A" * 20,
    )
    result = runner.PredTmbb2Result(
        sample_id="truncated_A",
        result="BARREL",
        score=3.0,
        tm_strands=3,
        decision_rule="LP_tm_strands>=3",
        prediction_field="LP",
        topology="MOMOM",
        length=20,
    )
    run = evaluate_dataset.SplitRun(
        "positive",
        sequences.GeneratedFastaSet(
            str(tmp_path), str(tmp_path / "input.fasta"), str(tmp_path / "map.csv"), [record]
        ),
        [result],
    )

    with pytest.raises(ValueError, match="topology length"):
        evaluate_dataset._chain_rows_for_split(run)


def test_pred_tmbb2_dataset_writes_fresh_complete_provenance_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roots = {name: tmp_path / name for name in ("positive", "negative")}
    for name, root in roots.items():
        root.mkdir()
        root.joinpath("sample.pdb").write_text(f"MODEL {name}\nEND\n", encoding="utf-8")
    checkout = tmp_path / "juchmme"
    checkout.mkdir()
    checkout.joinpath("model.dat").write_text("frozen model\n", encoding="utf-8")

    def fake_generate(input_path, output_dir, **_kwargs):
        input_root = Path(input_path).resolve()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        fasta = output / "sequences.fasta"
        fasta.write_text(
            f">{input_root.name}_A\nAAAAA\n>{input_root.name}_B\nAAAAA\n", encoding="utf-8"
        )
        records = [
            _synthetic_sequence(
                f"{input_root.name}_{chain}",
                input_root / "sample.pdb",
                chain,
                "AAAAA",
            )
            for chain in ("A", "B")
        ]
        return sequences.GeneratedFastaSet(
            str(output), str(fasta), str(output / "map.csv"), records
        )

    def fake_run(fasta_path, *, output_path, **_kwargs):
        ids = [
            line.removeprefix(">").strip()
            for line in Path(fasta_path).read_text(encoding="utf-8").splitlines()
            if line.startswith(">")
        ]
        assert output_path is None
        return [
            runner.PredTmbb2Result(
                sample_id=sample_id,
                result="BARREL" if sample_id.endswith("_B") else "NON_BARREL",
                score=3.0 if sample_id.endswith("_B") else 0.0,
                tm_strands=3 if sample_id.endswith("_B") else 0,
                decision_rule="unit",
                prediction_field="LP",
                topology="MOMOM" if sample_id.endswith("_B") else "OOOOO",
                length=5,
            )
            for sample_id in ids
        ]

    monkeypatch.setattr(evaluate_dataset, "generate_structure_fasta", fake_generate)
    monkeypatch.setattr(evaluate_dataset, "run_baseline", fake_run)
    run_dir = evaluate_dataset.run_dataset(
        roots["positive"],
        roots["negative"],
        tmp_path / "outputs",
        juchmme_dir=checkout,
        positive_target_manifest=None,
        negative_target_manifest=None,
        metric_level="file",
        min_residues=1,
        prediction_field="LP",
        min_tm_strands=3,
        java_executable=sys.executable,
        timeout=None,
        tag="file-only",
    )

    assert run_dir.parent == (tmp_path / "outputs").resolve()
    assert not (run_dir / "target_chain_results.csv").exists()
    with (run_dir / "chain_predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    assert all(row["y_true"] == "" and row["split"] == "" for row in predictions)
    assert all(row["is_target_author_chain"] == "False" for row in predictions)
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert [row["level"] for row in summary] == ["file"]
    assert summary[0]["n_used"] == "2"
    assert summary[0]["TP"] == "1"
    assert summary[0]["FP"] == "1"
    manifest = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"
    assert manifest["parameters"]["filtered_out_policy"] == "strict"
    assert manifest["external_software"]["juchmme_checkout"]["inventory_sha256"]
    assert all("sha256" in artifact for artifact in manifest["outputs"]["artifacts"])
