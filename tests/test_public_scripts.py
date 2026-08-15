from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cooper_beta.constants import DEFAULT_RESULT_COLUMNS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _complete_sequence_truth(annotate, relative_path: str, sequence: str = "AG"):
    return annotate.CompleteSequenceTruth(
        relative_path=relative_path,
        author_chain_id="A",
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        sequence_source="bfvd_catalog_fasta",
        source_accession=f"BFVD:{relative_path}",
        curation_evidence="frozen source FASTA exact accession mapping",
    )


def _write_complete_sequence_truth(
    path: Path,
    *,
    relative_path: str,
    sequence: str,
    author_chain_id: str = "A",
) -> Path:
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    path.write_text(
        "relative_path,author_chain_id,sequence,sequence_sha256,sequence_source,"
        "source_accession,curation_evidence\n"
        f"{relative_path},{author_chain_id},{sequence},{digest},bfvd_catalog_fasta,"
        "BFVD:TEST,frozen source FASTA exact accession mapping\n",
        encoding="utf-8",
    )
    return path


def _validated_detector_document(*input_paths: Path) -> dict[str, object]:
    return {
        "scientific_config_hash": "a" * 64,
        "producer_identity_hash": "b" * 64,
        "_validated_input_paths": [str(path.resolve()) for path in input_paths],
        "_validated_input_hashes": ["c" * 64 for _path in input_paths],
    }


def _detector_row(structure: Path, *, cycle_fraction: object = 1.0) -> dict[str, object]:
    return {
        "filename": structure.name,
        "source_path": str(structure.resolve()),
        "author_chain_id": "A",
        "result": "BARREL",
        "result_stage": "decision",
        "dssp_unassigned_residue_count": 0,
        "strand_count": 8,
        "strand_adjacency_count": 8,
        "cycle_strand_count": 8,
        "cycle_strand_fraction": cycle_fraction,
        "cycle_rank": 1,
        "reason": "All three strand-graph rules passed",
        "error_code": "",
        "degraded": False,
    }


def _write_detector_csv(path: Path, structure: Path, *, cycle_fraction: object = 1.0) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(_detector_row(structure, cycle_fraction=cycle_fraction))


def test_blast_annotation_structure_lookup_rejects_unsafe_and_ambiguous_paths(
    tmp_path: Path,
):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures_dir = tmp_path / "structures"
    (structures_dir / "a").mkdir(parents=True)
    (structures_dir / "b").mkdir(parents=True)
    (structures_dir / "a" / "same.pdb").write_text("HEADER a\n", encoding="utf-8")
    (structures_dir / "b" / "same.pdb").write_text("HEADER b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe"):
        annotate.resolve_structure_path(
            structures_dir,
            "../same.pdb",
            recursive=True,
            cache={},
        )

    with pytest.raises(ValueError, match="Ambiguous"):
        annotate.resolve_structure_path(
            structures_dir,
            "same.pdb",
            recursive=True,
            cache={},
        )

    (structures_dir / "same.pdb").write_text("HEADER wrong substitute\n", encoding="utf-8")
    assert (
        annotate.resolve_structure_path(
            structures_dir,
            "missing/same.pdb",
            recursive=True,
            cache={},
        )
        is None
    )


def test_blast_candidate_builder_keeps_duplicate_basenames_when_source_path_differs(
    tmp_path: Path,
):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures_dir = tmp_path / "structures"
    first = structures_dir / "a" / "same.pdb"
    second = structures_dir / "b" / "same.pdb"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    pdb_text = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  N   GLY A   2       2.000   0.000   0.000  1.00 80.00           N
ATOM      4  CA  GLY A   2       3.000   0.000   0.000  1.00 80.00           C
END
"""
    first.write_text(pdb_text, encoding="utf-8")
    second.write_text(pdb_text, encoding="utf-8")

    candidates, sequences, duplicates = annotate.build_candidates(
        [
            {
                "filename": "same.pdb",
                "source_path": str(first),
                "author_chain_id": "A",
            },
            {
                "filename": "same.pdb",
                "source_path": str(second),
                "author_chain_id": "A",
            },
        ],
        structures_dir,
        min_query_length=1,
        recursive=False,
        sequence_truth={
            ("a/same.pdb", "A"): _complete_sequence_truth(annotate, "a/same.pdb"),
            ("b/same.pdb", "A"): _complete_sequence_truth(annotate, "b/same.pdb"),
        },
    )

    assert duplicates == 0
    assert len(candidates) == 2
    assert len(sequences) == 2
    assert candidates[0].query_id != candidates[1].query_id


def test_bfvd_uses_complete_sequence_not_truncated_observed_residues(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    structure = structures_dir / "missing-coordinate.pdb"
    structure.write_text(
        "SEQRES   1 A    3  ALA GLY SER\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C\n"
        "ATOM      2  CA  SER A   3       2.000   0.000   0.000  1.00 80.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    truth = _complete_sequence_truth(annotate, structure.name, "AGS")

    candidates, sequences, duplicates = annotate.build_candidates(
        [
            {
                "filename": structure.name,
                "source_path": str(structure),
                "author_chain_id": "A",
            }
        ],
        structures_dir,
        min_query_length=1,
        recursive=False,
        sequence_truth={(structure.name, "A"): truth},
    )

    assert duplicates == 0
    assert list(sequences.values()) == ["AGS"]
    assert candidates[0].sequence_length == 3
    assert candidates[0].sequence_sha256 == hashlib.sha256(b"AGS").hexdigest()
    assert candidates[0].structure_declaration_status == "verified_against_pdb_seqres"
    annotation_rows, _ = annotate.annotate_candidates(
        candidates,
        {},
        SimpleNamespace(hit_evalue=1e-5, min_pident=25.0, min_qcov=80.0),
        tmp_path / "annotations.csv",
    )
    assert annotation_rows[0]["sequence_sha256"] == candidates[0].sequence_sha256
    assert annotation_rows[0]["sequence_source"] == "bfvd_catalog_fasta"
    assert annotation_rows[0]["sequence_source_accession"].startswith("BFVD:")
    assert annotation_rows[0]["sequence_completeness_policy"] == (
        annotate.BFVD_SEQUENCE_COMPLETENESS_POLICY
    )
    assert annotation_rows[0]["structure_declaration_status"] == "verified_against_pdb_seqres"

    # A two-residue alignment covers only 2/3 of the complete query, not 100% of
    # the two coordinate-observed residues used by the removed implementation.
    full_query_coverage = 2 / candidates[0].sequence_length * 100.0
    hit = annotate.BlastHit(
        qseqid=candidates[0].query_id,
        saccver="ACC.1",
        pident=100.0,
        length=2,
        qcovs=full_query_coverage,
        evalue=1e-20,
        bitscore=20.0,
        sscinames="Example species",
        sskingdoms="Bacteria",
        stitle="Example protein",
    )
    assert not annotate.hit_passes(hit, 1e-5, 25.0, 80.0)


def test_bfvd_atom_only_structure_requires_exact_frozen_complete_truth(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    structure = structures_dir / "atom-only.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C\nEND\n",
        encoding="utf-8",
    )
    selected = [
        {
            "filename": structure.name,
            "source_path": str(structure),
            "author_chain_id": "A",
        }
    ]

    with pytest.raises(ValueError, match="does not exactly cover"):
        annotate.build_candidates(
            selected,
            structures_dir,
            min_query_length=1,
            recursive=False,
            sequence_truth={},
        )

    truth = _complete_sequence_truth(annotate, structure.name, "AGS")
    with pytest.raises(ValueError, match="not a hash-validated detector input"):
        annotate.build_candidates(
            selected,
            structures_dir,
            min_query_length=1,
            recursive=False,
            sequence_truth={(structure.name, "A"): truth},
            detector_input_paths=frozenset({tmp_path / "different.pdb"}),
        )
    candidates, sequences, _ = annotate.build_candidates(
        selected,
        structures_dir,
        min_query_length=1,
        recursive=False,
        sequence_truth={(structure.name, "A"): truth},
    )
    assert list(sequences.values()) == ["AGS"]
    assert (
        candidates[0].structure_declaration_status
        == "independent_frozen_truth_no_structure_declaration"
    )


def test_bfvd_complete_sequence_truth_must_exactly_cover_final_candidates(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    structure = structures_dir / "candidate.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C\nEND\n",
        encoding="utf-8",
    )
    truths = {
        (structure.name, "A"): _complete_sequence_truth(annotate, structure.name, "A"),
        ("unselected.pdb", "A"): _complete_sequence_truth(annotate, "unselected.pdb", "A"),
    }

    with pytest.raises(ValueError, match="exactly cover the final unique candidate"):
        annotate.build_candidates(
            [
                {
                    "filename": structure.name,
                    "source_path": str(structure),
                    "author_chain_id": "A",
                }
            ],
            structures_dir,
            min_query_length=1,
            recursive=False,
            sequence_truth=truths,
        )


def test_bfvd_complete_sequence_truth_rejects_blank_and_duplicate_targets(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    digest = hashlib.sha256(b"AGS").hexdigest()
    header = (
        "relative_path,author_chain_id,sequence,sequence_sha256,sequence_source,"
        "source_accession,curation_evidence\n"
    )
    blank = tmp_path / "blank.csv"
    blank.write_text(
        header + f"candidate.pdb,,AGS,{digest},bfvd_catalog_fasta,BFVD:1,independent FASTA\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blank author_chain_id"):
        annotate.load_complete_sequence_truth(blank)

    duplicate = tmp_path / "duplicate.csv"
    row = f"candidate.pdb,A,AGS,{digest},bfvd_catalog_fasta,BFVD:1,independent FASTA\n"
    duplicate.write_text(header + row + row, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate complete-sequence truth target"):
        annotate.load_complete_sequence_truth(duplicate)


def test_blast_annotation_writes_headers_for_empty_outputs(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    candidates_csv = tmp_path / "candidates.csv"
    annotations_csv = tmp_path / "annotations.csv"

    annotate.write_candidate_manifest([], candidates_csv)
    annotate.annotate_candidates(
        [],
        {},
        type(
            "Args",
            (),
            {"hit_evalue": 1e-5, "min_pident": 25.0, "min_qcov": 50.0},
        )(),
        annotations_csv,
    )

    assert candidates_csv.read_text(encoding="utf-8").startswith(
        "query_id,filename,author_chain_id"
    )
    assert annotations_csv.read_text(encoding="utf-8").startswith(
        "query_id,filename,author_chain_id,source_path"
    )
    annotation_header = annotations_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {
        "sequence_sha256",
        "sequence_source",
        "sequence_source_accession",
        "sequence_evidence",
        "sequence_completeness_policy",
        "structure_declaration_status",
    }.issubset(annotation_header)


def _blast_args(
    *,
    database: str,
    blastp: str = "blastp",
    remote: bool = False,
    release_id: str | None = None,
    search_evalue: str = "1e-5",
):
    return SimpleNamespace(
        blastp=blastp,
        database_release_id=release_id,
        db=database,
        entrez_query=None,
        max_target_seqs=10,
        remote=remote,
        search_evalue=search_evalue,
        threads=2,
    )


def _valid_blast_row(query_id: str = "query_1") -> str:
    return (
        f"{query_id}\tACC.1\t50.0\t20\t100.0\t1e-20\t80.0\t"
        "Example species\tBacteria\tExample protein\n"
    )


def test_blast_tsv_parser_fails_fast_on_schema_numeric_and_query_errors(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    blast_tsv = tmp_path / "hits.tsv"

    blast_tsv.write_text("query_1\tACC.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected exactly"):
        annotate.read_blast_hits(blast_tsv)

    bad_numeric = _valid_blast_row().replace("\t50.0\t", "\tnot-a-number\t", 1)
    blast_tsv.write_text(bad_numeric, encoding="utf-8")
    with pytest.raises(ValueError, match="pident must be numeric"):
        annotate.read_blast_hits(blast_tsv)

    blast_tsv.write_text(_valid_blast_row("unexpected"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown query ID"):
        annotate.read_blast_hits(blast_tsv, expected_query_ids={"query_1"})


def test_candidate_results_csv_requires_public_schema_and_finite_graph_values(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    results = tmp_path / "results.csv"
    results.write_text(
        "filename,author_chain_id,result,reason\na.pdb,A,BARREL,OK\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="public schema"):
        annotate.read_candidate_rows(results, "BARREL")

    structure = tmp_path / "a.pdb"
    structure.write_text("HEADER\n", encoding="utf-8")
    _write_detector_csv(results, structure, cycle_fraction="nan")
    with pytest.raises(ValueError, match="finite"):
        annotate.read_candidate_rows(results, "BARREL")


def test_local_database_identity_hashes_key_index_files(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    database = tmp_path / "protein_db"
    index = tmp_path / "protein_db.pin"
    header = tmp_path / "protein_db.phr"
    sequence = tmp_path / "protein_db.psq"
    index.write_bytes(b"index-content")
    header.write_bytes(b"header-content")
    sequence.write_bytes(b"sequence-content")
    args = _blast_args(database=str(database), release_id="diagnostic-label")

    first = annotate.resolve_database_identity(args)
    index.write_bytes(b"index-current")
    second = annotate.resolve_database_identity(args)

    assert first["kind"] == "local_index_hashes"
    assert first["declared_release_label"] == "diagnostic-label"
    assert first["immutable_identity_available"] is True
    assert first["resolved_prefix"] == str(database.resolve())
    first_hashes = {
        Path(state["path"]).suffix: state["sha256"] for state in first["identity_files"]
    }
    second_hashes = {
        Path(state["path"]).suffix: state["sha256"] for state in second["identity_files"]
    }
    assert first_hashes[".pin"] != second_hashes[".pin"]


def test_local_database_alias_is_rejected_even_when_volumes_are_checksum_pinned(
    tmp_path: Path,
):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    database = tmp_path / "protein_db"
    for volume in ("00", "01"):
        for suffix in ("phr", "pin", "psq"):
            tmp_path.joinpath(f"protein_db.{volume}.{suffix}").write_bytes(
                f"{volume}-{suffix}".encode()
            )
    alias = tmp_path / "protein_db.pal"
    alias.write_text("TITLE proteins\nDBLIST protein_db.00 protein_db.01\n", encoding="utf-8")
    args = _blast_args(database=str(database), release_id="diagnostic-label")

    with pytest.raises(ValueError, match=r"\.pal alias.*not supported"):
        annotate.resolve_database_identity(args)

    alias.write_text("TITLE proteins\nDBLIST protein_db.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="concrete checksum-pinned"):
        annotate.resolve_database_identity(args)


def test_local_release_label_cannot_replace_index_checksums(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    args = _blast_args(
        database=str(tmp_path / "missing_database"),
        release_id="diagnostic-label",
    )

    with pytest.raises(ValueError, match="diagnostic only"):
        annotate.resolve_database_identity(args)


def test_blastp_identity_records_resolved_path_version_and_hash(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    executable = tmp_path / "blastp"
    executable.write_text("#!/bin/sh\necho 'blastp: 2.16.0+'\n", encoding="utf-8")
    executable.chmod(0o755)

    identity = annotate.resolve_blastp_identity(str(executable))

    assert identity["path"] == str(executable.resolve())
    assert identity["version"] == "blastp: 2.16.0+"
    assert len(identity["sha256"]) == 64


def test_blast_artifact_reuse_requires_matching_manifest_and_hashes(
    tmp_path: Path,
    monkeypatch,
):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    fasta = tmp_path / "queries.faa"
    candidate_manifest = tmp_path / "candidates.csv"
    results = tmp_path / "results.csv"
    results_manifest = tmp_path / "results.csv.manifest.json"
    structure = tmp_path / "candidate.pdb"
    sequence_truth = tmp_path / "sequence_truth.csv"
    blast_tsv = tmp_path / "blastp.tsv"
    sidecar = tmp_path / "blastp.tsv.artifact.json"
    fasta.write_text(">query_1\nAAAA\n", encoding="utf-8")
    candidate_manifest.write_text("query_id\nquery_1\n", encoding="utf-8")
    results.write_text("result\nBARREL\n", encoding="utf-8")
    results_manifest.write_text("manifest", encoding="utf-8")
    structure.write_text("structure", encoding="utf-8")
    sequence_truth.write_text("truth", encoding="utf-8")
    blast_tsv.write_text(_valid_blast_row(), encoding="utf-8")
    candidate = annotate.Candidate(
        query_id="query_1",
        filename="candidate.pdb",
        author_chain_id="A",
        result="BARREL",
        reason="OK",
        strand_adjacency_count="8",
        cycle_strand_count="8",
        cycle_strand_fraction="1.0",
        cycle_rank="1",
        source_path=str(structure),
        sequence_length=4,
        sequence_status="ok",
    )
    tool_identity = {
        "path": "/usr/bin/blastp",
        "sha256": "1" * 64,
        "version": "blastp: 2.16.0+",
    }
    database_identity = {
        "database": "nr",
        "kind": "local_index_hashes",
        "declared_release_label": "nr-2026-08-01",
        "immutable_identity_available": True,
        "identity_files": [{"path": "/db/nr.pin", "size": 1, "sha256": "4" * 64}],
        "resolved_prefix": "/db/nr",
    }
    monkeypatch.setattr(annotate, "resolve_blastp_identity", lambda _value: tool_identity)
    monkeypatch.setattr(annotate, "resolve_database_identity", lambda _args: database_identity)
    args = _blast_args(database="nr", release_id="nr-2026-08-01")
    context = annotate.build_blast_artifact_context(
        args,
        fasta_path=fasta,
        candidate_manifest=candidate_manifest,
        results_path=results,
        results_manifest_path=results_manifest,
        detector_manifest=_validated_detector_document(structure),
        sequence_truth_manifest=sequence_truth,
        candidates=[candidate],
    )
    document = annotate.build_blast_artifact_manifest(
        context,
        blast_tsv=blast_tsv,
        command=["/usr/bin/blastp", "-db", "nr"],
    )
    annotate.write_json(sidecar, document)

    validated = annotate.validate_blast_artifact_reuse(
        sidecar,
        blast_tsv=blast_tsv,
        context=context,
    )

    assert validated["artifact_key"] == context["artifact_key"]
    assert set(validated["inputs"]) == {
        "candidate_manifest",
        "fasta",
        "results_csv",
        "results_manifest",
        "sequence_truth_manifest",
        "structures",
    }
    assert len(validated["outputs"]["blast_tsv"]["sha256"]) == 64
    assert validated["generated_at_utc"].endswith("Z")

    changed_args = _blast_args(
        database="nr",
        release_id="nr-2026-08-01",
        search_evalue="1e-8",
    )
    changed_context = annotate.build_blast_artifact_context(
        changed_args,
        fasta_path=fasta,
        candidate_manifest=candidate_manifest,
        results_path=results,
        results_manifest_path=results_manifest,
        detector_manifest=_validated_detector_document(structure),
        sequence_truth_manifest=sequence_truth,
        candidates=[candidate],
    )
    with pytest.raises(ValueError, match="identity does not match"):
        annotate.validate_blast_artifact_reuse(
            sidecar,
            blast_tsv=blast_tsv,
            context=changed_context,
        )

    sequence_truth.write_text("truth-current", encoding="utf-8")
    changed_truth_context = annotate.build_blast_artifact_context(
        args,
        fasta_path=fasta,
        candidate_manifest=candidate_manifest,
        results_path=results,
        results_manifest_path=results_manifest,
        detector_manifest=_validated_detector_document(structure),
        sequence_truth_manifest=sequence_truth,
        candidates=[candidate],
    )
    assert changed_truth_context["artifact_key"] != context["artifact_key"]
    assert (
        context["identity"]["sequence_completeness_policy"]
        == annotate.BFVD_SEQUENCE_COMPLETENESS_POLICY
    )
    sequence_truth.write_text("truth", encoding="utf-8")

    blast_tsv.write_text(_valid_blast_row().replace("80.0", "81.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="path or hash"):
        annotate.validate_blast_artifact_reuse(
            sidecar,
            blast_tsv=blast_tsv,
            context=context,
        )


def test_existing_blast_tsv_without_sidecar_is_never_reused(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    blast_tsv = tmp_path / "blastp.tsv"
    blast_tsv.write_text(_valid_blast_row(), encoding="utf-8")

    with pytest.raises(ValueError, match="no artifact manifest"):
        annotate.validate_blast_artifact_reuse(
            tmp_path / "missing.artifact.json",
            blast_tsv=blast_tsv,
            context={"artifact_key": "x", "identity": {}, "inputs": {}, "reuse_allowed": True},
        )


@pytest.mark.parametrize("release_id", [None, "nr-2026-08-01"])
def test_remote_artifact_is_never_reusable_even_with_declared_release_label(
    tmp_path: Path,
    monkeypatch,
    release_id: str | None,
):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    fasta = tmp_path / "queries.faa"
    candidate_manifest = tmp_path / "candidates.csv"
    results = tmp_path / "results.csv"
    results_manifest = tmp_path / "results.csv.manifest.json"
    structure = tmp_path / "candidate.pdb"
    sequence_truth = tmp_path / "sequence_truth.csv"
    blast_tsv = tmp_path / "blastp.tsv"
    sidecar = tmp_path / "blastp.tsv.artifact.json"
    for path, content in (
        (fasta, ">query_1\nAAAA\n"),
        (candidate_manifest, "query_id\nquery_1\n"),
        (results, "result\nBARREL\n"),
        (results_manifest, "manifest"),
        (structure, "structure"),
        (sequence_truth, "truth"),
        (blast_tsv, _valid_blast_row()),
    ):
        path.write_text(content, encoding="utf-8")
    candidate = annotate.Candidate(
        "query_1",
        "candidate.pdb",
        "A",
        "BARREL",
        "OK",
        "8",
        "8",
        "1.0",
        "1",
        str(structure),
        4,
        "ok",
    )
    monkeypatch.setattr(
        annotate,
        "resolve_blastp_identity",
        lambda _value: {"path": "/blastp", "sha256": "2" * 64, "version": "2.16"},
    )
    args = _blast_args(database="nr", remote=True, release_id=release_id)
    context = annotate.build_blast_artifact_context(
        args,
        fasta_path=fasta,
        candidate_manifest=candidate_manifest,
        results_path=results,
        results_manifest_path=results_manifest,
        detector_manifest=_validated_detector_document(structure),
        sequence_truth_manifest=sequence_truth,
        candidates=[candidate],
    )
    annotate.write_json(
        sidecar,
        annotate.build_blast_artifact_manifest(
            context,
            blast_tsv=blast_tsv,
            command=["/blastp", "-remote", "-db", "nr"],
        ),
    )

    assert context["diagnostics"]["declared_database_release_label"] == release_id
    assert "declared_release_label" not in context["identity"]["database"]
    assert context["reuse_allowed"] is False
    with pytest.raises(ValueError, match="never reusable"):
        annotate.validate_blast_artifact_reuse(
            sidecar,
            blast_tsv=blast_tsv,
            context=context,
        )


def test_artifact_json_writer_rejects_non_finite_values(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")

    with pytest.raises(ValueError):
        annotate.write_json(tmp_path / "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()

    valid = tmp_path / "valid.json"
    annotate.write_json(valid, {"value": 1})
    assert json.loads(valid.read_text(encoding="utf-8")) == {"value": 1}


def test_blast_annotation_main_reuses_only_verified_matching_artifact(
    tmp_path: Path,
    monkeypatch,
):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures = tmp_path / "structures"
    structures.mkdir()
    structure = structures / "candidate.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C\n"
        "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00 80.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    results_manifest = tmp_path / "results.csv.manifest.json"
    results_manifest.write_text("validated detector manifest", encoding="utf-8")
    results = tmp_path / "results.csv"
    _write_detector_csv(results, structure)
    sequence_truth = _write_complete_sequence_truth(
        tmp_path / "sequence_truth.csv",
        relative_path="candidate.pdb",
        sequence="AG",
    )
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        annotate,
        "resolve_blastp_identity",
        lambda _value: {"path": "/blastp", "sha256": "3" * 64, "version": "2.16"},
    )
    monkeypatch.setattr(
        annotate,
        "resolve_database_identity",
        lambda _args: {
            "database": "nr",
            "kind": "local_index_hashes",
            "declared_release_label": "nr-2026-08-01",
            "immutable_identity_available": True,
            "identity_files": [{"path": "/db/nr.pin", "size": 1, "sha256": "5" * 64}],
            "resolved_prefix": "/db/nr",
        },
    )
    monkeypatch.setattr(
        annotate,
        "validate_detector_artifact_manifest",
        lambda manifest_path, *, expected_output: (
            _validated_detector_document(structure)
            if manifest_path == results_manifest.resolve() and expected_output == results.resolve()
            else pytest.fail("unexpected detector artifact identity")
        ),
    )
    calls = 0

    def fake_run_blastp(_args, fasta_path, blast_tsv, **_kwargs):
        nonlocal calls
        calls += 1
        query_id = fasta_path.read_text(encoding="utf-8").splitlines()[0].removeprefix(">")
        blast_tsv.write_text(_valid_blast_row(query_id), encoding="utf-8")
        return ["/blastp", "-db", "nr"]

    monkeypatch.setattr(annotate, "run_blastp", fake_run_blastp)
    common_args = [
        "--results",
        str(results),
        "--results-manifest",
        str(results_manifest),
        "--structures",
        str(structures),
        "--sequence-truth-manifest",
        str(sequence_truth),
        "--out-dir",
        str(output_dir),
        "--min-query-length",
        "1",
        "--db",
        "nr",
        "--database-release-id",
        "nr-2026-08-01",
    ]

    assert annotate.main(common_args) == 0
    assert annotate.main(common_args) == 0
    assert calls == 1
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["blast_status"] == "reused_verified_artifact"
    assert summary["blast_cache_reused"] is True
    assert (
        json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))["status"]
        == "complete"
    )

    with pytest.raises(ValueError, match="identity does not match"):
        annotate.main([*common_args, "--search-evalue", "1e-8"])
    assert calls == 1
    assert (
        json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))["status"]
        == "failed"
    )


def test_blast_run_rejects_database_mutation_during_search(tmp_path: Path, monkeypatch):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures = tmp_path / "structures"
    structures.mkdir()
    structure = structures / "candidate.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C\nEND\n",
        encoding="utf-8",
    )
    results = tmp_path / "results.csv"
    _write_detector_csv(results, structure)
    results_manifest = tmp_path / "results.csv.manifest.json"
    results_manifest.write_text("validated detector manifest", encoding="utf-8")
    sequence_truth = _write_complete_sequence_truth(
        tmp_path / "sequence_truth.csv",
        relative_path="candidate.pdb",
        sequence="A",
    )
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        annotate,
        "validate_detector_artifact_manifest",
        lambda _manifest, *, expected_output: _validated_detector_document(structure),
    )
    monkeypatch.setattr(
        annotate,
        "resolve_blastp_identity",
        lambda _value: {"path": "/blastp", "sha256": "3" * 64, "version": "2.16"},
    )
    database_calls = 0

    def changing_database(_args):
        nonlocal database_calls
        database_calls += 1
        digest = "4" * 64 if database_calls == 1 else "5" * 64
        return {
            "database": "nr",
            "kind": "local_index_hashes",
            "declared_release_label": "nr-2026-08-01",
            "immutable_identity_available": True,
            "identity_files": [{"path": "/db/nr.pin", "size": 1, "sha256": digest}],
            "resolved_prefix": "/db/nr",
        }

    monkeypatch.setattr(annotate, "resolve_database_identity", changing_database)

    def fake_run(_args, fasta_path, blast_tsv, **_kwargs):
        query_id = fasta_path.read_text(encoding="utf-8").splitlines()[0].removeprefix(">")
        blast_tsv.write_text(_valid_blast_row(query_id), encoding="utf-8")
        return ["/blastp", "-db", "nr"]

    monkeypatch.setattr(annotate, "run_blastp", fake_run)
    arguments = [
        "--results",
        str(results),
        "--results-manifest",
        str(results_manifest),
        "--structures",
        str(structures),
        "--sequence-truth-manifest",
        str(sequence_truth),
        "--out-dir",
        str(output_dir),
        "--min-query-length",
        "1",
        "--db",
        "nr",
    ]

    with pytest.raises(RuntimeError, match="changed during search"):
        annotate.main(arguments)

    assert not (output_dir / "blastp.tsv").exists()
    state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error"]["type"] == "RuntimeError"
