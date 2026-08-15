from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runner = importlib.import_module("external_methods.foldseek.runner")
structures = importlib.import_module("external_methods.foldseek.structures")
structure_search = importlib.import_module("external_methods.foldseek.structure_search")
evaluate_dataset = importlib.import_module("external_methods.foldseek.evaluate_dataset")
evaluation_common = importlib.import_module("external_methods.evaluation_common")

load_hits_tsv = runner.load_hits_tsv
summarize_hits = runner.summarize_hits
run_baseline = runner.run_baseline
generate_structure_chains = structures.generate_structure_chains
foldseek_query_aliases = structures.foldseek_query_aliases
run_structure_search_baseline = structure_search.run_structure_search_baseline

SMOKE_DATA = ROOT / "data" / "external_methods" / "foldseek_smoke"
STRUCTURE_SMOKE_DATA = ROOT / "data" / "external_methods" / "isitabarrel_structure_smoke"


def test_foldseek_runner_help_executes_from_source_checkout():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "external_methods" / "foldseek" / "runner.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage: python external_methods/foldseek/runner.py" in completed.stdout
    assert "Arguments after -- are passed to Foldseek easy-search." in completed.stdout
    assert completed.stderr == ""


def test_load_and_summarize_foldseek_hits_from_fixture():
    hits = load_hits_tsv(SMOKE_DATA / "hits.tsv")

    results = summarize_hits(
        hits,
        query_ids=["toy_barrel_A", "toy_nonbarrel_A", "missing_A"],
    )

    assert [result.sample_id for result in results] == [
        "toy_barrel_A",
        "toy_nonbarrel_A",
        "missing_A",
    ]
    assert [result.result for result in results] == ["BARREL", "NON_BARREL", "NON_BARREL"]
    assert results[0].score == 0.72
    assert results[0].decision_rule == "min_qtmscore_ttmscore>=0.5;qcov>=0.6;tcov>=0.6"
    assert results[1].eligible_hit_count == 0
    assert results[2].hit_count == 0


def test_summarize_foldseek_hits_can_alias_and_ignore_targets():
    hits = load_hits_tsv(SMOKE_DATA / "hits.tsv")

    aliased_results = summarize_hits(
        hits,
        query_ids=["toy_barrel_A"],
        target_aliases={"ref_barrel_A": "reference_barrel_A"},
    )
    assert aliased_results[0].best_target == "reference_barrel_A"

    filtered_results = summarize_hits(
        hits,
        query_ids=["toy_barrel_A"],
        target_aliases={"ref_barrel_A": "toy_barrel_A"},
        ignore_target_ids_by_query={"toy_barrel_A": {"toy_barrel_A"}},
    )

    assert filtered_results[0].result == "NON_BARREL"
    assert filtered_results[0].hit_count == 0
    assert filtered_results[0].ignored_target_hit_count == 1


def test_generate_structure_chains_from_pdb_fixture(tmp_path: Path):
    generated = generate_structure_chains(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "generated",
    )

    assert [record.sample_id for record in generated.records] == ["toy_barrel_A"]
    assert Path(generated.records[0].chain_path).exists()
    assert Path(generated.manifest_path).exists()
    assert Path(generated.residue_mapping_path).exists()

    chain_text = Path(generated.records[0].chain_path).read_text(encoding="utf-8")
    assert " A   1" in chain_text
    assert " A  16" in chain_text
    assert foldseek_query_aliases(generated.records)["toy_barrel_A.pdb_A"] == "toy_barrel_A"


def test_generate_structure_chains_rejects_blank_author_chain(tmp_path: Path):
    structure = tmp_path / "blank-chain.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA     1       1.000   0.000   0.000  1.00 80.00           C\nEND\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Blank author chain identifiers"):
        generate_structure_chains(structure, tmp_path / "generated", min_residues=1)


def test_generate_structure_chains_keeps_unknown_residues_with_complete_backbone(
    tmp_path: Path,
):
    structure = tmp_path / "unknown-backbone.pdb"
    lines = []
    serial = 1
    for residue_number in range(1, 16):
        for atom_name, element, offset in (
            ("N", "N", 0.0),
            ("CA", "C", 0.5),
            ("C", "C", 1.0),
        ):
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} UNK A{residue_number:4d}    "
                f"{residue_number * 3.0 + offset:8.3f}{0.0:8.3f}{0.0:8.3f}"
                f"  1.00 80.00          {element:>2s}\n"
            )
            serial += 1
    structure.write_text("".join(lines) + "END\n", encoding="utf-8")

    generated = generate_structure_chains(structure, tmp_path / "generated")

    assert len(generated.records) == 1
    assert generated.records[0].n_residues == 15
    assert Path(generated.records[0].chain_path).stat().st_size > 1000


def test_run_foldseek_adapter_with_fake_runner(tmp_path: Path):
    generated = generate_structure_chains(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "generated",
    )
    output_csv = tmp_path / "normalized.csv"

    results = run_baseline(
        generated.chain_dir,
        SMOKE_DATA,
        work_dir=tmp_path / "work",
        output_path=output_csv,
        query_ids=[record.sample_id for record in generated.records],
        query_aliases=foldseek_query_aliases(generated.records),
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_foldseek.py")],
    )

    assert [result.result for result in results] == ["BARREL"]
    assert results[0].best_target == "ref_barrel_A"

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "foldseek_tmalign_structure_search"
    assert rows[0]["sample_id"] == "toy_barrel_A"
    assert rows[0]["result"] == "BARREL"
    assert float(rows[0]["score"]) == 0.72


def test_run_foldseek_adapter_keeps_no_hit_structure_queries(tmp_path: Path):
    query_dir = tmp_path / "queries"
    query_dir.mkdir()
    query_text = (STRUCTURE_SMOKE_DATA / "toy_barrel.pdb").read_text(encoding="utf-8")
    (query_dir / "toy_barrel.pdb").write_text(query_text, encoding="utf-8")
    (query_dir / "toy_nonbarrel.pdb").write_text(query_text, encoding="utf-8")

    results = run_baseline(
        query_dir,
        SMOKE_DATA,
        work_dir=tmp_path / "work",
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_foldseek.py")],
    )

    assert [result.sample_id for result in results] == [
        "toy_barrel.pdb_A",
        "toy_nonbarrel.pdb_A",
    ]
    assert [result.result for result in results] == ["BARREL", "NON_BARREL"]
    assert results[1].hit_count == 0


def test_foldseek_db_prefix_sidecar_is_accepted(tmp_path: Path):
    db_prefix = tmp_path / "target_db"
    Path(f"{db_prefix}.dbtype").write_text("fake db marker\n", encoding="utf-8")

    assert runner._require_foldseek_input(db_prefix, "db") == db_prefix.resolve()


def test_run_structure_search_baseline_smoke(tmp_path: Path):
    output_csv = tmp_path / "structure_baseline.csv"

    run = run_structure_search_baseline(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "structure_work",
        reference_structures=STRUCTURE_SMOKE_DATA,
        output_path=output_csv,
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_foldseek.py")],
    )

    assert [record.sample_id for record in run.generated_chains.records] == ["toy_barrel_A"]
    assert [result.result for result in run.results] == ["BARREL"]

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "foldseek_tmalign_structure_search"
    assert rows[0]["sample_id"] == "toy_barrel_A"
    assert rows[0]["result"] == "BARREL"


def test_foldseek_cli_passthrough_requires_explicit_separator():
    parser = runner.build_arg_parser()

    args, extra_args = runner._parse_args_and_passthrough(
        parser,
        ["queries", "target", "--", "--threads", "2"],
    )
    assert args.query_structures == "queries"
    assert extra_args == ["--threads", "2"]

    with pytest.raises(SystemExit):
        runner._parse_args_and_passthrough(parser, ["queries", "target", "--threads", "2"])


def test_foldseek_chain_metrics_require_paired_target_manifests(tmp_path: Path):
    with pytest.raises(ValueError, match="paired positive and negative"):
        evaluate_dataset.run_dataset(
            tmp_path / "positive",
            tmp_path / "negative",
            tmp_path / "outputs",
            reference_dir=tmp_path / "reference",
            group_manifest=tmp_path / "groups.csv",
            foldseek_executable=sys.executable,
            positive_target_manifest=None,
            negative_target_manifest=None,
            metric_level="chain",
            min_residues=1,
            create_index=False,
            alignment_type=1,
            score_mode="min_qtmscore_ttmscore",
            score_threshold=0.5,
            min_query_coverage=0.6,
            min_target_coverage=0.6,
            evalue=1e-3,
            max_seqs=10,
            extra_args=None,
            timeout=None,
            tag="contract",
        )

    run_dir = next((tmp_path / "outputs").iterdir())
    manifest = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "failed"
    assert manifest["phase"] == "validation"
    assert manifest["error"]["type"] == "ValueError"


def test_foldseek_dataset_reports_missing_upstream_results(tmp_path: Path):
    record = structures.GeneratedStructureChain(
        sample_id="missing_A",
        source_path=str(tmp_path / "missing.pdb"),
        chain_id="A",
        n_residues=20,
        chain_path=str(tmp_path / "missing_A.pdb"),
    )
    run = evaluate_dataset.SplitRun(
        split_name="positive",
        generated=structures.GeneratedStructureSet(
            output_dir=str(tmp_path),
            chain_dir=str(tmp_path),
            manifest_path=str(tmp_path / "manifest.csv"),
            residue_mapping_path=str(tmp_path / "mapping.csv"),
            records=[record],
        ),
        results=[],
    )

    with pytest.raises(ValueError, match="result identity"):
        evaluate_dataset._chain_rows_for_split(
            run,
            reference_metadata={},
            alignment_type=1,
            reference_policy="unit",
        )


def test_foldseek_group_manifest_is_strict_and_excludes_entire_family(tmp_path: Path):
    query_root = tmp_path / "queries"
    reference_root = tmp_path / "references"
    query_root.mkdir()
    reference_root.mkdir()
    query_source = query_root / "query.pdb"
    reference_a = reference_root / "ref_a.pdb"
    reference_b = reference_root / "ref_b.pdb"
    for path in (query_source, reference_a, reference_b):
        path.write_text("fixture\n", encoding="utf-8")

    query = structures.GeneratedStructureChain("query_A", str(query_source), "A", 20, "q.pdb")
    references = [
        structures.GeneratedStructureChain("ref_a_A", str(reference_a), "A", 20, "a.pdb"),
        structures.GeneratedStructureChain("ref_b_A", str(reference_b), "A", 20, "b.pdb"),
    ]
    manifest = tmp_path / "groups.csv"
    manifest.write_text(
        "split,relative_path,author_chain_id,group_id\n"
        "positive,query.pdb,A,family-1\n"
        "reference,ref_a.pdb,A,family-1\n"
        "reference,ref_b.pdb,A,family-2\n",
        encoding="utf-8",
    )
    assignments = evaluate_dataset._group_assignments(manifest)
    query_groups = evaluate_dataset._group_ids_for_records(
        [query], split="positive", input_root=query_root, assignments=assignments
    )
    reference_groups = evaluate_dataset._group_ids_for_records(
        references, split="reference", input_root=reference_root, assignments=assignments
    )
    ignored = evaluate_dataset._ignore_map_for_queries(
        [query],
        query_groups,
        evaluate_dataset._target_ids_by_group(references, reference_groups),
        {},
    )

    assert ignored == {"query_A": {"ref_a_A"}}
    assert evaluate_dataset._pdb_id_from_filename("1abc_query.pdb.gz") == "1ABC"
    compressed_query = structures.GeneratedStructureChain(
        "compressed_A", "/queries/1abc_query.pdb.gz", "A", 20, "q.pdb"
    )
    compressed_reference = structures.GeneratedStructureChain(
        "compressed_ref_A", "/references/1abc_reference.cif", "A", 20, "r.pdb"
    )
    assert evaluate_dataset._ignore_map_for_queries(
        [compressed_query],
        {"compressed_A": "query-family"},
        {},
        evaluate_dataset._target_ids_by_pdb([compressed_reference]),
    ) == {"compressed_A": {"compressed_ref_A"}}


def test_foldseek_exact_normalized_chain_content_excludes_renamed_self_hit(
    tmp_path: Path,
) -> None:
    query_source = tmp_path / "9xyz_query.pdb"
    reference_source = tmp_path / "1abc_reference.pdb"
    query_source.write_text("query source identity\n", encoding="utf-8")
    reference_source.write_text("reference source identity\n", encoding="utf-8")
    query_dir = tmp_path / "query_chains"
    reference_dir = tmp_path / "reference_chains"
    query_dir.mkdir()
    reference_dir.mkdir()
    query_chain = query_dir / "renamed_query_A.pdb"
    reference_chain = reference_dir / "renamed_reference_A.pdb"
    identical_normalized_chain = "MODEL      1\nENDMDL\nEND\n"
    query_chain.write_text(identical_normalized_chain, encoding="utf-8")
    reference_chain.write_text(identical_normalized_chain, encoding="utf-8")
    query_record = structures.GeneratedStructureChain(
        "renamed_query_A", str(query_source), "A", 20, str(query_chain)
    )
    reference_record = structures.GeneratedStructureChain(
        "renamed_reference_A", str(reference_source), "A", 20, str(reference_chain)
    )
    query_generated = structures.GeneratedStructureSet(
        str(query_dir.parent), str(query_dir), "manifest.csv", "mapping.csv", [query_record]
    )
    reference_generated = structures.GeneratedStructureSet(
        str(reference_dir.parent),
        str(reference_dir),
        "manifest.csv",
        "mapping.csv",
        [reference_record],
    )
    query_inventory = evaluate_dataset._generated_chain_content_inventory(
        query_generated, label="positive"
    )
    reference_inventory = evaluate_dataset._generated_chain_content_inventory(
        reference_generated, label="reference"
    )
    query_hashes = evaluate_dataset._content_hashes_by_sample(query_inventory)
    reference_hashes = evaluate_dataset._content_hashes_by_sample(reference_inventory)

    ignored, details = evaluate_dataset._ignore_details_for_queries(
        [query_record],
        {query_record.sample_id: "incorrect-query-group"},
        {"incorrect-reference-group": {reference_record.sample_id}},
        evaluate_dataset._target_ids_by_pdb([reference_record]),
        query_content_hashes=query_hashes,
        reference_ids_by_content_hash=evaluate_dataset._target_ids_by_content_hash(
            [reference_record], reference_hashes
        ),
    )

    assert ignored == {query_record.sample_id: {reference_record.sample_id}}
    assert details[0]["same_pdb_reference_candidates_n"] == 0
    assert details[0]["curated_group_reference_candidates_n"] == 0
    assert details[0]["exact_content_reference_candidates_n"] == 1
    result = summarize_hits(
        [
            runner.FoldseekHit(
                query_record.sample_id,
                reference_record.sample_id,
                20,
                20,
                20,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                100.0,
            )
        ],
        query_ids=[query_record.sample_id],
        ignore_target_ids_by_query=ignored,
    )[0]
    assert result.result == "NON_BARREL"
    assert result.ignored_target_hit_count == 1


def test_foldseek_group_manifest_rejects_duplicate_or_missing_assignments(tmp_path: Path):
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "split,relative_path,author_chain_id,group_id\n"
        "positive,q.pdb,A,family-1\n"
        "positive,q.pdb,A,family-2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate assignment"):
        evaluate_dataset._group_assignments(duplicate)

    source_root = tmp_path / "queries"
    source_root.mkdir()
    source = source_root / "q.pdb"
    source.write_text("fixture\n", encoding="utf-8")
    record = structures.GeneratedStructureChain("q_A", str(source), "A", 20, "q_A.pdb")
    valid = tmp_path / "valid.csv"
    valid.write_text(
        "split,relative_path,author_chain_id,group_id\nreference,r.pdb,A,family-1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no assignment"):
        evaluate_dataset._group_ids_for_records(
            [record],
            split="positive",
            input_root=source_root,
            assignments=evaluate_dataset._group_assignments(valid),
        )


def test_foldseek_dataset_cli_requires_reference_and_group_manifest():
    parser = evaluate_dataset.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--out-dir", "out", "--tag", "run"])


def test_foldseek_dataset_uses_file_any_chain_and_manifest_target_author_chain_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roots = {name: tmp_path / name for name in ("positive", "negative", "reference")}
    for name, root in roots.items():
        root.mkdir()
        root.joinpath("sample.pdb").write_text(f"MODEL {name}\nEND\n", encoding="utf-8")
    groups = tmp_path / "groups.csv"
    groups.write_text(
        "split,relative_path,author_chain_id,group_id\n"
        "positive,sample.pdb,A,pos-a\n"
        "positive,sample.pdb,B,pos-b\n"
        "negative,sample.pdb,A,neg-a\n"
        "negative,sample.pdb,B,neg-b\n"
        "reference,sample.pdb,A,ref-a\n"
        "reference,sample.pdb,B,ref-b\n",
        encoding="utf-8",
    )
    manifests = {}
    for split in ("positive", "negative"):
        manifest = tmp_path / f"{split}_targets.csv"
        manifest.write_text("relative_path,author_chain_id\nsample.pdb,A\n", encoding="utf-8")
        manifests[split] = manifest

    def fake_generate(input_path, output_dir, **_kwargs):
        input_root = Path(input_path).resolve()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        chain_dir = output / "chains"
        chain_dir.mkdir()
        records = []
        for chain in ("A", "B"):
            chain_path = chain_dir / f"{input_root.name}_{chain}.pdb"
            chain_path.write_text("MODEL\nEND\n", encoding="utf-8")
            records.append(
                structures.GeneratedStructureChain(
                    f"{input_root.name}_{chain}",
                    str(input_root / "sample.pdb"),
                    chain,
                    20,
                    str(chain_path),
                )
            )
        return structures.GeneratedStructureSet(
            str(output),
            str(chain_dir),
            str(output / "manifest.csv"),
            str(output / "map.csv"),
            records,
        )

    def fake_run(_query, _target, *, query_ids, output_path, **_kwargs):
        assert output_path is None
        return [
            runner.FoldseekResult(
                sample_id=sample_id,
                result="BARREL" if sample_id.endswith("_B") else "NON_BARREL",
                score=0.8 if sample_id.endswith("_B") else 0.2,
                decision_rule="unit",
                score_mode="min_qtmscore_ttmscore",
                score_threshold=0.5,
                min_query_coverage=0.6,
                min_target_coverage=0.6,
                hit_count=1,
                eligible_hit_count=1,
            )
            for sample_id in query_ids
        ]

    monkeypatch.setattr(evaluate_dataset, "generate_structure_chains", fake_generate)
    monkeypatch.setattr(evaluate_dataset, "run_baseline", fake_run)
    run_dir = evaluate_dataset.run_dataset(
        roots["positive"],
        roots["negative"],
        tmp_path / "outputs",
        reference_dir=roots["reference"],
        group_manifest=groups,
        foldseek_executable=sys.executable,
        positive_target_manifest=manifests["positive"],
        negative_target_manifest=manifests["negative"],
        metric_level="both",
        min_residues=1,
        create_index=False,
        alignment_type=1,
        score_mode="min_qtmscore_ttmscore",
        score_threshold=0.5,
        min_query_coverage=0.6,
        min_target_coverage=0.6,
        evalue=1e-3,
        max_seqs=10,
        extra_args=None,
        timeout=None,
        tag="paired",
    )

    with (run_dir / "chain_predictions.csv").open(newline="", encoding="utf-8") as handle:
        chain_rows = list(csv.DictReader(handle))
    partner_rows = [row for row in chain_rows if row["author_chain_id"] == "B"]
    assert all(row["y_true"] == "" and row["split"] == "" for row in partner_rows)
    assert all(row["is_target_author_chain"] == "False" for row in partner_rows)
    with (run_dir / "file_results.csv").open(newline="", encoding="utf-8") as handle:
        file_rows = list(csv.DictReader(handle))
    assert len(file_rows) == 2
    assert all(row["pred_barrel_any"] == "True" for row in file_rows)
    with (run_dir / "target_chain_results.csv").open(newline="", encoding="utf-8") as handle:
        target_rows = list(csv.DictReader(handle))
    assert len(target_rows) == 2
    assert all(row["author_chain_id"] == "A" for row in target_rows)
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summaries = {row["level"]: row for row in csv.DictReader(handle)}
    assert summaries["file"]["n_used"] == "2"
    assert summaries["file"]["TP"] == "1"
    assert summaries["file"]["FP"] == "1"
    assert summaries["chain"]["n_used"] == "2"
    assert summaries["chain"]["FN"] == "1"
    assert summaries["chain"]["TN"] == "1"
    manifest = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["metric_sampling"]["partner_chain_labels"] == "unlabeled"
    assert manifest["metric_sampling"]["filtered_out_policy"] == "strict"
    leakage = manifest["reference_leakage_control"]
    assert leakage["hash_policy"] == evaluate_dataset.NORMALIZED_CHAIN_HASH_POLICY
    assert leakage["identity_failure_policy"] == "fail_closed"
    assert leakage["exact_content_reference_candidates_excluded_total"] > 0
    assert all(
        query["query_chain_sha256"]
        for split in leakage["splits"].values()
        for query in split["queries"]
    )
    assert manifest["outputs"]["artifacts"]
