from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cooper_beta.config import build_config
from cooper_beta.dssp_adapter import DsspAnnotation
from cooper_beta.loader import ProteinLoader
from cooper_beta.polymer_sequence import declared_polymer_sequences

SCRIPT_DIRECTORY = Path(__file__).parents[1] / "data" / "scripts"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import _dataset_provenance as dataset_provenance  # noqa: E402
import build_easy_negatives_from_pisces_cath as negative_builder  # noqa: E402
import mpstruc_download_and_classify as positive_builder  # noqa: E402


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _positive_xml(path: Path) -> Path:
    return _write(
        path,
        """
        <name>TRANSMEMBRANE PROTEINS: BETA-BARREL</name>
        <subgroup>
        <name>Porins</name>
        <protein>
        <pdbCode>1ABC</pdbCode>
        <name>Example porin</name>
        <species>Example species</species>
        <resolution>2.0</resolution>
        <relatedPdbEntries>
        <pdbCode>2ABC</pdbCode>
        </relatedPdbEntries>
        </protein>
        </subgroup>
        """,
    )


def _pisces(path: Path, *, pdb_id: str = "1ABC") -> Path:
    return _write(path, f"{pdb_id} A Xray 100 1.5 0.20 0.25")


def _cath(path: Path, *, class_num: int = 1, pdb_id: str = "1abc") -> Path:
    return _write(path, f"{pdb_id}A00 {class_num} 10 20 30 1 1 1 1 1 100 1.5")


def _nonoverlapping_mpstruc(path: Path) -> Path:
    return _write(path, "<root><pdbCode>9XYZ</pdbCode></root>")


def _negative_approval(path: Path, *, pdb_id: str = "1ABC", chain: str = "A") -> Path:
    return _write(
        path,
        f"""
        pdb_id,target_author_chain_id,group_id,curation_evidence
        {pdb_id},{chain},FAMILY:NEGATIVE-1,manual non-barrel topology review
        """,
    )


def _candidate(index: int, *, class_num: int = 1) -> negative_builder.CandidateRecord:
    pdb_id = f"{index % 10}{index // 10:03d}"[-4:]
    return negative_builder.CandidateRecord(
        pdb_id=pdb_id,
        pisces_chain="A",
        resolved_chain="A",
        map_status="exact",
        method="Xray",
        chain_length=100,
        resolution=1.5,
        r_value=0.2,
        r_free=0.25,
        cath_total_domain_len=100,
        cath_domain_count=1,
        cath_dominant_class=class_num,
        cath_dominant_class_name=negative_builder.CLASS_NAME_MAP[class_num],
        cath_dominant_class_fraction=1.0,
        cath_dominant_architecture=10,
        cath_dominant_topology=index + 1,
        cath_dominant_superfamily=1,
        cath_classes_present=str(class_num),
        cath_topologies_present=f"{class_num}.10.{index + 1}",
        cath_superfamilies_present=f"{class_num}.10.{index + 1}.1",
        cath_domain_ids=f"{pdb_id}A00",
        group_id=f"CATH:{class_num}.10.{index + 1}.1",
    )


def test_dataset_manifest_uses_source_version_when_distribution_is_uninstalled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "manifest"
    output.mkdir()

    def missing_distribution(_name: str) -> str:
        raise dataset_provenance.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(dataset_provenance.importlib.metadata, "version", missing_distribution)
    manifest = dataset_provenance.RunManifest(
        output,
        SCRIPT_DIRECTORY / "build_easy_negatives_from_pisces_cath.py",
        {},
        mode="exploratory",
        random_algorithm=None,
        seed=None,
    )

    software = manifest.data["software"]
    assert software["package_version"] == "1.0.0"
    assert software["installed_distribution_version"] is None
    assert software["package_version_source_path"].endswith("cooper_beta/_version.py")
    assert len(software["package_version_source_sha256"]) == 64


def _three_copy_cif(path: Path) -> Path:
    atom_rows = "\n".join(f"ATOM A A {index} ALA" for index in range(1, 11))
    sheet_rows = "\n".join(f"S{index} A A {index} {index}" for index in range(1, 9))
    return _write(
        path,
        f"""
        data_1ABC
        _entry.id 1ABC
        loop_
        _entity.id
        _entity.type
        _entity.pdbx_description
        1 polymer 'Example barrel'
        loop_
        _entity_poly.entity_id
        _entity_poly.type
        _entity_poly.pdbx_seq_one_letter_code_can
        1 'polypeptide(L)' AAAAAAAAAA
        loop_
        _struct_asym.id
        _struct_asym.entity_id
        A 1
        loop_
        _atom_site.group_PDB
        _atom_site.label_asym_id
        _atom_site.auth_asym_id
        _atom_site.label_seq_id
        _atom_site.label_comp_id
        {atom_rows}
        loop_
        _struct_sheet_range.sheet_id
        _struct_sheet_range.beg_label_asym_id
        _struct_sheet_range.end_label_asym_id
        _struct_sheet_range.beg_label_seq_id
        _struct_sheet_range.end_label_seq_id
        {sheet_rows}
        loop_
        _pdbx_struct_assembly.id
        1
        loop_
        _pdbx_struct_assembly_gen.assembly_id
        _pdbx_struct_assembly_gen.oper_expression
        _pdbx_struct_assembly_gen.asym_id_list
        1 '(1-3)' A
        """,
    )


def _extractable_cif(path: Path) -> Path:
    return _write(
        path,
        """
        data_1ABC
        _entry.id 1ABC
        loop_
        _entity.id
        _entity.type
        1 polymer
        loop_
        _entity_poly.entity_id
        _entity_poly.type
        _entity_poly.pdbx_seq_one_letter_code_can
        1 'polypeptide(L)' A
        loop_
        _entity_poly_seq.entity_id
        _entity_poly_seq.num
        _entity_poly_seq.mon_id
        1 1 ALA
        loop_
        _struct_asym.id
        _struct_asym.entity_id
        A 1
        loop_
        _pdbx_poly_seq_scheme.asym_id
        _pdbx_poly_seq_scheme.pdb_strand_id
        A A
        loop_
        _atom_site.group_PDB
        _atom_site.id
        _atom_site.type_symbol
        _atom_site.label_atom_id
        _atom_site.label_alt_id
        _atom_site.label_comp_id
        _atom_site.label_asym_id
        _atom_site.label_entity_id
        _atom_site.label_seq_id
        _atom_site.pdbx_PDB_ins_code
        _atom_site.Cartn_x
        _atom_site.Cartn_y
        _atom_site.Cartn_z
        _atom_site.occupancy
        _atom_site.B_iso_or_equiv
        _atom_site.pdbx_formal_charge
        _atom_site.auth_seq_id
        _atom_site.auth_comp_id
        _atom_site.auth_asym_id
        _atom_site.auth_atom_id
        _atom_site.pdbx_PDB_model_num
        ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.0 20.0 ? 1 ALA A CA 1
        """,
    )


def test_mpstruc_parser_handles_related_records(tmp_path: Path) -> None:
    records = positive_builder.parse_mpstruc_xml(str(_positive_xml(tmp_path / "mp.xml")))

    assert {(row["pdb_code"], row["source_type"]) for row in records} == {
        ("1ABC", "master"),
        ("2ABC", "related_master"),
    }
    assert all(row["subgroup_name"] == "Porins" for row in records)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("1", 1), ("1,2,3", 3), ("(1-3)", 3), ("(1-3)(4,5)", 6)],
)
def test_operation_expression_multiplicity(expression: str, expected: int) -> None:
    assert positive_builder.operation_expression_multiplicity(expression) == expected


@pytest.mark.parametrize("expression", ["(1-3", "3-1", "(1,,2)"])
def test_operation_expression_rejects_malformed_input(expression: str) -> None:
    with pytest.raises(ValueError):
        positive_builder.operation_expression_multiplicity(expression)


def test_assembly_operation_multiplicity_prevents_false_monomer(tmp_path: Path) -> None:
    summary = positive_builder.parse_cif_summary(str(_three_copy_cif(tmp_path / "three.cif")))
    result = positive_builder.classify_entry(
        {"pdb_code": "1ABC", "subgroup_name": "Porins"}, summary, {}
    )

    assert summary["preferred_assembly_chain_copy_counts"] == {"A": 3}
    assert summary["entity_summary"]["1"]["chain_count_preferred_assembly"] == 3
    assert result["class_label"] == "SELF_CONTAINED_HOMOOLIGOMER"


@pytest.mark.parametrize("mapping_failure", ["missing", "ambiguous"])
def test_positive_candidate_author_chain_mapping_fails_closed(
    tmp_path: Path,
    mapping_failure: str,
) -> None:
    source = _three_copy_cif(tmp_path / "source.cif").read_text(encoding="utf-8")
    if mapping_failure == "missing":
        source = source.replace("_atom_site.auth_asym_id\n", "")
        source = source.replace("ATOM A A ", "ATOM A ")
    else:
        source = source.replace("ATOM A A 10 ALA", "ATOM A B 10 ALA")
    cif_path = _write(tmp_path / f"{mapping_failure}.cif", source)

    if mapping_failure == "missing":
        with pytest.raises(ValueError, match="Inconsistent mmCIF author-chain mapping"):
            positive_builder.parse_cif_summary(str(cif_path))
        return

    summary = positive_builder.parse_cif_summary(str(cif_path))
    chain = summary["entity_summary"]["1"]["chains"][0]
    assert chain["auth_asym_id"] == ""
    assert chain["auth_asym_mapping_status"] == "ambiguous_multiple_author_chains"


def test_nonpolymer_label_chain_does_not_make_protein_author_chain_ambiguous(
    tmp_path: Path,
) -> None:
    source = _three_copy_cif(tmp_path / "source.cif").read_text(encoding="utf-8")
    source = source.replace(
        "1 polymer 'Example barrel'", "1 polymer 'Example barrel'\n2 non-polymer Ligand"
    )
    source = source.replace(
        "A 1\n        loop_\n        _atom_site",
        "A 1\n        B 2\n        loop_\n        _atom_site",
    )
    source = source.replace(
        "ATOM A A 10 ALA",
        "ATOM A A 10 ALA\n        HETATM B A . HEM",
    )
    summary = positive_builder.parse_cif_summary(str(_write(tmp_path / "ligand.cif", source)))

    chain = summary["entity_summary"]["1"]["chains"][0]
    assert chain["auth_asym_id"] == "A"
    assert chain["auth_asym_mapping_status"] == "exact"


def test_pisces_and_cath_parsers(tmp_path: Path) -> None:
    pisces = negative_builder.parse_pisces_list(_pisces(tmp_path / "pisces.txt"))
    cath = negative_builder.parse_cath_domain_list(_cath(tmp_path / "cath.txt"))

    assert [(row.pdb_id, row.chain_raw, row.length) for row in pisces] == [("1ABC", "A", 100)]
    assert [(row.pdb_id, row.chain_id, row.class_num) for row in cath] == [("1ABC", "A", 1)]


def test_pisces_parser_skips_current_official_header(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "pisces.txt",
        "PDBchain   len  method   resol   rfac  freerfac\n"
        "1ABCA      100  Xray     1.50    0.20  0.25\n",
    )

    records = negative_builder.parse_pisces_list(path)

    assert [(record.pdb_id, record.chain_raw, record.length) for record in records] == [
        ("1ABC", "A", 100)
    ]


def test_concrete_pisces_chain_is_never_replaced_by_unique_cath_chain(tmp_path: Path) -> None:
    pisces = negative_builder.parse_pisces_list(
        _write(tmp_path / "pisces.txt", "1ABC B Xray 100 1.5 0.20 0.25")
    )[0]
    domains = negative_builder.parse_cath_domain_list(_cath(tmp_path / "cath.txt"))
    summaries, entry_to_chains = negative_builder.build_cath_chain_summary(domains)

    summary, resolved_chain, status = negative_builder.resolve_pisces_to_cath_chain(
        pisces,
        summaries,
        entry_to_chains,
    )

    assert summary is None
    assert resolved_chain == ""
    assert status == "pisces_chain_not_in_cath"


@pytest.mark.parametrize(
    "content",
    ["1ABC A Xray 100 nan 0.20 0.25", "1ABC A Xray 100.5 1.5 0.20 0.25"],
)
def test_pisces_parser_fails_closed_on_invalid_numeric_input(tmp_path: Path, content: str) -> None:
    with pytest.raises(ValueError, match="Invalid PISCES record"):
        negative_builder.parse_pisces_list(_write(tmp_path / "pisces.txt", content))


@pytest.mark.parametrize(("classes", "dominant_class"), [("2", 2), ("1;6", 1)])
def test_any_unsupported_cath_class_is_not_automatically_labeled_negative(
    classes: str, dominant_class: int
) -> None:
    pisces = negative_builder.PiscesRecord("1ABC", "A", "Xray", 100, 1.5, 0.2, 0.25, "")
    summary = negative_builder.CathChainSummary(
        "1ABC",
        "A",
        100,
        1,
        dominant_class,
        100,
        1.0,
        10,
        20,
        30,
        100,
        classes,
        "1.10.20",
        "1.10.20.30",
        "1abcA00",
    )

    candidates, excluded = negative_builder.make_candidate_rows(
        [pisces], {("1ABC", "A"): summary}, {"1ABC": ["A"]}, set(), set(), 80, 1200, 0.7
    )

    assert candidates == []
    assert excluded[0]["reason"].startswith("contains_cath_class_not_safe_as_negative")


def test_seeded_selection_is_reproducible_and_seed_sensitive() -> None:
    candidates = [_candidate(index) for index in range(12)]
    kwargs = {
        "candidates": candidates,
        "n_total": 4,
        "class_quotas": {1: 4, 3: 0, 4: 0},
        "max_per_topology": 1,
        "max_per_pdb": 1,
    }

    first = negative_builder.select_diverse_easy_negatives(**kwargs, seed=1)
    repeated = negative_builder.select_diverse_easy_negatives(**kwargs, seed=1)
    other_seed = negative_builder.select_diverse_easy_negatives(**kwargs, seed=999)

    def identity(
        rows: list[negative_builder.CandidateRecord],
    ) -> list[tuple[str, str]]:
        return [(row.pdb_id, row.resolved_chain) for row in rows]

    assert identity(first) == identity(repeated)
    assert identity(first) != identity(other_seed)


def test_negative_approval_must_match_selected_chain_set_exactly(tmp_path: Path) -> None:
    selected = [_candidate(1)]
    exact = negative_builder.load_negative_approvals(
        _negative_approval(
            tmp_path / "exact.csv",
            pdb_id=selected[0].pdb_id,
            chain=selected[0].resolved_chain,
        )
    )
    assert negative_builder.match_negative_approvals(selected, exact) == exact

    extra_path = _write(
        tmp_path / "extra.csv",
        f"""
        pdb_id,target_author_chain_id,group_id,curation_evidence
        {selected[0].pdb_id},A,FAMILY:1,reviewed negative
        9XYZ,A,FAMILY:2,unselected reviewed negative
        """,
    )
    with pytest.raises(ValueError, match="cover the final selected panel exactly"):
        negative_builder.match_negative_approvals(
            selected,
            negative_builder.load_negative_approvals(extra_path),
        )


def test_negative_approval_rejects_duplicate_target(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "duplicate.csv",
        """
        pdb_id,target_author_chain_id,group_id,curation_evidence
        1ABC,A,FAMILY:1,first review
        1ABC,A,FAMILY:1,second review
        """,
    )
    with pytest.raises(ValueError, match="Duplicate negative approval"):
        negative_builder.load_negative_approvals(path)


def test_negative_approval_rejects_second_chain_for_same_pdb(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "duplicate-pdb.csv",
        """
        pdb_id,target_author_chain_id,group_id,curation_evidence
        1ABC,A,FAMILY:1,first reviewed chain
        1ABC,B,FAMILY:1,second reviewed chain
        """,
    )
    with pytest.raises(ValueError, match="exactly one target chain per PDB structure"):
        negative_builder.load_negative_approvals(path)


def test_negative_approval_rejects_blank_target_author_chain_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "blank-chain.csv",
        """
        pdb_id,target_author_chain_id,group_id,curation_evidence
        1ABC,,FAMILY:1,independently reviewed negative
        """,
    )

    with pytest.raises(ValueError, match="blank chain"):
        negative_builder.load_negative_approvals(path)


@pytest.mark.parametrize(
    ("quotas", "n_total", "message"),
    [("1:1,2:1,3:0,4:0", 2, "not an evidence-safe"), ("1:1,3:1,4:0", 3, "sum exactly")],
)
def test_quota_validation_rejects_unsafe_or_inconsistent_values(
    quotas: str, n_total: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        negative_builder.parse_class_quotas(quotas, n_total)


def test_publication_mode_requires_local_frozen_inputs(tmp_path: Path) -> None:
    args = negative_builder.build_arg_parser().parse_args(["--out", str(tmp_path / "output")])
    with pytest.raises(ValueError, match="publication mode requires"):
        negative_builder.validate_arguments(args)


def test_publication_mode_requires_frozen_structure_archive(tmp_path: Path) -> None:
    args = negative_builder.build_arg_parser().parse_args(
        [
            "--pisces-file",
            str(_pisces(tmp_path / "pisces.txt")),
            "--cath-domain-list",
            str(_cath(tmp_path / "cath.txt")),
            "--mpstruc-xml",
            str(_nonoverlapping_mpstruc(tmp_path / "mpstruc.xml")),
            "--negative-approval-manifest",
            str(_negative_approval(tmp_path / "negative-approval.csv")),
        ]
    )
    with pytest.raises(ValueError, match="structure-source-dir"):
        negative_builder.validate_arguments(args)


def test_publication_mode_requires_exactly_one_selected_chain_per_pdb(tmp_path: Path) -> None:
    archive = tmp_path / "coordinates"
    archive.mkdir()
    _extractable_cif(archive / "1abc.cif")
    args = negative_builder.build_arg_parser().parse_args(
        [
            "--pisces-file",
            str(_pisces(tmp_path / "pisces.txt")),
            "--cath-domain-list",
            str(_cath(tmp_path / "cath.txt")),
            "--mpstruc-xml",
            str(_nonoverlapping_mpstruc(tmp_path / "mpstruc.xml")),
            "--negative-approval-manifest",
            str(_negative_approval(tmp_path / "negative-approval.csv")),
            "--structure-source-dir",
            str(archive),
            "--max-per-pdb",
            "2",
        ]
    )

    with pytest.raises(ValueError, match="requires --max-per-pdb 1"):
        negative_builder.validate_arguments(args)


def test_validation_rejects_bool_and_nonfinite_values(tmp_path: Path) -> None:
    args = negative_builder.build_arg_parser().parse_args(
        [
            "--pisces-file",
            str(_pisces(tmp_path / "pisces.txt")),
            "--cath-domain-list",
            str(_cath(tmp_path / "cath.txt")),
        ]
    )
    args.seed = True
    with pytest.raises(ValueError, match="seed must be an integer"):
        negative_builder.validate_arguments(args)
    args.seed = 42
    args.rmax = float("nan")
    with pytest.raises(ValueError, match="rmax must be finite"):
        negative_builder.validate_arguments(args)
    with pytest.raises(ValueError, match="must be pinned"):
        negative_builder.require_pinned_source_id("pisces_source_id", "latest release")


def test_offline_negative_build_writes_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pisces = _pisces(tmp_path / "pisces.txt")
    cath = _cath(tmp_path / "cath.txt")
    output = tmp_path / "negative-output"
    monkeypatch.setattr(
        negative_builder,
        "http_get",
        lambda *_args, **_kwargs: pytest.fail("offline publication run attempted network access"),
    )

    negative_builder.main(
        [
            "--out",
            str(output),
            "--mode",
            "exploratory",
            "--pisces-file",
            str(pisces),
            "--cath-domain-list",
            str(cath),
            "--n-total",
            "1",
            "--class-quotas",
            "1:1,3:0,4:0",
            "--dry-run",
        ]
    )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["randomness"] == {
        "algorithm": negative_builder.SELECTION_ALGORITHM,
        "seed": 42,
    }
    assert manifest["inputs"]["pisces"]["pinned"] is True
    assert len(manifest["inputs"]["pisces"]["sha256"]) == 64
    assert any(row["path"].endswith("easy_negative_selected.csv") for row in manifest["outputs"])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        negative_builder.main(
            [
                "--out",
                str(output),
                "--mode",
                "exploratory",
                "--pisces-file",
                str(pisces),
                "--cath-domain-list",
                str(cath),
            ]
        )


def test_failed_negative_build_records_failure(tmp_path: Path) -> None:
    output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="Unable to satisfy"):
        negative_builder.main(
            [
                "--out",
                str(output),
                "--mode",
                "exploratory",
                "--pisces-file",
                str(_pisces(tmp_path / "pisces.txt")),
                "--cath-domain-list",
                str(_cath(tmp_path / "cath.txt")),
                "--n-total",
                "1",
                "--class-quotas",
                "1:0,3:1,4:0",
                "--dry-run",
            ]
        )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"


def test_publication_negative_build_uses_only_frozen_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "coordinates"
    archive.mkdir()
    source_structure = _extractable_cif(archive / "1abc.cif")
    output = tmp_path / "publication-negative"
    monkeypatch.setattr(
        negative_builder,
        "http_get",
        lambda *_args, **_kwargs: pytest.fail("publication build attempted network access"),
    )

    negative_builder.main(
        [
            "--out",
            str(output),
            "--pisces-file",
            str(_pisces(tmp_path / "pisces.txt")),
            "--cath-domain-list",
            str(_cath(tmp_path / "cath.txt")),
            "--structure-source-dir",
            str(archive),
            "--mpstruc-xml",
            str(_nonoverlapping_mpstruc(tmp_path / "mpstruc.xml")),
            "--negative-approval-manifest",
            str(_negative_approval(tmp_path / "negative-approval.csv")),
            "--pisces-source-id",
            "PISCES:2026-01-01:pc20",
            "--cath-source-id",
            "CATH:4.3.0",
            "--mpstruc-source-id",
            "mpstruc:2026-01-01",
            "--structure-source-id",
            "RCSB-snapshot:2026-01-01",
            "--n-total",
            "1",
            "--class-quotas",
            "1:1,3:0,4:0",
        ]
    )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["remote_sources"] == []
    assert manifest["inputs"]["structure_archive"]["file_count"] == 1
    assert manifest["summary"]["n_selected"] == 1
    selected_path = output / "metadata" / "easy_negative_selected.csv"
    with selected_path.open(newline="", encoding="utf-8") as handle:
        selected = next(csv.DictReader(handle))
    assert selected["curation_evidence"] == "manual non-barrel topology review"
    assert "chain_cif_path" not in selected
    assert "chain_pdb_path" not in selected
    published_structure = Path(selected["structure_cif_path"])
    assert published_structure == output / "full_entries" / "1abc.cif"
    assert published_structure.read_bytes() == source_structure.read_bytes()
    declarations = declared_polymer_sequences(published_structure)
    assert [(record.author_chain_id, record.sequence) for record in declarations] == [("A", "A")]

    config = build_config({"input.dssp_failure_policy": "degraded"})
    loader = ProteinLoader(
        published_structure,
        config.input,
        dssp_bin=config.runtime.dssp_bin_path,
    )
    loader._install_dssp_annotation(DsspAnnotation({}, (), ()))
    residues = loader.get_chain_data("A")
    assert [(residue["polymer_index"], residue["resseq"]) for residue in residues] == [(0, 1)]

    approval_path = output / "metadata" / "d2_approved_negatives.csv"
    with approval_path.open(newline="", encoding="utf-8") as handle:
        approval = next(csv.DictReader(handle))
    assert approval["filename"] == "1abc.cif"
    assert approval["target_author_chain_id"] == "A"


def test_publication_negative_build_fails_on_incomplete_archive(tmp_path: Path) -> None:
    archive = tmp_path / "coordinates"
    archive.mkdir()
    _extractable_cif(archive / "2abc.cif")
    output = tmp_path / "incomplete-negative"

    with pytest.raises(RuntimeError, match="incomplete"):
        negative_builder.main(
            [
                "--out",
                str(output),
                "--pisces-file",
                str(_pisces(tmp_path / "pisces.txt")),
                "--pisces-source-id",
                "PISCES:2026-01-01:pc20",
                "--cath-domain-list",
                str(_cath(tmp_path / "cath.txt")),
                "--cath-source-id",
                "CATH:4.3.0",
                "--mpstruc-xml",
                str(_nonoverlapping_mpstruc(tmp_path / "mpstruc.xml")),
                "--mpstruc-source-id",
                "mpstruc:2026-01-01",
                "--negative-approval-manifest",
                str(_negative_approval(tmp_path / "negative-approval.csv")),
                "--structure-source-dir",
                str(archive),
                "--structure-source-id",
                "RCSB-snapshot:2026-01-01",
                "--n-total",
                "1",
                "--class-quotas",
                "1:1,3:0,4:0",
            ]
        )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"


def test_offline_positive_parse_run_writes_complete_manifest(tmp_path: Path) -> None:
    output = tmp_path / "positive-output"
    positive_builder.main(
        [
            str(_positive_xml(tmp_path / "mp.xml")),
            "--out",
            str(output),
            "--mode",
            "exploratory",
            "--skip-download",
        ]
    )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["inputs"]["mpstruc_xml"]["pinned"] is True
    assert manifest["summary"]["n_unique_pdb_codes"] == 1


def test_publication_positive_build_requires_frozen_approval_manifest(tmp_path: Path) -> None:
    args = positive_builder.build_arg_parser().parse_args(
        [str(_positive_xml(tmp_path / "mp.xml")), "--out", str(tmp_path / "output")]
    )
    with pytest.raises(ValueError, match="positive-approval-manifest"):
        positive_builder.validate_arguments(args)


def test_positive_approval_requires_exact_candidate_chain(tmp_path: Path) -> None:
    approvals_path = _write(
        tmp_path / "approvals.csv",
        """
        filename,target_author_chain_id,group_id,curation_evidence
        1ABC.cif,X,FAMILY:1,manual closure and membrane-topology review
        """,
    )
    approvals = positive_builder.load_positive_approvals(str(approvals_path))
    entry = {
        "pdb_code": "1ABC",
        "automatic_candidate_stratum": "True",
        "candidate_label_asym_id": "A",
        "representative_author_chain_id": "X",
        "class_label": "SELF_CONTAINED_MONOMER",
        "entry_cif_path": "/archive/1ABC.cif",
    }

    matched = positive_builder.match_positive_approvals(approvals, [entry])

    assert matched[0]["truth_label"] == "BARREL"
    approvals[0]["target_author_chain_id"] = "A"
    with pytest.raises(ValueError, match="author-chain"):
        positive_builder.match_positive_approvals(approvals, [entry])


def test_positive_approval_rejects_multiple_targets_for_one_filename(tmp_path: Path) -> None:
    approvals_path = _write(
        tmp_path / "approvals.csv",
        """
        filename,target_author_chain_id,group_id,curation_evidence
        1ABC.cif,A,FAMILY:1,first reviewed chain
        1ABC.cif,B,FAMILY:1,second reviewed chain
        """,
    )

    with pytest.raises(ValueError, match="exactly one target chain per structure file"):
        positive_builder.load_positive_approvals(str(approvals_path))


def test_positive_approval_rejects_blank_target_author_chain_id(tmp_path: Path) -> None:
    approvals_path = _write(
        tmp_path / "blank-chain.csv",
        """
        filename,target_author_chain_id,group_id,curation_evidence
        1ABC.cif,,FAMILY:1,independently reviewed barrel
        """,
    )

    with pytest.raises(ValueError, match="blank required field 'target_author_chain_id'"):
        positive_builder.load_positive_approvals(str(approvals_path))


def test_publication_positive_build_uses_approval_and_frozen_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "coordinates"
    archive.mkdir()
    entry = _three_copy_cif(archive / "1ABC.cif")
    (archive / "1ABC-assembly1.cif").write_bytes(entry.read_bytes())
    approvals = _write(
        tmp_path / "approvals.csv",
        """
        filename,target_author_chain_id,group_id,curation_evidence
        1ABC.cif,A,FAMILY:1,manual closure and membrane-topology review
        """,
    )
    output = tmp_path / "publication-positive"
    monkeypatch.setattr(
        positive_builder,
        "urlopen_with_retries",
        lambda *_args, **_kwargs: pytest.fail("publication build attempted network access"),
    )

    positive_builder.main(
        [
            str(_positive_xml(tmp_path / "mp.xml")),
            "--out",
            str(output),
            "--positive-approval-manifest",
            str(approvals),
            "--structure-source-dir",
            str(archive),
            "--mpstruc-source-id",
            "mpstruc:2026-01-01",
            "--structure-source-id",
            "RCSB-snapshot:2026-01-01",
            "--link-mode",
            "copy",
        ]
    )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["remote_sources"] == []
    assert manifest["summary"]["n_approved_positives"] == 1
    assert (output / "approved_positives" / "1ABC.cif").is_file()


@pytest.mark.parametrize(
    "script_name",
    ["mpstruc_download_and_classify.py", "build_easy_negatives_from_pisces_cath.py"],
)
def test_dataset_builder_cli_help(script_name: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_DIRECTORY / script_name), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout
