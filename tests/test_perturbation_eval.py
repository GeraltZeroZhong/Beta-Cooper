from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from cooper_beta.config import build_config
from cooper_beta.dssp_adapter import DsspAnnotation
from cooper_beta.integrity import atomic_write_json, file_sha256
from cooper_beta.loader import ProteinLoader
from cooper_beta.polymer_sequence import declared_polymer_sequences

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


@pytest.fixture
def perturbation_module():
    return importlib.import_module("scripts.perturbation_eval")


def _input_dirs(tmp_path: Path) -> tuple[Path, Path]:
    positive = tmp_path / "positive"
    negative = tmp_path / "negative"
    positive.mkdir()
    negative.mkdir()
    positive.joinpath("a.pdb").write_text(
        "HEADER    SYNTHETIC POSITIVE\n"
        "SEQRES   1 A    1  ALA\n"
        "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    negative.joinpath("b.pdb").write_text(
        "HEADER    SYNTHETIC NEGATIVE\n"
        "SEQRES   1 B    1  GLY\n"
        "ATOM      1  CA  GLY B   1       2.000   0.000   0.000  1.00 80.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    return positive, negative


def _write_truth_manifest(path: Path, structures: dict[Path, str]) -> Path:
    rows = [
        "filename,source_path,structure_sha256,target_author_chain_id",
        *(
            f"{structure.name},{structure.resolve()},{file_sha256(structure)},{chain}"
            for structure, chain in structures.items()
        ),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _run_noise_suite(
    perturbation,
    tmp_path: Path,
    positive: Path,
    negative: Path,
    **overrides,
) -> Path:
    parameters = {
        "positive_dir": positive,
        "negative_dir": negative,
        "workers": 2,
        "prepare_workers": 1,
        "save_dir": tmp_path / "outputs",
        "metric_level": "file",
        "noise_sigmas": [0.0],
        "noise_seeds": [7],
        "noise_atoms": "ca",
        "max_files_per_split": None,
        "subset_seed": 11,
        "metric_error_policy": "strict",
    }
    parameters.update(overrides)
    return perturbation.run_perturbation_suite(**parameters)


def test_suite_manifest_records_parameters_truth_and_evaluation_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturbation_module,
):
    perturbation = perturbation_module
    positive, negative = _input_dirs(tmp_path)
    positive_manifest = _write_truth_manifest(
        tmp_path / "positive_truth.csv", {positive / "a.pdb": "A"}
    )
    negative_manifest = _write_truth_manifest(
        tmp_path / "negative_truth.csv", {negative / "b.pdb": "B"}
    )
    monkeypatch.setattr(perturbation, "_utc_run_token", lambda: "20260811T120000Z")
    calls: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        evaluation_manifest = Path(kwargs["save_dir"]) / (
            f"evaluation_manifest_{kwargs['tag']}.json"
        )
        atomic_write_json(
            evaluation_manifest,
            {"status": "complete", "tag": kwargs["tag"]},
        )
        return {
            "evaluation_manifest": str(evaluation_manifest),
            "file_recall": 0.75,
            "file_precision": 0.6,
            "file_f1": 2 / 3,
            "file_mcc": 0.5,
        }

    evaluation_runner = importlib.import_module("cooper_beta.evaluation.runner")
    monkeypatch.setattr(evaluation_runner, "evaluate", fake_evaluate)

    summary_path = _run_noise_suite(
        perturbation,
        tmp_path,
        positive,
        negative,
        metric_level="both",
        positive_manifest=positive_manifest,
        negative_manifest=negative_manifest,
    )

    output_dir = tmp_path / "outputs" / "perturbation_20260811T120000Z"
    suite_path = output_dir / "perturbation_suite_manifest.json"
    document = json.loads(suite_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["status"] == "complete"
    assert document["phase"] == "complete"
    assert document["started_at_utc"].endswith("Z")
    assert document["completed_at_utc"].endswith("Z")
    assert document["parameters"]["workers"] == 2
    assert document["parameters"]["prepare_workers"] == 1
    assert document["parameters"]["metric_error_policy"] == "strict"
    assert document["parameters"]["noise_sigmas_angstrom"] == [0.0]
    assert document["parameters"]["noise_seeds"] == [7]
    assert document["parameters"]["subset_seed"] == 11
    assert document["parameters"]["named_defaults"]["noise_sigmas"]
    assert document["parameters"]["named_defaults"]["metric_level"] == "file"
    assert document["parameters"]["randomness"]["bit_generator"] == "PCG64"
    assert document["parameters"]["randomness"]["seed_derivation_hash"] == "blake2b"
    for split, manifest in (
        ("positive", positive_manifest),
        ("negative", negative_manifest),
    ):
        state = document["inputs"][f"{split}_truth_manifest"]
        assert state["path"] == str(manifest.resolve())
        assert state["sha256"] == file_sha256(manifest)
        assert state["size"] == manifest.stat().st_size
        assert state["snapshot_policy"] == "descriptor_verified_private_copy"
    archives = document["inputs"]["structure_archives"]
    assert archives["selection"]["algorithm"].endswith("choice_without_replacement")
    for split in ("positive", "negative"):
        assert len(archives[split]["full_inventory"]) == 1
        selected = archives[split]["selected_inventory"][0]
        assert Path(selected["archived_path"]).is_file()
        assert selected["archived_sha256"] == file_sha256(selected["archived_path"])
    assert document["script"]["sha256"] == file_sha256(perturbation.__file__)
    assert document["software"]["python"]
    assert document["experiment_count"] == 1
    assert document["metric_sampling"] == {
        "file": "one_directory_labeled_structure_file_any_chain_prediction",
        "chain": "one_manifest_target_chain_per_positive_and_negative_file",
        "all_negative_detector_chains_allowed": False,
    }
    assert document["artifact_policy"] == {"evaluated_structure_retention": "always_persist"}
    artifact = document["experiments"][0]
    evaluation_manifest = Path(artifact["evaluation_manifest"])
    assert artifact["evaluation_manifest_sha256"] == file_sha256(evaluation_manifest)
    assert artifact["mode"] == "coordinate_noise"
    assert Path(artifact["evaluated_inputs"]["positive_dir"]).is_dir()
    assert Path(calls[0]["true_dir"]).is_relative_to(output_dir)
    assert Path(calls[0]["false_dir"]).is_relative_to(output_dir)
    assert summary_path == output_dir / "perturbation_summary_20260811T120000Z.csv"
    assert document["outputs"]["summary_csv"] == str(summary_path)
    assert document["outputs"]["summary_csv_sha256"] == file_sha256(summary_path)
    assert not list(output_dir.glob(".*.tmp"))
    generated_truth = artifact["effective_truth_manifests"]
    for split, argument in (("positive", "positive_manifest"), ("negative", "negative_manifest")):
        effective_path = Path(calls[0][argument])
        assert effective_path.is_relative_to(output_dir / "effective_truth_manifests" / "noise")
        assert effective_path == Path(generated_truth[split]["manifest"]["path"])
        assert file_sha256(effective_path) == generated_truth[split]["manifest"]["sha256"]
    assert calls[0]["detector_overrides"] is None


def test_stable_noise_seed_is_path_sensitive_and_version_locked(perturbation_module):
    perturbation = perturbation_module

    assert perturbation._stable_seed(7, "nested/a.pdb") == 310245853
    assert perturbation._stable_seed(7, "nested/a.pdb") != perturbation._stable_seed(
        7, "other/a.pdb"
    )


@pytest.mark.parametrize("sigma", [0.0, 0.25])
def test_pdb_noise_preserves_seqres_noncoordinate_bytes_and_loader_inventory(
    tmp_path: Path,
    perturbation_module,
    sigma: float,
) -> None:
    perturbation = perturbation_module
    positive, _ = _input_dirs(tmp_path)
    source = positive / "a.pdb"
    destination = tmp_path / f"generated-{sigma}.pdb"

    state = perturbation._perturb_structure_file(
        source,
        destination,
        sigma=sigma,
        seed=31,
        relative_name=source.name,
        atoms="ca",
    )

    source_lines = source.read_bytes().splitlines(keepends=True)
    generated_lines = destination.read_bytes().splitlines(keepends=True)
    assert len(source_lines) == len(generated_lines)
    for source_line, generated_line in zip(source_lines, generated_lines, strict=True):
        if source_line[:6] in {b"ATOM  ", b"HETATM"}:
            assert generated_line[:30] == source_line[:30]
            assert generated_line[54:] == source_line[54:]
        else:
            assert generated_line == source_line

    assert declared_polymer_sequences(destination) == declared_polymer_sequences(source)
    invariants = state["identity_invariants"]
    assert invariants["preserved"] is True
    assert invariants["non_coordinate_bytes_exact"] is True
    assert invariants["complete_polymer_sequences_equal"] is True

    config = build_config({"input.dssp_failure_policy": "degraded"})

    def loader_inventory(path: Path) -> list[dict[str, object]]:
        loader = ProteinLoader(path, config.input, dssp_bin=config.runtime.dssp_bin_path)
        loader._install_dssp_annotation(DsspAnnotation({}, (), ()))
        return [
            {
                key: value
                for key, value in residue.items()
                if key not in {"coord", "peptide_bond_distance_to_previous_angstrom"}
            }
            for residue in loader.get_chain_data("A")
        ]

    assert loader_inventory(destination) == loader_inventory(source)


def test_chain_metrics_require_symmetric_positive_and_negative_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturbation_module,
):
    perturbation = perturbation_module
    positive, negative = _input_dirs(tmp_path)
    positive_manifest = _write_truth_manifest(tmp_path / "positive.csv", {positive / "a.pdb": "A"})
    monkeypatch.setattr(perturbation, "_utc_run_token", lambda: "20260811T120011Z")

    with pytest.raises(ValueError, match="both positive_manifest and negative_manifest"):
        _run_noise_suite(
            perturbation,
            tmp_path,
            positive,
            negative,
            metric_level="chain",
        )

    first_manifest = (
        tmp_path / "outputs" / "perturbation_20260811T120011Z" / "perturbation_suite_manifest.json"
    )
    assert json.loads(first_manifest.read_text(encoding="utf-8"))["status"] == "failed"

    monkeypatch.setattr(perturbation, "_utc_run_token", lambda: "20260811T120012Z")
    with pytest.raises(ValueError, match="must either both be provided"):
        _run_noise_suite(
            perturbation,
            tmp_path,
            positive,
            negative,
            positive_manifest=positive_manifest,
        )


def test_suite_failure_manifest_records_phase_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturbation_module,
):
    perturbation = perturbation_module
    positive, negative = _input_dirs(tmp_path)
    monkeypatch.setattr(perturbation, "_utc_run_token", lambda: "20260811T120001Z")
    evaluation_runner = importlib.import_module("cooper_beta.evaluation.runner")

    def fail_evaluate(**_kwargs):
        raise RuntimeError("detector failed deliberately")

    monkeypatch.setattr(evaluation_runner, "evaluate", fail_evaluate)

    with pytest.raises(RuntimeError, match="failed deliberately"):
        _run_noise_suite(perturbation, tmp_path, positive, negative)

    output_dir = tmp_path / "outputs" / "perturbation_20260811T120001Z"
    document = json.loads(
        (output_dir / "perturbation_suite_manifest.json").read_text(encoding="utf-8")
    )
    assert document["status"] == "failed"
    assert document["phase"] == "evaluation"
    assert document["failed_at_utc"].endswith("Z")
    assert document["current_experiment"] == "noise_sigma_0_seed_7"
    assert document["error"] == {
        "type": "RuntimeError",
        "message": "detector failed deliberately",
    }
    assert document["outputs"]["summary_csv_sha256"] is None


def test_suite_refuses_timestamp_directory_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturbation_module,
):
    perturbation = perturbation_module
    positive, negative = _input_dirs(tmp_path)
    monkeypatch.setattr(perturbation, "_utc_run_token", lambda: "20260811T120002Z")
    existing = tmp_path / "outputs" / "perturbation_20260811T120002Z"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _run_noise_suite(perturbation, tmp_path, positive, negative)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (existing / "perturbation_suite_manifest.json").exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"noise_sigmas": [float("nan")]}, "finite"),
        ({"noise_sigmas": [1.0, 1.0]}, "duplicate"),
        ({"noise_sigmas": [1.0000001, 1.0000002]}, "duplicate experiment tags"),
        ({"workers": True}, "positive integer"),
        ({"metric_error_policy": "implicit"}, "strict, exclude"),
    ],
)
def test_invalid_parameters_fail_strictly_and_leave_failure_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturbation_module,
    override: dict[str, object],
    message: str,
):
    perturbation = perturbation_module
    positive, negative = _input_dirs(tmp_path)
    token = f"20260811T12000{len(override)}Z"
    monkeypatch.setattr(perturbation, "_utc_run_token", lambda: token)

    with pytest.raises(ValueError, match=message):
        _run_noise_suite(perturbation, tmp_path, positive, negative, **override)

    document = json.loads(
        (
            tmp_path / "outputs" / f"perturbation_{token}" / "perturbation_suite_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert document["status"] == "failed"
    assert document["phase"] == "validation"
    assert document["error"]["type"] == "ValueError"


def test_subset_selection_is_seeded_without_replacement_and_fully_audited(
    tmp_path: Path,
    perturbation_module,
):
    perturbation = perturbation_module
    positive, negative = _input_dirs(tmp_path)
    for index in range(1, 8):
        positive.joinpath(f"p{index}.pdb").write_text(f"positive-{index}\n", encoding="utf-8")
        negative.joinpath(f"n{index}.pdb").write_text(f"negative-{index}\n", encoding="utf-8")

    selections: list[dict[str, list[str]]] = []
    for run_name in ("first", "second"):
        _, _, state = perturbation._prepare_persistent_base_inputs(
            positive,
            negative,
            archive_root=tmp_path / run_name,
            max_files_per_split=3,
            subset_seed=2026,
        )
        selections.append(
            {
                split: [item["relative_path"] for item in state[split]["selected_inventory"]]
                for split in ("positive", "negative")
            }
        )
        assert len(state["positive"]["full_inventory"]) == 8
        assert len(state["negative"]["full_inventory"]) == 8
        assert len(set(selections[-1]["positive"])) == 3
        assert len(set(selections[-1]["negative"])) == 3

    assert selections[0] == selections[1]


def test_subset_chain_metrics_use_persistent_exact_truth_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturbation_module,
):
    perturbation = perturbation_module
    positive, negative = _input_dirs(tmp_path)
    for index in range(1, 6):
        positive.joinpath(f"p{index}.pdb").write_text(
            f"HEADER    POSITIVE {index}\n"
            "SEQRES   1 A    1  ALA\n"
            "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C\n"
            "END\n",
            encoding="utf-8",
        )
        negative.joinpath(f"n{index}.pdb").write_text(
            f"HEADER    NEGATIVE {index}\n"
            "SEQRES   1 B    1  GLY\n"
            "ATOM      1  CA  GLY B   1       2.000   0.000   0.000  1.00 80.00           C\n"
            "END\n",
            encoding="utf-8",
        )
    positive_manifest = _write_truth_manifest(
        tmp_path / "positive.csv",
        {structure: "A" for structure in sorted(positive.glob("*.pdb"))},
    )
    negative_manifest = _write_truth_manifest(
        tmp_path / "negative.csv",
        {structure: "B" for structure in sorted(negative.glob("*.pdb"))},
    )
    monkeypatch.setattr(perturbation, "_utc_run_token", lambda: "20260811T120020Z")
    calls: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        manifest = Path(kwargs["save_dir"]) / f"evaluation_manifest_{kwargs['tag']}.json"
        atomic_write_json(manifest, {"status": "complete", "tag": kwargs["tag"]})
        return {
            "evaluation_manifest": str(manifest),
            "chain_f1": 1.0,
            "chain_mcc": 1.0,
            "file_f1": 1.0,
            "file_mcc": 1.0,
        }

    evaluation_runner = importlib.import_module("cooper_beta.evaluation.runner")
    monkeypatch.setattr(evaluation_runner, "evaluate", fake_evaluate)
    _run_noise_suite(
        perturbation,
        tmp_path,
        positive,
        negative,
        metric_level="both",
        max_files_per_split=3,
        subset_seed=2026,
        positive_manifest=positive_manifest,
        negative_manifest=negative_manifest,
    )

    output_dir = tmp_path / "outputs" / "perturbation_20260811T120020Z"
    document = json.loads(
        (output_dir / "perturbation_suite_manifest.json").read_text(encoding="utf-8")
    )
    effective = document["inputs"]["effective_truth_manifests"]
    generated = document["experiments"][0]["effective_truth_manifests"]
    for split, call_key in (
        ("positive", "positive_manifest"),
        ("negative", "negative_manifest"),
    ):
        effective_path = Path(effective[split]["effective"]["path"])
        assert Path(calls[0][call_key]) == Path(generated[split]["manifest"]["path"])
        assert effective_path.is_relative_to(output_dir)
        rows = effective_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 4
        assert effective[split]["source_row_count"] == 6
        assert effective[split]["effective_row_count"] == 3
        selected = {
            Path(item["relative_path"]).name
            for item in document["inputs"]["structure_archives"][split]["selected_inventory"]
        }
        assert {row.split(",", 1)[0] for row in rows[1:]} == selected


@pytest.mark.parametrize("sigma", [0.0, 0.25])
def test_noise_experiment_keeps_generated_inputs_hashes_and_realized_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturbation_module,
    sigma: float,
):
    perturbation = perturbation_module
    positive, negative = _input_dirs(tmp_path)
    positive_manifest = _write_truth_manifest(
        tmp_path / "positive-noise.csv", {positive / "a.pdb": "A"}
    )
    negative_manifest = _write_truth_manifest(
        tmp_path / "negative-noise.csv", {negative / "b.pdb": "B"}
    )
    monkeypatch.setattr(perturbation, "_utc_run_token", lambda: "20260811T120010Z")
    evaluation_runner = importlib.import_module("cooper_beta.evaluation.runner")
    calls: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        manifest = Path(kwargs["save_dir"]) / f"evaluation_manifest_{kwargs['tag']}.json"
        atomic_write_json(manifest, {"status": "complete", "tag": kwargs["tag"]})
        return {"evaluation_manifest": str(manifest), "file_f1": 1.0, "file_mcc": 1.0}

    monkeypatch.setattr(evaluation_runner, "evaluate", fake_evaluate)
    _run_noise_suite(
        perturbation,
        tmp_path,
        positive,
        negative,
        noise_sigmas=[sigma],
        noise_seeds=[19],
        metric_level="both",
        positive_manifest=positive_manifest,
        negative_manifest=negative_manifest,
    )

    suite_path = (
        tmp_path / "outputs" / "perturbation_20260811T120010Z" / "perturbation_suite_manifest.json"
    )
    document = json.loads(suite_path.read_text(encoding="utf-8"))
    evaluated = document["experiments"][0]["evaluated_inputs"]
    truth_state = document["experiments"][0]["effective_truth_manifests"]
    assert evaluated["kind"] == "persistent_generated_noise_archive"
    assert evaluated["identity_invariants"]["all_preserved"] is True
    for split in ("positive", "negative"):
        generated = evaluated[f"{split}_inventory"][0]
        path = Path(generated["generated_path"])
        assert path.is_file()
        assert path.is_relative_to(suite_path.parent)
        assert generated["generated_sha256"] == file_sha256(path)
        realized = generated["perturbation"]["realized_delta"]
        assert realized["selected_atom_count"] == 1
        assert generated["perturbation"]["requested_sigma_angstrom"] == sigma
        assert generated["perturbation"]["identity_invariants"]["preserved"] is True
        if sigma == 0.0:
            assert realized["selected_delta_angstrom"]["max"] == 0.0
        truth_path = Path(calls[0][f"{split}_manifest"])
        assert truth_path.is_relative_to(suite_path.parent)
        assert truth_path == Path(truth_state[split]["manifest"]["path"])
        truth_rows = truth_path.read_text(encoding="utf-8").strip().splitlines()
        assert truth_rows[0] == ("filename,source_path,structure_sha256,target_author_chain_id")
        fields = truth_rows[1].split(",")
        assert Path(fields[1]) == path.resolve()
        assert fields[2] == generated["generated_sha256"]


@pytest.mark.parametrize("sigma", [0.0, 0.35])
def test_mmcif_noise_preserves_categories_atom_identity_and_loader_residues(
    tmp_path: Path,
    perturbation_module,
    sigma: float,
):
    perturbation = perturbation_module
    source = tmp_path / "unknown-polymer.cif"
    destination = tmp_path / f"generated-{sigma}.cif"
    source.write_text(
        """\
data_unknown
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
#
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
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
HETATM 1 N N  . ZZZ A 1 1 ? 0.000 0.000 0.000 1.00 80.00 7 ZZZ A N  1
HETATM 2 C CA . ZZZ A 1 1 ? 1.000 0.000 0.000 1.00 80.00 7 ZZZ A CA 1
HETATM 3 C C  . ZZZ A 1 1 ? 1.000 1.000 0.000 1.00 80.00 7 ZZZ A C  1
HETATM 4 O O  . ZZZ A 1 1 ? 1.000 1.500 0.800 1.00 80.00 7 ZZZ A O  1
#
""",
        encoding="utf-8",
    )
    before_document = MMCIF2Dict(str(source))

    perturbation_state = perturbation._perturb_structure_file(
        source,
        destination,
        sigma=sigma,
        seed=23,
        relative_name=source.name,
        atoms="ca",
    )

    after_document = MMCIF2Dict(str(destination))
    assert after_document["_entity_poly.entity_id"] == before_document["_entity_poly.entity_id"]
    assert after_document["_entity_poly.type"] == before_document["_entity_poly.type"]
    assert after_document["_atom_site.group_PDB"] == ["HETATM"] * 4
    assert perturbation_state["identity_invariants"]["preserved"] is True
    assert perturbation_state["realized_delta"]["selected_atom_count"] == 1
    assert perturbation_state["realized_delta"]["unselected_changed_atom_count"] == 0
    if sigma == 0.0:
        assert perturbation_state["realized_delta"]["selected_delta_angstrom"]["max"] == 0.0
    else:
        assert perturbation_state["realized_delta"]["selected_changed_atom_count"] == 1

    config = build_config(
        {
            "input.include_nonstandard_amino_acids": True,
            "input.dssp_failure_policy": "degraded",
        }
    )
    loaders = [
        ProteinLoader(path, config.input, dssp_bin=config.runtime.dssp_bin_path)
        for path in (source, destination)
    ]
    loader_inventories = []
    for loader in loaders:
        loader._install_dssp_annotation(DsspAnnotation({("A", ("H_ZZZ", 7, " ")): "-"}, (), ()))
        loader_inventories.append(
            [
                {
                    key: value
                    for key, value in residue.items()
                    if key not in {"coord", "peptide_bond_distance_to_previous_angstrom"}
                }
                for residue in loader.get_ca_data("A")
            ]
        )
    assert loader_inventories[0] == loader_inventories[1]
    assert loader_inventories[0][0]["resseq"] == 7


def test_noise_generation_preserves_manifest_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturbation_module,
):
    perturbation = perturbation_module
    source = tmp_path / "source"
    source.mkdir()
    mmcif = source / "candidate.mmcif"
    mmcif.write_text("data_candidate\n", encoding="utf-8")

    def copy_instead_of_perturb(source_path: Path, destination: Path, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())

    monkeypatch.setattr(perturbation, "_perturb_structure_file", copy_instead_of_perturb)
    generated = perturbation._write_perturbed_split(
        source,
        tmp_path / "generated",
        sigma=0.25,
        seed=7,
        atoms="ca",
    )

    assert Path(generated[0]["generated_path"]).name == "candidate.mmcif"
    assert generated[0]["relative_path"] == "candidate.mmcif"
