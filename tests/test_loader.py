from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import MMCIFIO, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from cooper_beta.chain_analysis import analyze_chain_payload
from cooper_beta.config import build_config
from cooper_beta.dssp_adapter import DsspAnnotation, DsspStrandRecord
from cooper_beta.exceptions import DsspError, DsspNotFoundError, StructureParseError
from cooper_beta.loader import (
    ProteinLoader,
    _infer_element_from_atom_name,
    _mmcif_polypeptide_residue_keys,
    _mmcif_polypeptide_residue_positions,
    _selected_model_mmcif_path,
)
from cooper_beta.preparation import PrepareFailure, prepare_one_file
from cooper_beta.runtime import require_dssp_binary
from cooper_beta.strand_graph import measure_strand_graph

MINIMAL_MODEL_PDB = """\
SEQRES   1 A    1  ALA
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 80.00           C
ATOM      4  O   ALA A   1       1.000   1.500   0.800  1.00 80.00           O
TER
ENDMDL
END
"""

TWO_RESIDUE_MODEL_PDB = """\
SEQRES   1 A    2  ALA GLY
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 80.00           C
ATOM      4  O   ALA A   1       1.000   1.500   0.800  1.00 80.00           O
ATOM      5  N   GLY A   2       2.000   0.000   0.000  1.00 80.00           N
ATOM      6  CA  GLY A   2       3.000   0.000   0.000  1.00 80.00           C
ATOM      7  C   GLY A   2       3.000   1.000   0.000  1.00 80.00           C
ATOM      8  O   GLY A   2       3.000   1.500   0.800  1.00 80.00           O
TER
ENDMDL
END
"""

TWO_CHAIN_MODEL_PDB = """\
SEQRES   1 A    1  ALA
SEQRES   1 B    1  GLY
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 80.00           C
ATOM      4  O   ALA A   1       1.000   1.500   0.800  1.00 80.00           O
TER
ATOM      5  N   GLY B   1       5.000   0.000   0.000  1.00 80.00           N
ATOM      6  CA  GLY B   1       6.000   0.000   0.000  1.00 80.00           C
ATOM      7  C   GLY B   1       6.000   1.000   0.000  1.00 80.00           C
ATOM      8  O   GLY B   1       6.000   1.500   0.800  1.00 80.00           O
TER
ENDMDL
END
"""


def _atom_site_only_mmcif(*, gly_author_chain: str = "X") -> str:
    residues = (
        ("ATOM", "ALA", "A", "1", "X", "1", 0.00),
        ("HETATM", "MSE", "B", ".", "X", "2", 2.33),
        ("ATOM", "GLY", "C", "2", gly_author_chain, "3", 4.66),
        # Biopython recognizes GLN as an amino acid. Its distance from the
        # protein makes this an unlinked small-molecule residue, not polymer.
        ("HETATM", "GLN", "D", ".", "X", "50", 100.00),
    )
    rows: list[str] = []
    atom_id = 1
    for group, component, label_chain, label_sequence, author_chain, auth_sequence, x in residues:
        for element, atom_name, offset_x, offset_y in (
            ("N", "N", 0.00, 0.00),
            ("C", "CA", 0.45, 0.40),
            ("C", "C", 1.00, 0.00),
            ("O", "O", 1.00, 1.00),
        ):
            rows.append(
                f"{group} {atom_id} {element} {atom_name} . {component} {label_chain} ? "
                f"{label_sequence} ? {x + offset_x:.2f} {offset_y:.2f} 0.00 1.0 20.0 ? "
                f"{auth_sequence} {component} {author_chain} {atom_name} 1"
            )
            atom_id += 1
    return (
        """\
data_atom_site_only
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
"""
        + "\n".join(rows)
        + "\n#\n"
    )


def _loader(
    file_path: Path,
    *,
    dssp_failure_policy: str = "error",
    input_overrides: dict[str, object] | None = None,
) -> ProteinLoader:
    overrides = dict(input_overrides or {})
    overrides["input.dssp_failure_policy"] = dssp_failure_policy
    cfg = build_config(overrides)
    return ProteinLoader(
        file_path,
        cfg.input,
        dssp_bin=cfg.runtime.dssp_bin_path,
    )


def _install_assignments(
    loader: ProteinLoader,
    assignments: dict[tuple[str, tuple[str, int, str]], str],
) -> None:
    loader._install_dssp_annotation(DsspAnnotation(assignments, (), ()))


def _supported_dssp_is_available() -> bool:
    try:
        require_dssp_binary()
    except (DsspError, DsspNotFoundError):
        return False
    return True


def test_pdb_atom_name_alignment_distinguishes_alpha_carbon_from_calcium():
    assert _infer_element_from_atom_name(" CA ") == "C"
    assert _infer_element_from_atom_name("CA  ") == "Ca"
    assert _infer_element_from_atom_name("1HG1") == "H"


@pytest.mark.parametrize(
    ("example_name", "expected_nodes", "expected_edges", "expected_cyclic_nodes"),
    [
        ("M4QT10.cif", 8, 8, 8),
        ("A0A2R4ALS6.cif", 9, 9, 8),
    ],
)
@pytest.mark.skipif(
    not _supported_dssp_is_available(),
    reason="DSSP 4.5.3 or newer is required for the real-structure regression test.",
)
def test_true_positive_examples_have_fresh_closed_strand_graphs(
    example_name: str,
    expected_nodes: int,
    expected_edges: int,
    expected_cyclic_nodes: int,
):
    example_path = Path(__file__).resolve().parents[1] / "examples" / example_name
    loader = _loader(example_path)

    residues = loader.get_chain_data("A")
    graph = loader.get_strand_graph("A")
    features = measure_strand_graph(graph)

    assert len(residues) > 0
    assert len(graph.nodes) == expected_nodes
    assert len(graph.edges) == expected_edges
    assert features.cycle_strand_count == expected_cyclic_nodes
    assert features.cycle_rank == 1


def test_loader_accepts_gzipped_pdb_with_plain_pdb_suffix(tmp_path: Path):
    pdb_path = tmp_path / "alphafold-style.pdb"
    pdb_path.write_bytes(gzip.compress(MINIMAL_MODEL_PDB.encode("utf-8")))

    loader = _loader(pdb_path)

    assert loader.available_chains() == ["A"]


def test_loader_rejects_blank_chain_id_instead_of_silently_renaming_it(tmp_path: Path):
    pdb_path = tmp_path / "blank-chain.pdb"
    pdb_path.write_text(MINIMAL_MODEL_PDB.replace("ALA A", "ALA  "), encoding="utf-8")

    with pytest.raises(StructureParseError, match="blank chain ID"):
        _loader(pdb_path)

    cfg = build_config(
        {
            "runtime.prepare_cache_enabled": False,
        }
    )
    prepared = prepare_one_file(str(pdb_path), cfg)
    assert isinstance(prepared, PrepareFailure)
    assert prepared.error_code == "STRUCTURE_PARSE_FAILED"


def test_loader_accepts_mmcif_with_explicit_gzip_suffix(tmp_path: Path):
    pdb_path = tmp_path / "source.pdb"
    pdb_path.write_text(MINIMAL_MODEL_PDB, encoding="utf-8")
    structure = PDBParser(QUIET=True).get_structure("source", pdb_path)
    mmcif_path = tmp_path / "source.cif"
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(mmcif_path))
    compressed_path = tmp_path / "source.cif.gz"
    compressed_path.write_bytes(gzip.compress(mmcif_path.read_bytes()))

    loader = _loader(compressed_path)

    assert loader.available_chains() == ["A"]


def test_loader_exports_cryst1_for_dssp_compatibility(tmp_path: Path):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(MINIMAL_MODEL_PDB, encoding="utf-8")
    loader = _loader(pdb_path)

    exported_path = loader._export_protein_only_pdb()
    try:
        exported_text = Path(exported_path).read_text(encoding="utf-8")
    finally:
        os.remove(exported_path)

    assert "CRYST1" in exported_text.splitlines()[1]


def test_loader_maps_dssp_blank_residue_key_to_nonstandard_amino_acid(tmp_path: Path):
    pdb_path = tmp_path / "mse.pdb"
    pdb_path.write_text(
        """\
SEQRES   1 A    1  MSE
MODEL        1
HETATM    1  N   MSE A   1       0.000   0.000   0.000  1.00 80.00           N
HETATM    2  CA  MSE A   1       1.000   0.000   0.000  1.00 80.00           C
HETATM    3  C   MSE A   1       1.000   1.000   0.000  1.00 80.00           C
HETATM    4  O   MSE A   1       1.000   1.500   0.800  1.00 80.00           O
TER
ENDMDL
END
""",
        encoding="utf-8",
    )
    loader = _loader(pdb_path, dssp_failure_policy="degraded")
    _install_assignments(loader, {("A", (" ", 1, " ")): "B"})

    residues = loader.get_chain_data("A")

    assert len(residues) == 1
    assert residues[0]["hetfield"] == "H_MSE"
    assert residues[0]["is_sheet"] is True


def test_loader_payload_preserves_polymer_positions_across_missing_ca(tmp_path: Path):
    pdb_path = tmp_path / "missing-ca.pdb"
    pdb_path.write_text(
        """\
SEQRES   1 A    3  ALA GLY SER
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 80.00           C
ATOM      4  N   GLY A   2       2.000   0.000   0.000  1.00 80.00           N
ATOM      5  C   GLY A   2       2.000   1.000   0.000  1.00 80.00           C
ATOM      6  N   SER A   3       3.000   0.000   0.000  1.00 80.00           N
ATOM      7  CA  SER A   3       4.000   0.000   0.000  1.00 80.00           C
ATOM      8  C   SER A   3       4.000   1.000   0.000  1.00 80.00           C
TER
ENDMDL
END
""",
        encoding="utf-8",
    )
    loader = _loader(pdb_path, dssp_failure_policy="degraded")
    _install_assignments(
        loader,
        {
            ("A", (" ", 1, " ")): "B",
            ("A", (" ", 3, " ")): "B",
        },
    )

    residues = loader.get_chain_data("A")

    assert [residue["resseq"] for residue in residues] == [1, 3]
    assert [residue["polymer_index"] for residue in residues] == [0, 2]
    assert residues[1]["peptide_bond_distance_to_previous_angstrom"] > 1.8
    assert [residue["dssp_assignment_available"] for residue in residues] == [False, False]
    assert [residue["is_sheet"] for residue in residues] == [False, False]
    assert all(residue["res_uid"]["chain"] == "A" for residue in residues)


def test_loader_maps_noncontiguous_e_residues_in_one_physical_strand(tmp_path: Path):
    pdb_path = tmp_path / "beta-bulge.pdb"
    pdb_path.write_text(
        """\
SEQRES   1 A    3  ALA GLY VAL
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 80.00           C
ATOM      4  O   ALA A   1       1.000   1.500   0.800  1.00 80.00           O
ATOM      5  N   GLY A   2       2.000   0.000   0.000  1.00 80.00           N
ATOM      6  CA  GLY A   2       3.000   0.000   0.000  1.00 80.00           C
ATOM      7  C   GLY A   2       3.000   1.000   0.000  1.00 80.00           C
ATOM      8  O   GLY A   2       3.000   1.500   0.800  1.00 80.00           O
ATOM      9  N   VAL A   3       4.000   0.000   0.000  1.00 80.00           N
ATOM     10  CA  VAL A   3       5.000   0.000   0.000  1.00 80.00           C
ATOM     11  C   VAL A   3       5.000   1.000   0.000  1.00 80.00           C
ATOM     12  O   VAL A   3       5.000   1.500   0.800  1.00 80.00           O
TER
ENDMDL
END
""",
        encoding="utf-8",
    )
    loader = _loader(pdb_path)
    loader._install_dssp_annotation(
        DsspAnnotation(
            residue_assignments={
                ("A", (" ", 1, " ")): "E",
                ("A", (" ", 2, " ")): "B",
                ("A", (" ", 3, " ")): "E",
            },
            strand_records=(
                DsspStrandRecord(
                    source_id=("S1", "R1"),
                    author_chain_id="A",
                    residue_keys=(
                        ("A", (" ", 1, " ")),
                        ("A", (" ", 3, " ")),
                    ),
                ),
            ),
            strand_edges=(),
        )
    )

    residues = loader.get_chain_data("A")
    graph = loader.get_strand_graph("A")

    assert [residue["strand_node_id"] for residue in residues] == [
        "strand_0",
        None,
        "strand_0",
    ]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].residue_range.start_polymer_index == 0
    assert graph.nodes[0].residue_range.end_polymer_index == 2


def test_loader_preserves_pdb_gap_when_residue_record_is_entirely_missing(tmp_path: Path):
    pdb_path = tmp_path / "missing-record.pdb"
    pdb_path.write_text(
        """\
SEQRES   1 A    3  ALA GLY SER
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 80.00           C
ATOM      4  N   SER A   3       1.000   2.270   0.000  1.00 80.00           N
ATOM      5  CA  SER A   3       2.000   2.270   0.000  1.00 80.00           C
ATOM      6  C   SER A   3       2.000   3.270   0.000  1.00 80.00           C
TER
ENDMDL
END
""",
        encoding="utf-8",
    )
    loader = _loader(pdb_path, dssp_failure_policy="degraded")
    _install_assignments(
        loader,
        {
            ("A", (" ", 1, " ")): "B",
            ("A", (" ", 3, " ")): "B",
        },
    )

    residues = loader.get_chain_data("A")

    assert [residue["polymer_index"] for residue in residues] == [0, 2]
    assert residues[1]["peptide_bond_distance_to_previous_angstrom"] == pytest.approx(1.27)


def test_loader_rejects_ambiguous_pdb_seqres_position_mapping(tmp_path: Path):
    pdb_path = tmp_path / "ambiguous-seqres.pdb"
    pdb_path.write_text(
        """\
SEQRES   1 A    2  ALA ALA
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 80.00           C
TER
ENDMDL
END
""",
        encoding="utf-8",
    )
    loader = _loader(pdb_path, dssp_failure_policy="degraded")
    _install_assignments(loader, {("A", (" ", 1, " ")): "B"})

    with pytest.raises(StructureParseError, match="more than one valid mapping"):
        loader.get_chain_data("A")


def test_mmcif_polymer_index_uses_declared_label_sequence_position(tmp_path: Path):
    mmcif_path = tmp_path / "positions.cif"
    mmcif_path.write_text(
        """\
data_positions
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
#
loop_
_atom_site.label_entity_id
_atom_site.pdbx_PDB_model_num
_atom_site.group_PDB
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
1 1 ATOM A 1 ALA A 101 ?
1 1 ATOM A 3 SER A 102 ?
#
""",
        encoding="utf-8",
    )

    assert _mmcif_polypeptide_residue_positions(mmcif_path) == {
        ("A", (" ", 101, " ")): 0,
        ("A", (" ", 102, " ")): 2,
    }


def test_mmcif_polymer_position_mapping_rejects_conflicting_atom_rows(tmp_path: Path):
    mmcif_path = tmp_path / "conflicting-positions.cif"
    mmcif_path.write_text(
        """\
data_conflict
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
#
loop_
_atom_site.label_entity_id
_atom_site.pdbx_PDB_model_num
_atom_site.group_PDB
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
1 1 ATOM A 1 ALA A 101 ?
1 1 ATOM A 2 ALA A 101 ?
#
""",
        encoding="utf-8",
    )

    with pytest.raises(StructureParseError, match="conflicting polymer positions"):
        _mmcif_polypeptide_residue_positions(mmcif_path)


def test_atom_site_only_mmcif_materializes_one_exact_polymer_for_dssp(tmp_path: Path):
    mmcif_path = tmp_path / "atom-site-only.cif"
    mmcif_path.write_text(
        _atom_site_only_mmcif()
        + "loop_\n_refine_ls_shell.d_res_high\n_refine_ls_shell.d_res_low\n2.0 2.2\n#\n",
        encoding="utf-8",
    )
    loader = _loader(mmcif_path, dssp_failure_policy="degraded")

    assert loader._mmcif_polypeptide_positions == {
        ("X", (" ", 1, " ")): 0,
        ("X", ("H_MSE", 2, " ")): 1,
        ("X", (" ", 3, " ")): 2,
    }
    assert loader._mmcif_polymer_mapping is not None
    with _selected_model_mmcif_path(
        mmcif_path,
        model_id=0,
        polymer_mapping=loader._mmcif_polymer_mapping,
    ) as selected_path:
        selected = MMCIF2Dict(selected_path)

    assert len(selected["_atom_site.group_PDB"]) == 12
    assert set(selected["_atom_site.label_asym_id"]) == {"A"}
    assert set(selected["_atom_site.label_entity_id"]) == {"1"}
    assert selected["_entity_poly.type"] == ["polypeptide(L)"]
    assert selected["_entity_poly.nstd_monomer"] == ["yes"]
    assert selected["_entity_poly_seq.mon_id"] == ["ALA", "MSE", "GLY"]
    assert selected["_pdbx_poly_seq_scheme.pdb_strand_id"] == ["X", "X", "X"]
    assert "GLN" not in selected["_atom_site.label_comp_id"]
    assert "_refine_ls_shell.d_res_high" not in selected


def test_atom_site_only_mmcif_rejects_multiple_author_chains(tmp_path: Path):
    mmcif_path = tmp_path / "multiple-author-chains.cif"
    mmcif_path.write_text(
        _atom_site_only_mmcif(gly_author_chain="Y"),
        encoding="utf-8",
    )

    with pytest.raises(StructureParseError, match="exactly one author chain"):
        _loader(mmcif_path, dssp_failure_policy="degraded")


def test_atom_site_only_mmcif_rejects_partial_polymer_metadata(tmp_path: Path):
    mmcif_path = tmp_path / "partial-polymer-metadata.cif"
    mmcif_path.write_text(
        _atom_site_only_mmcif() + "loop_\n_struct_asym.id\n_struct_asym.entity_id\nA 1\n#\n",
        encoding="utf-8",
    )

    with pytest.raises(StructureParseError, match="partial polymer metadata"):
        _mmcif_polypeptide_residue_positions(mmcif_path)


def test_mmcif_amino_acid_like_nonpolymer_is_not_included(tmp_path: Path):
    mmcif_path = tmp_path / "polymer-ligand-collision.cif"
    mmcif_path.write_text(
        """\
data_collision
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
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N N  . ALA P 1 1 ? 0.0 0.0 0.0 1.0 20.0 ? 101 ALA A N  1
ATOM   2 C CA . ALA P 1 1 ? 1.0 0.0 0.0 1.0 20.0 ? 101 ALA A CA 1
ATOM   3 C C  . ALA P 1 1 ? 1.0 1.0 0.0 1.0 20.0 ? 101 ALA A C  1
HETATM 4 C CA . GLU L 2 . ? 2.0 0.0 0.0 1.0 20.0 ? 1744 GLU A CA 1
#
""",
        encoding="utf-8",
    )
    loader = _loader(mmcif_path, dssp_failure_policy="degraded")
    _install_assignments(loader, {("A", (" ", 101, " ")): "B"})

    residues = loader.get_chain_data("A")

    assert [(residue["hetfield"], residue["polymer_index"]) for residue in residues] == [("", 0)]


def test_mmcif_residues_are_ordered_by_declared_polymer_position(tmp_path: Path):
    mmcif_path = tmp_path / "reordered-atoms.cif"
    mmcif_path.write_text(
        """\
data_reordered
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
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N  . SER A 1 3 ? 3.0 0.0 0.0 1.0 20.0 ? 103 SER A N  1
ATOM 2 C CA . SER A 1 3 ? 4.0 0.0 0.0 1.0 20.0 ? 103 SER A CA 1
ATOM 3 C C  . SER A 1 3 ? 4.0 1.0 0.0 1.0 20.0 ? 103 SER A C  1
ATOM 4 N N  . ALA A 1 1 ? 0.0 0.0 0.0 1.0 20.0 ? 101 ALA A N  1
ATOM 5 C CA . ALA A 1 1 ? 1.0 0.0 0.0 1.0 20.0 ? 101 ALA A CA 1
ATOM 6 C C  . ALA A 1 1 ? 1.0 1.0 0.0 1.0 20.0 ? 101 ALA A C  1
#
""",
        encoding="utf-8",
    )
    loader = _loader(mmcif_path, dssp_failure_policy="degraded")
    _install_assignments(
        loader,
        {
            ("A", (" ", 101, " ")): "B",
            ("A", (" ", 103, " ")): "B",
        },
    )

    residues = loader.get_chain_data("A")

    assert [(residue["resseq"], residue["polymer_index"]) for residue in residues] == [
        (101, 0),
        (103, 2),
    ]


def test_mmcif_unknown_residue_is_recognized_from_polypeptide_entity(tmp_path: Path):
    mmcif_path = tmp_path / "unknown-polymer.cif"
    mmcif_path.write_text(
        """\
data_unknown
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
#
loop_
_atom_site.label_entity_id
_atom_site.pdbx_PDB_model_num
_atom_site.group_PDB
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
1 1 ATOM A 1 UNK MX 7 ?
#
""",
        encoding="utf-8",
    )

    keys = _mmcif_polypeptide_residue_keys(mmcif_path)

    assert keys == {("MX", (" ", 7, " "))}


def test_loader_selects_highest_occupancy_altloc_deterministically(tmp_path: Path):
    pdb_path = tmp_path / "altloc.pdb"
    pdb_path.write_text(
        """\
SEQRES   1 A    1  ALA
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA AALA A   1       1.000   0.000   0.000  0.40 80.00           C
ATOM      3  CA BALA A   1       2.000   0.000   0.000  0.60 80.00           C
ATOM      4  C   ALA A   1       2.000   1.000   0.000  1.00 80.00           C
TER
ENDMDL
END
""",
        encoding="utf-8",
    )
    loader = _loader(pdb_path, dssp_failure_policy="degraded")
    _install_assignments(loader, {("A", (" ", 1, " ")): "B"})

    residues = loader.get_chain_data("A")

    assert len(residues) == 1
    assert np.allclose(residues[0]["coord"], [2.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "missing_atoms",
    [frozenset({"N"}), frozenset({"O"}), frozenset({"N", "O"})],
    ids=["missing-n", "missing-o", "missing-n-and-o"],
)
def test_incomplete_backbone_ca_residue_is_retained_without_dssp_assignment(
    tmp_path: Path,
    missing_atoms: frozenset[str],
):
    pdb_path = tmp_path / "incomplete-backbone.pdb"
    lines = [
        line
        for line in MINIMAL_MODEL_PDB.splitlines()
        if not (line.startswith("ATOM") and line[12:16].strip() in missing_atoms)
    ]
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loader = _loader(pdb_path)
    _install_assignments(loader, {})

    result = loader.prepare_chain("A")

    assert result.failed is False
    assert len(result.residues) == 1
    assert result.residues[0]["dssp_assignment_available"] is False
    assert result.residues[0]["is_sheet"] is False
    assert result.residues[0]["strand_node_id"] is None


def test_chain_local_coverage_failure_preserves_complete_sibling_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pdb_path = tmp_path / "two-chain-coverage.pdb"
    pdb_path.write_text(TWO_CHAIN_MODEL_PDB, encoding="utf-8")
    cfg = build_config({"runtime.prepare_cache_enabled": False})
    monkeypatch.setattr(
        ProteinLoader,
        "_run_dssp",
        lambda self, path: DsspAnnotation({("A", (" ", 1, " ")): "-"}, (), ()),
    )

    prepared = prepare_one_file(str(pdb_path), cfg)

    assert not isinstance(prepared, PrepareFailure)
    assert [payload["chain"] for payload in prepared] == ["A", "B"]
    assert prepared[0]["degraded"] is False
    assert len(prepared[0]["residues_data"]) == 1
    assert prepared[1]["degraded"] is True
    assert prepared[1]["degradation_code"] == "DSSP_FAILED"
    assert "assigned 0/1 required residues" in str(prepared[1]["degradation_reason"])
    assert prepared[1]["residues_data"] == []


def test_file_wide_dssp_failure_remains_prepare_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pdb_path = tmp_path / "file-wide-dssp-failure.pdb"
    pdb_path.write_text(TWO_CHAIN_MODEL_PDB, encoding="utf-8")
    cfg = build_config({"runtime.prepare_cache_enabled": False})

    def fail_dssp(self, path):
        raise RuntimeError("mkdssp process failed")

    monkeypatch.setattr(ProteinLoader, "_run_dssp", fail_dssp)

    prepared = prepare_one_file(str(pdb_path), cfg)

    assert isinstance(prepared, PrepareFailure)
    assert prepared.error_code == "DSSP_FAILED"
    assert "mkdssp process failed" in prepared.message


@pytest.mark.parametrize(
    "assignments",
    [
        {},
        {("B", (" ", 1, " ")): "B"},
    ],
    ids=["empty-mapping", "target-chain-entirely-missing"],
)
def test_loader_strict_coverage_rejects_chain_without_assignments(
    tmp_path: Path,
    assignments,
):
    pdb_path = tmp_path / "missing-chain-coverage.pdb"
    pdb_path.write_text(MINIMAL_MODEL_PDB, encoding="utf-8")
    loader = _loader(pdb_path)
    _install_assignments(loader, assignments)

    with pytest.raises(DsspError, match=r"assigned 0/1 required residues"):
        loader.get_chain_data("A")


def test_loader_strict_coverage_rejects_partial_chain_assignment(tmp_path: Path):
    pdb_path = tmp_path / "partial-coverage.pdb"
    pdb_path.write_text(TWO_RESIDUE_MODEL_PDB, encoding="utf-8")
    loader = _loader(pdb_path)
    _install_assignments(loader, {("A", (" ", 1, " ")): "B"})

    with pytest.raises(DsspError, match=r"assigned 1/2 required residues"):
        loader.get_chain_data("A")


def test_loader_degraded_coverage_records_failure_instead_of_normal_negative(
    tmp_path: Path,
):
    pdb_path = tmp_path / "partial-degraded.pdb"
    pdb_path.write_text(TWO_RESIDUE_MODEL_PDB, encoding="utf-8")
    loader = _loader(pdb_path, dssp_failure_policy="degraded")
    _install_assignments(loader, {("A", (" ", 1, " ")): "B"})

    chain_result = loader.prepare_chain("A")

    assert chain_result.residues == ()
    assert chain_result.error_code == "DSSP_FAILED"
    assert chain_result.error_message is not None
    assert "assigned 1/2 required residues" in chain_result.error_message
    assert "Missing residues cannot be interpreted as coil" in chain_result.error_message
    assert loader.get_chain_data("A") == []


def test_partial_dssp_mapping_becomes_degraded_error_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pdb_path = tmp_path / "partial-pipeline.pdb"
    pdb_path.write_text(TWO_RESIDUE_MODEL_PDB, encoding="utf-8")
    cfg = build_config(
        {
            "input.dssp_failure_policy": "degraded",
            "runtime.prepare_cache_enabled": False,
        }
    )
    monkeypatch.setattr(
        ProteinLoader,
        "_run_dssp",
        lambda self, path: DsspAnnotation({("A", (" ", 1, " ")): "B"}, (), ()),
    )

    prepared = prepare_one_file(str(pdb_path), cfg)

    assert not isinstance(prepared, PrepareFailure)
    assert len(prepared) == 1
    assert prepared[0]["degraded"] is True
    result = analyze_chain_payload(prepared[0], cfg)
    assert result["result"] == "ERROR"
    assert result["error_code"] == "DSSP_FAILED"


def test_loader_coverage_accepts_explicit_non_sheet_assignment(tmp_path: Path):
    pdb_path = tmp_path / "explicit-coil.pdb"
    pdb_path.write_text(MINIMAL_MODEL_PDB, encoding="utf-8")
    loader = _loader(pdb_path)
    _install_assignments(loader, {("A", (" ", 1, " ")): "-"})

    residues = loader.get_chain_data("A")

    assert len(residues) == 1
    assert residues[0]["is_sheet"] is False
    assert loader.secondary_structure_error is None


def test_loader_rejects_invalid_dssp_code_instead_of_treating_it_as_non_sheet(
    tmp_path: Path,
):
    pdb_path = tmp_path / "invalid-dssp-code.pdb"
    pdb_path.write_text(MINIMAL_MODEL_PDB, encoding="utf-8")
    loader = _loader(pdb_path)
    _install_assignments(loader, {("A", (" ", 1, " ")): "X"})

    with pytest.raises(DsspError, match="Invalid DSSP secondary-structure code 'X'"):
        loader.get_chain_data("A")
