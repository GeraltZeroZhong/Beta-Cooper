from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cooper_beta.dssp_adapter import (
    DsspAdapterError,
    parse_dssp_annotated_mmcif,
    run_dssp_annotation,
    validate_dssp_coverage,
)


def _annotated_mmcif(
    *,
    atom_rows: str,
    summary_rows: str | None,
    range_rows: str | None = None,
    ladder_rows: str | None = None,
) -> str:
    summary = ""
    if summary_rows is not None:
        summary = f"""\
loop_
_dssp_struct_summary.label_asym_id
_dssp_struct_summary.label_seq_id
_dssp_struct_summary.secondary_structure
{summary_rows}
#
"""
    ranges = ""
    if range_rows is not None:
        ranges = f"""\
loop_
_struct_sheet_range.sheet_id
_struct_sheet_range.id
_struct_sheet_range.beg_label_asym_id
_struct_sheet_range.beg_label_seq_id
_struct_sheet_range.end_label_asym_id
_struct_sheet_range.end_label_seq_id
{range_rows}
#
"""
    ladders = ""
    if ladder_rows is not None:
        ladders = f"""\
loop_
_dssp_struct_ladder.beg_1_label_asym_id
_dssp_struct_ladder.beg_1_label_seq_id
_dssp_struct_ladder.end_1_label_asym_id
_dssp_struct_ladder.end_1_label_seq_id
_dssp_struct_ladder.beg_2_label_asym_id
_dssp_struct_ladder.beg_2_label_seq_id
_dssp_struct_ladder.end_2_label_asym_id
_dssp_struct_ladder.end_2_label_seq_id
{ladder_rows}
#
"""
    return f"""\
data_dssp_test
#
loop_
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.group_PDB
_atom_site.label_comp_id
{atom_rows}
#
{summary}{ranges}{ladders}"""


def test_parse_dssp_annotation_maps_author_residues_and_ladder_graph(tmp_path: Path):
    output_path = tmp_path / "annotated.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows=(
                "AA 101 C1 150 ? ATOM GLN\n"
                "AA 102 C1 151 A HETATM MSE\n"
                "AA 110 C1 160 ? ATOM VAL\n"
                "AA 111 C1 161 ? ATOM ILE\n"
            ),
            summary_rows="AA 101 E\nAA 102 E\nAA 110 E\nAA 111 E\n",
            range_rows="S1 R1 AA 101 AA 102\nS1 R2 AA 110 AA 111\n",
            ladder_rows="AA 101 AA 102 AA 110 AA 111\n",
        ),
        encoding="utf-8",
    )

    annotation = parse_dssp_annotated_mmcif(output_path)

    assert annotation.residue_assignments == {
        ("C1", (" ", 150, " ")): "E",
        ("C1", ("H_MSE", 151, "A")): "E",
        ("C1", (" ", 160, " ")): "E",
        ("C1", (" ", 161, " ")): "E",
    }
    assert [record.source_id for record in annotation.strand_records] == [
        ("S1", "R1"),
        ("S1", "R2"),
    ]
    assert annotation.strand_records[0].author_chain_id == "C1"
    assert len(annotation.strand_edges) == 1
    assert annotation.strand_edges[0].first_source_id == ("S1", "R1")
    assert annotation.strand_edges[0].second_source_id == ("S1", "R2")


def test_ladder_side_merges_e_ranges_split_by_a_beta_bridge(tmp_path: Path):
    output_path = tmp_path / "split-strand.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows=(
                "A 1 A 1 ? ATOM ALA\n"
                "A 2 A 2 ? ATOM VAL\n"
                "A 3 A 3 ? ATOM GLY\n"
                "A 4 A 4 ? ATOM ILE\n"
                "A 10 A 10 ? ATOM LEU\n"
                "A 11 A 11 ? ATOM THR\n"
                "A 12 A 12 ? ATOM SER\n"
            ),
            summary_rows=("A 1 E\nA 2 E\nA 3 B\nA 4 E\nA 10 E\nA 11 E\nA 12 E\n"),
            range_rows=("S1 R1 A 1 A 2\nS1 R2 A 4 A 4\nS1 R3 A 10 A 12\nS2 B1 A 3 A 3\n"),
            # The first ladder side crosses the B residue. The second side is
            # reversed, as DSSP emits for an antiparallel ladder.
            ladder_rows="A 2 A 4 A 12 A 10\n",
        ),
        encoding="utf-8",
    )

    annotation = parse_dssp_annotated_mmcif(output_path)

    assert [record.source_id for record in annotation.strand_records] == [
        ("S1", "R1"),
        ("S1", "R3"),
    ]
    assert annotation.strand_records[0].residue_keys == (
        ("A", (" ", 1, " ")),
        ("A", (" ", 2, " ")),
        ("A", (" ", 4, " ")),
    )
    assert ("A", (" ", 3, " ")) not in annotation.strand_records[0].residue_keys
    assert [(edge.first_source_id, edge.second_source_id) for edge in annotation.strand_edges] == [
        (("S1", "R1"), ("S1", "R3"))
    ]


def test_ladder_side_rejects_endpoints_from_different_label_chains(tmp_path: Path):
    output_path = tmp_path / "cross-chain-side.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows=("LA 1 A 1 ? ATOM ALA\nLB 1 B 1 ? ATOM VAL\nLA 5 A 5 ? ATOM LEU\n"),
            summary_rows="LA 1 E\nLB 1 E\nLA 5 E\n",
            range_rows=("S1 R1 LA 1 LA 1\nS1 R2 LB 1 LB 1\nS1 R3 LA 5 LA 5\n"),
            ladder_rows="LA 1 LB 1 LA 5 LA 5\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DsspAdapterError, match="spans label chains"):
        parse_dssp_annotated_mmcif(output_path)


def test_ladder_row_ignores_two_sides_that_merge_to_one_physical_strand(tmp_path: Path):
    output_path = tmp_path / "self-adjacency.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="A 1 A 1 ? ATOM ALA\nA 2 A 2 ? ATOM VAL\n",
            summary_rows="A 1 E\nA 2 E\n",
            range_rows="S1 R1 A 1 A 1\nS1 R2 A 2 A 2\n",
            ladder_rows="A 1 A 2 A 2 A 2\n",
        ),
        encoding="utf-8",
    )

    annotation = parse_dssp_annotated_mmcif(output_path)
    assert len(annotation.strand_records) == 1
    assert annotation.strand_edges == ()


def test_inter_chain_ladder_is_excluded_from_chain_level_graph(tmp_path: Path):
    output_path = tmp_path / "inter-chain-ladder.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="LA 1 A 1 ? ATOM ALA\nLB 1 B 1 ? ATOM VAL\n",
            summary_rows="LA 1 E\nLB 1 E\n",
            range_rows="S1 R1 LA 1 LA 1\nS1 R2 LB 1 LB 1\n",
            ladder_rows="LA 1 LA 1 LB 1 LB 1\n",
        ),
        encoding="utf-8",
    )

    annotation = parse_dssp_annotated_mmcif(output_path)

    assert len(annotation.strand_records) == 2
    assert annotation.strand_edges == ()


def test_parse_dssp_annotation_excludes_pure_isolated_beta_bridge_range(tmp_path: Path):
    output_path = tmp_path / "bridge.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="A 1 A 1 ? ATOM ALA\n",
            summary_rows="A 1 B\n",
            range_rows="S1 R1 A 1 A 1\n",
        ),
        encoding="utf-8",
    )

    annotation = parse_dssp_annotated_mmcif(output_path)

    assert annotation.residue_assignments == {("A", (" ", 1, " ")): "B"}
    assert annotation.strand_records == ()
    assert annotation.strand_edges == ()


def test_parse_dssp_annotation_rejects_ambiguous_label_to_author_mapping(tmp_path: Path):
    output_path = tmp_path / "ambiguous.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="AA 101 C1 150 ? ATOM GLN\nAA 101 C2 150 ? ATOM GLN\n",
            summary_rows="AA 101 E\n",
            range_rows="S1 R1 AA 101 AA 101\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DsspAdapterError, match="maps ambiguously"):
        parse_dssp_annotated_mmcif(output_path)


def test_parse_dssp_annotation_rejects_two_labels_for_one_author_residue(tmp_path: Path):
    output_path = tmp_path / "duplicate-author.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="AA 101 C1 150 ? ATOM GLN\nAB 201 C1 150 ? ATOM GLN\n",
            summary_rows="AA 101 E\nAB 201 B\n",
            range_rows="S1 R1 AA 101 AA 101\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DsspAdapterError, match="both map to author residue"):
        parse_dssp_annotated_mmcif(output_path)


@pytest.mark.parametrize(
    ("summary_rows", "message"),
    [
        ("AA 101 E\nAA 101 E\n", "duplicate summary row"),
        ("AA 101 E\nAA 101 B\n", "conflicting summary assignments"),
    ],
)
def test_parse_dssp_annotation_rejects_duplicate_or_conflicting_summary_rows(
    tmp_path: Path,
    summary_rows: str,
    message: str,
):
    output_path = tmp_path / "duplicate-summary.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="AA 101 C1 150 ? ATOM GLN\n",
            summary_rows=summary_rows,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DsspAdapterError, match=message):
        parse_dssp_annotated_mmcif(output_path)


def test_parse_dssp_annotation_rejects_invalid_secondary_structure_code(tmp_path: Path):
    output_path = tmp_path / "invalid-code.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="AA 101 C1 150 ? ATOM GLN\n",
            summary_rows="AA 101 X\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DsspAdapterError, match="Invalid DSSP secondary-structure code 'X'"):
        parse_dssp_annotated_mmcif(output_path)


def test_parse_dssp_annotation_rejects_missing_per_residue_summary(tmp_path: Path):
    output_path = tmp_path / "without-summary.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="A 1 AAA 25 ? ATOM ALA\n",
            summary_rows=None,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DsspAdapterError, match="lacks exact per-residue DSSP data"):
        parse_dssp_annotated_mmcif(output_path)


def test_parse_dssp_annotation_rejects_unknown_ladder_endpoint(tmp_path: Path):
    output_path = tmp_path / "unknown-edge.cif"
    output_path.write_text(
        _annotated_mmcif(
            atom_rows="A 1 A 1 ? ATOM ALA\nA 5 A 5 ? ATOM VAL\n",
            summary_rows="A 1 E\nA 5 E\n",
            range_rows="S1 R1 A 1 A 1\nS1 R2 A 5 A 5\n",
            ladder_rows="A 1 A 1 A 5 A 6\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DsspAdapterError, match="absent from the DSSP summary"):
        parse_dssp_annotated_mmcif(output_path)


def test_dssp_coverage_requires_every_expected_residue_assignment():
    expected = {("C1", (" ", 150, " ")), ("C1", (" ", 151, " "))}

    with pytest.raises(DsspAdapterError, match=r"assigned 1/2 required residues"):
        validate_dssp_coverage(
            expected,
            {("C1", (" ", 150, " "))},
            context="test chain 'C1'",
        )


def test_dssp_coverage_accepts_explicit_coil_assignment():
    residue_key = ("C1", (" ", 150, " "))
    validate_dssp_coverage({residue_key}, {residue_key}, context="test chain 'C1'")


def test_run_dssp_annotation_uses_default_summary_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_path = tmp_path / "input.cif"
    input_path.write_text("data_input\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        calls.append(command)
        assert command == [
            "/opt/mkdssp",
            "--output-format=mmcif",
            str(input_path),
            command[-1],
        ]
        assert check is False
        assert capture_output is True
        assert text is True
        Path(command[-1]).write_text(
            _annotated_mmcif(
                atom_rows="GE 10 MX 42 ? ATOM UNK\n",
                summary_rows="GE 10 B\n",
                range_rows="S1 R1 GE 10 GE 10\n",
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="warning")

    monkeypatch.setattr("cooper_beta.dssp_adapter.subprocess.run", fake_run)

    annotation = run_dssp_annotation(input_path, dssp_executable="/opt/mkdssp")

    assert annotation.residue_assignments == {("MX", (" ", 42, " ")): "B"}
    assert len(calls) == 1


def test_run_dssp_annotation_rejects_missing_summary_after_one_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_path = tmp_path / "input.cif"
    input_path.write_text("data_input\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        calls.append(command)
        assert command == [
            "/opt/mkdssp",
            "--output-format=mmcif",
            str(input_path),
            command[-1],
        ]
        output = _annotated_mmcif(
            atom_rows="AA 101 C1 150 ? ATOM GLN\n",
            summary_rows=None,
        )
        Path(command[-1]).write_text(output, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("cooper_beta.dssp_adapter.subprocess.run", fake_run)

    with pytest.raises(DsspAdapterError, match="lacks exact per-residue DSSP data"):
        run_dssp_annotation(input_path, dssp_executable="/opt/mkdssp")

    assert len(calls) == 1


def test_run_dssp_annotation_does_not_hide_writer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_path = tmp_path / "input.cif"
    input_path.write_text("data_input\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Duplicate Key violation, cat: struct_sheet_range",
        )

    monkeypatch.setattr("cooper_beta.dssp_adapter.subprocess.run", fake_run)

    with pytest.raises(DsspAdapterError, match="Duplicate Key violation"):
        run_dssp_annotation(input_path, dssp_executable="/opt/mkdssp")

    assert len(calls) == 1
