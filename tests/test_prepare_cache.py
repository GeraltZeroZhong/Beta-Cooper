from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import cooper_beta.preparation as preparation
from cooper_beta.config import build_config
from cooper_beta.loader import ChainPreparationResult
from cooper_beta.preparation import PrepareFailure, prepare_file_batch, prepare_one_file
from cooper_beta.prepare_cache import _executable_state
from cooper_beta.strand_graph import StrandAdjacencyGraph


class FakeChain:
    def __init__(self, chain_id: str):
        self.id = chain_id


class FakeLoader:
    calls = 0

    def __init__(self, file_path, input_config, dssp_bin=None):
        del input_config, dssp_bin
        type(self).calls += 1
        self.file_path = file_path
        self.model = [FakeChain("A")]
        self.secondary_structure_error = None

    def get_chain_data(self, chain_id):
        assert chain_id == "A"
        return [
            {
                "res_id": index + 1,
                "coord": [float(index), 0.0, 0.0],
                "dssp_assignment_available": True,
                "is_sheet": True,
                "strand_node_id": None,
                "polymer_index": index,
                "peptide_bond_distance_to_previous_angstrom": (None if index == 0 else 1.33),
                "chain": "A",
                "resseq": index + 1,
                "icode": "",
                "hetfield": "",
                "res_uid": {
                    "chain": "A",
                    "hetfield": "",
                    "resseq": index + 1,
                    "icode": "",
                },
            }
            for index in range(2)
        ]

    def get_strand_graph(self, chain_id):
        assert chain_id == "A"
        return StrandAdjacencyGraph(author_chain_id="A", nodes=(), edges=())

    def available_chains(self):
        return ["A"]

    def prepare_chain(self, chain_id):
        return ChainPreparationResult(
            author_chain_id=chain_id,
            residues=tuple(self.get_chain_data(chain_id)),
            strand_graph=self.get_strand_graph(chain_id),
        )


def test_unexpected_preparation_exception_uses_stable_error_code(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "unexpected.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")
    cfg = build_config({"runtime.prepare_cache_enabled": False})

    class BrokenLoader:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(preparation, "ProteinLoader", BrokenLoader)
    result = prepare_one_file(str(input_file), cfg)

    assert result == PrepareFailure(
        source_path=str(input_file.resolve()),
        error_code="UNEXPECTED_PREPARATION_FAILURE",
        message="unexpected implementation failure",
    )


def test_prepare_one_file_reuses_cached_payloads(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")

    cfg = build_config(
        {
            "runtime.prepare_cache_enabled": True,
            "runtime.prepare_cache_dir": str(tmp_path / "cache"),
        }
    )

    monkeypatch.setattr("cooper_beta.preparation.ProteinLoader", FakeLoader)
    FakeLoader.calls = 0

    first = prepare_one_file(str(input_file), cfg)
    second = prepare_one_file(str(input_file), cfg)

    assert isinstance(first, list)
    assert first == second
    assert FakeLoader.calls == 1


def test_prepare_cache_invalidates_when_file_changes(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")

    cfg = build_config(
        {
            "runtime.prepare_cache_enabled": True,
            "runtime.prepare_cache_dir": str(tmp_path / "cache"),
        }
    )

    monkeypatch.setattr("cooper_beta.preparation.ProteinLoader", FakeLoader)
    FakeLoader.calls = 0

    first = prepare_one_file(str(input_file), cfg)
    input_file.write_text("HEADER UPDATED\n")
    second = prepare_one_file(str(input_file), cfg)

    assert isinstance(first, list)
    assert isinstance(second, list)
    assert FakeLoader.calls == 2


def test_prepare_cache_invalidates_when_same_size_file_content_changes(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")

    cfg = build_config(
        {
            "runtime.prepare_cache_enabled": True,
            "runtime.prepare_cache_dir": str(tmp_path / "cache"),
        }
    )

    monkeypatch.setattr("cooper_beta.preparation.ProteinLoader", FakeLoader)
    FakeLoader.calls = 0

    first = prepare_one_file(str(input_file), cfg)
    stat = input_file.stat()
    input_file.write_text("HEADEQ\n")
    os.utime(input_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = prepare_one_file(str(input_file), cfg)

    assert isinstance(first, list)
    assert isinstance(second, list)
    assert FakeLoader.calls == 2


def test_input_changed_before_snapshot_fails_without_caching_payload_under_new_identity(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("OLD\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cfg = build_config(
        {
            "runtime.prepare_cache_enabled": True,
            "runtime.prepare_cache_dir": str(cache_dir),
        }
    )

    original_snapshot = preparation.verified_input_snapshot

    @contextmanager
    def mutate_before_snapshot(identity):
        input_file.write_text("NEW\n", encoding="utf-8")
        with original_snapshot(identity) as snapshot:
            yield snapshot

    monkeypatch.setattr(preparation, "verified_input_snapshot", mutate_before_snapshot)

    result = prepare_one_file(str(input_file), cfg)

    assert result == PrepareFailure(
        source_path=str(input_file.resolve()),
        error_code="INPUT_CONTENT_CHANGED",
        message=(
            "Input content changed before its private snapshot was captured: "
            f"{input_file.resolve()}"
        ),
    )
    assert list(cache_dir.rglob("*.json")) == []


def test_swap_restore_during_parse_cannot_change_snapshot_bytes(
    tmp_path: Path,
    monkeypatch,
):
    input_file = tmp_path / "toy.pdb"
    input_file.write_bytes(b"A")
    cfg = build_config({"runtime.prepare_cache_enabled": False})

    class SwapRestoreLoader(FakeLoader):
        snapshot_path: Path | None = None
        parsed_bytes = b""

        def __init__(self, file_path, input_config, dssp_bin=None):
            input_file.write_bytes(b"B")
            type(self).snapshot_path = Path(file_path)
            type(self).parsed_bytes = Path(file_path).read_bytes()
            input_file.write_bytes(b"A")
            super().__init__(file_path, input_config, dssp_bin=dssp_bin)

    monkeypatch.setattr(preparation, "ProteinLoader", SwapRestoreLoader)

    result = prepare_one_file(str(input_file), cfg)

    assert isinstance(result, list)
    assert SwapRestoreLoader.parsed_bytes == b"A"
    assert result[0]["filename"] == input_file.name
    assert result[0]["source_path"] == str(input_file.resolve())
    assert input_file.read_bytes() == b"A"
    assert SwapRestoreLoader.snapshot_path is not None
    assert not SwapRestoreLoader.snapshot_path.exists()


def test_executable_cache_state_includes_content_hash(tmp_path: Path):
    executable = tmp_path / "mkdssp"
    executable.write_text("version-a\n", encoding="utf-8")
    executable.chmod(0o755)
    first = _executable_state(str(executable))
    assert first is not None

    stat = executable.stat()
    executable.write_text("version-b\n", encoding="utf-8")
    os.utime(executable, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = _executable_state(str(executable))

    assert second is not None
    assert first["size"] == second["size"]
    assert first["mtime_ns"] == second["mtime_ns"]
    assert first["sha256"] != second["sha256"]


def test_prepare_file_batch_aggregates_payloads_and_failures(monkeypatch):
    cfg = build_config()

    def fake_prepare_one_file(file_path: str, cfg):
        del cfg
        if file_path == "bad.pdb":
            return PrepareFailure(
                source_path="bad.pdb",
                error_code="STRUCTURE_PARSE_FAILED",
                message="parse failed",
            )
        return [{"filename": file_path, "chain": "A", "residues_data": []}]

    monkeypatch.setattr("cooper_beta.preparation.prepare_one_file", fake_prepare_one_file)

    result = prepare_file_batch(["good-1.pdb", "bad.pdb", "good-2.pdb"], cfg)

    assert result.processed_files == 3
    assert result.failures == [
        PrepareFailure(
            source_path="bad.pdb",
            error_code="STRUCTURE_PARSE_FAILED",
            message="parse failed",
        )
    ]
    assert result.payloads == [
        {"filename": "good-1.pdb", "chain": "A", "residues_data": []},
        {"filename": "good-2.pdb", "chain": "A", "residues_data": []},
    ]
