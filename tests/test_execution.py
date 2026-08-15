from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pytest

import cooper_beta.execution as execution
from cooper_beta.config import build_config
from cooper_beta.preparation import PreparationBatchResult, PrepareFailure


class _Progress:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.updates: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def update(self, value: int = 1) -> None:
        self.updates.append(value)


class _ImmediateExecutor:
    instances: list[_ImmediateExecutor] = []

    def __init__(self, *, max_workers, initializer, initargs) -> None:
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs
        self.shutdown_call: tuple[bool, bool] | None = None
        self.instances.append(self)

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:  # pragma: no cover - Future contract helper
            future.set_exception(exc)
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_call = (wait, cancel_futures)


def test_initialize_worker_configures_threads_and_worker_logging(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        execution,
        "configure_thread_environment",
        lambda value: calls.append(("threads", value)),
    )
    monkeypatch.setattr(
        execution,
        "configure_logging",
        lambda **kwargs: calls.append(("logging", kwargs)),
    )

    execution._initialize_worker(2, "INFO", True)

    assert calls == [
        ("threads", 2),
        ("logging", {"level": "INFO", "console": True, "jsonl_path": None}),
    ]


def test_batched_preserves_order_and_remainder() -> None:
    assert list(execution._batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_serial_preparation_yields_payloads_and_reports_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = build_config({"runtime.prepare_batch_size": 2})
    progress = _Progress()
    failures_seen: list[PrepareFailure] = []

    def fake_tqdm(*args, **kwargs):
        del args, kwargs
        return progress

    def fake_prepare(batch, config_arg):
        assert config_arg is config
        failure = (
            [PrepareFailure(str(tmp_path / batch[0]), "PARSE_FAILED", "bad input")]
            if batch[0] == "a.pdb"
            else []
        )
        return PreparationBatchResult(
            payloads=[{"filename": name, "chain": "A"} for name in batch],
            failures=failure,
            processed_files=len(batch),
        )

    monkeypatch.setattr(execution, "tqdm", fake_tqdm)
    monkeypatch.setattr(execution, "prepare_file_batch", fake_prepare)

    batches = list(
        execution.iter_prepared_payload_batches(
            ["a.pdb", "b.pdb", "c.pdb"],
            config,
            prepare_workers=1,
            on_failures=failures_seen.extend,
            show_progress=False,
        )
    )

    assert [[row["filename"] for row in batch] for batch in batches] == [
        ["a.pdb", "b.pdb"],
        ["c.pdb"],
    ]
    assert [failure.error_code for failure in failures_seen] == ["PARSE_FAILED"]
    assert progress.updates == [2, 1]


def test_parallel_preparation_is_bounded_and_shuts_down(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    config = build_config(
        {
            "runtime.prepare_batch_size": 1,
            "runtime.prepare_in_flight_multiplier": 1,
        }
    )
    monkeypatch.setattr(execution, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(execution, "tqdm", _Progress)
    monkeypatch.setattr(
        execution,
        "prepare_file_batch",
        lambda batch, config_arg: PreparationBatchResult(
            payloads=[{"filename": batch[0], "chain": "A"}],
            failures=[],
            processed_files=1,
        ),
    )

    batches = list(
        execution.iter_prepared_payload_batches(
            ["a.pdb", "b.pdb", "c.pdb"],
            config,
            prepare_workers=2,
            show_progress=False,
        )
    )

    assert sorted(batch[0]["filename"] for batch in batches) == ["a.pdb", "b.pdb", "c.pdb"]
    executor = _ImmediateExecutor.instances[-1]
    assert executor.max_workers == 2
    assert executor.shutdown_call == (True, True)


def test_analyze_payload_batch_preserves_success_and_contains_exception(monkeypatch) -> None:
    config = build_config()

    def fake_analyze(payload, config_arg):
        assert config_arg is config
        if payload["chain"] == "B":
            raise RuntimeError("numerical failure")
        return {
            "filename": payload["filename"],
            "author_chain_id": "A",
            "result": "NON_BARREL",
        }

    monkeypatch.setattr(execution, "analyze_chain_payload", fake_analyze)
    monkeypatch.setattr(
        execution,
        "unhandled_analysis_failure_row",
        lambda payload, config_arg, exc: {
            "filename": payload["filename"],
            "author_chain_id": payload["chain"],
            "result": "ERROR",
            "reason": str(exc),
        },
    )

    rows = execution.analyze_payload_batch(
        [
            {"filename": "one.pdb", "chain": "A"},
            {"filename": "two.pdb", "chain": "B"},
        ],
        config,
    )

    assert [row["result"] for row in rows] == ["NON_BARREL", "ERROR"]
    assert rows[1]["reason"] == "numerical failure"


def test_analysis_boundary_rejects_malformed_payload_instead_of_coercing_it() -> None:
    config = build_config()

    with pytest.raises(ValueError, match="is_sheet.*boolean"):
        execution.analyze_payload_batch(
            [
                {
                    "filename": "invalid.pdb",
                    "source_path": "/input/invalid.pdb",
                    "chain": "A",
                    "residues_data": [
                        {
                            "res_id": 1,
                            "coord": [0.0, 0.0, 0.0],
                            "dssp_assignment_available": True,
                            "is_sheet": "false",
                            "strand_node_id": None,
                            "polymer_index": 0,
                            "peptide_bond_distance_to_previous_angstrom": None,
                            "chain": "A",
                            "resseq": 1,
                            "icode": "",
                            "hetfield": "",
                            "res_uid": {
                                "chain": "A",
                                "hetfield": "",
                                "resseq": 1,
                                "icode": "",
                            },
                        }
                    ],
                    "strand_graph": {
                        "author_chain_id": "A",
                        "nodes": [],
                        "edges": [],
                    },
                }
            ],
            config,
        )


def test_parallel_analysis_stream_reports_error_rows_and_invokes_sink(monkeypatch) -> None:
    _ImmediateExecutor.instances.clear()
    config = build_config(
        {
            "runtime.analysis_batch_size": 1,
            "runtime.analysis_in_flight_multiplier": 1,
        }
    )
    monkeypatch.setattr(execution, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(execution, "tqdm", _Progress)

    def fake_analyze(batch, config_arg):
        assert config_arg is config
        payload = batch[0]
        return [
            {
                "filename": payload["filename"],
                "source_path": f"/input/{payload['filename']}",
                "author_chain_id": payload["chain"],
                "result": "ERROR" if payload["chain"] == "B" else "NON_BARREL",
                "result_stage": "analysis",
                "error_code": "FIT_FAILED" if payload["chain"] == "B" else "",
                "reason": "fixture",
            }
        ]

    monkeypatch.setattr(execution, "analyze_payload_batch", fake_analyze)
    observed: list[dict[str, object]] = []

    rows = execution.run_analysis_stream(
        [
            [
                {"filename": "one.pdb", "chain": "A"},
                {"filename": "two.pdb", "chain": "B"},
            ]
        ],
        config,
        workers=2,
        on_results=observed.extend,
        show_progress=False,
    )

    assert sorted(row["filename"] for row in rows) == ["one.pdb", "two.pdb"]
    assert sorted(row["filename"] for row in observed) == ["one.pdb", "two.pdb"]
    assert _ImmediateExecutor.instances[-1].shutdown_call == (True, True)
