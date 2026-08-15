from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from .exceptions import (
    OutputArtifactBusyError,
    OutputArtifactError,
    OutputArtifactExistsError,
)
from .integrity import atomic_write_json, freeze_input_identity

OUTPUT_MANIFEST_SCHEMA_VERSION = 1
OUTPUT_ARTIFACT_POLICIES = ("error", "replace")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_output_path(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if not requested.name:
        raise OutputArtifactError("Output CSV path must name a file.")
    parent = requested.parent.resolve()
    return parent / requested.name


def resolved_output_artifact_paths(path: str | Path) -> tuple[Path, Path, Path]:
    """Return the canonical CSV, sidecar, and coordination-lock paths without writing."""

    output_path = _canonical_output_path(path)
    return (
        output_path,
        Path(f"{output_path}.manifest.json"),
        output_path.parent / f".{output_path.name}.lock",
    )


def _fsync_parent_directory(path: Path) -> None:
    """Make a completed rename/unlink durable on POSIX filesystems."""
    if os.name == "nt":  # pragma: no cover - Windows has no directory fsync API
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _output_state(path: Path) -> dict[str, object] | None:
    if path.is_symlink():
        raise OutputArtifactError(f"Committed output cannot be a symlink: {path}")
    if not path.is_file():
        return None
    identity = freeze_input_identity(path)
    if identity.path != str(path):
        raise OutputArtifactError(
            f"Committed output resolved to an unexpected filesystem target: {path}"
        )
    return {
        "path": identity.path,
        "size": identity.size,
        "mtime_ns": identity.mtime_ns,
        "sha256": identity.sha256,
    }


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI/release installs
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            lock_nonblocking = msvcrt.LK_NBLCK  # type: ignore[attr-defined]
            msvcrt.locking(  # type: ignore[attr-defined]
                handle.fileno(),
                lock_nonblocking,
                1,
            )
        except OSError as exc:
            raise OutputArtifactBusyError(
                "Another process is publishing this output target."
            ) from exc
        return

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise OutputArtifactBusyError("Another process is publishing this output target.") from exc


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI/release installs
        import msvcrt

        handle.seek(0)
        unlock = msvcrt.LK_UNLCK  # type: ignore[attr-defined]
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            unlock,
            1,
        )
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class OutputArtifactTransaction:
    """Serialize and authenticate publication of one result CSV/manifest pair."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        write_manifest: bool,
        existing_artifact_policy: str,
        started_at_utc: str,
        run_id: str | None = None,
    ) -> None:
        if existing_artifact_policy not in OUTPUT_ARTIFACT_POLICIES:
            raise ValueError(
                f"`existing_artifact_policy` must be one of {OUTPUT_ARTIFACT_POLICIES!r}."
            )
        if not started_at_utc.strip():
            raise ValueError("`started_at_utc` cannot be blank.")
        resolved_run_id = uuid.uuid4().hex if run_id is None else run_id.strip()
        if not resolved_run_id:
            raise ValueError("`run_id` cannot be blank.")

        self.output_path, self.manifest_path, self.lock_path = resolved_output_artifact_paths(
            output_path
        )
        self.write_manifest = bool(write_manifest)
        self.existing_artifact_policy = existing_artifact_policy
        self.started_at_utc = started_at_utc
        self.run_id = resolved_run_id
        self._lock_handle: BinaryIO | None = None
        self._committed_output_state: dict[str, object] | None = None
        self._complete = False

    def _state_document(
        self,
        status: str,
        *,
        failure: BaseException | None = None,
    ) -> dict[str, object]:
        committed = self._committed_output_state
        generated_at_utc = _utc_now()
        document: dict[str, object] = {
            "schema_version": OUTPUT_MANIFEST_SCHEMA_VERSION,
            "manifest_kind": "cooper_beta_output_transaction",
            "status": status,
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": generated_at_utc if status == "failed" else None,
            "generated_at_utc": generated_at_utc,
            "output_path": str(self.output_path),
            "output_file_state": committed,
            "output_sha256": committed["sha256"] if committed is not None else None,
            "artifact_binding": {
                "run_id": self.run_id,
                "csv_path": str(self.output_path),
                "csv_size": committed["size"] if committed is not None else None,
                "csv_sha256": committed["sha256"] if committed is not None else None,
                "committed_by_run": committed is not None,
            },
            "transaction": {
                "existing_artifact_policy": self.existing_artifact_policy,
                "lock_path": str(self.lock_path),
                "pid": os.getpid(),
            },
        }
        if failure is not None:
            document["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
        return document

    def __enter__(self) -> OutputArtifactTransaction:
        if self._lock_handle is not None:
            raise RuntimeError("OutputArtifactTransaction is already active.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.output_path, self.manifest_path, self.lock_path):
            if path.is_symlink():
                raise OutputArtifactError(f"Output transaction paths cannot be symlinks: {path}")
        if self.output_path.is_dir():
            raise OutputArtifactError(f"Output CSV path points to a directory: {self.output_path}")
        if self.manifest_path.is_dir():
            raise OutputArtifactError(
                f"Output manifest path points to a directory: {self.manifest_path}"
            )

        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.lock_path, flags, 0o600)
        handle = os.fdopen(descriptor, "r+b")
        try:
            _lock_file(handle)
            existing = [path for path in (self.output_path, self.manifest_path) if path.exists()]
            if existing and self.existing_artifact_policy == "error":
                joined = ", ".join(str(path) for path in existing)
                raise OutputArtifactExistsError(
                    "Refusing to overwrite existing result artifact(s): "
                    f"{joined}. Move them or explicitly set "
                    "`output.existing_artifact_policy=replace`."
                )
            self._lock_handle = handle
            if self.write_manifest:
                atomic_write_json(self.manifest_path, self._state_document("running"), indent=2)
                _fsync_parent_directory(self.manifest_path)
            else:
                self.manifest_path.unlink(missing_ok=True)
                _fsync_parent_directory(self.manifest_path)
        except BaseException:
            try:
                _unlock_file(handle)
            except OSError:
                pass
            handle.close()
            raise
        return self

    def record_csv_commit(self) -> dict[str, object]:
        if self._lock_handle is None:
            raise RuntimeError("OutputArtifactTransaction is not active.")
        _fsync_parent_directory(self.output_path)
        state = _output_state(self.output_path)
        if state is None:
            raise OutputArtifactError(
                f"Result CSV was not committed as a regular file: {self.output_path}"
            )
        self._committed_output_state = state
        return dict(state)

    def mark_complete(self) -> None:
        if self._lock_handle is None:
            raise RuntimeError("OutputArtifactTransaction is not active.")
        expected = self._committed_output_state
        if expected is None:
            raise OutputArtifactError("Cannot complete an output transaction before CSV commit.")
        _fsync_parent_directory(self.manifest_path)
        observed = _output_state(self.output_path)
        if observed != expected:
            raise OutputArtifactError("Result CSV changed while its manifest was being published.")

        if self.write_manifest:
            try:
                document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise OutputArtifactError("Final output manifest is missing or invalid.") from exc
            binding = document.get("artifact_binding")
            if not isinstance(binding, dict):
                raise OutputArtifactError("Final output manifest lacks an artifact binding.")
            expected_binding = {
                "run_id": self.run_id,
                "csv_path": str(self.output_path),
                "csv_size": expected["size"],
                "csv_sha256": expected["sha256"],
                "committed_by_run": True,
            }
            if (
                document.get("status") != "complete"
                or document.get("run_id") != self.run_id
                or document.get("output_path") != str(self.output_path)
                or document.get("output_sha256") != expected["sha256"]
                or binding != expected_binding
            ):
                raise OutputArtifactError(
                    "Final manifest does not authenticate this run's committed CSV."
                )
        elif self.manifest_path.exists():
            raise OutputArtifactError("Manifest output is disabled but a sidecar still exists.")
        self._complete = True

    def _write_failed_state(self, failure: BaseException) -> None:
        if not self.write_manifest:
            return
        try:
            atomic_write_json(
                self.manifest_path,
                self._state_document("failed", failure=failure),
                indent=2,
            )
            _fsync_parent_directory(self.manifest_path)
        except OSError:
            # Preserve the already published `running` record when the filesystem
            # itself prevents the stronger `failed` transition.
            pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        handle = self._lock_handle
        if handle is None:
            return
        try:
            if exc is not None:
                self._write_failed_state(exc)
            elif not self._complete:
                failure = OutputArtifactError(
                    "Output transaction ended without a complete CSV/manifest binding."
                )
                self._write_failed_state(failure)
                raise failure
        finally:
            self._lock_handle = None
            try:
                _unlock_file(handle)
            finally:
                handle.close()
