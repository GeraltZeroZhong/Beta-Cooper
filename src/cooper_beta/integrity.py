from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from .exceptions import InputContentChangedError, InputValidationError

_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class FrozenInputIdentity:
    """Stable content identity captured before an input participates in a run."""

    path: str
    size: int
    sha256: str
    suffix: str
    mtime_ns: int

    def content_identity(self) -> dict[str, int | str]:
        """Return the path-independent fields that define the input bytes and format."""

        return {
            "size": self.size,
            "sha256": self.sha256,
            "suffix": self.suffix,
        }

    def manifest_state(self, *, include_hash: bool) -> dict[str, object]:
        """Return the frozen state, optionally redacting the content digest."""

        return {
            "path": self.path,
            "exists": True,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256 if include_hash else None,
        }


def freeze_input_identity(path: str | Path) -> FrozenInputIdentity:
    """Capture one regular file through a stable descriptor-backed SHA-256 read."""

    requested_path = Path(path).expanduser()
    try:
        resolved = requested_path.resolve(strict=True)
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not resolved.is_file():
                raise InputValidationError(f"Input is not a regular file: {resolved}")
            digest = hashlib.sha256()
            bytes_read = 0
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
                bytes_read += len(chunk)
            after = os.fstat(handle.fileno())
    except InputValidationError:
        raise
    except OSError as exc:
        raise InputValidationError(
            f"Could not freeze input file identity: {requested_path}"
        ) from exc

    stability_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stability_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stability_before != stability_after or bytes_read != int(after.st_size):
        raise InputContentChangedError(
            f"Input changed while its content identity was being captured: {resolved}"
        )
    return FrozenInputIdentity(
        path=str(resolved),
        size=int(after.st_size),
        sha256=digest.hexdigest(),
        # Compression is part of the parser-facing format identity.  Keeping
        # only ``.gz`` would let identical bytes named ``.pdb.gz`` and
        # ``.cif.gz`` share a preparation-cache key even though they select
        # different parsers after decompression.
        suffix=_snapshot_suffix(str(resolved)).lower(),
        mtime_ns=int(after.st_mtime_ns),
    )


def verify_input_identity(expected: FrozenInputIdentity) -> None:
    """Fail if the current resolved file bytes differ from a frozen identity."""

    try:
        observed = freeze_input_identity(expected.path)
    except InputContentChangedError:
        raise
    except InputValidationError as exc:
        raise InputContentChangedError(
            f"Frozen input became unavailable during the run: {expected.path}"
        ) from exc
    if (
        observed.path != expected.path
        or observed.size != expected.size
        or observed.sha256 != expected.sha256
        or observed.suffix != expected.suffix
    ):
        raise InputContentChangedError(f"Input content changed during the run: {expected.path}")


def freeze_input_identities(paths: list[str]) -> list[FrozenInputIdentity]:
    """Freeze an ordered input list before any file is parsed."""

    identities = [freeze_input_identity(path) for path in paths]
    resolved_paths = [identity.path for identity in identities]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise InputValidationError("Input discovery resolved the same file more than once.")
    return identities


def verify_input_identities(identities: list[FrozenInputIdentity]) -> None:
    """Revalidate every frozen input before publishing run artifacts."""

    for identity in identities:
        verify_input_identity(identity)


def _snapshot_suffix(path: str) -> str:
    """Preserve the coordinate suffix, including a trailing compression suffix."""

    suffixes = Path(path).suffixes
    if not suffixes:
        return ""
    if suffixes[-1].lower() == ".gz" and len(suffixes) >= 2:
        return "".join(suffixes[-2:])
    return suffixes[-1]


@contextmanager
def verified_input_snapshot(expected: FrozenInputIdentity) -> Iterator[Path]:
    """Yield a private immutable copy whose bytes match ``expected`` exactly.

    Freezing an identity and later reopening its path leaves a time-of-check/time-of-use
    window: another process can replace the path for the duration of parsing and restore
    it before the final verification.  This function closes that window by hashing while
    copying through one descriptor, validating the copy against the frozen identity, and
    requiring all parsers to consume only that private snapshot.
    """

    with tempfile.TemporaryDirectory(prefix="cooper-beta-input-") as directory_name:
        snapshot = Path(directory_name) / f"input{_snapshot_suffix(expected.path)}"
        try:
            with Path(expected.path).open("rb") as source, snapshot.open("xb") as target:
                before = os.fstat(source.fileno())
                digest = hashlib.sha256()
                bytes_read = 0
                for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
                    target.write(chunk)
                    bytes_read += len(chunk)
                target.flush()
                os.fsync(target.fileno())
                after = os.fstat(source.fileno())
        except OSError as exc:
            raise InputContentChangedError(
                f"Frozen input became unavailable while creating its private snapshot: "
                f"{expected.path}"
            ) from exc

        stability_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stability_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            stability_before != stability_after
            or bytes_read != int(after.st_size)
            or bytes_read != expected.size
            or digest.hexdigest() != expected.sha256
        ):
            raise InputContentChangedError(
                f"Input content changed before its private snapshot was captured: {expected.path}"
            )

        snapshot.chmod(0o400)
        yield snapshot


def to_jsonable(value: Any) -> Any:
    """Convert supported application values into a deterministic JSON structure."""
    if is_dataclass(value):
        # ``is_dataclass`` also accepts dataclass classes, whereas application
        # provenance serializes instances only. Preserve that contract here.
        if isinstance(value, type):
            raise TypeError("Dataclass classes cannot be serialized; pass an instance.")
        return to_jsonable(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            string_key = str(key)
            if string_key in converted:
                raise TypeError(
                    f"Mapping contains duplicate JSON key after conversion: {string_key!r}"
                )
            converted[string_key] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value to canonical UTF-8 JSON, rejecting NaN and infinity."""
    return json.dumps(
        to_jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(path: str | Path) -> str | None:
    try:
        return file_sha256(path)
    except OSError:
        return None


def atomic_write_text(
    path: str | Path,
    writer: Callable[[TextIO], None],
    *,
    newline: str | None = None,
) -> Path:
    """Atomically replace a UTF-8 text artifact produced by ``writer``."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline=newline) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    return output_path


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    """Atomically replace ``path`` with strict JSON from ``value``."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                to_jsonable(value),
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=indent,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    return output_path
