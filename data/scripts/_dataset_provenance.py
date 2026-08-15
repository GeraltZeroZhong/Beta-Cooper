"""Shared provenance and atomic-output primitives for dataset builders."""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
STRUCTURE_SUFFIXES = frozenset({".cif", ".mmcif", ".pdb"})
COORDINATE_NAME_ENDINGS = (".cif", ".mmcif", ".pdb", ".cif.gz", ".mmcif.gz", ".pdb.gz")


def source_package_version() -> tuple[str, Path]:
    """Read the distribution version from its single source without importing the package."""

    version_path = Path(__file__).resolve().parents[2] / "src" / "cooper_beta" / "_version.py"
    try:
        module = ast.parse(version_path.read_text(encoding="utf-8"), filename=str(version_path))
    except (OSError, SyntaxError) as error:
        raise RuntimeError(
            f"Could not read Cooper-Beta source version from {version_path}"
        ) from error
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in targets
        ):
            continue
        if statement.value is None:
            raise RuntimeError(f"Invalid __version__ declaration in {version_path}")
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"Invalid __version__ declaration in {version_path}") from error
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Invalid __version__ declaration in {version_path}")
        return value.strip(), version_path
    raise RuntimeError(f"Missing __version__ declaration in {version_path}")


def utc_now() -> str:
    """Return a timezone-explicit UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_path(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def atomic_write_json(value: object, path: os.PathLike[str] | str) -> None:
    destination = Path(path)
    file_descriptor, temporary_name = _atomic_path(destination)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_csv(
    rows: Sequence[Mapping[str, object]],
    path: os.PathLike[str] | str,
    fieldnames: Sequence[str] | None = None,
) -> None:
    destination = Path(path)
    header = list(fieldnames or _union_keys(rows))
    file_descriptor, temporary_name = _atomic_path(destination)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
            if header:
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _union_keys(rows: Sequence[Mapping[str, object]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row)
    return sorted(keys)


def prepare_new_output_directory(path: os.PathLike[str] | str) -> Path:
    """Create a run directory, refusing to mix with any existing output."""
    output = Path(path).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite or mix with non-empty output directory: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def require_exact_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer (boolean values are invalid)")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def require_finite_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number (boolean values are invalid)")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if minimum is not None:
        too_small = result < minimum if minimum_inclusive else result <= minimum
        if too_small:
            comparator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {comparator} {minimum}; got {result}")
    if maximum is not None:
        too_large = result > maximum if maximum_inclusive else result >= maximum
        if too_large:
            comparator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {comparator} {maximum}; got {result}")
    return result


def require_pinned_source_id(name: str, value: object) -> str:
    """Validate a user-supplied upstream release URL/DOI/version identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must identify the frozen upstream release")
    normalized = value.strip()
    if re.search(r"\b(?:latest|current|unknown|unspecified)\b", normalized, re.IGNORECASE):
        raise ValueError(f"{name} must be pinned, not {normalized!r}")
    return normalized


def input_file_record(
    path: os.PathLike[str] | str,
    *,
    source: str,
    pinned: bool,
    release: str | None = None,
) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"Input must be a non-empty regular file: {resolved}")
    return {
        "path": str(resolved),
        "source": source,
        "release": release or infer_release(source, resolved.name),
        "pinned": pinned,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def input_directory_record(
    path: os.PathLike[str] | str,
    *,
    source: str,
    pinned: bool,
    suffixes: Sequence[str] = COORDINATE_NAME_ENDINGS,
) -> dict[str, object]:
    """Hash every relevant file in a frozen local input archive."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Input must be a directory: {resolved}")
    normalized_suffixes = tuple(suffix.lower() for suffix in suffixes)
    files: list[dict[str, object]] = []
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(
                f"Frozen input archive must be self-contained; symlink found: {candidate}"
            )
        if not candidate.is_file() or not candidate.name.lower().endswith(normalized_suffixes):
            continue
        files.append(
            {
                "path": candidate.relative_to(resolved).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not files:
        raise ValueError(f"Input directory contains no coordinate files: {resolved}")
    inventory_json = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(resolved),
        "source": source,
        "pinned": pinned,
        "file_count": len(files),
        "inventory_sha256": hashlib.sha256(inventory_json).hexdigest(),
        "files": files,
    }


def infer_release(reference: str, filename: str) -> str:
    text = f"{reference}/{filename}".lower()
    if "latest" in text:
        return "latest (unpinned)"
    for pattern in (
        r"d(\d{4}_\d{2}_\d{2})",
        r"v(\d+(?:[_\.-]\d+){1,3})",
        r"release[_/-]?([0-9][0-9_.-]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace("_", ".")
    return filename


def artifact_inventory(
    root: os.PathLike[str] | str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base = Path(root).resolve()
    outputs: list[dict[str, object]] = []
    structures: list[dict[str, object]] = []
    for path in sorted(base.rglob("*")):
        if path.name == MANIFEST_FILENAME or not path.is_file():
            continue
        record = {
            "path": path.relative_to(base).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix.lower() in STRUCTURE_SUFFIXES:
            structures.append(record)
        elif path.suffix.lower() in {".csv", ".json", ".txt", ".tsv", ".xml"}:
            outputs.append(record)
    return outputs, structures


class RunManifest:
    """Atomically maintain the lifecycle record for one immutable run directory."""

    def __init__(
        self,
        output_directory: Path,
        script_path: os.PathLike[str] | str,
        arguments: Mapping[str, Any],
        *,
        mode: str,
        random_algorithm: str | None,
        seed: int | None,
    ) -> None:
        script = Path(script_path).resolve()
        package_version, version_source_path = source_package_version()
        try:
            installed_distribution_version: str | None = importlib.metadata.version("cooper-beta")
        except importlib.metadata.PackageNotFoundError:
            installed_distribution_version = None
        dependency_versions: dict[str, str] = {}
        for distribution in ("biopython",):
            try:
                dependency_versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                dependency_versions[distribution] = "not-installed"
        support_module = Path(__file__).resolve()
        self.output_directory = output_directory
        self.path = output_directory / MANIFEST_FILENAME
        self.data: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": utc_now(),
            "completed_at_utc": None,
            "mode": mode,
            "arguments": dict(arguments),
            "randomness": {"algorithm": random_algorithm, "seed": seed},
            "software": {
                "package": "cooper-beta",
                "package_version": package_version,
                "package_version_source_path": str(version_source_path),
                "package_version_source_sha256": sha256_file(version_source_path),
                "installed_distribution_version": installed_distribution_version,
                "dependency_versions": dependency_versions,
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "script_path": str(script),
                "script_sha256": sha256_file(script),
                "support_module_path": str(support_module),
                "support_module_sha256": sha256_file(support_module),
            },
            "inputs": {},
            "remote_sources": [],
            "summary": {},
            "outputs": [],
            "structures": [],
            "failure": None,
        }
        self._write()

    def set_provenance(
        self,
        *,
        inputs: Mapping[str, object],
        remote_sources: Sequence[Mapping[str, object]],
    ) -> None:
        self.data["inputs"] = dict(inputs)
        self.data["remote_sources"] = [dict(item) for item in remote_sources]
        self._write()

    def complete(self, summary: Mapping[str, object] | None = None) -> None:
        outputs, structures = artifact_inventory(self.output_directory)
        self.data.update(
            {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "summary": dict(summary or {}),
                "outputs": outputs,
                "structures": structures,
                "failure": None,
            }
        )
        self._write()

    def fail(self, error: BaseException) -> None:
        outputs, structures = artifact_inventory(self.output_directory)
        self.data.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "outputs": outputs,
                "structures": structures,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(traceback.format_exception(error)),
                },
            }
        )
        self._write()

    def _write(self) -> None:
        atomic_write_json(self.data, self.path)


class FrozenStructureArchive:
    """Resolve and atomically stage uniquely named coordinates from a local archive."""

    def __init__(self, root: os.PathLike[str] | str) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Structure source is not a directory: {self.root}")
        self._files: dict[str, Path] = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Frozen structure archive must not contain symlinks: {path}")
            if not path.is_file() or not path.name.lower().endswith(COORDINATE_NAME_ENDINGS):
                continue
            key = path.name.casefold()
            if key in self._files:
                raise ValueError(
                    "Frozen structure archive has ambiguous duplicate basenames: "
                    f"{self._files[key]} and {path}"
                )
            self._files[key] = path

    def stage(self, candidate_names: Sequence[str], destination: os.PathLike[str] | str) -> Path:
        matches = {
            self._files[name.casefold()]
            for name in candidate_names
            if name.casefold() in self._files
        }
        if not matches:
            raise FileNotFoundError(
                "Frozen structure archive has no file matching " + ", ".join(candidate_names)
            )
        if len(matches) != 1:
            raise ValueError(
                "Frozen structure archive matches multiple aliases for one structure: "
                + ", ".join(str(path) for path in sorted(matches))
            )
        source = matches.pop()
        content = source.read_bytes()
        if source.name.lower().endswith(".gz"):
            content = gzip.decompress(content)
        head = content[:2048].lstrip()
        if not content or not (head.startswith(b"data_") or b"_entry.id" in head):
            raise ValueError(f"Frozen coordinate is not recognizable mmCIF: {source}")

        output = Path(destination)
        file_descriptor, temporary_name = _atomic_path(output)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, output)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return source
