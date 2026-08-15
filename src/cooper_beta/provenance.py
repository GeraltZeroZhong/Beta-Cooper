from __future__ import annotations

import locale
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from ._version import __version__ as source_package_version
from .bootstrap import runtime_bootstrap_state
from .constants import (
    DSSP_RESIDUE_COVERAGE_POLICY,
    NATIVE_THREAD_ENV_NAMES,
    POLYMER_POSITION_POLICY,
)
from .integrity import (
    FrozenInputIdentity,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    freeze_input_identities,
    optional_file_sha256,
    sha256_bytes,
    to_jsonable,
    verify_input_identities,
)
from .runtime import dssp_version, find_dssp_binary

_PROJECT_NAME = "cooper-beta"
RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_KIND = "cooper_beta_run"
SCIENTIFIC_PRODUCER_PACKAGES = (
    "cooper-beta",
    "biopython",
    "numpy",
    "hydra-core",
    "omegaconf",
    "threadpoolctl",
)
_REQUIRED_SCIENTIFIC_PACKAGE_VERSIONS = frozenset(SCIENTIFIC_PRODUCER_PACKAGES) - {"cooper-beta"}
_NATIVE_POOL_EXECUTION_FIELDS = frozenset({"filepath", "num_threads"})
_PROJECT_TABLE_PATTERN = re.compile(
    r"^\s*\[project\]\s*$\n(?P<body>.*?)(?=^\s*\[[^]]+\]\s*$|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def config_hash(config: object) -> str:
    return canonical_json_sha256(config)


def _git_output(args: list[str], repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _parse_project_table(pyproject_path: Path) -> dict[str, str] | None:
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None
    table_match = _PROJECT_TABLE_PATTERN.search(text)
    if table_match is None:
        return None
    fields: dict[str, str] = {}
    for field_name in ("name", "version"):
        field_match = re.search(
            rf"^\s*{field_name}\s*=\s*(['\"])(?P<value>.*?)\1\s*(?:#.*)?$",
            table_match.group("body"),
            flags=re.MULTILINE,
        )
        if field_match is not None:
            fields[field_name] = field_match.group("value")
    return fields or None


def _source_project() -> tuple[Path | None, dict[str, str | None]]:
    for parent in Path(__file__).resolve().parents:
        pyproject_path = parent / "pyproject.toml"
        project_table = _parse_project_table(pyproject_path)
        if project_table is None:
            continue
        normalized_name = project_table.get("name", "").lower().replace("_", "-")
        if normalized_name != _PROJECT_NAME:
            continue
        return parent, {
            "name": project_table.get("name"),
            "version": project_table.get("version") or source_package_version,
        }
    return None, {"name": _PROJECT_NAME, "version": source_package_version}


def _path_state(path_value: str, *, include_hash: bool = True) -> dict[str, object]:
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return {
            "path": str(path),
            "exists": False,
            "size": None,
            "mtime_ns": None,
            "sha256": None,
        }
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": optional_file_sha256(resolved) if include_hash else None,
    }


def _git_untracked_file_state(repo_root: Path) -> list[dict[str, str | None]]:
    output = _git_output(["ls-files", "--others", "--exclude-standard"], repo_root)
    if not output:
        return []
    states: list[dict[str, str | None]] = []
    for relative_path in sorted(path for path in output.splitlines() if path.strip()):
        path = repo_root / relative_path
        if not path.is_file():
            continue
        states.append(
            {
                "path": relative_path,
                "sha256": optional_file_sha256(path),
            }
        )
    return states


def _git_state() -> dict[str, object]:
    repo_root, _ = _source_project()
    if repo_root is None:
        return {
            "commit": None,
            "dirty": None,
            "changed_path_count": None,
            "untracked_file_count": None,
            "diff_sha256": None,
        }
    commit = _git_output(["rev-parse", "HEAD"], repo_root)
    status = _git_output(["status", "--porcelain"], repo_root)
    diff = _git_output(["diff", "--binary", "HEAD"], repo_root)
    untracked_files = _git_untracked_file_state(repo_root)
    untracked_payload = canonical_json_bytes(untracked_files).decode("utf-8")
    diff_payload = "\n".join(part for part in (status, diff, untracked_payload) if part)
    return {
        "commit": commit,
        "dirty": bool(status),
        "changed_path_count": len(status.splitlines()) if status else 0,
        "untracked_file_count": len(untracked_files),
        "diff_sha256": (sha256_bytes(diff_payload.encode("utf-8")) if diff_payload else None),
    }


def resolved_config_sections(
    config: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    document = to_jsonable(config)
    if not isinstance(document, dict):
        raise TypeError("Run configuration must serialize to a JSON object.")
    runtime = document.get("runtime")
    input_config = document.get("input")
    output = document.get("output")
    if (
        not isinstance(runtime, dict)
        or not isinstance(input_config, dict)
        or not isinstance(output, dict)
    ):
        raise TypeError("Run configuration is missing runtime/input/output sections.")

    runtime_copy = dict(runtime)
    input_copy = dict(input_config)
    input_path = input_copy.pop("path", None)
    output_copy = dict(output)
    output_path = output_copy.pop("csv_path", None)
    typed_document = cast(dict[str, object], document)
    scientific: dict[str, object] = {
        "input_policy": input_copy,
        "strand_adjacency": document.get("strand_adjacency"),
        "rules": document.get("rules"),
    }
    execution: dict[str, object] = {
        "runtime": runtime_copy,
        "output_policy": output_copy,
    }
    io: dict[str, object] = {
        "input_path": input_path,
        "output_csv_path": output_path,
    }
    return typed_document, scientific, execution, io


def _package_source_state() -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    source_paths = set(package_root.rglob("*.py"))
    config_root = package_root / "conf"
    if config_root.is_dir():
        source_paths.update(config_root.rglob("*.yaml"))
        source_paths.update(config_root.rglob("*.yml"))
    files = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": optional_file_sha256(path),
        }
        for path in sorted(source_paths)
        if path.is_file()
    ]
    return {
        "algorithm": "sha256",
        "scope": "cooper_beta/**/*.py and cooper_beta/conf/**/*.{yaml,yml}",
        "file_count": len(files),
        "combined_sha256": canonical_json_sha256(files),
        "files": files,
    }


def _required_mapping(
    document: Mapping[str, object], field_name: str, *, context: str
) -> Mapping[str, object]:
    value = document.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{field_name} must be an object.")
    return value


def _optional_identity_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string or null.")
    return value.strip()


def _identity_sha256(value: object, *, context: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a SHA-256 string.")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{context} is not a valid SHA-256 digest.")
    return normalized


def scientific_producer_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return the stable producer identity that must match across scientific splits."""

    project = _required_mapping(manifest, "project", context="manifest")
    project_name = _optional_identity_string(project.get("name"), context="project.name")
    if project_name is None:
        raise ValueError("project.name cannot be null.")
    project_identity = {
        "name": project_name,
        "source_version": _optional_identity_string(
            project.get("source_version"), context="project.source_version"
        ),
        "installed_distribution_version": _optional_identity_string(
            project.get("installed_distribution_version"),
            context="project.installed_distribution_version",
        ),
    }

    packages = _required_mapping(manifest, "packages", context="manifest")
    missing_packages = sorted(set(SCIENTIFIC_PRODUCER_PACKAGES) - set(packages))
    if missing_packages:
        raise ValueError(
            "manifest.packages is missing scientific producer package(s): "
            + ", ".join(missing_packages)
        )
    package_identity: dict[str, object] = {}
    for package_name in SCIENTIFIC_PRODUCER_PACKAGES:
        package_version = _optional_identity_string(
            packages[package_name], context=f"packages.{package_name}"
        )
        if package_name in _REQUIRED_SCIENTIFIC_PACKAGE_VERSIONS and package_version is None:
            raise ValueError(f"packages.{package_name} cannot be null.")
        package_identity[package_name] = package_version

    source = _required_mapping(manifest, "source", context="manifest")
    source_content = _required_mapping(source, "package_content", context="source")
    if source_content.get("algorithm") != "sha256":
        raise ValueError("source.package_content.algorithm must be 'sha256'.")
    source_file_count = source_content.get("file_count")
    if (
        isinstance(source_file_count, bool)
        or not isinstance(source_file_count, int)
        or source_file_count <= 0
    ):
        raise ValueError("source.package_content.file_count must be a positive integer.")
    source_files = source_content.get("files")
    if not isinstance(source_files, list) or len(source_files) != source_file_count:
        raise ValueError(
            "source.package_content.files must match source.package_content.file_count."
        )
    source_paths: set[str] = set()
    for index, file_state in enumerate(source_files):
        if not isinstance(file_state, Mapping):
            raise ValueError(f"source.package_content.files[{index}] must be an object.")
        relative_path = _optional_identity_string(
            file_state.get("path"), context=f"source.package_content.files[{index}].path"
        )
        if relative_path is None or relative_path in source_paths:
            raise ValueError("source.package_content file paths must be non-null and unique.")
        source_paths.add(relative_path)
        file_size = file_state.get("size")
        if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size < 0:
            raise ValueError(
                f"source.package_content.files[{index}].size must be a non-negative integer."
            )
        _identity_sha256(
            file_state.get("sha256"),
            context=f"source.package_content.files[{index}].sha256",
        )
    source_combined_hash = _identity_sha256(
        source_content.get("combined_sha256"),
        context="source.package_content.combined_sha256",
    )
    if source_combined_hash != canonical_json_sha256(source_files):
        raise ValueError(
            "source.package_content.combined_sha256 does not match its file inventory."
        )
    source_identity = {
        "algorithm": "sha256",
        "file_count": source_file_count,
        "combined_sha256": source_combined_hash,
    }

    environment_lock = _required_mapping(manifest, "environment_lock", context="manifest")
    lock_available = environment_lock.get("available")
    if not isinstance(lock_available, bool):
        raise ValueError("environment_lock.available must be boolean.")
    lock_hash = _identity_sha256(
        environment_lock.get("combined_sha256"),
        context="environment_lock.combined_sha256",
        optional=True,
    )
    if lock_available != (lock_hash is not None):
        raise ValueError("environment_lock.available must agree with combined_sha256 availability.")
    lock_files = environment_lock.get("files")
    if not isinstance(lock_files, list):
        raise ValueError("environment_lock.files must be a list.")
    for index, file_state in enumerate(lock_files):
        if not isinstance(file_state, Mapping):
            raise ValueError(f"environment_lock.files[{index}] must be an object.")
        _optional_identity_string(
            file_state.get("path"), context=f"environment_lock.files[{index}].path"
        )
        file_size = file_state.get("size")
        if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size < 0:
            raise ValueError(
                f"environment_lock.files[{index}].size must be a non-negative integer."
            )
        _identity_sha256(
            file_state.get("sha256"), context=f"environment_lock.files[{index}].sha256"
        )
    if lock_available and lock_hash != canonical_json_sha256(lock_files):
        raise ValueError("environment_lock.combined_sha256 does not match its file inventory.")
    if not lock_available and lock_files:
        raise ValueError("environment_lock.files must be empty when no lock is available.")

    executables = _required_mapping(manifest, "executables", context="manifest")
    dssp_identity = {
        "version": _optional_identity_string(
            executables.get("dssp_version"), context="executables.dssp_version"
        ),
        "sha256": _identity_sha256(
            executables.get("dssp_sha256"),
            context="executables.dssp_sha256",
            optional=True,
        ),
    }

    runtime = _required_mapping(manifest, "runtime", context="manifest")
    python_version = _optional_identity_string(runtime.get("python"), context="runtime.python")
    platform_identity = _optional_identity_string(
        runtime.get("platform"), context="runtime.platform"
    )
    if python_version is None or platform_identity is None:
        raise ValueError("runtime.python and runtime.platform cannot be null.")
    pools = runtime.get("native_thread_pools")
    if isinstance(pools, list):
        pool_identity: list[dict[str, object]] = []
        for index, pool in enumerate(pools):
            if not isinstance(pool, Mapping):
                raise ValueError(f"runtime.native_thread_pools[{index}] must be an object.")
            normalized = to_jsonable(
                {
                    str(key): value
                    for key, value in pool.items()
                    if key not in _NATIVE_POOL_EXECUTION_FIELDS
                }
            )
            if not isinstance(normalized, dict):  # pragma: no cover - construction invariant
                raise TypeError("Native pool identity must serialize to an object.")
            pool_identity.append(cast(dict[str, object], normalized))
        pool_identity.sort(key=canonical_json_sha256)
    else:
        raise ValueError("runtime.native_thread_pools must be a list.")

    return {
        "project": project_identity,
        "packages": package_identity,
        "source_package_content": source_identity,
        "environment_lock": {
            "available": lock_available,
            "combined_sha256": lock_hash,
        },
        "dssp": dssp_identity,
        "runtime": {
            "python": python_version,
            "platform": platform_identity,
            "native_thread_pools": pool_identity,
        },
    }


def _lock_state(repo_root: Path | None) -> dict[str, object]:
    if repo_root is None:
        return {"files": [], "combined_sha256": None, "available": False}
    lock_files = []
    for relative_path in ("uv.lock", "conda-lock.yml"):
        lock_path = repo_root / relative_path
        if lock_path.is_file():
            lock_files.append(
                {
                    "path": relative_path,
                    "size": int(lock_path.stat().st_size),
                    "sha256": optional_file_sha256(lock_path),
                }
            )
    if not lock_files:
        return {"files": [], "combined_sha256": None, "available": False}
    return {
        "files": lock_files,
        "combined_sha256": canonical_json_sha256(lock_files),
        "available": True,
    }


def _thread_pool_state() -> list[dict[str, object]] | None:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    return [dict(item) for item in threadpool_info()]


def _loaded_pools_within_limit(
    pools: list[dict[str, object]] | None,
    requested_limit: int | None,
) -> bool | None:
    if pools is None or requested_limit is None:
        return None
    observed = [
        value
        for pool in pools
        if isinstance((value := pool.get("num_threads")), int) and not isinstance(value, bool)
    ]
    return all(value <= requested_limit for value in observed) if observed else None


def build_run_manifest(
    *,
    config: object,
    input_files: list[str],
    output_path: str | None,
    input_identities: list[FrozenInputIdentity] | None = None,
    input_identities_verified: bool = False,
    hash_input_files: bool | None = None,
    resolved_analysis_workers: int | None = None,
    resolved_prepare_workers: int | None = None,
    started_at_utc: str | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    configured_dssp = getattr(getattr(config, "runtime", None), "dssp_bin_path", None)
    dssp_path = str(configured_dssp) if configured_dssp else None
    resolved_dssp_path = find_dssp_binary(dssp_path)
    source_root, source_project = _source_project()
    installed_project_version = _package_version(_PROJECT_NAME)
    if hash_input_files is None:
        output_config = getattr(config, "output", None)
        hash_input_files = bool(getattr(output_config, "hash_input_files", True))
    config_document, scientific_config, execution_config, io_config = resolved_config_sections(
        config
    )
    frozen_inputs = (
        input_identities if input_identities is not None else freeze_input_identities(input_files)
    )
    expected_paths = [str(Path(path).expanduser().resolve()) for path in input_files]
    frozen_paths = [identity.path for identity in frozen_inputs]
    if expected_paths != frozen_paths:
        raise ValueError(
            "Frozen input identities must correspond one-to-one with manifest input files."
        )
    if input_identities_verified:
        verify_input_identities(frozen_inputs)
    input_states = [
        identity.manifest_state(include_hash=bool(hash_input_files)) for identity in frozen_inputs
    ]
    input_content_identities = [
        {"size": state["size"], "sha256": state["sha256"]} for state in input_states
    ]
    input_set_hash = (
        canonical_json_sha256(input_content_identities)
        if hash_input_files and all(state["sha256"] is not None for state in input_states)
        else None
    )
    output_state = _path_state(output_path) if output_path else None
    completed_at_utc = _utc_now()
    resolved_run_id = uuid.uuid4().hex if run_id is None else run_id.strip()
    if not resolved_run_id:
        raise ValueError("`run_id` cannot be blank.")
    output_committed = (
        isinstance(output_state, dict)
        and output_state.get("exists") is True
        and isinstance(output_state.get("size"), int)
        and isinstance(output_state.get("sha256"), str)
    )
    manifest_status = "complete" if output_committed else "incomplete"
    bound_output_path: str | None
    if output_committed and output_state is not None:
        bound_output_path = str(output_state["path"])
        bound_output_size = output_state["size"]
        bound_output_sha256 = output_state["sha256"]
    else:
        bound_output_path = output_path
        bound_output_size = None
        bound_output_sha256 = None
    artifact_binding = {
        "run_id": resolved_run_id,
        "csv_path": bound_output_path,
        "csv_size": bound_output_size,
        "csv_sha256": bound_output_sha256,
        "committed_by_run": output_committed,
    }
    lock_state = _lock_state(source_root)
    runtime_config = getattr(config, "runtime", None)
    requested_native_threads_value = getattr(runtime_config, "native_threads_per_process", None)
    requested_native_threads = (
        int(requested_native_threads_value)
        if isinstance(requested_native_threads_value, int)
        and not isinstance(requested_native_threads_value, bool)
        else None
    )
    thread_environment = {name: os.environ.get(name) for name in sorted(NATIVE_THREAD_ENV_NAMES)}
    native_thread_pools = _thread_pool_state()
    bootstrap_state = runtime_bootstrap_state()
    manifest: dict[str, object] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": RUN_MANIFEST_KIND,
        "status": manifest_status,
        "run_id": resolved_run_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc if output_committed else None,
        "generated_at_utc": completed_at_utc,
        "project": {
            "name": source_project.get("name") or _PROJECT_NAME,
            "source_version": source_project.get("version"),
            "installed_distribution_version": installed_project_version,
        },
        "config_hash": config_hash(config_document),
        "scientific_config_hash": canonical_json_sha256(scientific_config),
        "execution_config_hash": canonical_json_sha256(execution_config),
        "io_config_hash": canonical_json_sha256(io_config),
        "config": config_document,
        "config_partitions": {
            "scientific": scientific_config,
            "execution": execution_config,
            "io": io_config,
        },
        "input_files": frozen_paths,
        "input_file_hashing_enabled": hash_input_files,
        "input_identity_policy": {
            "algorithm": "sha256",
            "polymer_position_policy": POLYMER_POSITION_POLICY,
            "dssp_residue_coverage_policy": DSSP_RESIDUE_COVERAGE_POLICY,
            "frozen_before_parsing": input_identities is not None,
            "verified_before_artifact_publication": bool(input_identities_verified),
            "hash_redacted_from_manifest": not bool(hash_input_files),
        },
        "input_file_state": input_states,
        "input_set_hash": input_set_hash,
        "input_inventory_hash": canonical_json_sha256(input_states),
        "output_path": bound_output_path,
        "output_file_state": output_state,
        "output_sha256": output_state.get("sha256") if isinstance(output_state, dict) else None,
        "artifact_binding": artifact_binding,
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
            "locale": locale.setlocale(locale.LC_ALL, None),
            "timezone": os.environ.get("TZ"),
            "multiprocessing_start_method": multiprocessing.get_context().get_start_method(),
            "resolved_analysis_workers": (
                int(resolved_analysis_workers) if resolved_analysis_workers is not None else None
            ),
            "resolved_prepare_workers": (
                int(resolved_prepare_workers) if resolved_prepare_workers is not None else None
            ),
            "thread_environment": thread_environment,
            "native_thread_policy": {
                "requested_threads_per_process": requested_native_threads,
                "applied_limit_in_parent_process": bootstrap_state.native_threads_per_process,
                "applied_limit_matches_request": (
                    bootstrap_state.native_threads_per_process == requested_native_threads
                    if bootstrap_state.native_threads_per_process is not None
                    and requested_native_threads is not None
                    else False
                ),
                "environment_matches_request": (
                    all(
                        value == str(requested_native_threads)
                        for value in thread_environment.values()
                    )
                    if requested_native_threads is not None
                    else None
                ),
                "loaded_pools_within_request": _loaded_pools_within_limit(
                    native_thread_pools, requested_native_threads
                ),
                "worker_initializers_reapply_same_limit": True,
            },
            "python_hash_policy": {
                "environment_value_observed": os.environ.get("PYTHONHASHSEED"),
                "runtime_assignment_performed": False,
                "current_interpreter_hash_secret_reconfigured": False,
                "algorithm_depends_on_hash_iteration_order": False,
            },
            "native_thread_pools": native_thread_pools,
        },
        "packages": {
            "cooper-beta": installed_project_version,
            "biopython": _package_version("biopython"),
            "numpy": _package_version("numpy"),
            "hydra-core": _package_version("hydra-core"),
            "omegaconf": _package_version("omegaconf"),
            "threadpoolctl": _package_version("threadpoolctl"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "scikit-learn": _package_version("scikit-learn"),
        },
        "environment_lock": lock_state,
        "executables": {
            "dssp": dssp_path,
            "dssp_resolved_path": resolved_dssp_path,
            "dssp_version": dssp_version(resolved_dssp_path) if resolved_dssp_path else None,
            "dssp_sha256": (
                optional_file_sha256(resolved_dssp_path) if resolved_dssp_path else None
            ),
        },
        "source": {
            "git": _git_state(),
            "package_content": _package_source_state(),
        },
    }
    producer_identity = scientific_producer_identity(manifest)
    manifest["producer_identity"] = producer_identity
    manifest["producer_identity_hash"] = canonical_json_sha256(producer_identity)
    return manifest


def write_run_manifest(
    *,
    config: object,
    input_files: list[str],
    output_path: str,
    input_identities: list[FrozenInputIdentity] | None = None,
    input_identities_verified: bool = False,
    hash_input_files: bool | None = None,
    resolved_analysis_workers: int | None = None,
    resolved_prepare_workers: int | None = None,
    started_at_utc: str | None = None,
    run_id: str | None = None,
) -> Path:
    manifest_path = Path(f"{output_path}.manifest.json").expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        input_files=input_files,
        output_path=output_path,
        input_identities=input_identities,
        input_identities_verified=input_identities_verified,
        hash_input_files=hash_input_files,
        resolved_analysis_workers=resolved_analysis_workers,
        resolved_prepare_workers=resolved_prepare_workers,
        started_at_utc=started_at_utc,
        run_id=run_id,
    )
    if manifest["status"] != "complete":
        raise ValueError("A final run manifest requires an existing regular output CSV.")
    return atomic_write_json(manifest_path, manifest, indent=2)
