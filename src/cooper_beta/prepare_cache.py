from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ._version import __version__ as source_package_version
from .config import AppConfig
from .constants import DSSP_RESIDUE_COVERAGE_POLICY, POLYMER_POSITION_POLICY
from .exceptions import InputContentChangedError
from .integrity import (
    FrozenInputIdentity,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    freeze_input_identity,
    verify_input_identity,
)
from .models import PreparedChainPayload

LOGGER = logging.getLogger(__name__)

PREPARE_CACHE_VERSION = 1
PREPARE_PRODUCER_SCHEMA_VERSION = 1
_PREPARATION_PYTHON_SOURCE_FILES = (
    "integrity.py",
    "prepare_cache.py",
    "preparation.py",
    "loader.py",
    "dssp_adapter.py",
    "strand_graph.py",
    "polymer_sequence.py",
    "models.py",
    "runtime.py",
    "config.py",
    "exceptions.py",
)


def default_prepare_cache_dir() -> str:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache_home) if xdg_cache_home else (Path.home() / ".cache")
    return str(cache_root / "cooper-beta" / "prepare")


def resolve_prepare_cache_dir(configured_dir: str | None = None) -> Path:
    cache_dir = configured_dir or default_prepare_cache_dir()
    return Path(cache_dir).expanduser().resolve()


def _file_state(identity: FrozenInputIdentity) -> dict[str, int | str]:
    return identity.content_identity()


def _executable_state(path_value: str | None) -> dict[str, int | str] | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
        digest = file_sha256(resolved)
    except OSError:
        return {"path": str(path_value)}
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _preparation_source_state() -> dict[str, object]:
    """Fingerprint the implementation and config sources that produce cached payloads."""
    package_root = Path(__file__).resolve().parent
    relative_paths = [
        *_PREPARATION_PYTHON_SOURCE_FILES,
        *(
            path.relative_to(package_root).as_posix()
            for path in (package_root / "conf").rglob("*.yaml")
        ),
    ]
    files: list[dict[str, object]] = []
    for relative_path in sorted(set(relative_paths)):
        source_path = package_root / relative_path
        try:
            digest = file_sha256(source_path)
        except OSError:
            digest = None
        files.append(
            {
                "path": relative_path,
                "exists": source_path.is_file(),
                "sha256": digest,
            }
        )
    return {
        "algorithm": "sha256",
        "sha256": canonical_json_sha256(files),
        "files": files,
    }


def _prepare_config_state(cfg: AppConfig) -> dict[str, object]:
    from .runtime import find_dssp_binary

    configured_dssp_path = cfg.runtime.dssp_bin_path
    resolved_dssp_path = find_dssp_binary(configured_dssp_path)
    dssp_path = resolved_dssp_path or configured_dssp_path
    input_cfg = cfg.input
    return {
        "cache_version": PREPARE_CACHE_VERSION,
        "configured_dssp_bin_path": str(configured_dssp_path or ""),
        "dssp_bin_path": str(dssp_path or ""),
        "dssp_bin_state": _executable_state(dssp_path),
        "structure_loading": {
            "polymer_position_policy": POLYMER_POSITION_POLICY,
            "dssp_residue_coverage_policy": DSSP_RESIDUE_COVERAGE_POLICY,
            "model_id": int(input_cfg.model_id),
            "strict_chain": bool(input_cfg.strict_chain),
            "include_nonstandard_amino_acids": bool(input_cfg.include_nonstandard_amino_acids),
            "atom_altloc_policy": str(input_cfg.atom_altloc_policy),
            "disordered_residue_policy": str(input_cfg.disordered_residue_policy),
            "pdb_parser_permissive": bool(input_cfg.pdb_parser_permissive),
            "dssp_failure_policy": str(input_cfg.dssp_failure_policy),
            "dssp_pdb_export_cryst1_record": str(input_cfg.dssp_pdb_export_cryst1_record),
            "dssp_sheet_codes": [str(code) for code in input_cfg.dssp_sheet_codes],
        },
        "producer": {
            "schema_version": PREPARE_PRODUCER_SCHEMA_VERSION,
            "cooper_beta_source_version": source_package_version,
            "cooper_beta_installed_distribution_version": _package_version("cooper-beta"),
            "biopython": _package_version("biopython"),
            "source": _preparation_source_state(),
        },
    }


def _prepare_cache_key_payload(
    identity: FrozenInputIdentity,
    cfg: AppConfig,
) -> dict[str, object]:
    return {
        "schema_version": PREPARE_CACHE_VERSION,
        "file": _file_state(identity),
        "prepare": _prepare_config_state(cfg),
    }


def build_prepare_cache_key(file_path: str, cfg: AppConfig) -> str:
    identity = freeze_input_identity(file_path)
    return canonical_json_sha256(_prepare_cache_key_payload(identity, cfg))


def _prepare_cache_identity(
    identity: FrozenInputIdentity,
    cfg: AppConfig,
) -> tuple[Path, str, dict[str, object]]:
    key_payload = _prepare_cache_key_payload(identity, cfg)
    cache_key = canonical_json_sha256(key_payload)
    cache_dir = resolve_prepare_cache_dir(cfg.runtime.prepare_cache_dir)
    cache_path = cache_dir / cache_key[:2] / f"{cache_key}.json"
    return cache_path, cache_key, key_payload


def _require_requested_identity(
    file_path: str,
    identity: FrozenInputIdentity,
) -> None:
    requested_path = str(Path(file_path).expanduser().resolve())
    if requested_path != identity.path:
        raise ValueError(
            "Frozen input identity path does not match the requested cache input path."
        )


def prepare_cache_path(file_path: str, cfg: AppConfig) -> Path:
    cache_path, _, _ = _prepare_cache_identity(freeze_input_identity(file_path), cfg)
    return cache_path


def _discard_cache_file(cache_path: Path, *, reason: str) -> None:
    try:
        cache_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        LOGGER.warning(
            "Could not remove rejected preparation cache entry: %s",
            exc,
            extra={"stage": "prepare_cache", "source_path": str(cache_path)},
        )
        return
    LOGGER.warning(
        "Rejected preparation cache entry: %s",
        reason,
        extra={"stage": "prepare_cache", "source_path": str(cache_path)},
    )


def _normalize_payloads(payloads: object) -> list[dict[str, object]] | None:
    if not isinstance(payloads, list):
        return None

    normalized_payloads: list[dict[str, object]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            return None
        try:
            normalized_payloads.append(PreparedChainPayload.from_mapping(payload).to_dict())
        except (TypeError, ValueError):
            return None
    return normalized_payloads


def _contains_degraded_payload(payloads: list[dict[str, object]]) -> bool:
    """Return whether a valid normalized payload records a transient preparation failure."""
    return any(payload["degraded"] is True for payload in payloads)


def load_prepare_payloads(
    file_path: str,
    cfg: AppConfig,
    *,
    input_identity: FrozenInputIdentity | None = None,
) -> list[dict[str, object]] | None:
    if not cfg.runtime.prepare_cache_enabled:
        return None

    identity = input_identity or freeze_input_identity(file_path)
    _require_requested_identity(file_path, identity)
    cache_path, expected_key, expected_key_payload = _prepare_cache_identity(identity, cfg)
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)
    except FileNotFoundError:
        LOGGER.debug(
            "Preparation cache miss",
            extra={"stage": "prepare_cache", "source_path": str(cache_path)},
        )
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _discard_cache_file(cache_path, reason=f"unreadable envelope ({type(exc).__name__})")
        return None

    if not isinstance(envelope, dict) or envelope.get("cache_version") != PREPARE_CACHE_VERSION:
        _discard_cache_file(cache_path, reason="cache schema mismatch")
        return None

    stored_key = envelope.get("cache_key")
    stored_key_payload = envelope.get("cache_key_payload")
    try:
        stored_payload_key = canonical_json_sha256(stored_key_payload)
    except (TypeError, ValueError):
        stored_payload_key = None
    if (
        stored_key != expected_key
        or stored_payload_key != expected_key
        or stored_key_payload != expected_key_payload
    ):
        _discard_cache_file(cache_path, reason="cache key or producer fingerprint mismatch")
        return None

    payloads = _normalize_payloads(envelope.get("payloads"))
    if payloads is None:
        _discard_cache_file(cache_path, reason="invalid cached payload schema")
        return None
    if _contains_degraded_payload(payloads):
        _discard_cache_file(cache_path, reason="degraded payloads are not cacheable")
        return None
    try:
        payloads_hash = canonical_json_sha256(payloads)
    except (TypeError, ValueError):
        _discard_cache_file(cache_path, reason="cached payload is not canonical JSON")
        return None
    if envelope.get("payloads_sha256") != payloads_hash:
        _discard_cache_file(cache_path, reason="cached payload digest mismatch")
        return None
    verify_input_identity(identity)
    requested_path = Path(identity.path)
    for payload in payloads:
        payload["filename"] = requested_path.name
        payload["source_path"] = str(requested_path)
    LOGGER.debug(
        "Preparation cache hit",
        extra={"stage": "prepare_cache", "source_path": str(cache_path)},
    )
    return payloads


def store_prepare_payloads(
    file_path: str,
    cfg: AppConfig,
    payloads: list[dict[str, object]],
    *,
    input_identity: FrozenInputIdentity | None = None,
) -> None:
    if not cfg.runtime.prepare_cache_enabled:
        return

    identity = input_identity or freeze_input_identity(file_path)
    _require_requested_identity(file_path, identity)
    verify_input_identity(identity)
    cache_path, cache_key, cache_key_payload = _prepare_cache_identity(identity, cfg)
    normalized_payloads = _normalize_payloads(payloads)
    if normalized_payloads is None:
        LOGGER.error(
            "Refusing to cache an invalid preparation payload",
            extra={"stage": "prepare_cache", "source_path": str(file_path)},
        )
        return
    if _contains_degraded_payload(normalized_payloads):
        LOGGER.info(
            "Not caching a degraded preparation result",
            extra={"stage": "prepare_cache", "source_path": str(file_path)},
        )
        return

    try:
        cache_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            cache_path.parent.chmod(0o700)
        except OSError as exc:
            LOGGER.debug(
                "Could not tighten preparation cache directory permissions: %s",
                exc,
                extra={"stage": "prepare_cache", "source_path": str(cache_path.parent)},
            )
        envelope = {
            "cache_version": PREPARE_CACHE_VERSION,
            "cache_key": cache_key,
            "cache_key_payload": cache_key_payload,
            "payloads_sha256": canonical_json_sha256(normalized_payloads),
            "payloads": normalized_payloads,
            "source_diagnostics": {
                "path": identity.path,
                "mtime_ns": identity.mtime_ns,
            },
        }
        atomic_write_json(cache_path, envelope, indent=2)
        try:
            verify_input_identity(identity)
        except InputContentChangedError:
            _discard_cache_file(cache_path, reason="input changed during cache publication")
            raise
        LOGGER.debug(
            "Stored preparation cache entry",
            extra={"stage": "prepare_cache", "source_path": str(cache_path)},
        )
    except InputContentChangedError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning(
            "Could not store preparation cache entry: %s",
            exc,
            extra={"stage": "prepare_cache", "source_path": str(cache_path)},
        )
