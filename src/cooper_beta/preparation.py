from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .exceptions import InputContentChangedError
from .integrity import (
    FrozenInputIdentity,
    freeze_input_identity,
    verified_input_snapshot,
    verify_input_identity,
)
from .loader import ProteinLoader
from .models import strand_graph_to_mapping
from .prepare_cache import load_prepare_payloads, store_prepare_payloads

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepareFailure:
    """Structured file-level failure returned across a process boundary."""

    source_path: str
    error_code: str
    message: str


@dataclass(frozen=True)
class PreparationBatchResult:
    """Serializable result of preparing a deterministic file batch."""

    payloads: list[dict[str, object]]
    failures: list[PrepareFailure]
    processed_files: int


def _failure_for_exception(source_path: str, exception: Exception) -> PrepareFailure:
    error_code = getattr(exception, "error_code", None) or "UNEXPECTED_PREPARATION_FAILURE"
    return PrepareFailure(
        source_path=source_path,
        error_code=str(error_code),
        message=str(exception),
    )


def prepare_one_file(
    file_path: str,
    config: AppConfig,
    *,
    input_identity: FrozenInputIdentity | None = None,
) -> list[dict[str, object]] | PrepareFailure:
    """Parse a structure and execute DSSP exactly once for all selected chains."""
    source_path = str(Path(file_path).expanduser().resolve())
    try:
        identity = input_identity or freeze_input_identity(file_path)
        if identity.path != source_path:
            raise InputContentChangedError(
                f"Input path resolved to a different file after identities were frozen: {file_path}"
            )
        cached_payloads = load_prepare_payloads(
            file_path,
            config,
            input_identity=identity,
        )
        if cached_payloads is not None:
            return cached_payloads

        verify_input_identity(identity)
        payloads: list[dict[str, object]] = []
        with verified_input_snapshot(identity) as snapshot_path:
            loader = ProteinLoader(
                snapshot_path,
                config.input,
                dssp_bin=config.runtime.dssp_bin_path,
            )
            for chain_id in loader.available_chains():
                chain_result = loader.prepare_chain(chain_id)
                degraded = chain_result.failed
                payloads.append(
                    {
                        "filename": Path(identity.path).name,
                        "source_path": identity.path,
                        "chain": chain_result.author_chain_id,
                        "residues_data": [dict(residue) for residue in chain_result.residues],
                        "strand_graph": strand_graph_to_mapping(chain_result.strand_graph),
                        "degraded": degraded,
                        "degradation_code": chain_result.error_code or "",
                        "degradation_reason": chain_result.error_message or "",
                    }
                )
        verify_input_identity(identity)
        if not any(payload["degraded"] is True for payload in payloads):
            store_prepare_payloads(
                identity.path,
                config,
                payloads,
                input_identity=identity,
            )
        verify_input_identity(identity)
        return payloads
    except Exception as exc:
        failure = _failure_for_exception(source_path, exc)
        LOGGER.exception(
            "Structure preparation failed",
            extra={"source_path": source_path, "error_code": failure.error_code},
        )
        return failure


def prepare_file_batch(
    file_paths: list[str],
    config: AppConfig,
    input_identities: dict[str, FrozenInputIdentity] | None = None,
) -> PreparationBatchResult:
    """Prepare a batch while retaining every per-file failure as structured data."""
    payloads: list[dict[str, object]] = []
    failures: list[PrepareFailure] = []
    for file_path in file_paths:
        identity = None
        if input_identities is not None:
            identity = input_identities.get(str(Path(file_path).expanduser().resolve()))
            if identity is None:
                failures.append(
                    PrepareFailure(
                        source_path=str(Path(file_path).expanduser().resolve()),
                        error_code="MISSING_FROZEN_INPUT_IDENTITY",
                        message="No frozen input identity was supplied for the preparation task.",
                    )
                )
                continue
        result = (
            prepare_one_file(file_path, config)
            if identity is None
            else prepare_one_file(file_path, config, input_identity=identity)
        )
        if isinstance(result, PrepareFailure):
            failures.append(result)
        else:
            payloads.extend(result)
    return PreparationBatchResult(
        payloads=payloads,
        failures=failures,
        processed_files=len(file_paths),
    )
