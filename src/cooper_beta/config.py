from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Integral, Real
from types import NoneType, UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from .exceptions import ConfigValidationError


def _parameter(
    description: str,
    *,
    gt: float | None = None,
    ge: float | None = None,
    le: float | None = None,
    choices: tuple[object, ...] | None = None,
    item_choices: tuple[object, ...] | None = None,
) -> Any:
    return field(
        metadata={
            "description": description,
            "gt": gt,
            "ge": ge,
            "le": le,
            "choices": choices,
            "item_choices": item_choices,
        }
    )


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int | None = _parameter(
        "Analysis process count; null resolves from available CPUs.", gt=0
    )
    prepare_workers: int | None = _parameter(
        "Structure-preparation process count; null follows the analysis count.", gt=0
    )
    prepare_batch_size: int = _parameter("Input files per preparation task.", gt=0)
    analysis_batch_size: int = _parameter("Chains per analysis task.", gt=0)
    prepare_in_flight_multiplier: int = _parameter(
        "Maximum queued preparation batches per worker.", gt=0
    )
    analysis_in_flight_multiplier: int = _parameter(
        "Maximum queued analysis batches per worker.", gt=0
    )
    cpu_reserve: int = _parameter("CPUs left unused when worker count is automatic.", ge=0)
    native_threads_per_process: int = _parameter(
        "BLAS/OpenMP threads allowed in each process.", gt=0
    )
    dssp_bin_path: str | None = _parameter("Explicit DSSP executable path or command name.")
    prepare_cache_enabled: bool = _parameter("Enable the DSSP preparation cache.")
    prepare_cache_dir: str | None = _parameter("Preparation cache directory; null uses XDG cache.")
    log_level: str = _parameter(
        "Application log level.", choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
    )
    log_console: bool = _parameter("Emit application logs to stderr.")
    log_jsonl_path: str | None = _parameter("Optional JSON Lines log path.")
    check_env: bool = _parameter("Print resolved runtime dependencies and exit.")


@dataclass(frozen=True)
class InputConfig:
    path: str = _parameter("Structure file or directory to analyze.")
    allowed_suffixes: tuple[str, ...] = _parameter(
        "Accepted structure filename suffixes.",
        item_choices=(
            ".pdb",
            ".ent",
            ".cif",
            ".mmcif",
            ".pdb.gz",
            ".ent.gz",
            ".cif.gz",
            ".mmcif.gz",
        ),
    )
    model_id: int = _parameter("Zero-based Biopython model identifier.", ge=0)
    strict_chain: bool = _parameter("Require requested chain identifiers to exist exactly.")
    include_nonstandard_amino_acids: bool = _parameter(
        "Treat recognized non-standard amino acids as protein residues."
    )
    atom_altloc_policy: str = _parameter(
        "Alternate-location atom selection policy.",
        choices=("highest_occupancy", "biopython_selected", "error"),
    )
    disordered_residue_policy: str = _parameter(
        "Disordered-residue selection policy.",
        choices=("highest_ca_occupancy", "biopython_selected", "error"),
    )
    pdb_parser_permissive: bool = _parameter("Use Biopython's permissive PDB parser mode.")
    dssp_failure_policy: str = _parameter(
        "Behavior when DSSP cannot produce assignments.", choices=("error", "degraded")
    )
    dssp_pdb_export_cryst1_record: str = _parameter(
        "CRYST1 record used in temporary PDB input for DSSP."
    )
    dssp_sheet_codes: tuple[str, ...] = _parameter(
        "DSSP codes counted as sheet residues.", item_choices=("E", "B")
    )
    atom_site_only_max_peptide_bond_distance_angstrom: float = _parameter(
        "Maximum C-N distance for linked modified amino acids in atom-site-only mmCIF.",
        gt=0.0,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_suffixes", tuple(self.allowed_suffixes))
        object.__setattr__(self, "dssp_sheet_codes", tuple(self.dssp_sheet_codes))


@dataclass(frozen=True)
class OutputConfig:
    csv_path: str = _parameter("Detection result CSV path.")
    summary_limit: int = _parameter("Console row limit; -1 prints all rows.", ge=-1)
    write_manifest: bool = _parameter("Write a provenance sidecar with the result CSV.")
    hash_input_files: bool = _parameter("Hash input structures in the run manifest.")
    existing_artifact_policy: str = _parameter(
        "Behavior when output artifacts already exist.", choices=("error", "replace")
    )


@dataclass(frozen=True)
class StrandAdjacencyConfig:
    maximum_ca_distance_angstrom: float = _parameter(
        "Maximum C-alpha distance for a contact-supported strand adjacency.", gt=0.0
    )
    minimum_contact_pair_count: int = _parameter(
        "Minimum C-alpha contact pairs supporting an adjacency.", gt=0
    )
    minimum_contact_residue_count_per_strand: int = _parameter(
        "Minimum distinct contacting residues on each strand.", gt=0
    )


@dataclass(frozen=True)
class StrandAdjacencyCountRuleConfig:
    minimum: int = _parameter("Minimum strand-adjacency count.", gt=0)


@dataclass(frozen=True)
class CycleStrandCountFractionRuleConfig:
    minimum_count: int = _parameter("Minimum strands in the largest closed component.", gt=0)
    minimum_fraction: float = _parameter(
        "Minimum fraction of all strands in the largest closed component.", gt=0.0, le=1.0
    )


@dataclass(frozen=True)
class CycleRankRuleConfig:
    minimum: int = _parameter("Minimum independent-cycle count.", gt=0)


@dataclass(frozen=True)
class RuleConfig:
    strand_adjacency_count: StrandAdjacencyCountRuleConfig = _parameter(
        "Strand-adjacency-count rule."
    )
    cycle_strand_count_fraction: CycleStrandCountFractionRuleConfig = _parameter(
        "Cycle-strand-count and cycle-strand-fraction rule."
    )
    cycle_rank: CycleRankRuleConfig = _parameter("Independent-cycle-count rule.")


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    input: InputConfig
    output: OutputConfig
    strand_adjacency: StrandAdjacencyConfig
    rules: RuleConfig


def _validate_annotated_type(path: str, value: object, expected: object) -> None:
    if expected is Any:
        return
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin in {Union, UnionType}:
        if any(_annotation_accepts(value, option) for option in arguments):
            return
        raise ConfigValidationError(
            f"`{path}` has type {type(value).__name__}; expected {expected!r}."
        )
    if origin is tuple:
        if not isinstance(value, tuple):
            raise ConfigValidationError(f"`{path}` must be a tuple.")
        item_type = arguments[0] if arguments else Any
        for index, item in enumerate(value):
            _validate_annotated_type(f"{path}[{index}]", item, item_type)
        return
    if expected is NoneType:
        if value is not None:
            raise ConfigValidationError(f"`{path}` must be null.")
        return
    if expected is bool:
        if type(value) is not bool:
            raise ConfigValidationError(f"`{path}` must be a boolean.")
        return
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ConfigValidationError(f"`{path}` must be an integer.")
        return
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ConfigValidationError(f"`{path}` must be a real number.")
        if not math.isfinite(float(value)):
            raise ConfigValidationError(f"`{path}` must be finite.")
        return
    if expected is str:
        if not isinstance(value, str):
            raise ConfigValidationError(f"`{path}` must be a string.")
        return
    if isinstance(expected, type) and not isinstance(value, expected):
        raise ConfigValidationError(
            f"`{path}` has type {type(value).__name__}; expected {expected.__name__}."
        )


def _annotation_accepts(value: object, expected: object) -> bool:
    try:
        _validate_annotated_type("value", value, expected)
    except ConfigValidationError:
        return False
    return True


def _validate_dataclass(value: object, prefix: str = "") -> None:
    if not is_dataclass(value):
        raise ConfigValidationError("Configuration must be a structured AppConfig instance.")
    hints = get_type_hints(type(value))
    for definition in fields(value):
        item = getattr(value, definition.name)
        path = f"{prefix}.{definition.name}" if prefix else definition.name
        _validate_annotated_type(path, item, hints.get(definition.name, Any))
        if is_dataclass(item):
            _validate_dataclass(item, path)
            continue
        metadata = definition.metadata
        if item is None:
            continue
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            numeric = float(item)
            gt = metadata.get("gt")
            ge = metadata.get("ge")
            le = metadata.get("le")
            if gt is not None and numeric <= float(cast(Any, gt)):
                raise ConfigValidationError(f"`{path}` must be greater than {gt}.")
            if ge is not None and numeric < float(cast(Any, ge)):
                raise ConfigValidationError(f"`{path}` must be at least {ge}.")
            if le is not None and numeric > float(cast(Any, le)):
                raise ConfigValidationError(f"`{path}` must be at most {le}.")
        choices = metadata.get("choices")
        if choices is not None and item not in cast(tuple[object, ...], choices):
            raise ConfigValidationError(f"`{path}` must be one of {choices!r}.")
        item_choices = metadata.get("item_choices")
        if item_choices is not None:
            allowed = cast(tuple[object, ...], item_choices)
            invalid = [entry for entry in cast(tuple[object, ...], item) if entry not in allowed]
            if invalid:
                raise ConfigValidationError(f"`{path}` contains unsupported values {invalid!r}.")


def validate_config(cfg: AppConfig) -> None:
    if not isinstance(cfg, AppConfig):
        raise ConfigValidationError("Expected an AppConfig instance.")
    _validate_dataclass(cfg)
    if not cfg.input.allowed_suffixes:
        raise ConfigValidationError("`input.allowed_suffixes` cannot be empty.")
    normalized_suffixes = tuple(value.strip().lower() for value in cfg.input.allowed_suffixes)
    if normalized_suffixes != cfg.input.allowed_suffixes or any(
        not value.startswith(".") for value in normalized_suffixes
    ):
        raise ConfigValidationError(
            "`input.allowed_suffixes` must contain canonical lowercase suffixes beginning with '.'."
        )
    if len(set(normalized_suffixes)) != len(normalized_suffixes):
        raise ConfigValidationError("`input.allowed_suffixes` cannot contain duplicates.")
    if not cfg.input.dssp_sheet_codes or len(set(cfg.input.dssp_sheet_codes)) != len(
        cfg.input.dssp_sheet_codes
    ):
        raise ConfigValidationError("`input.dssp_sheet_codes` must be non-empty and unique.")


def _mapping_config(overrides: Mapping[str, Any]) -> DictConfig:
    result = OmegaConf.create({})
    for key, value in overrides.items():
        if "." in str(key):
            OmegaConf.update(result, str(key), value, merge=True, force_add=True)
        else:
            result[str(key)] = value
    return result


def compose_config(
    overrides: Mapping[str, Any] | list[str] | None = None,
    *,
    config_name: str = "config",
) -> DictConfig:
    cli_overrides = list(overrides) if isinstance(overrides, list) else []
    with initialize_config_module(config_module="cooper_beta.conf", version_base="1.3"):
        resolved = compose(config_name=config_name, overrides=cli_overrides)
    if isinstance(overrides, Mapping):
        resolved = cast(DictConfig, OmegaConf.merge(resolved, _mapping_config(overrides)))
    merged = cast(DictConfig, OmegaConf.merge(resolved, OmegaConf.structured(AppConfig)))
    OmegaConf.set_struct(merged, True)
    return merged


def build_config(
    overrides: Mapping[str, Any] | list[str] | None = None,
    *,
    config_name: str = "config",
) -> AppConfig:
    try:
        app_cfg = OmegaConf.to_object(compose_config(overrides, config_name=config_name))
    except ConfigValidationError:
        raise
    except Exception as exc:
        raise ConfigValidationError(
            f"Configuration is incomplete or has invalid types: {exc}"
        ) from exc
    if not isinstance(app_cfg, AppConfig):
        raise ConfigValidationError("Configuration composition returned an unexpected object.")
    validate_config(app_cfg)
    return app_cfg


def config_to_dict(cfg: AppConfig) -> dict[str, object]:
    validate_config(cfg)
    container = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True, enum_to_str=True)
    if not isinstance(container, dict):
        raise ConfigValidationError("Could not serialize configuration.")
    return cast(dict[str, object], container)


def config_to_yaml(cfg: AppConfig) -> str:
    validate_config(cfg)
    return OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=True, sort_keys=True)
