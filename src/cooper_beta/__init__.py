"""
cooper_beta

A small toolkit/pipeline to detect beta-barrel-like protein chains from PDB/mmCIF structures.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._version import __version__

__all__ = [
    "AppConfig",
    "ConfigValidationError",
    "ChainPreparationResult",
    "ChainNotFoundError",
    "CooperBetaError",
    "DetectionResult",
    "DsspError",
    "DsspNotFoundError",
    "InputValidationError",
    "InputContentChangedError",
    "PipelineRunResult",
    "PreparedChainPayload",
    "ProteinLoader",
    "ResidueRecord",
    "StrandAdjacencyGraph",
    "StrandGraphMeasurements",
    "StrandEdge",
    "StrandNode",
    "StrandRange",
    "StructureParseError",
    "build_config",
    "detect",
    "find_dssp_binary",
    "require_dssp_binary",
    "__version__",
]

_LAZY_EXPORTS = {
    "AppConfig": ("cooper_beta.config", "AppConfig"),
    "build_config": ("cooper_beta.config", "build_config"),
    "ChainNotFoundError": ("cooper_beta.exceptions", "ChainNotFoundError"),
    "ConfigValidationError": ("cooper_beta.exceptions", "ConfigValidationError"),
    "CooperBetaError": ("cooper_beta.exceptions", "CooperBetaError"),
    "DsspError": ("cooper_beta.exceptions", "DsspError"),
    "DsspNotFoundError": ("cooper_beta.exceptions", "DsspNotFoundError"),
    "InputValidationError": ("cooper_beta.exceptions", "InputValidationError"),
    "InputContentChangedError": ("cooper_beta.exceptions", "InputContentChangedError"),
    "StructureParseError": ("cooper_beta.exceptions", "StructureParseError"),
    "ProteinLoader": ("cooper_beta.loader", "ProteinLoader"),
    "ChainPreparationResult": ("cooper_beta.loader", "ChainPreparationResult"),
    "DetectionResult": ("cooper_beta.models", "DetectionResult"),
    "PipelineRunResult": ("cooper_beta.models", "PipelineRunResult"),
    "PreparedChainPayload": ("cooper_beta.models", "PreparedChainPayload"),
    "ResidueRecord": ("cooper_beta.models", "ResidueRecord"),
    "StrandAdjacencyGraph": ("cooper_beta.strand_graph", "StrandAdjacencyGraph"),
    "StrandGraphMeasurements": (
        "cooper_beta.strand_graph",
        "StrandGraphMeasurements",
    ),
    "StrandEdge": ("cooper_beta.strand_graph", "StrandEdge"),
    "StrandNode": ("cooper_beta.strand_graph", "StrandNode"),
    "StrandRange": ("cooper_beta.strand_graph", "StrandRange"),
    "detect": ("cooper_beta.pipeline", "detect"),
    "find_dssp_binary": ("cooper_beta.runtime", "find_dssp_binary"),
    "require_dssp_binary": ("cooper_beta.runtime", "require_dssp_binary"),
}


def __getattr__(name: str) -> Any:
    """Load public components lazily so runtime thread limits can be set first."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
