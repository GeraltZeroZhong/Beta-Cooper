from __future__ import annotations


class CooperBetaError(Exception):
    """Base exception for Cooper-Beta user-facing errors."""


class ConfigValidationError(CooperBetaError, ValueError):
    """Raised when a Cooper-Beta configuration contains invalid values."""

    error_code = "CONFIG_VALIDATION_FAILED"


class InputValidationError(CooperBetaError, ValueError):
    """Raised when an input path or file set cannot be used for analysis."""

    error_code = "INPUT_VALIDATION_FAILED"


class InputContentChangedError(InputValidationError):
    """Raised when a frozen input no longer has the content used by a run."""

    error_code = "INPUT_CONTENT_CHANGED"


class OutputArtifactError(CooperBetaError, RuntimeError):
    """Raised when a result CSV/manifest transaction cannot be completed safely."""

    error_code = "OUTPUT_ARTIFACT_FAILED"


class OutputArtifactExistsError(OutputArtifactError, FileExistsError):
    """Raised when publication would overwrite an existing result artifact."""

    error_code = "OUTPUT_ARTIFACT_EXISTS"


class OutputArtifactBusyError(OutputArtifactError):
    """Raised when another process holds the result target's publication lock."""

    error_code = "OUTPUT_ARTIFACT_BUSY"


class DsspNotFoundError(CooperBetaError, RuntimeError):
    """Raised when the DSSP executable cannot be found."""

    error_code = "DSSP_NOT_FOUND"


class DsspError(CooperBetaError, RuntimeError):
    """Raised when DSSP fails while preparing a structure."""

    error_code = "DSSP_FAILED"


class StructureParseError(CooperBetaError, ValueError):
    """Raised when a PDB/mmCIF structure cannot be parsed."""

    error_code = "STRUCTURE_PARSE_FAILED"


class ChainNotFoundError(CooperBetaError, KeyError):
    """Raised when a requested chain is not present in a structure."""

    error_code = "CHAIN_NOT_FOUND"
