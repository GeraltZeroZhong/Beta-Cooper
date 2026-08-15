from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from functools import cache

from .exceptions import DsspError, DsspNotFoundError

MINIMUM_DSSP_VERSION = (4, 5, 3)
DSSP_VERSION_QUERY_TIMEOUT_SECONDS = 5.0
_DSSP_VERSION_PATTERN = re.compile(
    r"mkdssp version (?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
)


def _resolve_executable(candidate: str | None) -> str | None:
    if not candidate:
        return None
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return os.path.abspath(candidate)
    return shutil.which(candidate)


def find_dssp_binary(explicit_path: str | None = None) -> str | None:
    if explicit_path:
        candidate_path = os.path.expanduser(explicit_path)
        if (
            os.path.isabs(candidate_path)
            or os.sep in candidate_path
            or (os.altsep is not None and os.altsep in candidate_path)
        ):
            if os.path.isfile(candidate_path) and os.access(candidate_path, os.X_OK):
                return os.path.abspath(candidate_path)
            return None
        return _resolve_executable(explicit_path)
    return shutil.which("mkdssp") or shutil.which("dssp")


def dssp_requirement_message() -> str:
    return (
        "Cooper-Beta requires DSSP (mkdssp 4.5.3 or newer) before analysis can run.\n"
        "Install a supported DSSP release and make sure `mkdssp` or `dssp` is on PATH.\n"
        "If DSSP is installed in a non-standard location, set `runtime.dssp_bin_path` "
        "to its executable path or command name in the resolved configuration."
    )


def _format_dssp_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(component) for component in version)


@cache
def _dssp_version_output(dssp_bin: str) -> str:
    try:
        completed = subprocess.run(
            [dssp_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=DSSP_VERSION_QUERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DsspError(f"Unable to query the DSSP version from {dssp_bin!r}: {error}") from error

    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        detail = output or f"exit status {completed.returncode}"
        raise DsspError(f"Unable to query the DSSP version from {dssp_bin!r}: {detail}")

    return output


def dssp_version(dssp_bin: str) -> str:
    """Return the first version line used in run provenance."""
    return _dssp_version_output(dssp_bin).splitlines()[0]


def validated_dssp_version(dssp_bin: str) -> str:
    output = _dssp_version_output(dssp_bin)
    match = _DSSP_VERSION_PATTERN.fullmatch(output)
    if match is None:
        received = repr(output) if output else "no output"
        raise DsspError(
            "Unable to verify the DSSP version from "
            f"{dssp_bin!r}: expected `mkdssp version X.Y.Z` from `--version`, "
            f"received {received}. Cooper-Beta requires mkdssp 4.5.3 or newer."
        )

    parsed_version = (
        int(match["major"]),
        int(match["minor"]),
        int(match["patch"]),
    )
    if parsed_version < MINIMUM_DSSP_VERSION:
        raise DsspError(
            "Unsupported DSSP version "
            f"{_format_dssp_version(parsed_version)} at {dssp_bin!r}. "
            "Cooper-Beta requires mkdssp 4.5.3 or newer because earlier releases can fail "
            "on supported mmCIF inputs. Use the repository's locked environment or install "
            "a supported DSSP release."
        )
    return output


def require_dssp_binary(explicit_path: str | None = None) -> str:
    dssp_bin = find_dssp_binary(explicit_path)
    if not dssp_bin:
        if explicit_path:
            raise DsspNotFoundError(
                f"Configured DSSP executable was not found or is not executable: {explicit_path}"
            )
        raise DsspNotFoundError(dssp_requirement_message())
    validated_dssp_version(dssp_bin)
    return dssp_bin


def runtime_summary(
    explicit_path: str | None = None, *, require_dssp: bool = True
) -> dict[str, str]:
    dssp_path = find_dssp_binary(explicit_path)
    if dssp_path is None and require_dssp:
        dssp_path = require_dssp_binary(explicit_path)
    dssp_display = "not found"
    if dssp_path is not None:
        dssp_display = f"{dssp_path} ({validated_dssp_version(dssp_path)})"
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "dssp": dssp_display,
    }
