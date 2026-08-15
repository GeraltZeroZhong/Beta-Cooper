from __future__ import annotations

import os
from dataclasses import dataclass

from .constants import NATIVE_THREAD_ENV_NAMES

_NATIVE_THREAD_LIMITER: object | None = None
_APPLIED_NATIVE_THREAD_LIMIT: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapState:
    """Process-local record of the applied native-thread limit."""

    native_threads_per_process: int | None


def runtime_bootstrap_state() -> RuntimeBootstrapState:
    return RuntimeBootstrapState(native_threads_per_process=_APPLIED_NATIVE_THREAD_LIMIT)


def configure_thread_environment(native_threads_per_process: int) -> None:
    """Limit future and already-loaded BLAS/OpenMP thread pools."""

    if native_threads_per_process <= 0:
        raise ValueError("`native_threads_per_process` must be greater than zero.")
    requested_limit = int(native_threads_per_process)
    for env_name in NATIVE_THREAD_ENV_NAMES:
        os.environ[env_name] = str(requested_limit)

    from threadpoolctl import threadpool_info, threadpool_limits

    limiter = threadpool_limits(limits=requested_limit)
    violations = [
        f"{pool.get('prefix') or pool.get('internal_api') or 'unknown'}={observed}"
        for pool in threadpool_info()
        if isinstance((observed := pool.get("num_threads")), int)
        and not isinstance(observed, bool)
        and observed > requested_limit
    ]
    if violations:
        raise RuntimeError(
            "Native thread pools exceeded the configured limit "
            f"{requested_limit}: {', '.join(violations)}."
        )

    global _APPLIED_NATIVE_THREAD_LIMIT, _NATIVE_THREAD_LIMITER
    _NATIVE_THREAD_LIMITER = limiter
    _APPLIED_NATIVE_THREAD_LIMIT = requested_limit
