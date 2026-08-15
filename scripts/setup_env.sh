#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$PROJECT_ROOT/conda-lock.yml"
ENV_NAME="cooperbeta"
WITH_DEV=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_env.sh [--name ENV_NAME] [--dev] [--dry-run]

Create a reproducible Cooper-Beta environment with DSSP 4.5.3 from
conda-lock.yml, then install the exact Python dependency graph from uv.lock.
There is deliberately no apt/pip fallback: a fallback would create a
scientifically different DSSP, Python, or BLAS environment while appearing to
use the same configuration.

Prerequisites:
  - conda, mamba, or micromamba
  - conda-lock 3.0.4
    (for example: uv tool install 'conda-lock==3.0.4')

Options:
  --name ENV_NAME  Environment name (default: cooperbeta)
  --dev            Install the locked development/test extra as well
  --dry-run        Print commands without executing them
  -h, --help       Show this help
EOF
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

choose_conda_frontend() {
  local candidate
  for candidate in micromamba mamba conda; do
    if have_cmd "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

environment_exists() {
  local frontend="$1"
  "$frontend" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ { print $1 }' | grep -Fxq "$ENV_NAME"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --dev)
      WITH_DEV=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$LOCK_FILE" || ! -f "$PROJECT_ROOT/uv.lock" ]]; then
  echo "Missing conda-lock.yml or uv.lock; refusing an unlocked installation." >&2
  exit 1
fi
if ! frontend="$(choose_conda_frontend)"; then
  echo "conda, mamba, or micromamba is required for the locked DSSP environment." >&2
  exit 1
fi
if ! have_cmd conda-lock; then
  echo "conda-lock 3.0.4 is required (uv tool install 'conda-lock==3.0.4')." >&2
  exit 1
fi
if [[ "$(conda-lock --version)" != "conda-lock, version 3.0.4" ]]; then
  echo "Expected conda-lock 3.0.4; refusing a lock-installer version drift." >&2
  exit 1
fi
if environment_exists "$frontend"; then
  echo "Environment '$ENV_NAME' already exists; choose a new name or remove it explicitly." >&2
  exit 1
fi

run_cmd conda-lock install --name "$ENV_NAME" --conda "$frontend" "$LOCK_FILE"

if [[ "$DRY_RUN" -eq 1 ]]; then
  env_prefix="<resolved-environment-prefix>"
else
  env_prefix="$("$frontend" run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
fi

sync_args=(sync --frozen --extra full)
if [[ "$WITH_DEV" -eq 1 ]]; then
  sync_args+=(--extra dev)
fi
run_cmd env UV_PROJECT_ENVIRONMENT="$env_prefix" "$env_prefix/bin/uv" "${sync_args[@]}"
run_cmd "$frontend" run -n "$ENV_NAME" cooper-beta --check-env

echo
echo "Environment '$ENV_NAME' is ready from the committed locks (DSSP 4.5.3)."
echo "Activate it with: $frontend activate $ENV_NAME"
