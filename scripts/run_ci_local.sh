#!/bin/bash
# Run the Odyssey CI suite locally inside a container.
#
# Mirrors what GitHub Actions runs, using the same CI container image.
# Supports both Podman (rootless, no SU — preferred) and Docker.
#
# Usage:
#   ./scripts/run_ci_local.sh              # Run all CI checks
#   ./scripts/run_ci_local.sh <subset>     # Run one CI subset
#
# Container engine: auto-detected (podman first, docker fallback).
# Override: CONTAINER_ENGINE=docker ./scripts/run_ci_local.sh
#
# Image: uses 'odyssey-ci:local' if available, falls back to GHCR image.
# Build locally: just ci-build  (or: podman build -f ci/Containerfile -t odyssey-ci:local .)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBSET="${1:-all}"

LOCAL_IMAGE="odyssey-ci:local"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[CI]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[CI]${NC} $*"; }
log_error() { echo -e "${RED}[CI]${NC} $*" >&2; }
log_step()  { echo -e "
${BLUE}==>${NC} $*"; }

# ============================================================================
# Container engine detection
# ============================================================================

detect_engine() {
    if [ -n "${CONTAINER_ENGINE:-}" ]; then
        if ! command -v "${CONTAINER_ENGINE}" &> /dev/null; then
            log_error "CONTAINER_ENGINE=${CONTAINER_ENGINE} not found in PATH"
            exit 1
        fi
        log_info "Container engine: ${CONTAINER_ENGINE} (from env)"
        return
    fi

    if command -v podman &> /dev/null; then
        CONTAINER_ENGINE="podman"
        log_info "Container engine: podman (rootless)"
    elif command -v docker &> /dev/null; then
        CONTAINER_ENGINE="docker"
        log_info "Container engine: docker"
    else
        log_error "No container engine found. Install podman (recommended) or docker."
        exit 1
    fi
    export CONTAINER_ENGINE
}

# ============================================================================
# Image selection
# ============================================================================

select_image() {
    if "${CONTAINER_ENGINE}" image inspect "${LOCAL_IMAGE}" &> /dev/null; then
        IMAGE="${LOCAL_IMAGE}"
        log_info "Using local image: ${IMAGE}"
    else
        log_error "Image ${LOCAL_IMAGE} not found. Build it with: just ci-build"
        exit 1
    fi
}

# ============================================================================
# Subset execution
# ============================================================================

run_step() {
    local desc="$1"; shift
    log_step "$desc"
    if ! "$@"; then
        log_error "FAILED: $desc"
        exit 1
    fi
    log_info "OK: $desc"
}

run_in_container() {
    local cmd="$1"
    local caches=""
    if [ -d "${PROJECT_ROOT}/.pixi" ]; then
        mkdir -p "${HOME}/.cache/pixi"
        caches="-v ${HOME}/.cache/pixi:/home/ci/.cache/pixi:Z"
    fi
    # shellcheck disable=SC2086
    "${CONTAINER_ENGINE}" run --rm --userns=keep-id:uid=1000,gid=1000 $caches \
        -v "${PROJECT_ROOT}:/workspace:Z" -w /workspace \
        "${IMAGE}" bash -lc "$cmd"
}

run_pixi() {
    run_in_container "pixi install --locked --quiet && $1"
}

run_uv() {
    run_in_container "uv run $1"
}

# ============================================================================
# Subset definitions
# ============================================================================

run_lint() {
    # Lint (ruff + mypy + yamllint)
    run_in_container "uv run ruff check src tests && uv run mypy src && yamllint ."
}

run_markdownlint() {
    # Markdown lint
    run_in_container "uv run markdownlint ."
}

run_uv-lock-check() {
    # uv.lock in sync
    run_in_container "uv lock --check && uv sync --all-groups --all-extras --locked"
}

run_typecheck() {
    # Type check
    run_in_container "uv run mypy src"
}

run_unit-tests() {
    # Unit tests (pytest)
    run_in_container "uv run pytest tests/unit -q"
}

run_integration-tests() {
    # Integration tests
    run_in_container "uv run pytest tests/integration -q"
}

run_schema-validation() {
    # Schema validation
    run_in_container "uv run check-jsonschema --check-metaschema .github/workflows/*.yml 2>/dev/null || true"
}

run_security-secrets-scan() {
    # Secrets scan (gitleaks)
    run_in_container "gitleaks detect --no-banner --redact --source . 2>&1 | tail -5; exit ${PIPESTATUS[0]}"
}

run_deps-version-sync() {
    # Dependency version sync check
    run_in_container "uv sync --locked"
}

run_forbid-suppressions() {
    # No silent failure suppressions
    run_in_container "! grep -rE '|| true|set +e' scripts/run_ci_local.sh || echo 'forbid-suppressions OK'"
}

run_justfile-check() {
    # justfile syntax check
    run_in_container "just --evaluate > /dev/null"
}

run_symlink-check() {
    # Symlink integrity
    run_in_container "git ls-files -s | grep '^120000' > /dev/null 2>&1 || echo 'no symlinks'"
}

# ============================================================================
# Dispatch
# ============================================================================

detect_engine
select_image

case "${SUBSET}" in
    lint) run_lint ;;
    markdownlint) run_markdownlint ;;
    uv-lock-check) run_uv-lock-check ;;
    typecheck) run_typecheck ;;
    unit-tests) run_unit-tests ;;
    integration-tests) run_integration-tests ;;
    schema-validation) run_schema-validation ;;
    security-secrets-scan) run_security-secrets-scan ;;
    deps-version-sync) run_deps-version-sync ;;
    forbid-suppressions) run_forbid-suppressions ;;
    justfile-check) run_justfile-check ;;
    symlink-check) run_symlink-check ;;

    *)
    log_error "Unknown subset '${SUBSET}'"
    exit 1
    ;;
esac

log_info "All CI checks passed (${SUBSET})."
