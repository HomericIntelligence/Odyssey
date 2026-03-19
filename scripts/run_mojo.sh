#!/usr/bin/env bash
# Transparently run any mojo subcommand inside a Podman container.
# The host machine may lack a compatible GLIBC for Mojo — the container provides it.
#
# Usage:
#   ./scripts/run_mojo.sh --version
#   ./scripts/run_mojo.sh test -I . tests/shared/core/test_utility.mojo
#   ./scripts/run_mojo.sh build -I . shared/core/extensor.mojo
#   ./scripts/run_mojo.sh format --check shared/
#
# Environment variables:
#   MOJO_IMAGE   Override the container image (default: projectodyssey:dev)
#   MOJO_ENGINE  Override the container engine (default: podman)

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

IMAGE="${MOJO_IMAGE:-projectodyssey:dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
# Engine detection with version gate
# ============================================================

check_podman_version() {
    if ! command -v podman &>/dev/null; then
        return 1
    fi
    local major
    major=$(podman --version | grep -oP '\d+' | head -1)
    [ "$major" -ge 4 ]
}

if [ -n "${MOJO_ENGINE:-}" ]; then
    ENGINE="$MOJO_ENGINE"
elif check_podman_version; then
    ENGINE="podman"
else
    echo "Error: Podman 4.0+ not found." >&2
    echo "Install Podman: ./scripts/install-podman.sh" >&2
    exit 1
fi

# ============================================================
# Engine-specific flags
# ============================================================

ENGINE_FLAGS=()
if [[ "$(basename "$ENGINE")" == "podman" ]]; then
    # --userns=keep-id maps host UID/GID into the container (rootless Podman)
    ENGINE_FLAGS+=(--userns=keep-id)
fi

# ============================================================
# Run mojo inside the container
# ============================================================

exec "$ENGINE" run --rm \
    "${ENGINE_FLAGS[@]}" \
    -v "${REPO_ROOT}:/workspace:Z" \
    -w /workspace \
    "$IMAGE" \
    pixi run mojo "$@"
