#!/usr/bin/env bash
set -e

# Ensure the workspace .pixi environment is functional.
# The named volume at /workspace/.pixi shadows the bind-mounted host .pixi/,
# but starts empty on first run. Run pixi install to populate it if needed.
if [ ! -x ".pixi/envs/default/bin/mojo" ]; then
    echo "Initializing pixi environment inside container..."
    pixi install
fi

# ---------------------------------------------------------------------------
# Ensure test fixture directories are writable.
#
# When the workspace is bind-mounted from the host, directory ownership may
# not match the container user.  Several Mojo tests create temp files in
# their own fixture directories (relative to $PWD which is /workspace).
# We also create a shared /tmp/mojo-tests scratch area for tests that need
# an absolute writable path.
# ---------------------------------------------------------------------------
_ensure_writable() {
    for dir in "$@"; do
        mkdir -p "$dir" 2>/dev/null || true
        # If we own the directory already, nothing to do.
        if [ -w "$dir" ]; then
            continue
        fi
        # Try to make it writable (will only succeed if we have permission).
        chmod u+w "$dir" 2>/dev/null || true
    done
}

_ensure_writable \
    tests/configs/fixtures \
    tests/shared/fixtures \
    /tmp/mojo-tests

# ---------------------------------------------------------------------------
# Install pre-commit hooks into the git repo if not already installed.
# This must run at container startup (not Dockerfile build) because the
# workspace is bind-mounted at runtime and .git/hooks is inside the mount.
# ---------------------------------------------------------------------------
if [ -d ".git" ] && [ ! -f ".git/hooks/pre-commit" ]; then
    echo "Installing pre-commit git hooks..."
    pixi run pre-commit install --install-hooks 2>/dev/null || true
fi

exec "$@"
