#!/usr/bin/env bash
set -e

# ---------------------------------------------------------------------------
# Ensure $HOME/.modular exists and is writable by the current user.
#
# Mojo's runtime (libAsyncRTMojoBindings.so) calls std::filesystem::status()
# on $HOME/.modular during startup via getAcceleratorArchOrEmpty(). If the
# path is inaccessible (e.g., HOME is owned by a different UID), the uncaught
# filesystem_error aborts the process before any user code runs.
#
# This is a Mojo upstream bug (filed: modular/modular). Workaround: ensure
# $HOME/.modular exists and is owned by the current UID before invoking mojo.
# ---------------------------------------------------------------------------
if [ ! -d "${HOME}/.modular" ]; then
    mkdir -p "${HOME}/.modular" 2>/dev/null || {
        # HOME is not writable by this UID (e.g., CI UID mismatch).
        # Redirect HOME to a writable location for the duration of this session.
        export HOME="/tmp/mojo-home-$(id -u)"
        mkdir -p "${HOME}/.modular"
        export PIXI_HOME="${HOME}/.pixi"
    }
fi

# Ensure the workspace pixi environment is installed.
# With detached-environments = true (.pixi/config.toml), pixi stores each env
# in the per-user cache dir instead of <workspace>/.pixi/envs/. The path varies
# by user and workspace hash, so we probe via `pixi info` rather than checking
# a hardcoded path.
_pixi_default_prefix() {
    pixi info --json 2>/dev/null \
        | python3 -c "
import json, sys
d = json.load(sys.stdin)
for e in d.get('environments_info', []):
    if e.get('name') == 'default':
        print(e.get('prefix', ''))
        break
" 2>/dev/null || echo ""
}
_prefix=$(_pixi_default_prefix)
if [ -z "$_prefix" ] || [ ! -x "$_prefix/bin/mojo" ]; then
    echo "Initializing pixi environment inside container..."
    pixi install
fi
unset _prefix

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
        mkdir -p "$dir" 2>/dev/null || sudo -n mkdir -p "$dir" 2>/dev/null || true
        # If we own the directory already, nothing to do.
        if [ -w "$dir" ]; then
            continue
        fi
        # chmod only succeeds if we already own the directory.
        # When the workspace is bind-mounted as root:root, fall back to
        # sudo chown -R on the specific subdir only (never the whole workspace).
        chmod u+w "$dir" 2>/dev/null || \
            sudo -n chown -R "$(id -u):$(id -g)" "$dir" 2>/dev/null || true
    done
}

_ensure_writable \
    build \
    .pixi \
    datasets \
    lenet5_weights \
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
