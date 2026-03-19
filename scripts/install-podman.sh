#!/usr/bin/env bash
# Install Podman 4.0+ via static binary from GitHub releases.
# Works on any Linux x86_64 regardless of distro/GLIBC version.
#
# Usage:
#   ./scripts/install-podman.sh [version]
#   ./scripts/install-podman.sh 5.8.1

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${GREEN}=== $* ===${NC}"; }

# ============================================================
# Step 0: Early exit if already installed
# ============================================================

section "Checking existing Podman installation"

if command -v podman &>/dev/null; then
    major=$(podman --version | grep -oP '\d+' | head -1)
    version=$(podman --version | grep -oP '\d+\.\d+\.\d+' | head -1)
    if [ "$major" -ge 4 ]; then
        info "Podman $version already installed — nothing to do"
        exit 0
    else
        warn "Podman $version found but 4.0+ required — will install newer version"
    fi
fi

# ============================================================
# Step 1: Determine version to install
# ============================================================

section "Determining version"

if [ -n "${1:-}" ]; then
    VERSION="$1"
    info "Using requested version: $VERSION"
else
    info "Fetching latest release from GitHub..."
    VERSION=$(curl -fsSL "https://api.github.com/repos/containers/podman/releases/latest" \
        | grep '"tag_name"' | grep -oP '\d+\.\d+\.\d+')
    info "Latest version: $VERSION"
fi

INSTALL_DIR="${HOME}/.local/bin"
INSTALL_PATH="${INSTALL_DIR}/podman"

# ============================================================
# Step 2: Download static binary
# ============================================================

section "Downloading Podman $VERSION static binary"

DOWNLOAD_URL="https://github.com/containers/podman/releases/download/v${VERSION}/podman-remote-static-linux_amd64.tar.gz"
info "URL: $DOWNLOAD_URL"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

info "Downloading..."
curl -fsSL --progress-bar "$DOWNLOAD_URL" -o "$TMPDIR/podman.tar.gz"

info "Extracting..."
tar -xz -C "$TMPDIR" -f "$TMPDIR/podman.tar.gz"

# Find the binary (name varies by version)
BINARY=$(find "$TMPDIR" -name "podman*" -type f -perm /111 | head -1)
if [ -z "$BINARY" ]; then
    error "Could not find podman binary in archive"
    ls -la "$TMPDIR/"
    exit 1
fi

info "Found binary: $(basename "$BINARY")"

# ============================================================
# Step 3: Install to ~/.local/bin
# ============================================================

section "Installing to $INSTALL_PATH"

mkdir -p "$INSTALL_DIR"
cp "$BINARY" "$INSTALL_PATH"
chmod 755 "$INSTALL_PATH"

info "Installed: $INSTALL_PATH"

# ============================================================
# Step 4: PATH check
# ============================================================

section "Checking PATH"

if ! echo "$PATH" | grep -q "${HOME}/.local/bin"; then
    warn "~/.local/bin is not in your PATH"
    echo ""
    echo "Add it to your shell config:"
    echo ""
    echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo ""
    echo "Or for this session only:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    export PATH="${HOME}/.local/bin:${PATH}"
    info "Added to PATH for this session"
fi

# ============================================================
# Step 5: Verify
# ============================================================

section "Verifying installation"

installed_ver=$("$INSTALL_PATH" --version | grep -oP '\d+\.\d+\.\d+' | head -1)
installed_major=$("$INSTALL_PATH" --version | grep -oP '\d+' | head -1)

info "Installed: podman $installed_ver at $INSTALL_PATH"

if [ "$installed_major" -lt 4 ]; then
    error "Version $installed_ver is older than 4.0 — something went wrong"
    exit 1
fi

echo ""
info "Installation complete!"
echo ""
echo "Next steps:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\"   # if not already in PATH"
echo "  pixi run just podman-build               # Build dev image (~5-10 min)"
echo "  pixi run just test-mojo                  # Run all tests"
echo ""
