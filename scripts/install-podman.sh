#!/usr/bin/env bash
# Build and install full Podman 4.0+ from source.
# Required on Debian 11 / PureOS 10 where apt repos only ship 3.x.
#
# Runs WITHOUT sudo by default. Only apt-get calls use sudo.
# When run as root: installs globally to /usr/local/bin.
# When run as user: installs to ~/.local/bin.
#
# Usage:
#   bash scripts/install-podman.sh            # latest version
#   bash scripts/install-podman.sh 5.8.1      # specific version
#   sudo bash scripts/install-podman.sh       # global install

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${GREEN}=== $* ===${NC}"; }

PODMAN_VERSION="${1:-}"
GO_VERSION="${2:-}"

# Install paths: global when root, user-local otherwise
if [ "$(id -u)" -eq 0 ]; then
    INSTALL_DIR="/usr/local/bin"
    BUILD_DIR="/usr/local/src"
    info "Running as root — installing globally to $INSTALL_DIR"
else
    INSTALL_DIR="${HOME}/.local/bin"
    BUILD_DIR="${HOME}/.local/src"
fi

# ============================================================
# Step 0: Early exit if already installed
# ============================================================

section "Checking existing Podman installation"

INSTALLED_BINARY="${INSTALL_DIR}/podman"

# podman-remote fails `podman info` with a socket error; full podman succeeds.
if [ -x "$INSTALLED_BINARY" ]; then
    installed_major=$("$INSTALLED_BINARY" --version 2>/dev/null | grep -oP '\d+' | head -1 || echo "0")
    installed_ver=$("$INSTALLED_BINARY" --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "unknown")
    if [ "$installed_major" -ge 4 ] && "$INSTALLED_BINARY" info &>/dev/null; then
        info "Full Podman $installed_ver already installed at $INSTALLED_BINARY — nothing to do"
        exit 0
    elif [ "$installed_major" -ge 4 ]; then
        warn "Podman $installed_ver at $INSTALLED_BINARY cannot run (podman-remote or broken) — rebuilding"
        rm -f "$INSTALLED_BINARY"
    fi
fi

# ============================================================
# Step 1: Resolve versions
# ============================================================

section "Resolving versions"

if [ -z "$PODMAN_VERSION" ]; then
    info "Fetching latest podman release..."
    PODMAN_VERSION=$(curl -fsSL "https://api.github.com/repos/containers/podman/releases/latest" \
        | grep '"tag_name"' | grep -oP '\d+\.\d+\.\d+')
fi
info "Podman version: $PODMAN_VERSION"

if [ -z "$GO_VERSION" ]; then
    info "Fetching required Go version from podman go.mod..."
    GO_VERSION=$(curl -fsSL "https://raw.githubusercontent.com/containers/podman/v${PODMAN_VERSION}/go.mod" \
        | grep "^go " | grep -oP '\d+\.\d+(\.\d+)?')
fi
info "Go version: $GO_VERSION"

# ============================================================
# Step 2: Install Go (no sudo — downloads to BUILD_DIR)
# ============================================================

section "Installing Go $GO_VERSION"

GO_INSTALL_DIR="${BUILD_DIR}/go-${GO_VERSION}"
GO_BIN="${GO_INSTALL_DIR}/bin/go"

if [ -x "$GO_BIN" ] && "$GO_BIN" version 2>/dev/null | grep -q "$GO_VERSION"; then
    info "Go $GO_VERSION already installed at $GO_BIN"
else
    GO_TARBALL="go${GO_VERSION}.linux-amd64.tar.gz"
    GO_URL="https://go.dev/dl/${GO_TARBALL}"
    info "Downloading Go from $GO_URL..."
    TMPDIR=$(mktemp -d)
    trap 'rm -rf "$TMPDIR"' EXIT
    curl -fsSL --progress-bar "$GO_URL" -o "$TMPDIR/$GO_TARBALL"
    info "Extracting..."
    mkdir -p "$GO_INSTALL_DIR"
    tar -xz -C "$GO_INSTALL_DIR" -f "$TMPDIR/$GO_TARBALL" --strip-components=1
    info "Go $GO_VERSION installed at $GO_INSTALL_DIR"
fi

export PATH="${GO_INSTALL_DIR}/bin:${PATH}"
go version

# ============================================================
# Step 3: Install build dependencies (only this step uses sudo)
# ============================================================

section "Installing build dependencies"

if [ "$(id -u)" -ne 0 ]; then
    info "apt-get requires sudo for this step only..."
fi

sudo apt-get install -y \
    libgpgme-dev \
    libassuan-dev \
    libbtrfs-dev \
    libdevmapper-dev \
    libseccomp-dev \
    pkg-config \
    iptables \
    slirp4netns \
    fuse-overlayfs \
    containernetworking-plugins

info "Build dependencies installed"

# ============================================================
# Step 4: Clone and build podman (no sudo)
# ============================================================

section "Building Podman $PODMAN_VERSION from source"

mkdir -p "$BUILD_DIR"
PODMAN_SRC="${BUILD_DIR}/podman-${PODMAN_VERSION}"

if [ ! -d "$PODMAN_SRC" ]; then
    info "Cloning podman v${PODMAN_VERSION}..."
    git clone --depth=1 --branch "v${PODMAN_VERSION}" \
        https://github.com/containers/podman.git \
        "$PODMAN_SRC"
else
    info "Source already cloned — reusing (delete $PODMAN_SRC to force re-clone)"
fi

info "Building (this takes ~5 minutes)..."
cd "$PODMAN_SRC"
make GOFLAGS="-trimpath" \
    CGO_ENABLED=1 \
    BUILDTAGS="exclude_graphdriver_devicemapper selinux seccomp" \
    binaries 2>&1

info "Build complete"

# ============================================================
# Step 5: Install binary (no sudo for user install)
# ============================================================

section "Installing to $INSTALL_DIR"

mkdir -p "$INSTALL_DIR"
cp bin/podman "$INSTALL_DIR/podman"
chmod 755 "$INSTALL_DIR/podman"
info "Installed: $INSTALL_DIR/podman"

# ============================================================
# Step 6: PATH check
# ============================================================

section "Checking PATH"

if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    warn "$INSTALL_DIR is not in your PATH"
    if [ "$(id -u)" -ne 0 ]; then
        echo "Add it permanently:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
    fi
fi

# ============================================================
# Step 7: Verify
# ============================================================

section "Verifying installation"

installed_ver=$("$INSTALL_DIR/podman" --version | grep -oP '\d+\.\d+\.\d+' | head -1)
installed_major=$("$INSTALL_DIR/podman" --version | grep -oP '\d+' | head -1)

if [ "$installed_major" -lt 4 ]; then
    error "Built podman $installed_ver is older than 4.0 — something went wrong"
    exit 1
fi

info "podman $installed_ver installed at $INSTALL_DIR/podman"

if "$INSTALL_DIR/podman" info &>/dev/null; then
    info "Podman is working (rootless)"
else
    warn "podman info failed — rootless networking may need configuration:"
    echo "  https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md"
fi

echo ""
info "Installation complete!"
echo ""
echo "Next steps:"
echo "  pixi run just podman-build   # Build dev image (~5-10 min)"
echo "  pixi run just test-mojo      # Run all tests"
echo ""
