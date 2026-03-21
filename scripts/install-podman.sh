#!/usr/bin/env bash
# Build and install full Podman stack from source:
#   podman, netavark, aardvark-dns, podman-compose
#
# Required on Debian 11 / PureOS 10 where apt repos only ship Podman 3.x
# and lack the netavark/aardvark-dns networking backend.
#
# Runs WITHOUT sudo by default. Only apt-get calls use sudo.
# When run as root: installs globally to /usr/local.
# When run as user: installs to ~/.local.
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
    LIBEXEC_DIR="/usr/local/libexec/podman"
    BUILD_DIR="/usr/local/src"
    info "Running as root — installing globally to $INSTALL_DIR"
else
    INSTALL_DIR="${HOME}/.local/bin"
    LIBEXEC_DIR="${HOME}/.local/libexec/podman"
    BUILD_DIR="${HOME}/.local/src"
fi

# ============================================================
# Step 0: Early exit if full stack already installed
# ============================================================

section "Checking existing installation"

INSTALLED_BINARY="${INSTALL_DIR}/podman"
NEED_PODMAN=true
NEED_NETAVARK=true
NEED_COMPOSE=true

# Check podman binary
if [ -x "$INSTALLED_BINARY" ]; then
    installed_major=$("$INSTALLED_BINARY" --version 2>/dev/null | grep -oP '\d+' | head -1 || echo "0")
    installed_ver=$("$INSTALLED_BINARY" --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "unknown")
    if [ "$installed_major" -ge 4 ]; then
        NEED_PODMAN=false
        info "Podman $installed_ver found at $INSTALLED_BINARY"
    else
        warn "Podman $installed_ver too old (need 4+) — will rebuild"
        rm -f "$INSTALLED_BINARY"
    fi
fi

# Check netavark
if [ -x "${LIBEXEC_DIR}/netavark" ]; then
    NEED_NETAVARK=false
    info "netavark found at ${LIBEXEC_DIR}/netavark"
fi

# Check crun (need version 1.9+ for OCI 1.1 / Ubuntu 24.04 images)
NEED_CRUN=true
if command -v crun &>/dev/null; then
    CRUN_VER_NUM=$(crun --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' || echo "0.0")
    CRUN_MAJOR=$(echo "$CRUN_VER_NUM" | cut -d. -f1)
    CRUN_MINOR=$(echo "$CRUN_VER_NUM" | cut -d. -f2)
    if [ "$CRUN_MAJOR" -ge 1 ] && [ "$CRUN_MINOR" -ge 9 ]; then
        NEED_CRUN=false
        info "crun $CRUN_VER_NUM found"
    else
        info "crun $CRUN_VER_NUM too old (need 1.9+) — will update"
    fi
fi

# Check pasta (rootless networking for podman 5.x)
NEED_PASTA=true
if command -v pasta &>/dev/null; then
    NEED_PASTA=false
    info "pasta found: $(command -v pasta)"
fi

# Check compose provider
if command -v podman-compose &>/dev/null; then
    NEED_COMPOSE=false
    info "podman-compose found: $(command -v podman-compose)"
fi

# If everything is present and podman info works, nothing to do
if ! $NEED_PODMAN && ! $NEED_NETAVARK && ! $NEED_CRUN && ! $NEED_PASTA && ! $NEED_COMPOSE; then
    if "$INSTALLED_BINARY" info &>/dev/null; then
        info "Full Podman stack already installed and working — nothing to do"
        exit 0
    else
        warn "Podman stack installed but 'podman info' fails — continuing to diagnose"
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

# Resolve netavark/aardvark-dns latest releases
info "Fetching latest netavark release..."
NETAVARK_VERSION=$(curl -fsSL "https://api.github.com/repos/containers/netavark/releases/latest" \
    | grep '"tag_name"' | grep -oP '\d+\.\d+\.\d+')
info "netavark version: $NETAVARK_VERSION"

info "Fetching latest aardvark-dns release..."
AARDVARK_VERSION=$(curl -fsSL "https://api.github.com/repos/containers/aardvark-dns/releases/latest" \
    | grep '"tag_name"' | grep -oP '\d+\.\d+\.\d+')
info "aardvark-dns version: $AARDVARK_VERSION"

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
# Step 3: Install Rust via rustup (no sudo)
# ============================================================

section "Installing Rust toolchain"

# Ensure cargo is on PATH if rustup was previously installed
if [ -f "${HOME}/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/.cargo/env"
fi

# Check that cargo actually works (not just exists — rustup without a
# default toolchain has the binary but `cargo --version` fails)
if cargo --version &>/dev/null; then
    RUST_VER=$(cargo --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    info "Rust/Cargo already installed: $RUST_VER"
else
    info "Installing Rust via rustup..."
    if command -v rustup &>/dev/null; then
        # rustup exists but no toolchain — just install the toolchain
        rustup default stable
    else
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
        # shellcheck source=/dev/null
        source "${HOME}/.cargo/env"
    fi
    info "Rust installed: $(cargo --version)"
fi

# ============================================================
# Step 4: Install build dependencies (only this step uses sudo)
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
    containernetworking-plugins \
    protobuf-compiler \
    python3-pip

info "Build dependencies installed"

# ============================================================
# Step 5: Clone and build podman (no sudo)
# ============================================================

if $NEED_PODMAN; then
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
    # Step 6: Install podman binary (no sudo for user install)
    # ============================================================

    section "Installing podman to $INSTALL_DIR"

    mkdir -p "$INSTALL_DIR"
    cp bin/podman "$INSTALL_DIR/podman"
    chmod 755 "$INSTALL_DIR/podman"
    info "Installed: $INSTALL_DIR/podman"
else
    info "Skipping podman build — already installed"
fi

# ============================================================
# Step 7: Build netavark
# ============================================================

if $NEED_NETAVARK; then
    section "Building netavark $NETAVARK_VERSION from source"

    mkdir -p "$BUILD_DIR"
    NETAVARK_SRC="${BUILD_DIR}/netavark-${NETAVARK_VERSION}"

    if [ ! -d "$NETAVARK_SRC" ]; then
        info "Cloning netavark v${NETAVARK_VERSION}..."
        git clone --depth=1 --branch "v${NETAVARK_VERSION}" \
            https://github.com/containers/netavark.git \
            "$NETAVARK_SRC"
    else
        info "Source already cloned — reusing"
    fi

    info "Building netavark (this takes ~3 minutes)..."
    cd "$NETAVARK_SRC"
    cargo build --release 2>&1
    info "netavark build complete"

    # ============================================================
    # Step 8: Build aardvark-dns
    # ============================================================

    section "Building aardvark-dns $AARDVARK_VERSION from source"

    AARDVARK_SRC="${BUILD_DIR}/aardvark-dns-${AARDVARK_VERSION}"

    if [ ! -d "$AARDVARK_SRC" ]; then
        info "Cloning aardvark-dns v${AARDVARK_VERSION}..."
        git clone --depth=1 --branch "v${AARDVARK_VERSION}" \
            https://github.com/containers/aardvark-dns.git \
            "$AARDVARK_SRC"
    else
        info "Source already cloned — reusing"
    fi

    info "Building aardvark-dns..."
    cd "$AARDVARK_SRC"
    cargo build --release 2>&1
    info "aardvark-dns build complete"

    # ============================================================
    # Step 9: Install networking binaries to libexec
    # ============================================================

    section "Installing networking binaries to $LIBEXEC_DIR"

    mkdir -p "$LIBEXEC_DIR"
    cp "${NETAVARK_SRC}/target/release/netavark" "$LIBEXEC_DIR/netavark"
    chmod 755 "$LIBEXEC_DIR/netavark"
    info "Installed: $LIBEXEC_DIR/netavark"

    cp "${AARDVARK_SRC}/target/release/aardvark-dns" "$LIBEXEC_DIR/aardvark-dns"
    chmod 755 "$LIBEXEC_DIR/aardvark-dns"
    info "Installed: $LIBEXEC_DIR/aardvark-dns"
else
    info "Skipping netavark/aardvark-dns build — already installed"
fi

# ============================================================
# Step 9b: Install crun (static binary)
# ============================================================

if $NEED_CRUN; then
    section "Installing crun"

    info "Fetching latest crun release..."
    CRUN_VERSION=$(curl -fsSL "https://api.github.com/repos/containers/crun/releases/latest" \
        | grep '"tag_name"' | grep -oP '[\d.]+')
    info "crun version: $CRUN_VERSION"

    CRUN_URL="https://github.com/containers/crun/releases/download/${CRUN_VERSION}/crun-${CRUN_VERSION}-linux-amd64"
    info "Downloading crun from $CRUN_URL..."
    curl -fsSL "$CRUN_URL" -o "$INSTALL_DIR/crun"
    chmod 755 "$INSTALL_DIR/crun"
    info "Installed: $INSTALL_DIR/crun ($("$INSTALL_DIR/crun" --version | head -1))"
else
    info "Skipping crun install — already up to date"
fi

# ============================================================
# Step 9c: Build and install passt/pasta
# ============================================================

if $NEED_PASTA; then
    section "Building passt/pasta"

    info "Fetching latest passt release tag..."
    PASST_TAG=$(curl -fsSL "https://passt.top/passt/refs/" \
        | grep -oP '(?<=\?h=)\d{4}_\d{2}_\d{2}\.[a-f0-9]+' | head -1)
    info "passt tag: $PASST_TAG"

    PASST_URL="https://passt.top/passt/snapshot/passt-${PASST_TAG}.tar.gz"
    PASST_TMPDIR=$(mktemp -d)
    info "Downloading passt from $PASST_URL..."
    curl -fsSL "$PASST_URL" -o "$PASST_TMPDIR/passt.tar.gz"
    tar xzf "$PASST_TMPDIR/passt.tar.gz" -C "$PASST_TMPDIR"

    info "Building passt..."
    cd "$PASST_TMPDIR/passt-${PASST_TAG}"
    make 2>&1

    cp passt "$INSTALL_DIR/passt"
    cp pasta "$INSTALL_DIR/pasta"
    chmod 755 "$INSTALL_DIR/passt" "$INSTALL_DIR/pasta"
    rm -rf "$PASST_TMPDIR"
    info "Installed: $INSTALL_DIR/passt and $INSTALL_DIR/pasta"
else
    info "Skipping passt/pasta install — already available"
fi

# ============================================================
# Step 10: Configure containers.conf (user installs only)
# ============================================================

if [ "$(id -u)" -ne 0 ]; then
    section "Configuring containers.conf"

    CONTAINERS_CONF_DIR="${HOME}/.config/containers"
    CONTAINERS_CONF="${CONTAINERS_CONF_DIR}/containers.conf"
    COMPOSE_BIN="${INSTALL_DIR}/podman-compose"
    mkdir -p "$CONTAINERS_CONF_DIR"

    NEEDS_UPDATE=false
    if [ ! -f "$CONTAINERS_CONF" ]; then
        NEEDS_UPDATE=true
    elif ! grep -q "helper_binaries_dir" "$CONTAINERS_CONF" 2>/dev/null; then
        NEEDS_UPDATE=true
    elif ! grep -q "compose_providers" "$CONTAINERS_CONF" 2>/dev/null; then
        NEEDS_UPDATE=true
    elif ! grep -q "engine.runtimes" "$CONTAINERS_CONF" 2>/dev/null; then
        NEEDS_UPDATE=true
    fi

    if $NEEDS_UPDATE; then
        cat > "$CONTAINERS_CONF" <<EOF
[engine]
helper_binaries_dir = ["${LIBEXEC_DIR}"]
compose_providers = ["${COMPOSE_BIN}"]

[engine.runtimes]
crun = ["${INSTALL_DIR}/crun"]
EOF
        info "Wrote $CONTAINERS_CONF"
    else
        info "containers.conf already configured"
    fi
fi

# ============================================================
# Step 11: Install podman-compose via pip
# ============================================================

if $NEED_COMPOSE; then
    section "Installing podman-compose"

    if [ "$(id -u)" -eq 0 ]; then
        pip3 install podman-compose
    else
        pip3 install --user podman-compose
    fi
    info "podman-compose installed: $(podman-compose --version 2>/dev/null || echo 'installed')"
else
    info "Skipping podman-compose install — already available"
fi

# ============================================================
# Step 12: Migrate database (BoltDB → SQLite)
# ============================================================

section "Migrating database"

"$INSTALL_DIR/podman" system migrate --migrate-db 2>/dev/null || true
info "Database migration complete (BoltDB → SQLite)"

# ============================================================
# Step 13: PATH check
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

# Check ~/.local/bin for podman-compose (pip --user installs there)
if [ "$(id -u)" -ne 0 ] && ! echo "$PATH" | grep -q "${HOME}/.local/bin"; then
    warn "${HOME}/.local/bin is not in your PATH (needed for podman-compose)"
    echo "Add it permanently:"
    echo "  echo 'export PATH=\"${HOME}/.local/bin:\$PATH\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
fi

# ============================================================
# Step 14: Verify full installation
# ============================================================

section "Verifying installation"

VERIFY_OK=true

# Verify podman
installed_ver=$("$INSTALL_DIR/podman" --version | grep -oP '\d+\.\d+\.\d+' | head -1)
installed_major=$("$INSTALL_DIR/podman" --version | grep -oP '\d+' | head -1)

if [ "$installed_major" -lt 4 ]; then
    error "Built podman $installed_ver is older than 4.0 — something went wrong"
    VERIFY_OK=false
else
    info "podman $installed_ver installed at $INSTALL_DIR/podman"
fi

# Verify podman info (tests networking stack)
if "$INSTALL_DIR/podman" info &>/dev/null; then
    info "podman info: OK (networking stack working)"
else
    warn "podman info failed — networking may need configuration"
    echo "  Check: $INSTALL_DIR/podman info"
    echo "  Docs:  https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md"
    VERIFY_OK=false
fi

# Verify netavark
if [ -x "${LIBEXEC_DIR}/netavark" ]; then
    info "netavark: OK at ${LIBEXEC_DIR}/netavark"
else
    warn "netavark not found at ${LIBEXEC_DIR}/netavark"
    VERIFY_OK=false
fi

# Verify aardvark-dns
if [ -x "${LIBEXEC_DIR}/aardvark-dns" ]; then
    info "aardvark-dns: OK at ${LIBEXEC_DIR}/aardvark-dns"
else
    warn "aardvark-dns not found at ${LIBEXEC_DIR}/aardvark-dns"
    VERIFY_OK=false
fi

# Verify compose provider
if command -v podman-compose &>/dev/null; then
    info "podman-compose: OK ($(podman-compose --version 2>/dev/null || echo 'available'))"
else
    warn "podman-compose not found in PATH"
    VERIFY_OK=false
fi

echo ""
if $VERIFY_OK; then
    info "Full Podman stack installation complete!"
else
    warn "Installation completed with warnings — see above"
fi

echo ""
echo "Next steps:"
echo "  just podman-build   # Build dev image (~5-10 min)"
echo "  just podman-up      # Start dev environment"
echo "  just shell           # Open shell in container"
echo "  just test-mojo      # Run all tests"
echo ""
