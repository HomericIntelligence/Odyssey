#!/usr/bin/env bash
# Install Podman 4.0+ from official repositories on Debian/Ubuntu/PureOS
#
# Usage:
#   ./scripts/install-podman.sh

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
        podman --version
        exit 0
    else
        warn "Podman $version found but 4.0+ required — upgrading"
    fi
fi

# ============================================================
# Step 1: Detect OS
# ============================================================

section "Detecting OS"

if [ ! -f /etc/os-release ]; then
    error "Cannot detect OS: /etc/os-release not found"
    exit 1
fi

# shellcheck source=/dev/null
. /etc/os-release

info "OS: ${NAME:-unknown} ${VERSION_ID:-unknown} (ID: ${ID:-unknown})"

# Map distro to Kubic repo path
# PureOS 10 = Debian 11 base; PureOS has no ID_LIKE so we handle it explicitly
case "${ID:-}" in
    debian)
        KUBIC_DISTRO="Debian_${VERSION_ID}"
        ;;
    ubuntu)
        KUBIC_DISTRO="xUbuntu_${VERSION_ID}"
        ;;
    pureos)
        # PureOS 10 (Byzantium) is based on Debian 11
        KUBIC_DISTRO="Debian_11"
        ;;
    *)
        # Try ID_LIKE fallback
        if [[ "${ID_LIKE:-}" == *"debian"* ]]; then
            KUBIC_DISTRO="Debian_${VERSION_ID}"
        else
            error "Unsupported distro: ${ID:-unknown}"
            echo "For other distros, see: https://podman.io/docs/installation"
            exit 1
        fi
        ;;
esac

info "Mapped to Kubic distro: $KUBIC_DISTRO"

# ============================================================
# Step 2: Find a working Kubic repository
# ============================================================

section "Finding Podman repository"

KUBIC_CANDIDATES=(
    "https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/unstable/${KUBIC_DISTRO}"
    "https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/${KUBIC_DISTRO}"
)

KUBIC_BASE=""
for candidate in "${KUBIC_CANDIDATES[@]}"; do
    if curl -fsS --head "${candidate}/Release" &>/dev/null; then
        KUBIC_BASE="$candidate"
        info "Using repository: $KUBIC_BASE"
        break
    else
        warn "Not available: $candidate"
    fi
done

if [ -z "$KUBIC_BASE" ]; then
    error "No Kubic repository found for ${KUBIC_DISTRO}"
    echo ""
    echo "Manual alternatives:"
    echo "  Option A: Static binary (no sudo needed)"
    echo "    PODMAN_VERSION=5.3.1"
    echo "    curl -fsSL https://github.com/containers/podman/releases/download/v\${PODMAN_VERSION}/podman-remote-static-linux_amd64.tar.gz | tar -xz"
    echo "    sudo install -m755 bin/podman-remote-static-linux_amd64 /usr/local/bin/podman"
    echo ""
    echo "  Option B: Build from source"
    echo "    https://podman.io/docs/installation#building-from-scratch"
    exit 1
fi

# ============================================================
# Step 3: Install via apt
# ============================================================

section "Installing Podman"
info "This requires sudo. You may be prompted for your password."

echo "deb ${KUBIC_BASE}/ /" \
    | sudo tee /etc/apt/sources.list.d/devel:kubic:libcontainers.list >/dev/null

curl -fsSL "${KUBIC_BASE}/Release.key" \
    | gpg --dearmor \
    | sudo tee /etc/apt/trusted.gpg.d/devel_kubic_libcontainers.gpg >/dev/null

info "Running apt update..."
sudo apt-get update -qq

info "Installing podman..."
sudo apt-get install -y podman

# ============================================================
# Step 4: Verify
# ============================================================

section "Verifying installation"

if ! command -v podman &>/dev/null; then
    error "podman not found after installation — something went wrong"
    exit 1
fi

podman --version

installed_major=$(podman --version | grep -oP '\d+' | head -1)
installed_ver=$(podman --version | grep -oP '\d+\.\d+\.\d+' | head -1)

if [ "$installed_major" -lt 4 ]; then
    error "Installed Podman $installed_ver is older than 4.0"
    exit 1
fi

info "Podman $installed_ver installed successfully"

section "Rootless check"
if podman info 2>/dev/null | grep -q "rootless: true"; then
    info "Running in rootless mode — good"
else
    warn "Not running in rootless mode. For rootless setup:"
    echo "  https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md"
fi

echo ""
info "Installation complete!"
echo ""
echo "Next steps:"
echo "  pixi run just podman-build   # Build the development image (~5-10 min first time)"
echo "  pixi run just shell          # Open interactive shell"
echo "  pixi run just test-mojo      # Run all tests inside container"
echo ""
