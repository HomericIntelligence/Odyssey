#!/usr/bin/env bash
# Install Podman 4.0+ from official repositories on Debian/Ubuntu
# Supports: Debian 11 (Bullseye), Debian 12 (Bookworm), Ubuntu 20.04+
#
# Usage:
#   ./scripts/install-podman.sh
#
# What this does:
#   1. Checks if Podman 4.0+ is already installed — exits early if so
#   2. Adds the official podman.io (Kubic/unstable) apt repository
#   3. Installs podman via apt
#   4. Verifies the installation
#
# Fallback: If apt method fails, see the "Manual Installation" section below.

set -euo pipefail

# ============================================================
# Helpers
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${GREEN}=== $* ===${NC}"; }

print_fallback_instructions() {
    cat <<'EOF'

Manual Installation Options
============================

Option A: Build from source
  See: https://podman.io/docs/installation#building-from-scratch
  Requires: golang, libgpgme-dev, libbtrfs-dev, libassuan-dev, etc.

Option B: Static binary from GitHub releases
  PODMAN_VERSION="5.3.1"  # or latest from https://github.com/containers/podman/releases
  curl -fsSL "https://github.com/containers/podman/releases/download/v${PODMAN_VERSION}/podman-remote-static-linux_amd64.tar.gz" \
      | tar -xz
  sudo install -m 755 bin/podman-remote-static-linux_amd64 /usr/local/bin/podman
  Note: The remote binary requires a running podman socket (systemd service)

Option C: Use Docker instead
  Podman recipes use --userns=keep-id for rootless operation.
  Docker requires running as root or being in the 'docker' group.
  Existing docker-* justfile recipes remain available.

EOF
}

# ============================================================
# Version check
# ============================================================

check_podman_version() {
    if ! command -v podman &>/dev/null; then
        return 1
    fi
    local version major
    version=$(podman --version | grep -oP '\d+\.\d+' | head -1)
    major=$(echo "$version" | cut -d. -f1)
    if [ "$major" -lt 4 ]; then
        warn "Podman $version found but 4.0+ required"
        return 1
    fi
    info "Podman $version already installed — nothing to do"
    return 0
}

# ============================================================
# Step 0: Early exit if already installed
# ============================================================

section "Checking existing Podman installation"
if check_podman_version; then
    podman --version
    exit 0
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

info "OS: $NAME $VERSION_ID (ID: $ID)"

if [[ "$ID" != "debian" && "$ID_LIKE" != *"debian"* && "$ID" != "ubuntu" ]]; then
    error "This script supports Debian/Ubuntu only. Detected: $ID"
    echo ""
    echo "For other distros, see: https://podman.io/docs/installation"
    exit 1
fi

# ============================================================
# Step 2: Install via official Kubic repository
# ============================================================

section "Adding official Podman repository (kubic)"

info "This requires sudo. You may be prompted for your password."
echo ""

# Determine repository URL based on distro
if [[ "$ID" == "ubuntu" ]]; then
    # Ubuntu uses a different repo path
    KUBIC_BASE="https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/unstable/xUbuntu_${VERSION_ID}"
elif [[ "$ID" == "debian" || "$ID_LIKE" == *"debian"* ]]; then
    KUBIC_BASE="https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/unstable/Debian_${VERSION_ID}"
else
    error "Unsupported distro: $ID"
    exit 1
fi

info "Repository: $KUBIC_BASE"

# Check if repo is reachable
if ! curl -fsS --head "${KUBIC_BASE}/Release" &>/dev/null; then
    warn "Kubic repo not reachable for ${ID} ${VERSION_ID}"
    warn "Your distro version may not be supported yet."
    echo ""
    print_fallback_instructions
    exit 1
fi

# Add apt sources list
echo "deb ${KUBIC_BASE}/ /" \
    | sudo tee /etc/apt/sources.list.d/devel:kubic:libcontainers:unstable.list >/dev/null

# Add GPG key
curl -fsSL "${KUBIC_BASE}/Release.key" \
    | gpg --dearmor \
    | sudo tee /etc/apt/trusted.gpg.d/devel_kubic_libcontainers_unstable.gpg >/dev/null

info "Repository added successfully"

# Update apt and install
info "Running apt update..."
sudo apt-get update -qq

info "Installing podman..."
sudo apt-get install -y podman

# ============================================================
# Step 3: Verify installation
# ============================================================

section "Verifying installation"

if ! command -v podman &>/dev/null; then
    error "podman not found after installation — something went wrong"
    exit 1
fi

podman --version

PODMAN_VER=$(podman --version | grep -oP '\d+\.\d+' | head -1)
PODMAN_MAJOR=$(echo "$PODMAN_VER" | cut -d. -f1)

if [ "$PODMAN_MAJOR" -lt 4 ]; then
    error "Installed Podman $PODMAN_VER is older than 4.0"
    echo ""
    print_fallback_instructions
    exit 1
fi

info "Podman $PODMAN_VER installed successfully"

# Basic info check
section "Podman info"
podman info --format='Host OS: {{.Host.OS}}
Kernel: {{.Host.Kernel}}
Rootless: {{.Host.Security.Rootless}}' 2>/dev/null || podman info | head -20

# Rootless check
section "Rootless configuration"
if podman info 2>/dev/null | grep -q "rootless: true"; then
    info "Running in rootless mode — recommended for development"
else
    warn "Not running in rootless mode. For rootless setup:"
    echo "  https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md"
fi

echo ""
info "Installation complete!"
echo ""
echo "Next steps:"
echo "  just podman-build    # Build the development image (~5-10 min first time)"
echo "  just podman-mojo --version    # Verify mojo works inside container"
echo "  just podman-shell    # Open interactive shell"
echo ""
