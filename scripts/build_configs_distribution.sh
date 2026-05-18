#!/usr/bin/env bash
#
# Build distributable configuration package
#
# Creates configs-VERSION.tar.gz containing:
# - All YAML configuration files
# - Documentation (README.md, MIGRATION.md)
# - Templates for new papers/experiments
#
# Usage:
#   ./scripts/build_configs_distribution.sh [VERSION]
#
# Output:
#   dist/configs-VERSION.tar.gz

set -euo pipefail

VERSION="${1:-0.1.0}"
DIST_DIR="dist"
BUILD_DIR="${DIST_DIR}/configs-${VERSION}"
TARBALL="${DIST_DIR}/configs-${VERSION}.tar.gz"

echo "Building configs distribution package v${VERSION}..."

# Create clean build directory
rm -rf "${BUILD_DIR}" "${TARBALL}"
mkdir -p "${BUILD_DIR}"

# Copy configuration files
echo "Copying configuration files..."
cp -r configs "${BUILD_DIR}/"

# Copy integration utilities
echo "Copying integration utilities..."
mkdir -p "${BUILD_DIR}/utils"
cp src/projectodyssey/utils/config_loader.mojo "${BUILD_DIR}/utils/"
cp src/projectodyssey/utils/config.mojo "${BUILD_DIR}/utils/"

# Copy documentation
echo "Copying documentation..."
cp README.md "${BUILD_DIR}/" || echo "Warning: README.md not found"
cp LICENSE "${BUILD_DIR}/" || echo "Warning: LICENSE not found"

# Copy example
echo "Copying examples..."
mkdir -p "${BUILD_DIR}/examples"
if [ -f "papers/_template/examples/train.mojo" ]; then
    cp papers/_template/examples/train.mojo "${BUILD_DIR}/examples/"
fi

# Create installation instructions
cat > "${BUILD_DIR}/INSTALL.md" << 'EOF'
# Configuration Package Installation

## Quick Install

Extract the tarball to your ML Odyssey installation:

```bash
tar -xzf configs-VERSION.tar.gz
cd configs-VERSION
./install.sh
```

## Manual Installation

1. Copy configs/ directory to your ML Odyssey root:
   ```bash
   cp -r configs /path/to/ProjectOdyssey/
   ```

2. Copy utilities to src/projectodyssey/utils/:
   ```bash
   cp utils/*.mojo /path/to/ProjectOdyssey/shared/utils/
   ```

3. Verify installation:
   ```bash
   ./scripts/verify_configs_install.sh
   ```

## Usage

See configs/README.md for usage documentation.

## Environment Variables

Set these for custom paths:
- `ML_ODYSSEY_DATA` - Data directory
- `ML_ODYSSEY_CHECKPOINTS` - Checkpoint directory
- `ML_ODYSSEY_LOGS` - Log directory
- `ML_ODYSSEY_OUTPUT` - Output directory
- `ML_ODYSSEY_CACHE` - Cache directory
EOF

# Create installation script
cat > "${BUILD_DIR}/install.sh" << 'EOF'
#!/usr/bin/env bash
# Install configuration package to ML Odyssey

set -euo pipefail

ML_ODYSSEY_ROOT="${ML_ODYSSEY_ROOT:-.}"

echo "Installing configs package to ${ML_ODYSSEY_ROOT}..."

# Install configs
if [ -d "${ML_ODYSSEY_ROOT}/configs" ]; then
    echo "Warning: configs/ already exists. Backing up to configs.bak/"
    mv "${ML_ODYSSEY_ROOT}/configs" "${ML_ODYSSEY_ROOT}/configs.bak"
fi
cp -r configs "${ML_ODYSSEY_ROOT}/"

# Install utilities (nullglob → empty list when no .mojo files present)
mkdir -p "${ML_ODYSSEY_ROOT}/shared/utils"
shopt -s nullglob
utils_files=(utils/*.mojo)
if [ "${#utils_files[@]}" -gt 0 ]; then
    cp "${utils_files[@]}" "${ML_ODYSSEY_ROOT}/shared/utils/"
fi

# Install examples
mkdir -p "${ML_ODYSSEY_ROOT}/examples"
examples_files=(examples/*.mojo)
if [ "${#examples_files[@]}" -gt 0 ]; then
    cp "${examples_files[@]}" "${ML_ODYSSEY_ROOT}/examples/"
fi
shopt -u nullglob

echo "Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Review configs/README.md for usage"
echo "  2. Set environment variables (see INSTALL.md)"
echo "  3. Run: ./scripts/verify_configs_install.sh"
EOF
chmod +x "${BUILD_DIR}/install.sh"

# Create tarball
echo "Creating tarball..."
cd "${DIST_DIR}"
tar -czf "configs-${VERSION}.tar.gz" "configs-${VERSION}"
cd - > /dev/null

# Calculate checksum
if command -v sha256sum &> /dev/null; then
    sha256sum "${TARBALL}" > "${TARBALL}.sha256"
    echo "Checksum: $(cat ${TARBALL}.sha256)"
fi

# Show package contents
echo ""
echo "Package contents:"
tar -tzf "${TARBALL}" | head -20
echo "..."

# Show summary
TARBALL_SIZE=$(du -h "${TARBALL}" | cut -f1)
echo ""
echo "Distribution package created:"
echo "  File: ${TARBALL}"
echo "  Size: ${TARBALL_SIZE}"
echo "  Version: ${VERSION}"

# Cleanup build directory
rm -rf "${BUILD_DIR}"

echo ""
echo "Build complete!"
