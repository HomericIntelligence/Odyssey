#!/bin/bash
set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

VERSION="0.1.0"
PACKAGE_NAME="utils"
OUTPUT_DIR="dist"

# Validate Mojo is installed
command -v mojo >/dev/null 2>&1 || { echo "❌ Mojo not found in PATH"; exit 1; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

cd "$PROJECT_ROOT"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build package
echo "Building ${PACKAGE_NAME}-${VERSION}.mojopkg..."
mojo package "src/odyssey/${PACKAGE_NAME}" -o "${OUTPUT_DIR}/${PACKAGE_NAME}-${VERSION}.mojopkg"

# Verify package was created
if [[ ! -f "${OUTPUT_DIR}/${PACKAGE_NAME}-${VERSION}.mojopkg" ]]; then
    echo "❌ ERROR: Package file not created"
    exit 1
fi

echo "✅ Package built successfully: ${OUTPUT_DIR}/${PACKAGE_NAME}-${VERSION}.mojopkg"
ls -lh "${OUTPUT_DIR}/${PACKAGE_NAME}-${VERSION}.mojopkg"
