#!/bin/bash
set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

# Constants
VERSION="0.1.0"
PACKAGE_NAME="data"
OUTPUT_DIR="dist"

# Validate environment
command -v mojo >/dev/null 2>&1 || {
    echo "❌ ERROR: Mojo not found in PATH"
    echo "Install Mojo from: https://docs.modular.com/mojo/manual/get-started/"
    exit 1
}

# Validate package source exists
if [[ ! -d "src/odyssey/${PACKAGE_NAME}" ]]; then
    echo "❌ ERROR: Package source directory not found: src/odyssey/${PACKAGE_NAME}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build package
echo "Building ${PACKAGE_NAME}-${VERSION}.mojopkg..."
mojo package "src/odyssey/${PACKAGE_NAME}" -o "${OUTPUT_DIR}/${PACKAGE_NAME}-${VERSION}.mojopkg" || {
    echo "❌ ERROR: Package build failed"
    exit 1
}

# Verify package was created
if [[ ! -f "${OUTPUT_DIR}/${PACKAGE_NAME}-${VERSION}.mojopkg" ]]; then
    echo "❌ ERROR: Package file not created"
    exit 1
fi

# Display package info
echo "✅ Package built successfully!"
echo "Package: ${OUTPUT_DIR}/${PACKAGE_NAME}-${VERSION}.mojopkg"
ls -lh "${OUTPUT_DIR}/${PACKAGE_NAME}-${VERSION}.mojopkg"
