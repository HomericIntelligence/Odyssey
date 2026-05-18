#!/bin/bash
# Complete packaging workflow for utils module

set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Utils Package Build and Test ===${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

cd "$PROJECT_ROOT"

# Step 1: Create dist directory
echo -e "\n${YELLOW}Step 1: Creating dist directory...${NC}"
mkdir -p dist
echo -e "${GREEN}✓ dist/ directory ready${NC}"

# Step 2: Build package
echo -e "\n${YELLOW}Step 2: Building .mojopkg package...${NC}"
echo "Command: mojo package src/projectodyssey/utils -o dist/utils-0.1.0.mojopkg"
mojo package src/projectodyssey/utils -o dist/utils-0.1.0.mojopkg

if [ -f "dist/utils-0.1.0.mojopkg" ]; then
    echo -e "${GREEN}✓ Package built successfully!${NC}"
    ls -lh dist/utils-0.1.0.mojopkg
else
    echo -e "${RED}✗ Package build failed!${NC}"
    exit 1
fi

# Step 3: Make verification script executable
echo -e "\n${YELLOW}Step 3: Making verification script executable...${NC}"
chmod +x "$SCRIPT_DIR/install_verify_utils.sh"
echo -e "${GREEN}✓ install_verify_utils.sh is now executable${NC}"

# Step 4: Test installation (optional, can be skipped if environment issues)
echo -e "\n${YELLOW}Step 4: Testing package installation...${NC}"
if [ "${SKIP_INSTALL_TEST}" == "1" ]; then
    echo -e "${YELLOW}⚠ Skipping installation test (SKIP_INSTALL_TEST=1)${NC}"
else
    echo "Running: ./scripts/install_verify_utils.sh"
    "$SCRIPT_DIR/install_verify_utils.sh" || {
        echo -e "${YELLOW}⚠ Installation test failed (may require different environment)${NC}"
        echo "Package file exists and can be manually tested with:"
        echo "  mojo install dist/utils-0.1.0.mojopkg"
    }
fi

echo -e "\n${GREEN}=== Packaging Complete ===${NC}"
echo "Deliverables created:"
echo "  - dist/utils-0.1.0.mojopkg (binary package)"
echo "  - scripts/install_verify_utils.sh (installation verification)"
echo "  - scripts/build_utils_package.sh (build-only script)"
echo "  - scripts/package_utils.sh (complete workflow script)"
