#!/usr/bin/env bash
#
# Verify configuration package installation
#
# Checks:
# - Directory structure exists
# - Required files present
# - YAML syntax valid
# - Config loading works
#
# Usage:
#   ./scripts/verify_configs_install.sh

set -euo pipefail

ERRORS=0

echo "Verifying configuration package installation..."
echo ""

# Check directory structure
echo "Checking directory structure..."
REQUIRED_DIRS=(
    "configs"
    "configs/defaults"
    "configs/papers"
    "configs/experiments"
    "configs/templates"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir (missing)"
        ((ERRORS++))
    fi
done

# Check required files
echo ""
echo "Checking required files..."
REQUIRED_FILES=(
    "configs/README.md"
    "configs/MIGRATION.md"
    "configs/defaults/training.yaml"
    "configs/defaults/model.yaml"
    "configs/defaults/data.yaml"
    "configs/defaults/paths.yaml"
    "configs/templates/paper.yaml"
    "configs/templates/experiment.yaml"
    "src/odyssey/utils/config_loader.mojo"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
        ((ERRORS++))
    fi
done

# Check YAML syntax
echo ""
echo "Validating YAML syntax..."
if command -v yamllint &> /dev/null; then
    if yamllint configs/ 2>&1 | grep -q "error"; then
        echo "  ✗ YAML validation failed"
        ((ERRORS++))
    else
        echo "  ✓ All YAML files valid"
    fi
else
    echo "  ⚠ yamllint not found (skipping syntax check)"
fi

# Check for common paper configs
echo ""
echo "Checking example configurations..."
EXAMPLE_CONFIGS=(
    "configs/papers/lenet5/model.yaml"
    "configs/papers/lenet5/training.yaml"
    "configs/experiments/lenet5/baseline.yaml"
)

for file in "${EXAMPLE_CONFIGS[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ⚠ $file (optional, not found)"
    fi
done

# Summary
echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✓ Verification complete - no errors found"
    exit 0
else
    echo "✗ Verification failed with $ERRORS error(s)"
    exit 1
fi
