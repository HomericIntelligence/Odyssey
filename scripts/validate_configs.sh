#!/usr/bin/env bash
# Validate repository configuration files
# Tests YAML syntax and basic structure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================="
echo "Repository Configuration Validation"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test function
test_file() {
    local file="$1"
    local description="$2"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo -n "Testing: ${description}... "

    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✓ EXISTS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ MISSING${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Validate YAML syntax
validate_yaml() {
    local file="$1"
    local description="$2"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo -n "Validating YAML: ${description}... "

    if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
        echo -e "${GREEN}✓ VALID${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ INVALID${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Check file content
check_content() {
    local file="$1"
    local pattern="$2"
    local description="$3"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo -n "Checking content: ${description}... "

    if grep -q "$pattern" "$file"; then
        echo -e "${GREEN}✓ FOUND${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ NOT FOUND${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Change to repo root
cd "$REPO_ROOT"

# ============================================================================
# Test 1: File Existence
# ============================================================================
echo "Test Suite 1: File Existence"
echo "-----------------------------"
test_file ".github/dependabot.yml" "Dependabot configuration"
test_file ".github/CODEOWNERS" "CODEOWNERS configuration"
test_file ".github/FUNDING.yml" "FUNDING configuration"
echo ""

# ============================================================================
# Test 2: YAML Syntax Validation
# ============================================================================
echo "Test Suite 2: YAML Syntax Validation"
echo "-------------------------------------"
validate_yaml ".github/dependabot.yml" "dependabot.yml syntax"
validate_yaml ".github/FUNDING.yml" "FUNDING.yml syntax"
echo ""

# ============================================================================
# Test 3: Dependabot Configuration Content
# ============================================================================
echo "Test Suite 3: Dependabot Configuration"
echo "---------------------------------------"
check_content ".github/dependabot.yml" "package-ecosystem: \"pip\"" "Python ecosystem configured"
check_content ".github/dependabot.yml" "package-ecosystem: \"github-actions\"" "GitHub Actions ecosystem configured"
check_content ".github/dependabot.yml" "interval: \"weekly\"" "Weekly update schedule"
check_content ".github/dependabot.yml" "day: \"tuesday\"" "Tuesday schedule"
check_content ".github/dependabot.yml" "dependencies" "Dependencies label"
check_content ".github/dependabot.yml" "mvillmow" "Assignee configured"
echo ""

# ============================================================================
# Test 4: CODEOWNERS Configuration Content
# ============================================================================
echo "Test Suite 4: CODEOWNERS Configuration"
echo "--------------------------------------"
check_content ".github/CODEOWNERS" "^\*  @mvillmow" "Default ownership"
check_content ".github/CODEOWNERS" "pyproject.toml" "Project configuration ownership"
check_content ".github/CODEOWNERS" ".github/workflows/" "Workflows ownership"
check_content ".github/CODEOWNERS" "papers/" "Papers directory ownership"
check_content ".github/CODEOWNERS" "\*\*\/\*security\*" "Security pattern"
check_content ".github/CODEOWNERS" "\.mojo" "Mojo files ownership"
echo ""

# ============================================================================
# Test 5: FUNDING Configuration Content
# ============================================================================
echo "Test Suite 5: FUNDING Configuration"
echo "------------------------------------"
check_content ".github/FUNDING.yml" "github:" "GitHub Sponsors configured"
check_content ".github/FUNDING.yml" "mvillmow" "GitHub username present"
echo ""

# ============================================================================
# Test 6: Additional Checks
# ============================================================================
echo "Test Suite 6: Additional Checks"
echo "--------------------------------"

# Check file permissions
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo -n "Checking file permissions... "
if [[ -r ".github/dependabot.yml" ]] && [[ -r ".github/CODEOWNERS" ]] && [[ -r ".github/FUNDING.yml" ]]; then
    echo -e "${GREEN}✓ READABLE${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}✗ NOT READABLE${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Check for trailing whitespace (should be clean)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo -n "Checking for trailing whitespace... "
if ! grep -n '[[:space:]]$' .github/dependabot.yml .github/CODEOWNERS .github/FUNDING.yml 2>/dev/null; then
    echo -e "${GREEN}✓ CLEAN${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠ FOUND${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Check file ends with newline
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo -n "Checking files end with newline... "
if [[ -n "$(tail -c 1 .github/dependabot.yml)" ]] || [[ -n "$(tail -c 1 .github/CODEOWNERS)" ]] || [[ -n "$(tail -c 1 .github/FUNDING.yml)" ]]; then
    echo -e "${YELLOW}⚠ MISSING NEWLINE${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
else
    echo -e "${GREEN}✓ PRESENT${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

echo ""

# ============================================================================
# Test Summary
# ============================================================================
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total Tests:  ${TOTAL_TESTS}"
echo -e "Passed:       ${GREEN}${PASSED_TESTS}${NC}"
if [[ ${FAILED_TESTS} -gt 0 ]]; then
    echo -e "Failed:       ${RED}${FAILED_TESTS}${NC}"
else
    echo -e "Failed:       ${GREEN}${FAILED_TESTS}${NC}"
fi
echo ""

# Exit with appropriate code
if [[ ${FAILED_TESTS} -eq 0 ]]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed! ✗${NC}"
    exit 1
fi
