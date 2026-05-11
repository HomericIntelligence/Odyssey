#!/usr/bin/env bash
#
# agent_health_check.sh - Verify agent system health and integrity
#
# This script performs comprehensive health checks on the agent system including:
# - File existence and permissions
# - Broken link detection
# - YAML frontmatter parsing
# - Agent configuration validation
#
# Usage:
#   ./scripts/agents/agent_health_check.sh [options]
#
# Options:
#   --verbose    Show detailed output for each check
#   --help       Show this help message
#
# Examples:
#   # Run basic health check
#   ./scripts/agents/agent_health_check.sh
#
#   # Run with detailed output
#   ./scripts/agents/agent_health_check.sh --verbose
#
# Exit codes:
#   0 - All health checks passed
#   1 - One or more health checks failed
#   2 - Invalid arguments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AGENTS_DIR="${REPO_ROOT}/.claude/agents"
TEMPLATES_DIR="${REPO_ROOT}/agents/templates"
DOCS_DIR="${REPO_ROOT}/agents"
REPORT_FILE="${REPO_ROOT}/logs/agent_health_$(date +%Y%m%d_%H%M%S).txt"

# Flags
VERBOSE=false

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "  $1"
    fi
}

show_help() {
    cat << EOF
agent_health_check.sh - Verify agent system health and integrity

USAGE:
    ./scripts/agents/agent_health_check.sh [options]

OPTIONS:
    --verbose    Show detailed output for each check
    --help       Show this help message

DESCRIPTION:
    Performs comprehensive health checks on the agent system:

    1. File Existence - Verifies all agent files are present
    2. File Permissions - Checks that files are readable
    3. YAML Frontmatter - Validates YAML parsing in agent files
    4. Broken Links - Detects broken markdown links to other agents
    5. Required Fields - Ensures all agents have required fields
    6. Naming Conventions - Validates file naming standards

    A detailed report is saved to logs/agent_health_*.txt

EXAMPLES:
    # Run basic health check
    ./scripts/agents/agent_health_check.sh

    # Run with detailed output
    ./scripts/agents/agent_health_check.sh --verbose

EXIT CODES:
    0 - All health checks passed
    1 - One or more health checks failed
    2 - Invalid arguments

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}" >&2
            echo "Use --help for usage information"
            exit 2
            ;;
    esac
done

# Initialize report
mkdir -p "$(dirname "$REPORT_FILE")"

# Note: To save output to report, redirect when calling this script:
# ./agent_health_check.sh 2>&1 | tee logs/agent_health_report.txt

# Start health check
print_header "Agent System Health Check"
echo "Repository: $REPO_ROOT"
echo "Date: $(date)"
echo "Verbose: $VERBOSE"
echo ""

# Track issues
ERRORS=0
WARNINGS=0

# Check 1: File Existence
print_header "Check 1: Agent File Existence"
if [[ ! -d "$AGENTS_DIR" ]]; then
    print_error "Agent directory not found: $AGENTS_DIR"
    ((ERRORS++))
    exit 1
fi

AGENT_FILES=($(find "$AGENTS_DIR" -maxdepth 1 -type f -name "*.md" | sort))
AGENT_COUNT=${#AGENT_FILES[@]}

print_success "Found $AGENT_COUNT agent files"

if [[ "$VERBOSE" == "true" ]]; then
    for agent_file in "${AGENT_FILES[@]}"; do
        print_verbose "$(basename "$agent_file")"
    done
fi

# Check 2: File Permissions
print_header "Check 2: File Permissions"
PERMISSION_ERRORS=0

for agent_file in "${AGENT_FILES[@]}"; do
    if [[ ! -r "$agent_file" ]]; then
        print_error "Not readable: $(basename "$agent_file")"
        ((PERMISSION_ERRORS++))
        ((ERRORS++))
    else
        print_verbose "✓ $(basename "$agent_file")"
    fi
done

if [[ $PERMISSION_ERRORS -eq 0 ]]; then
    print_success "All agent files are readable"
else
    print_error "$PERMISSION_ERRORS files have permission issues"
fi

# Check 3: YAML Frontmatter Validation
print_header "Check 3: YAML Frontmatter"
YAML_ERRORS=0
REQUIRED_FIELDS=("name" "description" "tools" "model")

for agent_file in "${AGENT_FILES[@]}"; do
    agent_name=$(basename "$agent_file")
    print_verbose "Checking $agent_name..."

    # Check if file starts with ---
    if ! head -n 1 "$agent_file" | grep -q "^---$"; then
        print_error "$agent_name: Missing YAML frontmatter delimiter"
        ((YAML_ERRORS++))
        ((ERRORS++))
        continue
    fi

    # Extract YAML frontmatter (between first two ---)
    frontmatter=$(sed -n '/^---$/,/^---$/p' "$agent_file" | sed '1d;$d')

    # Check for required fields
    for field in "${REQUIRED_FIELDS[@]}"; do
        if ! echo "$frontmatter" | grep -q "^${field}:"; then
            print_warning "$agent_name: Missing required field '$field'"
            ((YAML_ERRORS++))
            ((WARNINGS++))
        else
            print_verbose "  ✓ $field present"
        fi
    done
done

if [[ $YAML_ERRORS -eq 0 ]]; then
    print_success "All agent files have valid YAML frontmatter"
else
    print_warning "$YAML_ERRORS YAML validation issues found"
fi

# Check 4: Broken Links
print_header "Check 4: Broken Links Detection"
BROKEN_LINKS=0

for agent_file in "${AGENT_FILES[@]}"; do
    agent_name=$(basename "$agent_file")
    print_verbose "Checking links in $agent_name..."

    # Find all markdown links to other agents: [text](./file.md).
    # Bucket D: under pipefail, two `grep -oP` calls that find nothing return
    # rc=1 and would tank the whole script. Disable -e around the pipeline so
    # "no links in this file" stays empty without masking other errors.
    set +e
    links=$(grep -oP '\[.*?\]\(\./.*?\.md\)' "$agent_file" 2>/dev/null \
              | grep -oP '\(\./\K[^)]+')
    set -e

    if [[ -n "$links" ]]; then
        while IFS= read -r link; do
            # Resolve the path relative to the agent file
            link_path="${AGENTS_DIR}/${link}"

            if [[ ! -f "$link_path" ]]; then
                print_error "$agent_name: Broken link to $link"
                ((BROKEN_LINKS++))
                ((ERRORS++))
            else
                print_verbose "  ✓ $link"
            fi
        done <<< "$links"
    fi
done

if [[ $BROKEN_LINKS -eq 0 ]]; then
    print_success "No broken links detected"
else
    print_error "$BROKEN_LINKS broken links found"
fi

# Check 5: Naming Conventions
print_header "Check 5: Naming Conventions"
NAMING_ERRORS=0

for agent_file in "${AGENT_FILES[@]}"; do
    agent_name=$(basename "$agent_file" .md)

    # Check naming convention: lowercase with hyphens
    if [[ ! "$agent_name" =~ ^[a-z]+(-[a-z]+)*$ ]]; then
        print_warning "Non-standard naming: $agent_name (should be lowercase-with-hyphens)"
        ((NAMING_ERRORS++))
        ((WARNINGS++))
    else
        print_verbose "✓ $agent_name"
    fi
done

if [[ $NAMING_ERRORS -eq 0 ]]; then
    print_success "All agent files follow naming conventions"
else
    print_warning "$NAMING_ERRORS files have non-standard names"
fi

# Check 6: Agent Level Assignment
print_header "Check 6: Agent Level Assignment"
declare -A LEVEL_COUNTS
# Initialize all levels to 0
for i in {0..5}; do
    LEVEL_COUNTS[$i]=0
done

for agent_file in "${AGENT_FILES[@]}"; do
    agent_name=$(basename "$agent_file")

    # Try to extract level from the file content
    if grep -q "^## Role" "$agent_file"; then
        level_line=$(grep -A 1 "^## Role" "$agent_file" | tail -n 1)

        if echo "$level_line" | grep -qi "Level [0-5]"; then
            level=$(echo "$level_line" | grep -oP 'Level \K[0-5]')
            ((LEVEL_COUNTS[$level]++)) || LEVEL_COUNTS[$level]=1
            print_verbose "$agent_name → Level $level"
        else
            print_warning "$agent_name: No level assignment found"
            ((WARNINGS++))
        fi
    else
        print_verbose "$agent_name: No Role section (may be template)"
    fi
done

echo ""
print_info "Agent distribution by level:"
for level in {0..5}; do
    count=${LEVEL_COUNTS[$level]:-0}
    echo "  Level $level: $count agents"
done

# Check 7: Template Files
print_header "Check 7: Template Files"
if [[ -d "$TEMPLATES_DIR" ]]; then
    TEMPLATE_COUNT=$(find "$TEMPLATES_DIR" -type f -name "*.md" | wc -l)
    print_success "Found $TEMPLATE_COUNT template files"

    if [[ "$VERBOSE" == "true" ]]; then
        find "$TEMPLATES_DIR" -type f -name "*.md" -exec basename {} \; | while read -r template; do
            print_verbose "$template"
        done
    fi
else
    print_warning "Templates directory not found: $TEMPLATES_DIR"
    ((WARNINGS++))
fi

# Summary
print_header "Health Check Summary"
echo ""
echo "Total agents:    $AGENT_COUNT"
echo "Errors found:    $ERRORS"
echo "Warnings found:  $WARNINGS"
echo ""

if [[ $ERRORS -eq 0 ]]; then
    if [[ $WARNINGS -eq 0 ]]; then
        print_success "Perfect health! All checks passed with no warnings."
        RESULT=0
    else
        print_success "Health check passed with $WARNINGS warnings."
        RESULT=0
    fi
else
    print_error "Health check failed with $ERRORS errors and $WARNINGS warnings."
    RESULT=1
fi

# Note: To save output to report, run:
# ./agent_health_check.sh 2>&1 | tee logs/agent_health_report.txt

exit $RESULT
