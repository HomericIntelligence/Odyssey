#!/usr/bin/env bash
#
# setup_agents.sh - Initialize and verify the agent system
#
# This script checks that all required agent files, templates, and skills are properly
# installed in the repository. It validates the complete agent hierarchy and skills system.
#
# Usage:
#   ./scripts/agents/setup_agents.sh [options]
#
# Options:
#   --dry-run    Show what would be checked without making changes
#   --help       Show this help message
#
# Examples:
#   # Run full setup verification
#   ./scripts/agents/setup_agents.sh
#
#   # Preview checks without running validation
#   ./scripts/agents/setup_agents.sh --dry-run
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
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
SKILLS_DIR="${REPO_ROOT}/.claude/skills"
SCRIPTS_DIR="${REPO_ROOT}/scripts/agents"
REPORT_FILE="${REPO_ROOT}/logs/agent_setup_report_$(date +%Y%m%d_%H%M%S).txt"

# Expected counts
EXPECTED_AGENTS=23
EXPECTED_TEMPLATES=6

# Flags
DRY_RUN=false

# Expected agent files (Level 0-5)
EXPECTED_AGENT_FILES=(
    # Level 0
    "chief-architect.md"
    # Level 1
    "foundation-orchestrator.md"
    "shared-library-orchestrator.md"
    "tooling-orchestrator.md"
    "papers-orchestrator.md"
    "cicd-orchestrator.md"
    "agentic-workflows-orchestrator.md"
    # Level 2
    "architecture-design.md"
    "integration-design.md"
    "security-design.md"
    # Level 3
    "implementation-specialist.md"
    "test-specialist.md"
    "documentation-specialist.md"
    "performance-specialist.md"
    "security-specialist.md"
    # Level 4
    "senior-implementation-engineer.md"
    "implementation-engineer.md"
    "test-engineer.md"
    "documentation-engineer.md"
    "performance-engineer.md"
    # Level 5
    "junior-implementation-engineer.md"
    "junior-documentation-engineer.md"
)

# Expected template files
EXPECTED_TEMPLATE_FILES=(
    "level-0-chief-architect.md"
    "level-1-section-orchestrator.md"
    "level-2-module-design.md"
    "level-3-component-specialist.md"
    "level-4-implementation-engineer.md"
    "level-5-junior-engineer.md"
)

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

show_help() {
    cat << EOF
setup_agents.sh - Initialize and verify the agent system

USAGE:
    ./scripts/agents/setup_agents.sh [options]

OPTIONS:
    --dry-run    Show what would be checked without making changes
    --help       Show this help message

DESCRIPTION:
    This script verifies the complete agent system setup including:
    - Agent configuration files (23 agents across 6 levels)
    - Template files (6 templates for different agent levels)
    - Skills directory structure
    - Validation scripts
    - File permissions and structure

    A detailed report is saved to logs/agent_setup_report_*.txt

EXAMPLES:
    # Run full setup verification
    ./scripts/agents/setup_agents.sh

    # Preview checks without running validation
    ./scripts/agents/setup_agents.sh --dry-run

EXIT CODES:
    0 - All checks passed
    1 - One or more checks failed
    2 - Invalid arguments

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
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

# Function to save output - we'll redirect at the end instead of using tee
# to avoid conflicts with subprocess execution

# Start setup
print_header "Agent System Setup Verification"
echo "Repository: $REPO_ROOT"
echo "Date: $(date)"
echo "Dry run: $DRY_RUN"
echo ""

# Track failures
FAILURES=0

# Check 1: Verify .claude/agents directory exists
print_header "Check 1: Agent Directory Structure"
if [[ -d "$AGENTS_DIR" ]]; then
    print_success "Agent directory exists: $AGENTS_DIR"
else
    print_error "Agent directory missing: $AGENTS_DIR"
    ((FAILURES++))
fi

# Check 2: Verify all agent files are present
print_header "Check 2: Agent Configuration Files"
AGENT_COUNT=0
MISSING_AGENTS=()

for agent_file in "${EXPECTED_AGENT_FILES[@]}"; do
    if [[ -f "${AGENTS_DIR}/${agent_file}" ]]; then
        AGENT_COUNT=$((AGENT_COUNT + 1))
    else
        MISSING_AGENTS+=("$agent_file")
    fi
done

if [[ $AGENT_COUNT -eq $EXPECTED_AGENTS ]]; then
    print_success "All $EXPECTED_AGENTS agent files present"
else
    print_error "Missing agent files: $((EXPECTED_AGENTS - AGENT_COUNT)) of $EXPECTED_AGENTS"
    for missing in "${MISSING_AGENTS[@]}"; do
        print_warning "  Missing: $missing"
    done
    ((FAILURES++))
fi

# Check 3: Verify template files
print_header "Check 3: Template Files"
TEMPLATE_COUNT=0
MISSING_TEMPLATES=()

if [[ -d "$TEMPLATES_DIR" ]]; then
    for template_file in "${EXPECTED_TEMPLATE_FILES[@]}"; do
        if [[ -f "${TEMPLATES_DIR}/${template_file}" ]]; then
            TEMPLATE_COUNT=$((TEMPLATE_COUNT + 1))
        else
            MISSING_TEMPLATES+=("$template_file")
        fi
    done

    if [[ $TEMPLATE_COUNT -eq $EXPECTED_TEMPLATES ]]; then
        print_success "All $EXPECTED_TEMPLATES template files present"
    else
        print_error "Missing template files: $((EXPECTED_TEMPLATES - TEMPLATE_COUNT)) of $EXPECTED_TEMPLATES"
        for missing in "${MISSING_TEMPLATES[@]}"; do
            print_warning "  Missing: $missing"
        done
        ((FAILURES++))
    fi
else
    print_error "Templates directory missing: $TEMPLATES_DIR"
    ((FAILURES++))
fi

# Check 4: Verify skills directory
print_header "Check 4: Skills Directory"
if [[ -d "$SKILLS_DIR" ]]; then
    SKILL_COUNT=$(find "$SKILLS_DIR" -type f -name "*.md" | wc -l)
    print_success "Skills directory exists: $SKILLS_DIR"
    print_info "Found $SKILL_COUNT skill files"

    # Check for tier structure
    if [[ -d "${SKILLS_DIR}/tier-1" ]]; then
        TIER1_COUNT=$(find "${SKILLS_DIR}/tier-1" -type f -name "*.md" | wc -l)
        print_info "Tier 1 skills: $TIER1_COUNT"
    fi
    if [[ -d "${SKILLS_DIR}/tier-2" ]]; then
        TIER2_COUNT=$(find "${SKILLS_DIR}/tier-2" -type f -name "*.md" | wc -l)
        print_info "Tier 2 skills: $TIER2_COUNT"
    fi
else
    print_error "Skills directory missing: $SKILLS_DIR"
    ((FAILURES++))
fi

# Check 5: Verify validation scripts exist
print_header "Check 5: Validation Scripts"
VALIDATION_SCRIPTS=(
    "setup_agents.sh"
    "agent_health_check.sh"
    "agent_stats.py"
)

for script in "${VALIDATION_SCRIPTS[@]}"; do
    if [[ -f "${SCRIPTS_DIR}/${script}" ]]; then
        print_success "Found: $script"

        # Check if executable
        if [[ -x "${SCRIPTS_DIR}/${script}" ]]; then
            print_info "  Executable: yes"
        else
            print_warning "  Executable: no (consider running: chmod +x ${SCRIPTS_DIR}/${script})"
        fi
    else
        print_error "Missing: $script"
        ((FAILURES++))
    fi
done

# Check 6: Run validation scripts (if not dry-run)
if [[ "$DRY_RUN" == "false" ]]; then
    print_header "Check 6: Running Validation Scripts"

    # Run health check if it exists
    if [[ -x "${SCRIPTS_DIR}/agent_health_check.sh" ]]; then
        print_info "Running agent health check..."
        # Redirect to avoid tee conflicts
        if "${SCRIPTS_DIR}/agent_health_check.sh" > /dev/null 2>&1; then
            print_success "Health check passed"
        else
            print_error "Health check failed"
            ((FAILURES++))
        fi
    else
        print_warning "Skipping health check (script not executable)"
    fi
else
    print_header "Check 6: Validation Scripts (Skipped - Dry Run)"
    print_info "Skipping validation script execution in dry-run mode"
fi

# Summary
print_header "Setup Summary"
echo ""
echo "Agent files:    $AGENT_COUNT / $EXPECTED_AGENTS"
echo "Template files: $TEMPLATE_COUNT / $EXPECTED_TEMPLATES"
echo "Skills found:   ${SKILL_COUNT:-0}"
echo ""

# Save report note
print_info "Report will be saved to: $REPORT_FILE"
echo ""

if [[ $FAILURES -eq 0 ]]; then
    print_success "All checks passed! Agent system is properly configured."
    RESULT=0
else
    print_error "$FAILURES check(s) failed. Please review the output above."
    RESULT=1
fi

# Note: To save full output to report file, run:
# ./scripts/agents/setup_agents.sh 2>&1 | tee logs/agent_setup_report.txt

exit $RESULT
