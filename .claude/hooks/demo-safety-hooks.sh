#!/usr/bin/env bash
#
# Quick demonstration of safety hooks (NOT production-deployed).
#
# This script is illustrative only — running it shows how the safety hooks
# block dangerous commands and allow safe ones. Production hook deployment
# is wired up via `.claude/hooks/IMPLEMENTATION.md` and the user's Claude
# Code settings, not by this file. The "demo-" prefix in the filename
# signals that this is a developer-facing example, not a runtime hook
# (closes #5325).

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Safety Hooks Demonstration${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PROJECT_ROOT

HOOK_SCRIPT=".claude/hooks/pre-bash-exec.sh"

echo "Project Root: $PROJECT_ROOT"
echo ""

# Demo dangerous commands
echo -e "${RED}=== DANGEROUS COMMANDS (BLOCKED) ===${NC}"
echo ""

echo -e "${YELLOW}Attempting: rm -rf /${NC}"
if bash "$HOOK_SCRIPT" "rm -rf /" 2>&1; then
    echo -e "${RED}✗ ERROR: Should have been blocked!${NC}"
else
    echo -e "${GREEN}✓ Blocked successfully${NC}"
fi
echo ""

echo -e "${YELLOW}Attempting: rm -rf .git${NC}"
if bash "$HOOK_SCRIPT" "rm -rf .git" 2>&1; then
    echo -e "${RED}✗ ERROR: Should have been blocked!${NC}"
else
    echo -e "${GREEN}✓ Blocked successfully${NC}"
fi
echo ""

echo -e "${YELLOW}Attempting: rm /etc/passwd${NC}"
if bash "$HOOK_SCRIPT" "rm /etc/passwd" 2>&1; then
    echo -e "${RED}✗ ERROR: Should have been blocked!${NC}"
else
    echo -e "${GREEN}✓ Blocked successfully${NC}"
fi
echo ""

# Demo safe commands
echo -e "${GREEN}=== SAFE COMMANDS (ALLOWED) ===${NC}"
echo ""

echo -e "${YELLOW}Attempting: rm temp.txt${NC}"
if bash "$HOOK_SCRIPT" "rm temp.txt" 2>&1; then
    echo -e "${GREEN}✓ Allowed successfully${NC}"
else
    echo -e "${RED}✗ ERROR: Should have been allowed!${NC}"
fi
echo ""

echo -e "${YELLOW}Attempting: rm -rf build/${NC}"
if bash "$HOOK_SCRIPT" "rm -rf build/" 2>&1; then
    echo -e "${GREEN}✓ Allowed successfully${NC}"
else
    echo -e "${RED}✗ ERROR: Should have been allowed!${NC}"
fi
echo ""

echo -e "${YELLOW}Attempting: git status${NC}"
if bash "$HOOK_SCRIPT" "git status" 2>&1; then
    echo -e "${GREEN}✓ Allowed successfully${NC}"
else
    echo -e "${RED}✗ ERROR: Should have been allowed!${NC}"
fi
echo ""

echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}Safety hooks are working correctly!${NC}"
echo -e "${BLUE}======================================${NC}"
