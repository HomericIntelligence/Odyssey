#!/bin/bash

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
SUCCEEDED=0
FAILED=0
SKIPPED=0

# Store original branch to return to it later
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo -e "${BLUE}=== Rebasing all branches against main ===${NC}\n"

# Ensure main is up to date
echo -e "${BLUE}Fetching latest changes...${NC}"
git fetch origin main > /dev/null 2>&1

# Get all local branches except main
BRANCHES=$(git branch --format='%(refname:short)' | grep -v '^main$')

if [ -z "$BRANCHES" ]; then
    echo -e "${YELLOW}No branches to rebase (only main exists)${NC}"
    exit 0
fi

for BRANCH in $BRANCHES; do
    echo -e "${BLUE}Processing branch: ${BRANCH}${NC}"

    # Check if the branch is tracking a remote
    TRACKING=$(git rev-parse --abbrev-ref "${BRANCH}@{u}" 2>/dev/null)
    IS_TRACKING=$?

    # Checkout the branch
    if ! git checkout "$BRANCH" > /dev/null 2>&1; then
        echo -e "${RED}✗ Failed to checkout branch: ${BRANCH}${NC}"
        ((FAILED++))
        continue
    fi

    # Check if branch is already up to date with main
    if git merge-base --is-ancestor main "$BRANCH"; then
        MERGE_BASE=$(git merge-base main "$BRANCH")
        CURRENT_HEAD=$(git rev-parse HEAD)

        if [ "$MERGE_BASE" = "$CURRENT_HEAD" ]; then
            echo -e "${YELLOW}⊘ Branch is already up to date with main${NC}\n"
            ((SKIPPED++))
            continue
        fi
    fi

    # Attempt rebase
    if git rebase origin/main > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Rebase successful${NC}"

        # Push to remote if branch is tracking a remote
        if [ $IS_TRACKING -eq 0 ]; then
            if git push origin "$BRANCH" --force-with-lease > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Pushed to remote${NC}"
            else
                echo -e "${RED}✗ Failed to push (force-with-lease rejected)${NC}"
                echo -e "${YELLOW}  Note: Someone else may have pushed changes. Please handle manually.${NC}"
                ((FAILED++))
                echo ""
                continue
            fi
        else
            echo -e "${YELLOW}⊘ Branch is not tracking a remote, skipping push${NC}"
        fi

        echo -e "${GREEN}Successfully rebased and pushed${NC}\n"
        ((SUCCEEDED++))
    else
        # Rebase failed - check for conflicts
        if git diff --name-only --diff-filter=U | grep -q .; then
            echo -e "${RED}✗ Rebase conflict detected${NC}"
        else
            echo -e "${RED}✗ Rebase failed${NC}"
        fi

        # Abort the rebase
        git rebase --abort 2>/dev/null
        echo -e "${YELLOW}  Aborted rebase for: ${BRANCH}${NC}\n"
        ((FAILED++))
    fi
done

# Return to original branch
if [ "$ORIGINAL_BRANCH" != "HEAD" ]; then
    git checkout "$ORIGINAL_BRANCH" > /dev/null 2>&1
fi

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Succeeded: ${SUCCEEDED}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
