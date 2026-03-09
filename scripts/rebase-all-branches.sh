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

# Get the main repository root
MAIN_REPO_ROOT=$(git rev-parse --show-toplevel)

# Store original branch/worktree to return to it later
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
ORIGINAL_DIR=$(pwd)

echo -e "${BLUE}=== Rebasing all branches against main ===${NC}\n"

# Ensure main is up to date
echo -e "${BLUE}Fetching latest changes...${NC}"
git fetch origin main > /dev/null 2>&1

# Get all local branches except main
BRANCHES=$(git branch --format='%(refname:short)' | grep -v '^main$')

# Get all worktree branches (excluding main)
WORKTREES=$(git worktree list --porcelain | grep '^worktree' | awk '{print $2}' | xargs -I {} basename {} | grep -v '^main$' 2>/dev/null)

# Combine branches and worktrees
ALL_BRANCHES=$(echo -e "$BRANCHES\n$WORKTREES" | grep -v '^$' | sort -u)

if [ -z "$ALL_BRANCHES" ]; then
    echo -e "${YELLOW}No branches to rebase (only main exists)${NC}"
    exit 0
fi

for BRANCH in $ALL_BRANCHES; do
    echo -e "${BLUE}Processing branch: ${BRANCH}${NC}"

    # Check if this branch has a worktree
    WORKTREE_PATH=$(git worktree list --porcelain 2>/dev/null | grep "branch.*/$BRANCH$" | awk '{print $2}')

    # Determine where to work from
    if [ -n "$WORKTREE_PATH" ] && [ -d "$WORKTREE_PATH" ]; then
        WORK_DIR="$WORKTREE_PATH"
        IS_WORKTREE=1
    else
        WORK_DIR="$MAIN_REPO_ROOT"
        IS_WORKTREE=0
    fi

    # Check if the branch is tracking a remote
    TRACKING=$(git -C "$WORK_DIR" rev-parse --abbrev-ref "${BRANCH}@{u}" 2>/dev/null)
    IS_TRACKING=$?

    # Checkout the branch (either in worktree or main repo)
    if [ $IS_WORKTREE -eq 1 ]; then
        # For worktrees, just cd into them
        if ! cd "$WORK_DIR"; then
            echo -e "${RED}✗ Failed to enter worktree: ${WORK_DIR}${NC}"
            ((FAILED++))
            continue
        fi
    else
        # For regular branches, checkout in main repo
        if ! git -C "$MAIN_REPO_ROOT" checkout "$BRANCH" > /dev/null 2>&1; then
            echo -e "${RED}✗ Failed to checkout branch: ${BRANCH}${NC}"
            ((FAILED++))
            continue
        fi
    fi

    # Check if branch is already up to date with main
    if git merge-base --is-ancestor main "$BRANCH"; then
        MERGE_BASE=$(git merge-base main "$BRANCH")
        CURRENT_HEAD=$(git rev-parse HEAD)

        if [ "$MERGE_BASE" = "$CURRENT_HEAD" ]; then
            echo -e "${YELLOW}⊘ Branch is already up to date with main${NC}\n"
            if [ $IS_WORKTREE -eq 0 ]; then
                cd "$MAIN_REPO_ROOT"
            fi
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
                if [ $IS_WORKTREE -eq 0 ]; then
                    cd "$MAIN_REPO_ROOT"
                fi
                ((FAILED++))
                echo ""
                continue
            fi
        else
            echo -e "${YELLOW}⊘ Branch is not tracking a remote, skipping push${NC}"
        fi

        echo -e "${GREEN}Successfully rebased and pushed${NC}\n"
        if [ $IS_WORKTREE -eq 0 ]; then
            cd "$MAIN_REPO_ROOT"
        fi
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
        if [ $IS_WORKTREE -eq 0 ]; then
            cd "$MAIN_REPO_ROOT"
        fi
        ((FAILED++))
    fi
done

# Return to original branch/directory
cd "$ORIGINAL_DIR" 2>/dev/null || cd "$MAIN_REPO_ROOT"

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
