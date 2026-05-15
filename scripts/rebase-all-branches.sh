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

# Arrays to track issues
declare -a CONFLICTS_BRANCHES
declare -a DIRTY_BRANCHES
declare -a CLEANED_WORKTREES

# Remove the worktree at $2 (for branch $1) if it's safe to do so:
#   - path is non-empty and not the main repo
#   - no uncommitted/untracked changes
#   - no open PR for the branch
# Branch is intentionally NOT deleted — that is deferred to the user.
# Caller must already have cd'd out of the worktree.
maybe_remove_worktree() {
    local branch="$1"
    local wt_path="$2"

    [ -z "$wt_path" ] && return 0
    [ "$wt_path" = "$MAIN_REPO_ROOT" ] && return 0
    [ ! -d "$wt_path" ] && return 0

    if [ -n "$(git -C "$wt_path" status --porcelain 2>/dev/null)" ]; then
        return 0
    fi

    local open_prs
    if ! open_prs=$(gh pr list --head "$branch" --state open --json number 2>/dev/null); then
        # gh failed (unauthenticated, offline, etc.) — preserve the worktree.
        return 0
    fi
    if [ -n "$open_prs" ] && [ "$open_prs" != "[]" ]; then
        return 0
    fi

    if git worktree remove "$wt_path" 2>/dev/null; then
        CLEANED_WORKTREES+=("$branch")
        echo -e "${GREEN}✓ Removed worktree for ${branch} (branch kept)${NC}"
    else
        echo -e "${YELLOW}  Could not remove worktree at ${wt_path}${NC}"
    fi
}

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
    WORKTREE_PATH=$(git worktree list --porcelain 2>/dev/null | \
      awk -v branch="$BRANCH" '/^worktree /{path=$2} /^branch / && $2 ~ "/" branch "$" {print path}')

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
    if git -C "$WORK_DIR" merge-base --is-ancestor main "$BRANCH"; then
        MERGE_BASE=$(git -C "$WORK_DIR" merge-base main "$BRANCH")
        CURRENT_HEAD=$(git rev-parse HEAD)

        if [ "$MERGE_BASE" = "$CURRENT_HEAD" ]; then
            echo -e "${YELLOW}⊘ Branch is already up to date with main${NC}\n"
            cd "$MAIN_REPO_ROOT"
            if [ $IS_WORKTREE -eq 1 ]; then
                maybe_remove_worktree "$BRANCH" "$WORK_DIR"
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
        cd "$MAIN_REPO_ROOT"
        if [ $IS_WORKTREE -eq 1 ]; then
            maybe_remove_worktree "$BRANCH" "$WORK_DIR"
        fi
        ((SUCCEEDED++))
    else
        # Rebase failed - check for conflicts
        if git diff --name-only --diff-filter=U | grep -q .; then
            echo -e "${RED}✗ Rebase conflict detected${NC}"
            CONFLICTS_BRANCHES+=("$BRANCH")
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

    # Check for uncommitted changes in the branch.
    # Skip if the worktree was just removed inline (directory no longer exists).
    if [ -d "$WORK_DIR" ]; then
        if ! git -C "$WORK_DIR" diff-index --quiet HEAD -- 2>/dev/null \
           || [ -n "$(git -C "$WORK_DIR" ls-files --others --exclude-standard 2>/dev/null)" ]; then
            DIRTY_BRANCHES+=("$BRANCH")
        fi
    fi
done

# Return to original branch/directory
cd "$ORIGINAL_DIR" 2>/dev/null || cd "$MAIN_REPO_ROOT"

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Succeeded: ${SUCCEEDED}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo ""

# Print conflicts table
if [ ${#CONFLICTS_BRANCHES[@]} -gt 0 ]; then
    echo -e "${RED}=== Branches with Merge Conflicts ===${NC}"
    printf "%-40s %s\n" "Branch" "Type"
    printf "%-40s %s\n" "$(printf '=%.0s' {1..40})" "$(printf '=%.0s' {1..10})"
    for BRANCH in "${CONFLICTS_BRANCHES[@]}"; do
        WORKTREE_PATH=$(git worktree list --porcelain 2>/dev/null | \
          awk -v branch="$BRANCH" '/^worktree /{path=$2} /^branch / && $2 ~ "/" branch "$" {print path}')
        if [ -n "$WORKTREE_PATH" ] && [ -d "$WORKTREE_PATH" ]; then
            TYPE="Worktree"
        else
            TYPE="Branch"
        fi
        printf "%-40s %s\n" "$BRANCH" "$TYPE"
    done
    echo ""
fi

# Print dirty branches table
if [ ${#DIRTY_BRANCHES[@]} -gt 0 ]; then
    echo -e "${YELLOW}=== Branches with Uncommitted Changes ===${NC}"
    printf "%-40s %s\n" "Branch" "Type"
    printf "%-40s %s\n" "$(printf '=%.0s' {1..40})" "$(printf '=%.0s' {1..10})"
    for BRANCH in "${DIRTY_BRANCHES[@]}"; do
        WORKTREE_PATH=$(git worktree list --porcelain 2>/dev/null | \
          awk -v branch="$BRANCH" '/^worktree /{path=$2} /^branch / && $2 ~ "/" branch "$" {print path}')
        if [ -n "$WORKTREE_PATH" ] && [ -d "$WORKTREE_PATH" ]; then
            TYPE="Worktree"
        else
            TYPE="Branch"
        fi
        printf "%-40s %s\n" "$BRANCH" "$TYPE"
    done
    echo ""
fi

# Prune any dangling worktree references left over from inline removals
git worktree prune 2>/dev/null

if [ ${#CLEANED_WORKTREES[@]} -gt 0 ]; then
    echo -e "${GREEN}=== Worktrees Removed Inline: ${#CLEANED_WORKTREES[@]} ===${NC}"
    for BRANCH in "${CLEANED_WORKTREES[@]}"; do
        echo -e "  ${GREEN}✓ ${BRANCH}${NC}"
    done
    echo ""
fi

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
