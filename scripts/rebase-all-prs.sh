#!/bin/bash

set -o pipefail

# Rebase all open PRs against main, run pre-commit, commit fixes, and push.
#
# For each open PR:
#   1. Check out the PR branch in an existing worktree or create one
#   2. Rebase against origin/main
#   3. Run uv run pre-commit run --all-files
#   4. If pre-commit modified files, commit them
#   5. If everything succeeded, force-push (with lease) to update the PR
#
# Usage:
#   ./scripts/rebase-all-prs.sh [--dry-run] [--max N] [--skip-precommit]

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
declare -a CONFLICT_PRS
declare -a PRECOMMIT_FAIL_PRS
declare -a PUSH_FAIL_PRS
declare -a CREATED_WORKTREES

# Options
DRY_RUN=0
MAX_PRS=0
SKIP_PRECOMMIT=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --max)
            MAX_PRS="$2"
            shift 2
            ;;
        --skip-precommit)
            SKIP_PRECOMMIT=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--max N] [--skip-precommit]"
            echo ""
            echo "  --dry-run         Show what would be done without making changes"
            echo "  --max N           Process at most N PRs"
            echo "  --skip-precommit  Skip the pre-commit step"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Get the main repository root
MAIN_REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE_DIR="${MAIN_REPO_ROOT}/.worktrees"
ORIGINAL_DIR=$(pwd)

echo -e "${BLUE}=== Rebasing all open PRs against main ===${NC}"
if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}(dry-run mode — no changes will be made)${NC}"
fi
echo ""

# Ensure main is up to date
echo -e "${BLUE}Fetching latest changes...${NC}"
git fetch origin main > /dev/null 2>&1

# Get all open PR numbers and their head branch names
echo -e "${BLUE}Listing open PRs...${NC}"
PR_DATA=$(gh pr list --state open --json number,headRefName --limit 500)
if [ $? -ne 0 ] || [ -z "$PR_DATA" ]; then
    echo -e "${RED}Failed to list PRs. Check gh auth status.${NC}"
    exit 1
fi

PR_COUNT=$(echo "$PR_DATA" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
echo -e "${BLUE}Found ${PR_COUNT} open PRs${NC}"
echo ""

if [ "$PR_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No open PRs to process.${NC}"
    exit 0
fi

# Parse PR data into parallel arrays
mapfile -t PR_NUMBERS < <(echo "$PR_DATA" | python3 -c "
import json, sys
for pr in json.load(sys.stdin):
    print(pr['number'])
")
mapfile -t PR_BRANCHES < <(echo "$PR_DATA" | python3 -c "
import json, sys
for pr in json.load(sys.stdin):
    print(pr['headRefName'])
")

PROCESSED=0

for i in "${!PR_NUMBERS[@]}"; do
    PR_NUM="${PR_NUMBERS[$i]}"
    BRANCH="${PR_BRANCHES[$i]}"

    # Respect --max
    if [ "$MAX_PRS" -gt 0 ] && [ "$PROCESSED" -ge "$MAX_PRS" ]; then
        echo -e "${YELLOW}Reached --max ${MAX_PRS}, stopping.${NC}"
        break
    fi

    echo -e "${BLUE}──────────────────────────────────────────${NC}"
    echo -e "${BLUE}PR #${PR_NUM}: ${BRANCH}${NC}"

    if [ $DRY_RUN -eq 1 ]; then
        echo -e "${YELLOW}  [dry-run] Would rebase, run pre-commit, and push${NC}"
        ((PROCESSED++))
        ((SKIPPED++))
        continue
    fi

    # Fetch the PR branch
    if ! git fetch origin "$BRANCH" > /dev/null 2>&1; then
        echo -e "${RED}  ✗ Failed to fetch branch: ${BRANCH}${NC}"
        ((FAILED++))
        ((PROCESSED++))
        continue
    fi

    # Find existing worktree for this branch
    WORKTREE_PATH=""
    while IFS= read -r line; do
        if [[ "$line" == worktree\ * ]]; then
            current_wt="${line#worktree }"
        elif [[ "$line" == "branch refs/heads/$BRANCH" ]]; then
            WORKTREE_PATH="$current_wt"
            break
        fi
    done < <(git worktree list --porcelain 2>/dev/null)

    CREATED_WT=0

    if [ -n "$WORKTREE_PATH" ] && [ -d "$WORKTREE_PATH" ]; then
        echo -e "  Using existing worktree: ${WORKTREE_PATH}"
    else
        # Create a worktree for this PR
        WORKTREE_PATH="${WORKTREE_DIR}/pr-${PR_NUM}"
        mkdir -p "$WORKTREE_DIR"

        # Ensure local branch exists tracking the remote
        if ! git show-ref --verify --quiet "refs/heads/$BRANCH"; then
            git branch "$BRANCH" "origin/$BRANCH" > /dev/null 2>&1
        fi

        if ! git worktree add "$WORKTREE_PATH" "$BRANCH" > /dev/null 2>&1; then
            echo -e "${RED}  ✗ Failed to create worktree for ${BRANCH}${NC}"
            ((FAILED++))
            ((PROCESSED++))
            continue
        fi
        echo -e "  Created worktree: ${WORKTREE_PATH}"
        CREATED_WT=1
        CREATED_WORKTREES+=("$WORKTREE_PATH")
    fi

    cd "$WORKTREE_PATH" || {
        echo -e "${RED}  ✗ Failed to enter worktree${NC}"
        ((FAILED++))
        ((PROCESSED++))
        continue
    }

    # Make sure we're on the right branch
    CURRENT=$(git rev-parse --abbrev-ref HEAD)
    if [ "$CURRENT" != "$BRANCH" ]; then
        git checkout "$BRANCH" > /dev/null 2>&1
    fi

    # Reset to match remote (in case local is stale)
    git reset --hard "origin/$BRANCH" > /dev/null 2>&1

    # Check if already up to date with main
    MAIN_HEAD=$(git rev-parse origin/main)
    MERGE_BASE=$(git merge-base origin/main HEAD)
    if [ "$MAIN_HEAD" = "$MERGE_BASE" ]; then
        if [ $SKIP_PRECOMMIT -eq 1 ]; then
            echo -e "${YELLOW}  ⊘ Already up to date with main, skipping${NC}"
            cd "$ORIGINAL_DIR"
            ((SKIPPED++))
            ((PROCESSED++))
            # Clean up worktree if we just created it
            if [ $CREATED_WT -eq 1 ]; then
                git -C "$MAIN_REPO_ROOT" worktree remove "$WORKTREE_PATH" --force > /dev/null 2>&1
            fi
            continue
        fi
        echo -e "${YELLOW}  ⊘ Already up to date with main${NC}"
        REBASE_DONE=0
    else
        REBASE_DONE=1
    fi

    # Attempt rebase (only if not already up to date)
    if [ $REBASE_DONE -eq 1 ]; then
        if git rebase origin/main > /dev/null 2>&1; then
            echo -e "${GREEN}  ✓ Rebase successful${NC}"
        else
            # Check for conflicts
            if git diff --name-only --diff-filter=U 2>/dev/null | grep -q .; then
                echo -e "${RED}  ✗ Rebase conflict${NC}"
                CONFLICT_PRS+=("PR #${PR_NUM} (${BRANCH})")
            else
                echo -e "${RED}  ✗ Rebase failed${NC}"
            fi
            git rebase --abort > /dev/null 2>&1
            cd "$ORIGINAL_DIR"
            ((FAILED++))
            ((PROCESSED++))
            # Clean up worktree if we just created it
            if [ $CREATED_WT -eq 1 ]; then
                git -C "$MAIN_REPO_ROOT" worktree remove "$WORKTREE_PATH" --force > /dev/null 2>&1
            fi
            continue
        fi
    fi

    # Run pre-commit
    if [ $SKIP_PRECOMMIT -eq 0 ]; then
        echo -e "  Running pre-commit..."
        PRECOMMIT_OUTPUT=$(uv run pre-commit run --all-files 2>&1)
        PRECOMMIT_EXIT=$?

        # Check if pre-commit modified any files
        MODIFIED_FILES=$(git diff --name-only 2>/dev/null)
        if [ -n "$MODIFIED_FILES" ]; then
            echo -e "${YELLOW}  Pre-commit modified files, committing fixes...${NC}"
            git add -A > /dev/null 2>&1
            git commit -m "$(cat <<'EOF'
style: auto-fix pre-commit formatting

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)" > /dev/null 2>&1
            echo -e "${GREEN}  ✓ Committed pre-commit fixes${NC}"

            # Run pre-commit again to verify fixes are clean
            PRECOMMIT_OUTPUT=$(uv run pre-commit run --all-files 2>&1)
            PRECOMMIT_EXIT=$?
        fi

        if [ $PRECOMMIT_EXIT -ne 0 ]; then
            echo -e "${RED}  ✗ Pre-commit failed (exit code ${PRECOMMIT_EXIT})${NC}"
            PRECOMMIT_FAIL_PRS+=("PR #${PR_NUM} (${BRANCH})")
            # Reset to remote state to undo partial changes
            git reset --hard "origin/$BRANCH" > /dev/null 2>&1
            cd "$ORIGINAL_DIR"
            ((FAILED++))
            ((PROCESSED++))
            # Clean up worktree if we just created it
            if [ $CREATED_WT -eq 1 ]; then
                git -C "$MAIN_REPO_ROOT" worktree remove "$WORKTREE_PATH" --force > /dev/null 2>&1
            fi
            continue
        else
            echo -e "${GREEN}  ✓ Pre-commit passed${NC}"
        fi
    fi

    # Push to remote
    if git push origin "$BRANCH" --force-with-lease > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Pushed to remote${NC}"
        ((SUCCEEDED++))
    else
        echo -e "${RED}  ✗ Push failed (force-with-lease rejected)${NC}"
        PUSH_FAIL_PRS+=("PR #${PR_NUM} (${BRANCH})")
        ((FAILED++))
    fi

    cd "$ORIGINAL_DIR"

    # Clean up worktree if we created it
    if [ $CREATED_WT -eq 1 ]; then
        git -C "$MAIN_REPO_ROOT" worktree remove "$WORKTREE_PATH" --force > /dev/null 2>&1
    fi

    ((PROCESSED++))
done

# Return to original directory
cd "$ORIGINAL_DIR" 2>/dev/null || cd "$MAIN_REPO_ROOT"

# Summary
echo ""
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Succeeded: ${SUCCEEDED}${NC}"
echo -e "${YELLOW}Skipped:   ${SKIPPED}${NC}"
echo -e "${RED}Failed:    ${FAILED}${NC}"
echo ""

if [ ${#CONFLICT_PRS[@]} -gt 0 ]; then
    echo -e "${RED}=== PRs with Rebase Conflicts ===${NC}"
    for pr in "${CONFLICT_PRS[@]}"; do
        echo -e "  ${RED}✗ ${pr}${NC}"
    done
    echo ""
fi

if [ ${#PRECOMMIT_FAIL_PRS[@]} -gt 0 ]; then
    echo -e "${RED}=== PRs with Pre-commit Failures ===${NC}"
    for pr in "${PRECOMMIT_FAIL_PRS[@]}"; do
        echo -e "  ${RED}✗ ${pr}${NC}"
    done
    echo ""
fi

if [ ${#PUSH_FAIL_PRS[@]} -gt 0 ]; then
    echo -e "${RED}=== PRs with Push Failures ===${NC}"
    for pr in "${PUSH_FAIL_PRS[@]}"; do
        echo -e "  ${RED}✗ ${pr}${NC}"
    done
    echo ""
fi

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
