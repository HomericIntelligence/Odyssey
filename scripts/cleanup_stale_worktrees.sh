#!/usr/bin/env bash
# cleanup_stale_worktrees.sh - Remove stale git worktrees and their branches
#
# Usage:
#   ./scripts/cleanup_stale_worktrees.sh [--dry-run] [--force] [--include-unmerged]
#
# Options:
#   --dry-run           Show what would be removed without removing anything
#   --force             Remove worktrees even if they have uncommitted changes
#   --include-unmerged  Also remove worktrees whose branches are NOT merged to main
#                       (default: only remove worktrees with merged branches)
#
# This script identifies and removes stale git worktrees created by agents
# in .claude/worktrees/ and .worktrees/ directories. It:
#   1. Lists all worktrees (excluding the main working tree)
#   2. Classifies each as MERGED or NOT-MERGED relative to origin/main
#   3. Removes worktrees (merged by default, all with --include-unmerged)
#   4. Deletes local branches for removed worktrees
#   5. Deletes remote branches that were merged
#   6. Runs git worktree prune
#
# Safety:
#   - Skips the worktree the script is running from
#   - Requires --include-unmerged to remove unmerged branches
#   - Dry-run mode shows plan without executing

set -euo pipefail

# --- Configuration ---
DRY_RUN=false
FORCE=false
INCLUDE_UNMERGED=false
REPO_ROOT=""

# --- Colors (disabled if not a terminal) ---
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# --- Functions ---
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_dry()   { echo -e "${YELLOW}[DRY-RUN]${NC} $*"; }

usage() {
    sed -n '2,/^$/s/^# \?//p' "$0"
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)       DRY_RUN=true ;;
            --force)         FORCE=true ;;
            --include-unmerged) INCLUDE_UNMERGED=true ;;
            -h|--help)       usage ;;
            *)               log_error "Unknown option: $1"; usage ;;
        esac
        shift
    done
}

find_repo_root() {
    # Use git-common-dir to find the main repo root (works from any worktree).
    # git-common-dir returns the shared .git dir, e.g. /repo/.git
    local git_common_dir
    # `git rev-parse` exits non-zero outside a git repo. Capture rc separately
    # so the empty-string check below remains authoritative.
    set +e
    git_common_dir="$(git rev-parse --git-common-dir 2>/dev/null)"
    set -e
    if [[ -z "$git_common_dir" ]]; then
        log_error "Not inside a git repository."
        exit 1
    fi
    # Resolve to absolute path and get parent (the main working tree root)
    REPO_ROOT="$(cd "$git_common_dir/.." && pwd)"
}

get_current_worktree() {
    # Return the absolute path of the worktree we are running inside
    git rev-parse --show-toplevel 2>/dev/null || echo ""
}

is_branch_merged() {
    local branch="$1"
    git merge-base --is-ancestor "refs/heads/$branch" origin/main 2>/dev/null
}

remote_branch_exists() {
    local branch="$1"
    git ls-remote --heads origin "$branch" 2>/dev/null | grep -q "$branch"
}

remove_worktree() {
    local wt_path="$1"
    local branch="$2"
    local merged="$3"

    if $DRY_RUN; then
        log_dry "Would remove worktree: $wt_path (branch: $branch, $merged)"
        return 0
    fi

    # Remove the worktree
    local remove_flags=""
    if $FORCE; then
        remove_flags="--force"
    fi

    if git worktree remove $remove_flags "$wt_path" 2>/dev/null; then
        log_ok "Removed worktree: $wt_path"
    else
        log_warn "Failed to remove worktree: $wt_path (try --force if it has changes)"
        return 1
    fi

    # Delete local branch
    if git rev-parse --verify "refs/heads/$branch" &>/dev/null; then
        if [[ "$merged" == "MERGED" ]]; then
            git branch -d "$branch" 2>/dev/null && log_ok "Deleted local branch: $branch" \
                || log_warn "Could not delete local branch: $branch"
        else
            git branch -D "$branch" 2>/dev/null && log_ok "Force-deleted local branch: $branch" \
                || log_warn "Could not delete local branch: $branch"
        fi
    fi

    # Delete remote branch if merged
    if [[ "$merged" == "MERGED" ]] && remote_branch_exists "$branch"; then
        if git push origin --delete "$branch" 2>/dev/null; then
            log_ok "Deleted remote branch: $branch"
        else
            log_warn "Could not delete remote branch: $branch"
        fi
    fi

    return 0
}

# --- Main ---
main() {
    parse_args "$@"
    find_repo_root

    local current_wt
    current_wt="$(get_current_worktree)"

    log_info "Repository root: $REPO_ROOT"
    log_info "Current worktree: $current_wt"
    if $DRY_RUN; then
        log_info "Mode: DRY-RUN (no changes will be made)"
    fi
    if $INCLUDE_UNMERGED; then
        log_warn "Including unmerged branches for removal"
    fi
    echo ""

    # Fetch latest main for accurate merge checking
    log_info "Fetching origin/main..."
    git fetch origin main --quiet 2>/dev/null || log_warn "Could not fetch origin/main"
    echo ""

    # Collect worktree info using porcelain format
    local wt_path=""
    local wt_branch=""
    local removed=0
    local skipped=0
    local failed=0
    local total=0

    while IFS= read -r line; do
        if [[ "$line" =~ ^worktree\ (.+) ]]; then
            wt_path="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^branch\ refs/heads/(.+) ]]; then
            wt_branch="${BASH_REMATCH[1]}"
        elif [[ -z "$line" && -n "$wt_path" ]]; then
            # End of a worktree entry - process it

            # Skip main working tree
            if [[ "$wt_path" == "$REPO_ROOT" ]]; then
                wt_path=""
                wt_branch=""
                continue
            fi

            # Skip the worktree we're running from
            if [[ "$wt_path" == "$current_wt" ]]; then
                log_warn "Skipping current worktree: $wt_path ($wt_branch)"
                wt_path=""
                wt_branch=""
                skipped=$((skipped + 1))
                continue
            fi

            total=$((total + 1))

            # Check merge status
            local merged="NOT-MERGED"
            if [[ -n "$wt_branch" ]] && is_branch_merged "$wt_branch"; then
                merged="MERGED"
            fi

            # Decide whether to remove
            if [[ "$merged" == "MERGED" ]] || $INCLUDE_UNMERGED; then
                if remove_worktree "$wt_path" "$wt_branch" "$merged"; then
                    removed=$((removed + 1))
                else
                    failed=$((failed + 1))
                fi
            else
                log_info "Skipping unmerged worktree: $wt_path (branch: $wt_branch)"
                skipped=$((skipped + 1))
            fi

            wt_path=""
            wt_branch=""
        fi
    done < <(git worktree list --porcelain; echo "")

    # Prune stale worktree references
    echo ""
    if $DRY_RUN; then
        log_dry "Would run: git worktree prune"
    else
        git worktree prune
        log_ok "Pruned stale worktree references"
    fi

    # Summary
    echo ""
    echo "==============================="
    echo "  Worktree Cleanup Summary"
    echo "==============================="
    echo "  Total worktrees found: $total"
    echo "  Removed:               $removed"
    echo "  Skipped:               $skipped"
    echo "  Failed:                $failed"
    echo "==============================="

    if $DRY_RUN; then
        echo ""
        log_info "This was a dry run. Re-run without --dry-run to apply changes."
    fi

    # Verify remaining worktrees
    echo ""
    log_info "Remaining worktrees:"
    git worktree list
}

main "$@"
