# Pull Request Workflow

Shared PR workflow for all agents. Reference this file instead of duplicating.

## Creating a PR

```bash
git checkout -b {issue-number}-{description}
# Make changes, commit
git push -u origin {branch-name}
gh pr create --title "[Type] Description" --body "Closes #{issue}"
gh pr merge --auto --rebase  # Always enable auto-merge
```

## Verification

After creating PR:

1. **Verify** the PR is linked to the issue (check issue page in GitHub)
2. **Confirm** link appears in issue's "Development" section
3. **If link missing**: Edit PR description to add "Closes #{number}"

## PR Requirements

- **One PR per issue** - Each GitHub issue gets exactly ONE pull request
- PR must be linked to GitHub issue
- PR title should be clear and descriptive
- PR description should summarize changes
- **Always enable auto-merge** (`gh pr merge --auto --rebase`)
- Do NOT create PR without linking to issue
- Do NOT combine multiple issues into a single PR

## Responding to Review Comments

Reply to EACH review comment individually using GitHub API:

```bash
# Get review comment IDs
gh api repos/OWNER/REPO/pulls/PR/comments --jq '.[] | {id, path, body}'

# Reply to each comment
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST -f body="Fixed - [brief description]"

# Verify replies posted
gh api repos/OWNER/REPO/pulls/PR/comments --jq '.[] | select(.in_reply_to_id)'
```

## Response Format

Keep responses SHORT and CONCISE (1 line preferred):

- `Fixed - Updated conftest.py to use real repository root`
- `Fixed - Deleted redundant test file`
- `Fixed - Removed deprecated section from README`

## Post-Merge Cleanup

After a PR is merged/rebased to main, clean up the worktree and branch:

```bash
# 1. Remove the worktree
git worktree remove worktrees/<issue-number>-<description>

# 2. Delete local branch
git branch -d <branch-name>

# 3. Delete remote branch
git push origin --delete <branch-name>

# 4. Prune stale references
git worktree prune
```

**Important:** Always clean up after merge to avoid accumulating stale worktrees and branches.

See [AGENTS.md](../../AGENTS.md#git-workflow) for complete PR creation instructions.
