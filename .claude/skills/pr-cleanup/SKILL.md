---
name: pr-cleanup
description: Handle PR cleanup operations including commit squashing, rebasing, conflict resolution, and pre-merge verification. Ensures PR is merge-ready with clean history and passing CI. Use for PR finalization and cleanup.
mcp_fallback: none
category: github
---

# PR Cleanup Skill

Prepare pull requests for merge by organizing commits, resolving conflicts, and verifying
all pre-merge requirements are met.

## When to Use

- After PR review feedback has been addressed
- When commits need squashing or reorganization
- When PR has conflicts with main branch
- Before requesting final merge approval
- Verifying PR has clean commit history

## Quick Reference

```bash
# Check CI status
gh pr checks <pr>

# Check for conflicts
gh pr view <pr> --json mergeable

# Squash last N commits
git rebase -i HEAD~N

# Rebase onto main
git fetch origin main && git rebase origin/main

# Push after rebase
git push --force-with-lease
```

## Workflow

1. **Verify CI passing** - All checks must pass before cleanup
2. **Check for conflicts** - Identify any conflicts with main branch
3. **Plan commit consolidation** - Decide squash strategy
4. **Execute squashing/rebasing** - Organize commit history
5. **Verify force push safety** - Confirm no one else has pushed
6. **Perform final pre-merge checklist** - All requirements met
7. **Confirm PR ready** - Report status

## Pre-Merge Checklist

- [ ] All review feedback addressed
- [ ] All CI checks passing (no red X)
- [ ] No merge conflicts with main
- [ ] Commits organized logically
- [ ] Commit messages clear and descriptive
- [ ] No extraneous commits (debug, WIP, etc.)
- [ ] Branch based on latest main
- [ ] No unnecessary merges from main
- [ ] Force push safety verified

## Cleanup Operations

**Commit Squashing**:

```bash
# Squash last N commits
git rebase -i HEAD~N
# Mark all but first as 'squash' (s)
# Edit final commit message
git push --force-with-lease
```

**Rebase onto Main**:

```bash
# Rebase feature branch onto main
git fetch origin main
git rebase origin/main
# Resolve conflicts if any
git push --force-with-lease
```

**Conflict Resolution**:

```bash
# Identify conflicts
git status

# Resolve in editor, then mark as resolved
git add <file>

# Continue rebase
git rebase --continue
```

**Commit Message Cleanup**:

```bash
# Interactive rebase for message editing
git rebase -i HEAD~N
# Mark commits to edit with 'e'
# Edit message when prompted
git rebase --continue
```

## Squash Strategy Example

**Scenario**: PR with 8 commits needs consolidation

```text
Before:
  - Commits 1-2: Feature implementation
  - Commits 3-4: Bug fixes from review
  - Commits 5-6: Documentation updates
  - Commits 7-8: Pre-commit formatting

After squash:
  - "feat: Implement feature X"
  - "fix: Address review feedback"
  - "docs: Update feature documentation"
  - "style: Auto-format with pre-commit"
```

## Output Format

Report cleanup status with:

```markdown
[EMOJI] [SEVERITY]: [Issue summary]

Actions Taken:
- Action 1 (result)
- Action 2 (result)

Current Status:
- All CI checks passing
- 0 conflicts with main
- N commits organized as: [description]

Ready for merge: [Yes|No - reason if no]

See: [Link to PR]
```

## Error Handling

| Problem | Solution |
|---------|----------|
| CI failing | Fix root cause before cleanup (see fix-ci-failures skill) |
| Unresolvable conflicts | Escalate to implementation team |
| Complex rebase | Break into smaller steps, resolve one conflict at a time |
| Push rejected | Check if others pushed, use `--force-with-lease` not `--force` |
| Lost commits | Check `git reflog` to recover |

## References

- See `verify-pr-ready` skill for pre-merge validation checklist
- See `gh-batch-merge-by-labels` skill for batch merge operations
- See `fix-ci-failures` skill for CI failure diagnosis
- See `CLAUDE.md` for PR workflow and branch protection rules
