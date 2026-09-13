Implement GitHub issue #$ARGUMENTS.

1. Read issue context: `gh issue view $ARGUMENTS --comments`
2. Understand requirements and acceptance criteria
3. Check if a worktree exists for this issue, create if needed:
   - `git worktree add worktree/$ARGUMENTS-<desc> -b $ARGUMENTS-<desc>`
4. Implement the changes following project standards from AGENTS.md using sub-agents in parallel
5. Spawn the dedicated test-runner sub-agent described in `AGENTS.md` to run
   `just test-mojo`; triage its PASS/FAIL report and delegate any repair
   or rerun. Do not execute test commands directly.
6. Commit with conventional commit format
7. Create PR linked to issue: `gh pr create --body "Closes #$ARGUMENTS"`
8. Enable auto-merge: `gh pr merge --auto --rebase`
