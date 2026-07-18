# GitHub Issue Workflow

This document defines the standard workflow for reading from and writing to GitHub issues.
All agents and skills should follow these patterns for issue-based documentation.

## Reading from GitHub Issues

### Get Issue Details

```bash
# Get issue description and metadata
gh issue view <number>

# Get issue with all comments (implementation history)
gh issue view <number> --comments

# Get structured JSON for parsing
gh issue view <number> --json title,body,comments,labels,assignees,milestone,state

# Get specific fields only
gh issue view <number> --json body --jq '.body'
```

### Check Related Items

```bash
# Get linked PRs
gh pr list --search "issue:<number>"

# Get issue timeline (events, references)
gh api repos/{owner}/{repo}/issues/<number>/timeline

# Search for related issues
gh issue list --search "in:body #<number>"
```

### Before Starting Work

1. Run `gh issue view <number>` to understand requirements
2. Run `gh issue view <number> --comments` for prior context and decisions
3. Check linked PRs: `gh pr list --search "issue:<number>"`
4. Note any blockers or dependencies mentioned in comments

## Writing to GitHub Issues

### Status Updates

```bash
# Short status update
gh issue comment <number> --body "Status: Implementation in progress - 3/5 tests passing"

# Progress checkpoint
gh issue comment <number> --body "Checkpoint: Completed tensor operations, starting gradient computation"
```

### Detailed Implementation Notes

```bash
gh issue comment <number> --body "$(cat <<'EOF'
## Implementation Notes

### Changes Made
- Created `shared/core/tensor.mojo` with basic tensor operations
- Added unit tests in `tests/shared/core/test_tensor.mojo`

### Design Decisions
1. Used SIMD for vectorized operations
2. Implemented lazy evaluation for chain operations

### Testing
- All 15 unit tests pass
- Manual verification complete

### Next Steps
- Integration with existing modules
- Performance benchmarking
EOF
)"
```

### Implementation Complete

```bash
gh issue comment <number> --body "$(cat <<'EOF'
## Implementation Complete

**PR**: #<pr-number>

### Summary
[Brief description of what was implemented]

### Files Changed
- `path/to/file1.mojo` - [what changed]
- `path/to/file2.mojo` - [what changed]

### Testing
- All tests pass
- Coverage: [percentage if known]

### Verification
- [x] `uv run pytest` passes
- [x] `just pre-commit-all` passes
- [x] Manual testing complete
EOF
)"
```

### Work In Progress

```bash
gh issue comment <number> --body "$(cat <<'EOF'
## Progress Update

### Completed
- [x] Item 1
- [x] Item 2

### In Progress
- [ ] Item 3 (70% complete)

### Blocked
- Item 4 - waiting on #<other-issue>

### Next Steps
1. Complete item 3
2. Start item 5
EOF
)"
```

### Using Body Files

For longer content or content with special characters:

```bash
# Write content to temp file
cat > /tmp/issue_update.md << 'EOF'
## Update

Content here...
EOF

# Post using body file
gh issue comment <number> --body-file /tmp/issue_update.md
```

## Referencing Issues

### In Commits

```bash
# Reference issue in commit message
git commit -m "feat(tensor): Add SIMD operations

Implements vectorized tensor operations with automatic SIMD detection.

Refs #123"

# Close issue via commit
git commit -m "fix(gradient): Correct backward pass computation

Closes #456"
```

### In Pull Requests

```bash
# Create PR linked to issue
gh pr create --title "feat: Add tensor operations" --body "Closes #123"

# Reference multiple issues
gh pr create --title "refactor: Update core module" --body "
## Summary
Refactored core module for better performance.

Closes #123
Refs #124, #125
"
```

### In Code Comments

```mojo
# Implementation for issue #123
# See https://github.com/owner/repo/issues/123 for design decisions
fn tensor_multiply(a: AnyTensor, b: AnyTensor) -> AnyTensor:
    ...
```

### In Documentation

```markdown
For implementation details, see [Issue #123](https://github.com/owner/repo/issues/123).

Related issues:
- #124 - Test coverage improvements
- #125 - Performance benchmarks
```

## Templates

### New Feature Implementation

```markdown
## Implementation Started

**Issue**: #<number>
**Branch**: `<branch-name>`

### Approach
[Brief description of implementation approach]

### Files to Create/Modify
- [ ] `path/to/file1.mojo`
- [ ] `path/to/file2.mojo`

### Estimated Scope
- [X] Small (< 100 lines)
- [ ] Medium (100-500 lines)
- [ ] Large (> 500 lines)
```

### Bug Fix

```markdown
## Bug Investigation

### Root Cause
[Description of what's causing the bug]

### Fix Approach
[How we'll fix it]

### Testing Plan
- [ ] Add regression test
- [ ] Verify fix in isolation
- [ ] Run full test suite
```

### Review Findings

```markdown
## Review Complete

### Summary
[1-2 sentence summary]

### Findings
1. [Finding 1]
2. [Finding 2]

### Recommendations
- [Recommendation 1]
- [Recommendation 2]

### Action Items
- [ ] [Action 1]
- [ ] [Action 2]
```

## Best Practices

1. **Be Concise**: Keep updates focused and actionable
2. **Use Checklists**: Track progress with `- [ ]` and `- [x]`
3. **Link Related Items**: Reference other issues, PRs, and commits
4. **Update Regularly**: Post progress updates, don't wait until completion
5. **Include Context**: Future readers should understand decisions made
6. **Use Formatting**: Headers, code blocks, and lists improve readability
