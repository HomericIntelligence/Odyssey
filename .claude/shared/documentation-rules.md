# Documentation Rules

Shared documentation rules for all agents. Reference this file instead of duplicating.

## Output Location

**All implementation notes must be posted as comments on the GitHub issue.**

## Before Starting Work

1. **Verify GitHub issue number** is provided
2. **Read issue context**: `gh issue view <issue> --comments`
3. **Check for prior work**: Look for linked PRs and existing comments
4. **If no issue number provided**: STOP and escalate - request issue creation first

## Reading from GitHub Issues

```bash
# Get issue details
gh issue view <number>

# Get all comments (implementation history)
gh issue view <number> --comments

# Get structured data
gh issue view <number> --json title,body,comments,labels,state

# Check linked PRs
gh pr list --search "issue:<number>"
```

## Writing to GitHub Issues

```bash
# Short status update
gh issue comment <number> --body "Status: [brief update]"

# Detailed implementation notes
gh issue comment <number> --body "$(cat <<'EOF'
## Implementation Notes

### Summary
[what was done]

### Files Changed
- path/to/file.mojo

### Verification
- [x] Tests pass
- [x] Linting passes
EOF
)"
```

## What Goes Where

| Content | Location |
| --- | --- |
| Issue-specific work | GitHub issue comments |
| Architectural decisions | `/notes/review/` |
| Team guides | `/agents/` |
| Comprehensive specs | `/notes/review/` |

## Rules

- Write ALL findings, decisions, and outputs as GitHub issue comments
- Link to comprehensive docs in `/notes/review/` and `/agents/` (don't duplicate)
- Keep issue-specific content focused and concise
- Reference related issues with `#<number>` format
- Post completion summaries before creating PR

## Templates

### Implementation Started

```markdown
## Implementation Started

**Branch**: `<branch-name>`

### Approach
[Brief description]

### Files to Modify
- `path/to/file1`
- `path/to/file2`
```

### Implementation Complete

```markdown
## Implementation Complete

**PR**: #<pr-number>

### Summary
[What was implemented]

### Verification
- [x] Tests pass
- [x] Pre-commit passes
```

See [github-issue-workflow.md](./github-issue-workflow.md) for complete workflow patterns.
