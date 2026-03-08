# Skill: gh-deduplicate-issues

## Overview

| Field | Value |
| ----- | ----- |
| **Date** | 2026-03-08 |
| **Category** | tooling |
| **Objective** | Close duplicate GitHub issues with "Duplicate of #XXXX" comment, keeping oldest as canonical survivor |
| **Outcome** | ✅ Success - 114 issues closed across 9 groups, open count reduced 683→570 |

## When to Use

Invoke this skill when:

- The repository has accumulated many near-duplicate issues from repeated sessions
- You need to consolidate issues with the same intent but slightly different wording
- Backlog clutter is making it hard to track actual work
- Issues have been recreated 10+ times across multiple sessions (common with ADR-related issues)

## Verified Workflow

### Step 1: Identify Duplicate Groups

Before closing anything, enumerate groups. For each group identify:

- **Survivor**: the oldest (lowest-numbered) open issue
- **Duplicates**: all other open issues with the same intent

Use `gh issue list` with JSON output to gather data efficiently:

```bash
gh issue list --repo OWNER/REPO --state open --json number,title --limit 1000 \
  | jq '.[] | select(.title | test("ADR-009"))' | head -50
```

### Step 2: Write the Close Function

```bash
close_duplicate() {
    local issue=$1
    local survivor=$2
    local repo=$3
    echo -n "Closing #$issue (dup of #$survivor)... "
    gh issue comment "$issue" --repo "$repo" --body "Duplicate of #$survivor" 2>/dev/null && \
    gh issue close "$issue" --repo "$repo" --reason "not planned" 2>/dev/null && \
    echo "done" || echo "FAILED"
}
```

**Key details:**
- Always comment BEFORE closing (comment on closed issues still works, but comment first is cleaner)
- Use `--reason "not planned"` for duplicate closures
- Suppress stderr with `2>/dev/null` to avoid noise from already-closed issues

### Step 3: Execute in Batches of 30

GitHub's API handles this volume fine (~5000 req/hr limit), but batching in 30s keeps
output manageable and makes it easy to spot failures:

```bash
REPO="OWNER/REPO"
SURVIVOR=3962

for issue in 4098 4103 4104 4105 ... ; do
    close_duplicate "$issue" "$SURVIVOR" "$REPO"
done
```

### Step 4: Verify Results

```bash
# Check all survivors are still open
for issue in 3962 3776 4100; do
    state=$(gh issue view "$issue" --repo "$REPO" --json state --jq '.state')
    echo "#$issue: $state"
done

# Spot-check closed issues have the right comment
for issue in 4098 4320; do
    gh issue view "$issue" --repo "$REPO" \
        --json state,comments \
        --jq '{state: .state, last_comment: .comments[-1].body}'
done

# Count remaining open issues
gh issue list --repo "$REPO" --state open --json number --limit 1000 \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Open: {len(d)}')"
```

## Failed Attempts

### 1. Closing Before Commenting

**What was tried:** `gh issue close` first, then `gh issue comment`

**Why it mattered:** Not actually a failure — GitHub allows comments on closed issues.
But commenting first is the canonical order because:

- The "Duplicate of #X" comment explains WHY it was closed
- If close fails but comment succeeds, the issue is at least annotated
- GitHub's duplicate closure UI does it in this order

**Resolution:** Always comment first, close second.

### 2. Using `--reason "duplicate"` flag

**What was tried:** `gh issue close "$issue" --repo "$REPO" --reason "duplicate"`

**Why it failed:** GitHub's CLI only accepts `"completed"` or `"not planned"` as close reasons.
`"duplicate"` is a valid state reason in the GitHub web UI but not exposed as a CLI option.

**Resolution:** Use `--reason "not planned"` with the comment providing the duplicate reference.

### 3. Single Giant Loop Without Batching

**What was tried:** Running all 114 issues in a single loop

**Why it didn't fail but could:** If rate limits kick in mid-loop, you get partial state with
no easy way to see which issues were processed. Batching by group makes it easy to resume.

**Resolution:** Group by intent (9 groups), run each group separately with echo markers.

## Results & Parameters

### Session Metrics

| Metric | Value |
| ------ | ----- |
| Total issues closed | 114 |
| Groups processed | 9 |
| Survivors kept open | 9 |
| Open count before | 683 |
| Open count after | 570 |
| Success rate | 100% |
| Time to execute | ~8 minutes |

### Group Summary

| Group | Theme | Survivor | Closed |
| ----- | ----- | -------- | ------ |
| 1 | ADR-009 compliance/enforcement | #3962 | 79 |
| 2 | ADR-009 documentation | #3776 | 11 |
| 3 | Remove continue-on-error Core Tensors | #4100 | 9 |
| 4 | Core Utilities CI splitting | #4116 | 7 |
| 5 | ADR-009 apply-split-to-others | #4150 | 4 |
| 6 | Core Activations CI glob | #4157 | 1 |
| 7 | Remove continue-on-error Core Loss | #4172 | 1 |
| 8 | Negative index `__setitem__` | #3387 | 2 |
| 9 | validate_test_coverage track splits | #4109 | 1 |

### Complete Close Script Template

```bash
#!/bin/bash
REPO="OWNER/REPO"

close_duplicate() {
    local issue=$1
    local survivor=$2
    echo -n "Closing #$issue (dup of #$survivor)... "
    gh issue comment "$issue" --repo "$REPO" --body "Duplicate of #$survivor" 2>/dev/null && \
    gh issue close "$issue" --repo "$REPO" --reason "not planned" 2>/dev/null && \
    echo "done" || echo "FAILED"
}

# Group 1: <theme> (survivor: #XXXX)
SURVIVOR=XXXX
for issue in A B C D; do
    close_duplicate "$issue" "$SURVIVOR"
done

# Verify survivors
for survivor in XXXX YYYY; do
    state=$(gh issue view "$survivor" --repo "$REPO" --json state --jq '.state')
    echo "#$survivor: $state"
done
```

## Consolidation Rules

When deciding what counts as a duplicate, apply these rules:

1. **Same intent = duplicate** — "Add ADR-009 pre-commit hook" and "Enforce ADR-009 fn test_ limit via CI" are the same goal
2. **Oldest issue wins** — Always keep the lowest issue number as survivor
3. **Unique feature requests stay** — Even if related to the same area, keep issues that request distinct features
4. **When in doubt, keep separate** — If two issues could produce different PRs, they're not duplicates

## Platform Notes

- GitHub CLI (`gh`) handles rate limiting automatically at ~5000 req/hr
- At 2 API calls per issue (comment + close), 114 issues = 228 calls — well within limits
- The `2>/dev/null` suppression on already-closed issues prevents false alarm output
- Batch size of 30 is arbitrary but provides good progress visibility
