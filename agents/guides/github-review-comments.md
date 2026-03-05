# GitHub Review Comment Handling Guide

This guide explains how to properly handle GitHub pull request review comments, including the critical distinction
between PR-level comments and review comment replies.

## Overview

When addressing PR review feedback, there are two types of comments:

1. **PR-level comments** - General comments visible in the PR timeline
1. **Review comment replies** - Specific replies to inline code review comments

**CRITICAL**: These are NOT interchangeable. Using the wrong type will result in incomplete work.

## The Critical Difference

### PR-Level Comments

### What they are

- General comments on the entire PR
- Visible in the PR conversation/timeline tab
- Posted using `gh pr comment`

### When to use

- Providing general updates about the PR
- Asking questions about the PR as a whole
- Posting summary information

### Command

```bash
gh pr comment <pr-number> --body "Your comment here"
```text

### Example:

```bash
gh pr comment 1559 --body "All review comments have been addressed"
```text

### Review Comment Replies

### What they are

- Specific replies to inline code review comments
- Visible in the "Files changed" tab under the specific line of code
- Nested under the original review comment
- Marked as "in reply to" the original comment

### When to use

- Responding to specific code review feedback
- Confirming fixes to specific issues
- THIS IS WHAT YOU NEED when a reviewer leaves comments on specific lines of code

### Command

```bash
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments/COMMENT_ID/replies \
  --method POST \
  -f body="Your reply here"
```text

### Example:

```bash
gh api repos/HomericIntelligence/ProjectOdyssey/pulls/1559/comments/2524809837/replies \
  --method POST \
  -f body="✅ Fixed - Updated conftest.py to use real repository root"
```text

## Complete Workflow: Addressing Review Comments

### Step 1: List All Review Comments

```bash
# Get all review comments on a PR
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments \
  --jq '.[] | {id: .id, path: .path, line: .line, author: .user.login, body: .body}'
```text

### Example output:

```json
{
  "id": 2524809837,
  "path": "tests/foundation/docs/test_advanced_docs.py",
  "line": null,
  "author": "mvillmow",
  "body": "Don't use a mock repository root, use the real repository."
}
```text

### Step 2: Filter for Specific Reviewer's Comments

```bash
# Get only comments from a specific reviewer (e.g., the user)
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments \
  --jq '.[] | select(.user.login == "mvillmow") | {id: .id, path: .path, body: .body}'
```text

### Step 3: Make Your Code Changes

- Check out the PR branch
- Make the requested changes
- Commit and push

### Step 4: Reply to EACH Review Comment

**IMPORTANT**: You must reply to EACH comment individually, not post one general comment.

```bash
# For each comment ID from Step 1, post a reply
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments/COMMENT_ID_1/replies \
  --method POST \
  -f body="✅ Fixed - [brief description of what you did]"

gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments/COMMENT_ID_2/replies \
  --method POST \
  -f body="✅ Fixed - [brief description of what you did]"

# Continue for all comments
```text

### Reply format:

```text
✅ Fixed - [brief description]
```text

### Good examples:

- `✅ Fixed - Updated conftest.py to use real repository root instead of mock tmp_path`
- `✅ Fixed - Deleted test_link_validation.py (421 lines) since link validation is handled by pre-commit`
- `✅ Fixed - Removed markdown linting section from README.md`

### Bad examples:

- `Fixed` (too vague)
- `Done` (doesn't explain what was done)
- `See commit abc123` (make it self-contained)

### Step 5: Verify Replies Were Posted

```bash
# Check that your replies show up with in_reply_to_id set
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments \
  --jq '.[] | select(.user.login == "YOUR_USERNAME" and .in_reply_to_id) | {replying_to: .in_reply_to_id, body: .body}'
```text

### Expected output:

```json
{
  "replying_to": 2524809837,
  "body": "✅ Fixed - Updated conftest.py to use real repository root"
}
{
  "replying_to": 2524810446,
  "body": "✅ Fixed - Removed mock directories from fixtures"
}
```text

**If you see no output**: Your replies were NOT posted correctly. Try again.

### Step 6: Check CI Status

After pushing changes, verify CI passes:

```bash
# Wait 30 seconds for CI to start
sleep 30

# Check CI status
gh pr checks PR_NUMBER

# Look for any failing checks
```text

### Expected output:

```text
All checks have passed
```text

**If you see failures**: Investigate and fix before reporting completion.

## Common Mistakes to Avoid

### ❌ Mistake 1: Using PR Comment Instead of Review Reply

```bash
# WRONG - This does NOT reply to review comments
gh pr comment 1559 --body "Fixed all review comments"
```text

**Why it's wrong**: This creates a general PR comment, not replies to specific review comments.

### ❌ Mistake 2: Not Replying to Each Comment

```bash
# WRONG - Only replied to one comment when there were 5
gh api repos/.../pulls/1559/comments/2524809837/replies --method POST -f body="Fixed everything"
```text

**Why it's wrong**: Each review comment needs its own individual reply.

### ❌ Mistake 3: Not Verifying Replies Posted

**Why it's wrong**: You can't confirm your work succeeded without verification.

### ❌ Mistake 4: Using Wrong API Endpoint

```bash
# WRONG - This endpoint doesn't exist
gh api repos/.../pulls/comments/2524809837/replies --method POST -f body="Fixed"

# CORRECT - Include full path
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments/2524809837/replies --method POST -f body="Fixed"
```text

## Quick Reference

### Get review comments

```bash
gh api repos/OWNER/REPO/pulls/PR/comments
```text

### Get comment IDs for a specific reviewer

```bash
gh api repos/OWNER/REPO/pulls/PR/comments --jq '.[] | select(.user.login == "REVIEWER") | .id'
```text

### Reply to a review comment

```bash
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies --method POST -f body="✅ Fixed - [description]"
```text

### Verify replies posted

```bash
gh api repos/OWNER/REPO/pulls/PR/comments --jq '.[] | select(.in_reply_to_id)'
```text

### Check CI status

```bash
gh pr checks PR
```text

## When to Use Each Type

| Situation | Use This |
|-----------|----------|
| Reviewer left comments on specific lines of code | Review comment replies |
| Need to reply to each review comment | Review comment replies |
| General update about the PR | PR-level comment |
| Asking a question about the PR direction | PR-level comment |
| Summary of changes made | PR-level comment |

## Real Example from PR #1559

**Scenario**: User left 5 review comments on PR #1559

### Review comments:

1. Comment ID 2524809837: "Don't use a mock repository root"
1. Comment ID 2524810446: "don't use mock directories"
1. Comment ID 2525151897: "This is handled by precommit, so can be deleted"
1. Comment ID 2525152000: "This is handled by precommit, so can be deleted"
1. Comment ID 2525154295: "This isn't needed, it is handled by precommit"

### Correct approach:

```bash
# Reply to each comment individually
gh api repos/HomericIntelligence/ProjectOdyssey/pulls/1559/comments/2524809837/replies \
  --method POST \
  -f body="✅ Fixed - Updated conftest.py to use real repository root instead of mock tmp_path"

gh api repos/HomericIntelligence/ProjectOdyssey/pulls/1559/comments/2524810446/replies \
  --method POST \
  -f body="✅ Fixed - Updated conftest.py to use real directories by navigating from tests/foundation/docs/ to repository root"

gh api repos/HomericIntelligence/ProjectOdyssey/pulls/1559/comments/2525151897/replies \
  --method POST \
  -f body="✅ Fixed - Deleted test_link_validation.py (421 lines) since link validation is handled by pre-commit"

gh api repos/HomericIntelligence/ProjectOdyssey/pulls/1559/comments/2525152000/replies \
  --method POST \
  -f body="✅ Fixed - Deleted test_markdown_linting.py (778 lines) since markdown linting is handled by pre-commit"

gh api repos/HomericIntelligence/ProjectOdyssey/pulls/1559/comments/2525154295/replies \
  --method POST \
  -f body="✅ Fixed - Removed markdown linting section from README.md since it's handled by pre-commit hooks"
```text

### Verify:

```bash
gh api repos/HomericIntelligence/ProjectOdyssey/pulls/1559/comments \
  --jq '.[] | select(.in_reply_to_id) | {replying_to: .in_reply_to_id, body: .body}'
```text

### Expected: 5 replies, one for each comment.
