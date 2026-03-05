# Agent Verification Checklist

This checklist ensures agents verify their work before reporting completion. Following these steps prevents
false positive reporting and ensures tasks are actually complete.

## Core Principle

### Report actual results, not attempted actions

- ❌ Bad: "I pushed the changes"
- ✅ Good: "I pushed the changes and verified CI is passing"

- ❌ Bad: "I replied to all review comments"
- ✅ Good: "I replied to all 5 review comments and verified the replies are visible"

## Standard Verification Workflow

### 1. Before Starting Work

#### Understand the Requirements

- [ ] Read the complete task description
- [ ] Identify all deliverables
- [ ] Note any verification requirements mentioned
- [ ] Clarify any ambiguous requirements with the user

#### Identify Success Criteria

- [ ] What does "done" mean for this task?
- [ ] How will I verify each requirement was met?
- [ ] What evidence will prove the work is complete?

### 2. During Work

#### Track Progress

- [ ] Check off each requirement as it's completed (not attempted)
- [ ] Keep notes of any issues encountered
- [ ] Document any deviations from the plan

#### Verify Each Step

- [ ] After each significant action, verify it succeeded
- [ ] Don't assume commands worked - check the results
- [ ] If an API call returns, parse the response to confirm success

### 3. After Pushing Code Changes

#### Wait for CI to Start

```bash
# Wait 30 seconds for CI to begin
sleep 30
```text

**Why**: CI doesn't start instantly. Checking too early shows incomplete status.

#### Check CI Status

```bash
# View all CI checks for a PR
gh pr checks <pr-number>
```text

### Expected output:

```text
All checks have passed
✓ pre-commit        pass  21s  https://...
✓ build-validation  pass  15s  https://...
✓ test-python       pass  33s  https://...
```text

### If you see failures:

```bash
# Get detailed failure information
gh pr checks <pr-number> --web
```text

- [ ] All checks passing OR
- [ ] Failures are expected (document why) OR
- [ ] Failures are unrelated to your changes (document evidence)

#### Verify Pre-commit Locally Matches CI

```bash
# Run the same pre-commit checks that CI runs
just pre-commit-all

# Or if using pixi
just pre-commit-all
```text

**Why**: Sometimes local pre-commit passes but CI fails (different Mojo version, different environment).

- [ ] Local pre-commit matches CI behavior
- [ ] If they differ, understand why and address it

### 4. After Posting Review Comment Replies

#### Verify Each Reply Posted

```bash
# Get all your review comment replies
gh api repos/OWNER/REPO/pulls/PR/comments \
  --jq '.[] | select(.user.login == "YOUR_USERNAME" and .in_reply_to_id) | {replying_to: .in_reply_to_id, body: .body[:60]}'
```text

**Expected**: One entry for each review comment you replied to.

### Verification checklist:

- [ ] Count of replies matches count of review comments
- [ ] Each reply has correct `replying_to` ID
- [ ] Reply text is what you intended
- [ ] No duplicate replies

### If missing replies:

- Check the original review comments again
- Identify which ones are missing
- Post replies to the missing comments
- Verify again

#### Cross-Reference with Original Comments

```bash
# Get the original review comments you were replying to
gh api repos/OWNER/REPO/pulls/PR/comments \
  --jq '.[] | select(.user.login == "REVIEWER" and .in_reply_to_id == null) | .id'

# Compare this list with your replies' replying_to IDs
```text

- [ ] Every original comment has a reply
- [ ] No comments were missed

### 5. After Making File Changes

#### Verify Files Exist

```bash
# Check that files you created actually exist
ls -la path/to/file

# Check file contents if needed
head -20 path/to/file
```text

- [ ] All expected files exist
- [ ] File contents are correct
- [ ] File permissions are appropriate

#### Verify Files Were Deleted

```bash
# Check that files you deleted are actually gone
test -f path/to/file && echo "STILL EXISTS" || echo "Deleted successfully"
```text

- [ ] All files marked for deletion are gone
- [ ] No orphaned references to deleted files

#### Verify Git Tracking

```bash
# Check what git sees
git status

# Verify staged changes
git diff --cached --stat
```text

- [ ] All intended changes are staged
- [ ] No unintended changes are included
- [ ] No untracked files that should be tracked

### 6. After Creating GitHub Issues

#### Verify Issue Was Created

```bash
# Get the issue you just created
gh issue view <issue-number>
```text

- [ ] Issue number is valid
- [ ] Title is correct
- [ ] Body content is complete
- [ ] Labels are correct
- [ ] Issue is in correct state (open/closed)

#### Verify Issue Content

- [ ] All sections are present
- [ ] Code blocks render correctly
- [ ] Links work
- [ ] No formatting issues

### 7. After Updating Documentation

#### Verify Links Work

```bash
# Test internal links (if file exists)
# Example: If doc references "./other-file.md"
test -f other-file.md && echo "Link target exists" || echo "BROKEN LINK"
```text

- [ ] All internal links point to existing files
- [ ] Relative paths are correct
- [ ] No absolute paths (unless intentional)

#### Verify Code Examples

```bash
# If the documentation includes bash examples, test them
bash -n script-in-docs.sh  # Syntax check
```text

- [ ] Code examples are valid
- [ ] Commands are correct
- [ ] Examples match current codebase

### 8. Before Reporting Completion

#### Final Requirements Check

- [ ] Re-read the original task description
- [ ] Verify EACH requirement was met (not just attempted)
- [ ] Check for any "false positive" completions

### Question checklist:

- Did I actually DO what I said I did?
- Can I prove it with evidence?
- If someone checks my work, will they find it complete?
- Are there any edge cases I missed?

#### Final Verification

```bash
# Summary check
git status                          # What's staged?
gh pr checks <pr>                   # CI passing?
gh api repos/.../pulls/<pr>/comments --jq '.[] | select(.in_reply_to_id)' | wc -l  # Review replies count?
```text

- [ ] All changes committed and pushed
- [ ] All CI checks passing
- [ ] All review comments replied to
- [ ] All files created/modified as intended
- [ ] All verification steps completed

#### Evidence Collection

Gather evidence for your completion report:

- [ ] CI status screenshot or output
- [ ] Review comment reply verification
- [ ] File change verification
- [ ] Test results (if applicable)

## Common Verification Mistakes

### Mistake 1: Checking Too Early

❌ **Wrong:**

```bash
git push
gh pr checks <pr>  # Checked immediately - CI not started yet!
```text

✅ **Correct:**

```bash
git push
sleep 30  # Wait for CI to start
gh pr checks <pr>
```text

### Mistake 2: Assuming API Success

❌ **Wrong:**

```bash
gh api repos/.../pulls/1559/comments/123/replies --method POST -f body="Fixed"
# Assumed it worked without checking
```text

✅ **Correct:**

```bash
result=$(gh api repos/.../pulls/1559/comments/123/replies --method POST -f body="Fixed")
echo "$result" | jq .id  # Verify response has an ID
```text

### Mistake 3: Not Counting Results

❌ **Wrong:**

"I replied to all review comments" (without checking count)

✅ **Correct:**

```bash
# Original comments: 5
gh api repos/.../pulls/PR/comments --jq '.[] | select(.user.login == "mvillmow" and .in_reply_to_id == null)' | jq -s 'length'
# Output: 5

# My replies: 5
gh api repos/.../pulls/PR/comments --jq '.[] | select(.in_reply_to_id)' | jq -s 'length'
# Output: 5

# ✓ Counts match - all comments replied to
```text

### Mistake 4: Confusing Local and CI

❌ **Wrong:**

"Pre-commit passed" (only checked locally, didn't check CI)

✅ **Correct:**

```bash
# Check local
just pre-commit-all
# ✓ Passed

# Check CI
gh pr checks <pr>
# ✓ All checks passing

# Both must pass
```text

### Mistake 5: Reporting Attempts as Results

❌ **Wrong:**

- "I tried to push the changes"
- "I attempted to reply to comments"
- "I ran the tests"

✅ **Correct:**

- "I pushed the changes and verified CI is passing"
- "I replied to all 5 comments and verified the replies are visible"
- "I ran the tests and all 233 tests passed"

## Verification Scripts

### Quick Verification Script for PR Work

```bash
#!/bin/bash
# verify-pr-work.sh <pr-number>

PR=$1

echo "=== Verifying PR #$PR ==="

echo -e "\n1. Git status:"
git status --short

echo -e "\n2. CI checks:"
gh pr checks $PR

echo -e "\n3. Review comment replies:"
reply_count=$(gh api repos/HomericIntelligence/ProjectOdyssey/pulls/$PR/comments --jq '.[] | select(.in_reply_to_id)' | jq -s 'length')
echo "Found $reply_count review comment replies"

echo -e "\n4. Original review comments:"
comment_count=$(gh api repos/HomericIntelligence/ProjectOdyssey/pulls/$PR/comments --jq '.[] | select(.in_reply_to_id == null and .user.login == "mvillmow")' | jq -s 'length')
echo "Found $comment_count original review comments"

if [ "$reply_count" -eq "$comment_count" ]; then
    echo "✓ All review comments have replies"
else
    echo "✗ Missing replies: replied to $reply_count of $comment_count comments"
fi
```text

### Quick Verification Script for File Changes

```bash
#!/bin/bash
# verify-file-changes.sh

echo "=== Verifying File Changes ==="

echo -e "\n1. Staged changes:"
git diff --cached --name-status

echo -e "\n2. Unstaged changes:"
git diff --name-status

echo -e "\n3. Untracked files:"
git ls-files --others --exclude-standard

echo -e "\n4. Files to be committed:"
git diff --cached --stat
```text

## Summary

Before reporting any task as complete:

1. ✓ Verify each action succeeded (don't assume)
1. ✓ Check CI status after pushing
1. ✓ Verify review comments were replied to
1. ✓ Count results and compare with expectations
1. ✓ Re-read requirements and confirm completion
1. ✓ Report actual results with evidence

**Remember**: It's better to report "I completed X but need to fix Y" than to report "All done" when work is incomplete.

## Related

- Issue #1573: Agent Improvements: PR Review Comment Handling
- GitHub Review Comments Guide: [github-review-comments.md](./github-review-comments.md)
