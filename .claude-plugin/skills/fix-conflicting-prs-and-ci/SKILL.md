---
name: fix-conflicting-prs-and-ci
description: Batch rebase conflicting PR branches and fix CI failures on main
category: ci-cd
user-invocable: false
date: 2026-03-11
objective: Resolve merge conflicts on 7 PR branches and fix 3 CI failures on main
outcome: All branches rebased and mergeable; CI fix PR created
---

# fix-conflicting-prs-and-ci

## When to Use

- Multiple PR branches have merge conflicts with main
- CI is failing on main and needs fixing before PRs can merge
- Test files were split but CI workflow wasn't updated
- Mojo nightly breaks `py=` keyword args or deprecates `alias`

## Verified Workflow

### 1. Batch Rebase Conflicting Branches

```bash
git fetch origin
for branch in branch1 branch2 branch3; do
  git checkout $branch
  git rebase origin/main
  # Resolve conflicts
  git rebase --continue
done
git push --force-with-lease origin branch1 branch2 branch3
```

### 2. Resolve Common Conflict Patterns

- **Add/add conflicts**: Keep the richer version (more tests, better docstrings)
- **Modify/delete**: If main deleted the file, accept deletion (`git rm`)
- **New method additions (HEAD empty)**: Keep the PR's addition
- **Import ordering**: Merge both, remove duplicates

### 3. Fix Mojo Nightly Compilation Errors

```mojo
# BROKEN (py= keyword removed in nightly)
var int_val = Int(py=val_obj)
var float_val = Float64(py=val_obj)

# FIXED (portable workaround)
var int_val = atol(String(val_obj))
var float_val = Float64(atof(String(val_obj)))
```

### 4. Fix `alias` Deprecation

```mojo
# BROKEN (deprecated in nightly)
alias THRESHOLD = 1000

# FIXED
comptime THRESHOLD = 1000
```

### 5. Apply Formatting from CI When Local Formatter Crashes

Local `mojo format` may crash with `'_python_symbols' object has no attribute
'comptime_assert_stmt'`. Extract the diff from CI logs instead:

```bash
# Get failed run ID
gh run list --branch <branch> --status completed --json name,conclusion,databaseId \
  --jq '.[] | select(.conclusion == "failure") | {name, id: .databaseId}'

# Extract clean patch from CI logs
gh run view <run-id> --log-failed 2>&1 \
  | grep "Run pre-commit hooks" \
  | sed 's/^.*Run pre-commit hooks\t[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T[0-9:\.]*Z //' \
  > /tmp/ci_clean.txt

sed -n '/^diff --git/,/^##\[error\]/p' /tmp/ci_clean.txt \
  | grep -v '^\(##\[error\]\)' > /tmp/format.patch

git apply /tmp/format.patch
```

### 6. Update CI Workflow for Split Test Files

When test files are split (`test_foo.mojo` -> `test_foo_part1.mojo`, etc.),
update `comprehensive-tests.yml` to use wildcards:

```yaml
# BEFORE (breaks when files are split)
pattern: "test_tensors.mojo test_broadcasting.mojo test_creation.mojo"

# AFTER (auto-discovers split files)
pattern: "test_tensors*.mojo test_broadcasting*.mojo test_creation*.mojo"
```

### 7. Verify

```bash
# Check all PRs are mergeable
for pr in 1234 5678; do
  echo -n "PR #$pr: "
  gh pr view $pr --json mergeable --jq .mergeable
done
```

## Failed Attempts

| Attempt | Why Failed | Lesson Learned |
| --- | --- | --- |
| First CI fix only addressed `py=` keyword | Missed `view_with_strides` error, 34 formatting files, test coverage gaps | Always get ALL failure logs from ALL failed jobs before fixing |
| Ran local `mojo format` on files | Crashes with `comptime_assert_stmt` error on nightly-formatted files | Extract formatting patch from CI logs instead of running locally |
| Listed specific test files in CI workflow | Breaks every time files are split | Use wildcard patterns like `test_foo*.mojo` |

## Results & Parameters

- **Branches rebased**: 7 (across 2 sessions)
- **Conflict types**: add/add, modify/delete, content merge
- **CI failures fixed**: compilation (py= keyword, view_with_strides), formatting (34 files), test coverage validation
- **Key files**: `shared/utils/toml_loader.mojo`, `shared/core/extensor.mojo`, `shared/core/matrix.mojo`, `.github/workflows/comprehensive-tests.yml`
- **Safe push**: Always use `git push --force-with-lease` (never `--force`)
