# Fix Conflicting PRs and CI - Raw Notes

## Session Date: 2026-03-11

## Branches Fixed

### Session 1 (3 branches)

| PR | Branch | Conflict Type | Resolution |
|----|--------|---------------|------------|
| #4317 | 3476-auto-impl | add/add in test_extensor_abs_ops.mojo, rename/delete in test_extensor_slicing.mojo.DEPRECATED | Kept incoming tests (more comprehensive), accepted main's deletion |
| #3933 | 3311-auto-impl | content in test_migrate_odyssey_skills.py | Merged import ordering, kept new test classes from PR |
| #3885 | 3285-auto-impl | modify/delete in test_alexnet_layers.mojo | Accepted main's deletion (file was split) |

### Session 2 (4 branches)

| PR | Branch | Conflict Type | Resolution |
|----|--------|---------------|------------|
| #3836 | 3275-auto-impl | 3 conflicts in extensor.mojo | Kept PR's new __setitem__ method, kept HEAD's richer __str__ with truncation |
| #4053 | 3379-auto-impl | 2 conflicts in test_utility.mojo | Kept PR's new test function + call |
| #4054 | 3380-auto-impl | 2 conflicts in test_utility.mojo | Kept PR's new test function + call |
| #4059 | 3383-auto-impl | 2 conflicts in test_utility.mojo | Kept PR's new test function + call |

## CI Failures on Main

### 1. Compilation Error - `py=` keyword removed

```text
shared/utils/toml_loader.mojo:104:36: error: no matching function in initialization
    var float_val = Float64(py=val_obj)
```

**Root cause**: Mojo nightly removed `py=` keyword argument from `Float64()` and `Int()`.

**Fix**: `atol(String(val_obj))` for Int, `Float64(atof(String(val_obj)))` for Float64.

### 2. Compilation Error - missing `view_with_strides`

```text
shared/core/matrix.mojo:824:18: error: 'ExTensor' value has no attribute 'view_with_strides'
    return tensor.view_with_strides(result_shape, result_strides)
```

**Root cause**: `transpose()` called a method that was never implemented on ExTensor.

**Fix**: Replaced with element-by-element data copy using multi-index decomposition
and permuted indices.

### 3. Pre-commit Formatting - 34+ files

**Root cause**: CI uses nightly `mojo format` which has stricter line-length enforcement.
Local `mojo format` (stable) crashes on these files with `comptime_assert_stmt` error.

**Fix**: Extracted diff from CI logs as a patch and applied with `git apply`.

### 4. Test Coverage Validation

**Root cause**: Test files split per ADR-009 (e.g., `test_tensors.mojo` ->
`test_tensors_part1.mojo` + `test_tensors_part2.mojo` + `test_tensors_part3.mojo`)
but `comprehensive-tests.yml` still listed old names.

**Fix**: Changed explicit file lists to wildcard patterns (`test_tensors*.mojo`).

### 5. Test Count Badge Stale

**Root cause**: Badge showed 379+ tests but actual count was 498 after splits.

**Fix**: `python3 scripts/check_test_count_badge.py --fix`

## Key Commands

```bash
# Batch push multiple rebased branches
git push --force-with-lease origin branch1 branch2 branch3

# Check mergeability of multiple PRs
for pr in 1234 5678; do
  gh pr view $pr --json mergeable --jq .mergeable
done

# Extract formatting patch from CI
gh run view <id> --log-failed 2>&1 | grep "Run pre-commit hooks" | ...

# Fix test count badge
python3 scripts/check_test_count_badge.py --fix
```

## Worktree Management

```bash
# Remove worktree after PR work is done
git worktree remove .worktrees/pr-XXXX
git worktree prune

# Verify only main worktree remains
git worktree list
```
