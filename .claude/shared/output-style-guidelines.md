# Output Style Guidelines

Consistent output styles improve clarity and actionability.
Follow these guidelines for different contexts.

## Code References

**DO**: Use repo-relative file paths with line numbers when referencing code:

```markdown
GOOD: Updated CLAUDE.md:173-185

GOOD: Modified ExTensor initialization in shared/core/extensor.mojo:45

BAD: Updated CLAUDE.md (missing line numbers)

BAD: Fixed the tensor file (too vague)
```

**DO**: Include relevant code snippets with context:

```markdown
GOOD:
File: /home/user/ProjectOdyssey/shared/core/extensor.mojo:45-52

fn __init__(out self, shape: List[Int], dtype: DType):
    """Initialize tensor with given shape and dtype."""
    self._shape = shape^
    self._dtype = dtype
    var numel = 1
    for dim in shape:
        numel *= dim
    self._data = DTypePointer[dtype].alloc(numel)

BAD: Changed the constructor
```

## Issue and PR Formatting

**DO**: Use structured markdown with clear sections:

```markdown
GOOD:

## Summary
Added comprehensive Claude 4 optimization guidance to CLAUDE.md

## Changes Made
- Added "Extended Thinking" section with when/when-not guidelines
- Added "Thinking Budget Guidelines" table with 5 task types
- Added "Agent Skills vs Sub-Agents" decision tree

## Files Modified
- `/home/user/ProjectOdyssey/CLAUDE.md` (lines 173-500, added 327 lines)

## Verification
- [x] Markdown linting passes
- [x] All code examples use correct syntax
- [x] Cross-references point to existing sections

BAD: Added some docs
```

**DO**: Link to related issues and PRs explicitly:

```markdown
GOOD:
Related Issues:
- Closes #2549
- Related to #2548 (Markdown standards)
- Depends on #2544 (Agent hierarchy)

BAD: Fixes the issue about Claude docs
```

## Code Review Output

**DO**: Provide specific, actionable feedback with examples:

```markdown
GOOD:

**Issue**: Inconsistent parameter naming in ExTensor methods

**Location**: `/home/user/ProjectOdyssey/shared/core/extensor.mojo:120-145`

**Problem**: Methods use both `mut self` and `self` inconsistently

**Recommendation**: Use implicit `read` (just `self`) for read-only methods:

# Current (line 120)
fn shape(mut self) -> List[Int]:  # mut not needed
    return self._shape

# Should be
fn shape(self) -> List[Int]:  # Implicit read
    return self._shape

**Impact**: Misleading API - suggests mutation when none occurs

BAD: The shape method is wrong
```

**DO**: Prioritize feedback by severity:

```markdown
GOOD:

### Critical (Must Fix Before Merge)
1. Memory leak in ExTensor.__del__() - data not freed
2. Missing bounds check in __getitem__() - potential segfault

### Important (Should Fix)
1. Inconsistent parameter naming (mut vs read)
2. Missing docstrings on public methods

### Nice to Have (Consider for Future)
1. Add SIMD optimization to fill() method
2. Consider caching numel() computation

BAD: Here's 20 random issues in no particular order
```

## Terminal Output

**DO**: Use structured formatting for command output:

```bash
$ mojo test tests/shared/core/test_tensor.mojo
Testing: /home/user/ProjectOdyssey/tests/shared/core/test_tensor.mojo
  test_tensor_creation ... PASSED
  test_tensor_indexing ... PASSED
  test_tensor_reshape ... PASSED
All tests passed (3/3)
```

**DO**: Include error context when reporting failures:

```bash
$ mojo build shared/core/extensor.mojo
error: ExTensor.mojo:145:16: cannot transfer ownership of
  non-copyable type
    return self._data
           ^
```
