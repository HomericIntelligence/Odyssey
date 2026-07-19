# Mojo Test Patterns and Common Fixes

## Overview

This guide documents common patterns and fixes discovered during comprehensive test suite migration to Mojo v0.26.1+.
Use these patterns when writing new tests or fixing existing test failures.

## Test Entry Points

### Pattern: Main Function Required

**Problem**: Test modules must define a `main()` function as an entry point.

**Solution**: Add a main function that calls all test functions:

```mojo
fn main() raises:
    """Run all tests."""
    test_function_1()
    test_function_2()
    test_function_3()
    # ... all test functions
```

**When to Use**:

- All test files executed with `mojo test`
- Any module that needs an entry point

**Example Files**:

- tests/shared/utils/test_config.mojo (33 test functions)
- tests/shared/utils/test_io.mojo (36 test functions)
- tests/shared/utils/test_logging.mojo (23 test functions)

## Python Interop Patterns

### Pattern: Time Module Access

**Problem**: Mojo doesn't have native `time.now()`, need Python interop.

**Wrong**:

```mojo
from time import now  # Module doesn't exist in Mojo
var timestamp = now()
```

**Correct**:

```mojo
from python import Python

var time_module = Python.import_module("time")
var timestamp = Int(time_module.time())
```

### Pattern: Dictionary Assignment

**Problem**: Using `__setitem__` directly doesn't work with Mojo syntax.

**Wrong**:

```mojo
_ = python.environ.__setitem__("VAR", "value")
```

**Correct**:

```mojo
python.environ["VAR"] = "value"
```

## Import Path Patterns

### Pattern: Correct Module Imports

**Problem**: Test files can't use absolute imports like `from tests.shared.conftest import ...`

**Wrong**:

```mojo
from tests.shared.conftest import (
    assert_equal,
    assert_true,
)
```

**Correct**:

```mojo
# Use relative imports or correct module path structure
from testing import assert_equal, assert_true
```

**Key Points**:

- Mojo doesn't treat the repository as a package by default
- Use relative imports within the same module tree
- Ensure imports match actual file structure

## Boolean Literal Pattern

### Pattern: Mojo-Style Booleans

**Problem**: Python-style boolean literals (`true`, `false`) don't exist in Mojo.

**Wrong**:

```mojo
var is_ready = true
var has_error = false
```

**Correct**:

```mojo
var is_ready = True
var has_error = False
```

**Impact**: 60+ instances needed fixing across benchmark tests.

## Function Signature Patterns

### Pattern: Raises Keyword

**Problem**: Functions that can raise exceptions must declare `raises`.

**Wrong**:

```mojo
fn test_something():
    var result = may_raise_error()
```

**Correct**:

```mojo
fn test_something() raises:
    var result = may_raise_error()
```

### Pattern: Escaping Keyword

**Problem**: Functions passed as arguments that capture variables need `escaping`.

**Wrong**:

```mojo
fn accept_function(func: fn() -> None):
    func()
```

**Correct**:

```mojo
fn accept_function(func: fn() raises escaping -> None):
    func()
```

## Data Structure Patterns

### Pattern: GradientPair Field Access

**Problem**: Tuple unpacking doesn't work with struct return values.

**Wrong**:

```mojo
var (grad_a, grad_b) = compute_gradients()
```

**Correct**:

```mojo
var grads = compute_gradients()
var grad_a = grads.grad_a
var grad_b = grads.grad_b
```

### Pattern: Tensor Operations

**Problem**: Cannot divide tensors directly for scalar results.

**Wrong**:

```mojo
var ratio = tensor_a / tensor_b
```

**Correct**:

```mojo
# Extract scalars first
var scalar_a = tensor_a.item()
var scalar_b = tensor_b.item()
var ratio = scalar_a / scalar_b
```

### Pattern: Ownership Transfer with var Parameters

**Problem**: Non-copyable types (List, Dict) need ownership transfer in constructors and methods.

**Wrong**:

```mojo
fn __init__(out self, value: List[String]):
    self.list_val = value  # Error: cannot copy List
```

**Correct**:

```mojo
fn __init__(out self, var value: List[String]):
    self.list_val = value^  # Transfer ownership with ^
```

**When to Use**:

- Parameters of type List, Dict, or other non-copyable types
- Methods that take ownership of data structures
- Constructors that initialize non-copyable fields

### Pattern: Explicit Copy Constructors

**Problem**: Structs with `ImplicitlyCopyable` trait containing non-copyable fields need explicit `__copyinit__`.

**Wrong**:

```mojo
struct Config(Copyable, Movable, ImplicitlyCopyable):
    var data: Dict[String, Value]  # Non-copyable type

    fn __init__(out self):
        self.data = Dict[String, Value]()
    # Missing __copyinit__ - will fail at copy sites
```

**Correct**:

```mojo
struct Config(Copyable, Movable, ImplicitlyCopyable):
    var data: Dict[String, Value]

    fn __init__(out self):
        self.data = Dict[String, Value]()

    fn __copyinit__(out self, existing: Self):
        """Explicit copy constructor."""
        self.data = existing.data.copy()
```

**When to Use**:

- Structs with `ImplicitlyCopyable` trait
- Structs containing List, Dict, or other non-copyable types
- Any struct that needs custom copy semantics

### Pattern: StringSlice to String Conversion

**Problem**: String methods like `split()` and `strip()` return `StringSlice` in Mojo v0.26.1+.

**Wrong**:

```mojo
var parts = line.split(":")
var key = parts[0].strip()  # Returns StringSlice, not String
self.data[key] = value  # Error: Dict expects String key
```

**Correct**:

```mojo
var parts = line.split(":")
var key = String(parts[0].strip())  # Convert to String
self.data[key] = value  # Now works
```

**Common Operations Returning StringSlice**:

- `split()` - Returns list of StringSlice
- `strip()`, `lstrip()`, `rstrip()` - Returns StringSlice
- String slicing - `string[0:5]` returns StringSlice

**When to Use**:

- When using string manipulation results as Dict keys
- When passing to functions expecting String
- When storing in String fields

## Assertion Import Pattern

### Pattern: Complete Assertion Set

**Problem**: Missing assertion function imports cause compilation errors.

**Common Assertions Needed**:

```mojo
from testing import (
    assert_equal,
    assert_not_equal,
    assert_true,
    assert_false,
    assert_greater,
    assert_greater_equal,
    assert_less,
    assert_less_equal,
)
```

**When to Use**:

- Always import all assertions your tests might need
- Don't assume assertions are globally available
- Check imports when adding new test cases

## Markdown Documentation Pattern

### Pattern: Code Block Language Tags

**Problem**: Code blocks without language tags fail linting.

**Wrong**:

````markdown
```
def example():
    pass
```
````

**Correct**:

````markdown
```python
def example():
    pass
```
````

### Pattern: Line Length

**Problem**: Lines exceeding 120 characters fail linting.

**Fix**: Wrap long lines at natural boundaries:

```markdown
This is a very long sentence that exceeds 120 characters and should be wrapped at a natural
boundary point for better readability.
```

## Migration Checklist

When fixing test files, check these items:

- [ ] Main entry point defined (`fn main() raises:`)
- [ ] All test functions called from main
- [ ] Python interop uses correct syntax (subscripts, module imports)
- [ ] Import paths match actual file structure
- [ ] Boolean literals are `True`/`False` (not `true`/`false`)
- [ ] Functions that can raise have `raises` keyword
- [ ] Functions passed as arguments have `escaping` if needed
- [ ] Struct return values use field access (not tuple unpacking)
- [ ] All required assertion functions imported
- [ ] StringSlice converted to String when needed (`String(...)`)
- [ ] Non-copyable parameters use `var` with ownership transfer (`^`)
- [ ] Structs with non-copyable fields have `__copyinit__`
- [ ] Markdown code blocks have language tags
- [ ] Markdown lines don't exceed 120 characters

## Quick Reference

### Most Common Fixes (by frequency)

1. **Boolean literals** (60+ instances) - `true`/`false` → `True`/`False`
2. **String conversion** (22+ instances) - `str()` → `String()`
3. **Parameter conventions** (50+ instances) - `inout` → `mut`
4. **Main entry points** (6 instances) - Add `fn main() raises:`
5. **Import paths** (5 instances) - Fix absolute imports
6. **Assertion imports** (5 instances) - Add missing assertions
7. **Function signatures** (10+ instances) - Add `raises` keyword
8. **StringSlice conversion** (5+ instances) - `String(parts[0].strip())`
9. **Ownership transfer** (3+ instances) - `var` parameters + `value^`
10. **Copy constructors** (2+ instances) - Add `__copyinit__` for non-copyable fields

## References

- [AGENTS.md - Mojo Syntax Standards](../../AGENTS.md#mojo-syntax-standards-v0257)
- [Mojo Manual - Value Ownership](https://docs.modular.com/mojo/manual/values/ownership/)
- [Mojo Manual - Python Integration](https://docs.modular.com/mojo/manual/python/)

## Related Documentation

- [Test Specialist Agent](.claude/agents/test-specialist.md) - Overall testing strategy
- [Test Engineer Agent](.claude/agents/test-engineer.md) - Implementation patterns
- [Mojo Language Review Specialist](.claude/agents/mojo-language-review-specialist.md) - Language patterns

---

**Last Updated**: 2025-11-24
**Migration Version**: Mojo v0.26.1+
**Files Fixed**: 287 Mojo files, 17 test files, 1884 markdown files
