# Mojo Anti-Patterns

Common mistakes from 64+ test failures (PRs #2037-#2046). Flag ALL occurrences immediately.

## Use-After-Free via UnsafePointer.bitcast (CRITICAL)

### Dangling pointer from bitcast on struct field

```mojo
# WRONG — use-after-free: compiler destroys `target` after bitcast extraction
# because `target` is never referenced again. `td` becomes a dangling pointer.
var target = zeros(shape, DType.float32)
var td = target._data.bitcast[Float32]()
td[0] = 0.0   # WRITES TO FREED MEMORY
td[1] = 1.0   # WRITES TO FREED MEMORY

# CORRECT — use __setitem__ (safe API, keeps tensor alive)
var target = zeros(shape, DType.float32)
target[0] = Float32(0.0)
target[1] = Float32(1.0)

# ALSO CORRECT — keep original variable alive past the writes
var target = zeros(shape, DType.float32)
var td = target._data.bitcast[Float32]()
td[0] = 0.0
td[1] = 1.0
_ = target  # prevents early destruction
```

**Why**: `UnsafePointer` does not participate in Mojo's ownership system.
Extracting a pointer from a struct field does NOT extend the struct's lifetime.
The compiler may destroy the struct as soon as the variable is no longer
referenced, even if a derived pointer is still in use.

**ASAN proof**: Building with `mojo build --sanitize address` shows:
`ERROR: AddressSanitizer: heap-use-after-free` on the bitcast write line.

**Safe patterns**: Bitcast inside methods (where `self` stays alive) or when the
result tensor is returned (`return result^`) are safe because the variable is
used after the bitcast writes.

**Fix**: Replace `tensor._data.bitcast[T]()[i] = val` with `tensor[i] = val`.
For reads, replace `tensor._data.bitcast[T]()[i]` with `tensor[i]`.

## Ownership Violations (40% of failures)

### Temporary rvalue ownership transfer

```mojo
# WRONG - Cannot transfer ownership of temporary
var tensor = AnyTensor(List[Int](), DType.int32)

# CORRECT - Create named variable first
var shape = List[Int]()
var tensor = AnyTensor(shape, DType.int32)
```

**Fix**: Create named variable for ALL ownership transfers to `var` parameters.

### ImplicitlyCopyable with non-copyable fields

```mojo
# WRONG - List is NOT implicitly copyable
struct Foo(Copyable, Movable, ImplicitlyCopyable):
    var data: List[Int]

# CORRECT - Only Copyable and Movable
struct Foo(Copyable, Movable):
    var data: List[Int]
    fn get_data(self) -> List[Int]:
        return self.data^  # Explicit transfer
```

**Fix**: NEVER add `ImplicitlyCopyable` to structs with `List`/`Dict`/`String` fields.

### Missing transfer operator

```mojo
# WRONG - Implicit copy fails
fn get_params(self) -> List[Float32]:
    return self.params

# CORRECT - Explicit transfer
fn get_params(self) -> List[Float32]:
    return self.params^
```

**Fix**: ALL returns of `List`/`Dict`/`String` MUST use `^` operator.

## Constructor Signatures (25% of failures)

```mojo
# WRONG - mut self in constructor
fn __init__(mut self, value: Int):
    self.value = value

# CORRECT - out self for constructors
fn __init__(out self, value: Int):
    self.value = value
```

**Constructor Convention Table:**

| Method | Parameter | Example |
| --- | --- | --- |
| `__init__` | `out self` | `fn __init__(out self, value: Int)` |
| `__moveinit__` | `out self, owned existing` | `fn __moveinit__(out self, owned existing: Self)` |
| `__copyinit__` | `out self, existing` | `fn __copyinit__(out self, existing: Self)` |
| Mutating methods | `mut self` | `fn modify(mut self)` |

## Uninitialized Data (20% of failures)

### Uninitialized list access

```mojo
# WRONG - Cannot assign to uninitialized index
var list = List[Int]()
list[0] = 42  # Runtime error

# CORRECT - append creates the element
var list = List[Int]()
list.append(42)
```

### Empty tensor shape

```mojo
# WRONG - Empty shape is 0D scalar (1 element only)
var shape = List[Int]()
var tensor = AnyTensor(shape, DType.float32)
tensor._data[0] = 1.0
tensor._data[1] = 2.0  # SEGFAULT - out of bounds!

# CORRECT - Initialize shape dimensions
var shape = List[Int]()
shape.append(4)
var tensor = AnyTensor(shape, DType.float32)
# Now can safely access indices 0-3
```

## Syntax Errors (10% of failures)

### Missing space after `var`

```mojo
# WRONG - Typo (vara seen as variable name)
vara = 1.0
varb = ones(shape, DType.float32)

# CORRECT - Space required
var a = 1.0
var b = ones(shape, DType.float32)
```

**Detection**: Search for `var[a-z]` pattern.

## Type System Issues (5% of failures)

### DType comparisons

```mojo
# WRONG - DType doesn't conform to Comparable
assert_equal(tensor._dtype, DType.float32)

# CORRECT - Use assert_true with ==
assert_true(tensor._dtype == DType.float32, "Expected float32")
```

### Method vs property access

```mojo
# WRONG - dtype is a method
if tensor.dtype == DType.float32:

# CORRECT - Call method with ()
if tensor.dtype() == DType.float32:
```

## v0.26.1 Compilation Anti-Patterns

### Trying to compile library files standalone

```bash
# DON'T
mojo build shared/core/__init__.mojo
mojo build shared/core/activation.mojo
```

**Why it fails**: Library files use relative imports (`from ..version import VERSION`), which require being part of a package.

**Correct approach**:

```bash
# DO
mojo package shared                    # Build the package
mojo build -I . examples/train.mojo   # Build executable that imports from shared
```

### Expecting all .mojo files to have main()

```bash
# This will fail for library modules
mojo build shared/core/extensor.mojo
# Error: module does not contain a 'main' function
```

**Why it's expected**: Library modules are meant to be imported, not executed.

**Correct understanding**: Only executable files need `main()`.

### Forgetting -I . flag

```bash
# DON'T
mojo build examples/train.mojo
# Error: unable to locate module 'shared'
```

**Correct approach**:

```bash
# DO
mojo build -I . examples/train.mojo
```

## Deprecated Keyword Usage

### Using `Bool(x)` on types with raising `__bool__`

```mojo
# WRONG - Bool() constructor requires Boolable (non-raising __bool__)
var val = Bool(tensor)  # Compile error if __bool__ raises

# CORRECT - call __bool__() directly for raising implementations
var val = tensor.__bool__()  # Works with raising __bool__
```

**Fix**: If a type's `__bool__` method raises, call `.__bool__()` directly instead of `Bool()`.

## Quick Detection Checklist

Search codebase for these patterns:

- `fn __init__(mut self` → Change to `out self`
- `inout self` → Change to `mut self`
- `ImplicitlyCopyable` → Check if fields are copyable
- `return self.*` without `^` → Add transfer operator
- `var[a-z]` → Add space after `var`
- `Bool(` on non-Boolable types → Use `.__bool__()` directly

See [notes/review/mojo-test-failure-learnings.md](../../notes/review/mojo-test-failure-learnings.md) for complete analysis.
