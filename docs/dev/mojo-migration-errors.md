# Mojo v0.26.1+ Migration - Comprehensive Error Summary

**Date**: 2025-01-23
**Context**: Post-rebase compilation fixes for PR #1922
**Total Errors Fixed**: 1,013 / 1,143 (88.6%)

---

## v0.26.3 Quality Audit (2026-04-09)

**Pinned version**: `0.26.3.0.dev2026040705` (April 7 nightly)
**Scope**: `src/projectodyssey/` and `tests/` directories, all `*.mojo` files
**Standards**: Modular mojo-syntax skill v26.2

### Results Summary

All deprecated pattern counts are measured against **actual code** (not comment text or docstrings).

| Pattern | Grep Count | Actual Code | Status |
| --- | --- | --- | --- |
| `fn` declarations (prefer `def`) | 150 | **0** | Clean — all 150 matches are inside comments/docstrings/quoted examples |
| `inout` parameter | 0 | 0 | Clean |
| `owned` parameter | 1 | **0** | Match is in a docstring comment |
| `borrowed` parameter | 2 | **0** | Both matches are in prose comments |
| `@value` decorator | 0 | 0 | Clean |
| `__copyinit__` method | 7 | **0** | All matches are docstring references explaining why explicit copy was needed previously |
| `__moveinit__` method | 3 | **0** | All matches are docstring references |
| `Stringable` trait | 0 | 0 | Clean |
| `@parameter if/for` | 1 | **0** | Match is inside a prose comment |
| `constrained(` | 0 | 0 | Clean |

**Total deprecated patterns in actual code: 0**

### Key Findings

- The codebase has **fully migrated** from `fn` to `def` for all function and method definitions.
  The 150-match grep count is a false positive: the pattern `fn` appears in docstrings, comments,
  and quoted code examples but not in live Mojo source.
- `__copyinit__` and `__moveinit__` appear only in `src/projectodyssey/base/memory_pool.mojo` docstrings
  explaining the historical rationale for explicit copy/move constructors. The actual struct
  implementations use Mojo's synthesized copy/move semantics via `(Copyable, Movable)` traits.
- No action required. The codebase is compliant with v0.26.3 standards.

---

## Executive Summary

After rebasing against main, discovered 1,143 compilation errors due to Mojo v0.26.1+ breaking
changes. Systematically fixed 1,013 errors (88.6%) across 10 batches. This document catalogs
all error patterns to prevent recurrence and guide agent updates.

## Error Categories

### 1. Keyword & Parameter Convention Changes (69 errors)

#### 1.1 `inout` → `mut` Parameter Convention (8 errors)

**Pattern**: Mojo v0.26.1+ renamed `inout` to `mut` for mutable parameters.

**Error**:

```text
error: use of unknown declaration 'inout'
```

**Fix**:

```mojo
// WRONG (deprecated):
fn modify(inout self):
fn process(inout data: AnyTensor):

// CORRECT:
fn modify(mut self):
fn process(mut data: AnyTensor):
```

**Files Affected**: 8 files in `src/projectodyssey/core/arithmetic_simd.mojo`

**Agent Guidance**: Always use `mut` for mutable parameters, never `inout`.

---

#### 1.2 `__init__` Parameter Convention: `mut self` → `out self` (75 errors)

**Pattern**: Constructor methods must use `out` parameter convention.

**Error**:

```text
error: __init__ method must return Self type with 'out' argument
    fn __init__(mut self):
       ^
```

**Fix**:

```mojo
// WRONG:
fn __init__(mut self):
fn __init__(mut self, value: Int):

// CORRECT:
fn __init__(out self):
fn __init__(out self, value: Int):
```

**Files Affected**: 34 files across src/projectodyssey/, examples/, tools/

**Agent Guidance**: ALL `__init__` methods MUST use `out self`, not `mut self`.

---

### 2. Stdlib Reorganization (75 errors)

#### 2.1 DType Import Location Change (4 errors)

**Pattern**: DType moved from `sys` to `memory` module.

**Error**:

```text
error: package 'sys' does not contain 'DType'
```

**Fix**:

```mojo
// WRONG:
from sys import DType

// CORRECT:
from memory import DType
```

**Files Affected**: 4 legacy test files

---

#### 2.2 Builtin Functions (No Import Needed)

**Pattern**: Common functions moved to builtins, no import required.

**Removed Imports**:

- `from collections import Tuple` → Tuple is now builtin
- `from math import abs, round` → abs, round are now builtins
- `from math import max, min` → max, min are now builtins (3 files)
- `from math import pow` → Use `**` operator instead

**Files Affected**: 20+ files

**Agent Guidance**: Don't import Tuple, abs, round, max, min - they're builtins.

---

#### 2.3 simdwidthof Import Location (1 error)

**Pattern**: simdwidthof moved from `sys` to `sys.info`.

**Error**:

```text
error: package 'sys' does not contain 'simdwidthof'
```

**Fix**:

```mojo
// WRONG:
from sys import simdwidthof

// CORRECT:
from sys.info import simdwidthof
```

**Files Affected**: `src/projectodyssey/core/any_tensor.mojo`

---

#### 2.4 str() → String() (31 errors)

**Pattern**: `str()` function deprecated, use `String()` constructor.

**Error**:

```text
error: use of unknown declaration 'str'
```

**Fix**:

```mojo
// WRONG:
var s = str(value)
var msg = "Count: " + str(count)

// CORRECT:
var s = String(value)
var msg = "Count: " + String(count)
```

**Files Affected**: 14 files + 11 in src/projectodyssey/data/

**Agent Guidance**: Always use `String(value)`, never `str(value)`.

---

### 3. Type System Changes (75 errors)

#### 3.1 @value Decorator Removed (10 errors)

**Pattern**: `@value` decorator replaced with `@fieldwise_init` + explicit traits.

**Error**:

```text
error: use of unknown declaration '@value'
```

**Fix**:

```mojo
// WRONG:
@value
struct Transform:
    var name: String

// CORRECT:
@fieldwise_init
struct Transform(Copyable, Movable):
    var name: String
```

**Files Affected**: 10 files in src/projectodyssey/core/types/, src/projectodyssey/training/

**Agent Guidance**: Use `@fieldwise_init` with explicit `(Copyable, Movable)` traits.

---

#### 3.2 Trait Conformance Requirements (28 errors)

**Pattern**: Structs used in List/parameters must explicitly declare Copyable & Movable.

**Error**:

```text
error: cannot bind type 'TypeName' to trait 'Copyable & Movable'
    var list_var: List[TypeName]
                       ^~~~~~~~~
```

**Fix**:

```mojo
// WRONG:
struct MetricResult:
    var value: Float64

// CORRECT:
struct MetricResult(Copyable, Movable):
    var value: Float64
```

**Affected Patterns**:

- Struct stored in `List[StructType]`
- Struct passed as function parameter
- Struct returned from function

**Files Affected**: 10 files (28 structs total)

- src/projectodyssey/training/metrics/
- src/projectodyssey/autograd/
- src/projectodyssey/utils/
- src/projectodyssey/core/gradient_types.mojo

**Agent Guidance**: Any struct used in collections or parameters needs `(Copyable, Movable)`.

---

#### 3.3 @fieldwise_init Conflicts (32 errors)

**Pattern**: Cannot have both `@fieldwise_init` decorator and manual `__init__` method.

**Error**:

```text
error: 'StructName' has an explicitly declared fieldwise initializer
```

**Fix**:

```mojo
// WRONG:
@fieldwise_init
struct Dataset(Copyable, Movable):
    var size: Int

    fn __init__(out self, size: Int):
        self.size = size

// CORRECT (remove decorator):
struct Dataset(Copyable, Movable):
    var size: Int

    fn __init__(out self, size: Int):
        self.size = size
```

**Files Affected**: 7 files in src/projectodyssey/data/ (25 structs)

**Agent Guidance**: Use `@fieldwise_init` OR manual `__init__`, never both.

---

### 4. Ownership & Memory Safety (50 errors)

#### 4.1 ImplicitlyCopyable Removed (21 errors)

**Pattern**: AnyTensor, List[T] no longer implicitly copyable - need explicit transfer or copy.

**Error**:

```text
error: value of type 'AnyTensor' cannot be implicitly copied,
       it does not conform to 'ImplicitlyCopyable'
```

**Fixes**:

**Pattern 1: Ownership Transfer (most common)**

```mojo
// WRONG:
var copy = some_tensor
self.field = tensor
return result

// CORRECT:
var copy = some_tensor^
self.field = tensor^
return result^
```

**Pattern 2: Explicit Copy (when copy needed)**

```mojo
// WRONG:
var copy = some_list

// CORRECT:
var copy = List[Int](some_list)
```

**Files Affected**: 8 files in src/projectodyssey/training/metrics/, src/projectodyssey/autograd/, src/projectodyssey/core/

**Agent Guidance**: Use `^` for transfer, explicit constructor for copying.

---

#### 4.2 UnsafePointer API Changes (3 errors)

**Pattern**: `UnsafePointer.address_of()` removed, use `Pointer.address_of()`.

**Error**:

```text
error: 'UnsafePointer[?, ?, address_space=?]' value has no attribute 'address_of'
```

**Fix**:

```mojo
// WRONG:
var ptr = UnsafePointer.address_of(variable)
var value = ptr.bitcast[Type]()[0]

// CORRECT:
from memory import Pointer
var ptr = Pointer.address_of(variable)
var value = ptr.bitcast[Type]()[]
```

**Files Affected**: src/projectodyssey/core/bfloat16.mojo (3 locations)

**Agent Guidance**: Use `Pointer.address_of()` and `[]` array access syntax.

---

### 5. API Method Changes (214 errors)

#### 5.1 AnyTensor Method → Function Migration (21 errors)

**Pattern**: Instance methods moved to standalone functions.

**Errors & Fixes**:

**1. AnyTensor.from_scalar() removed (8 errors)**

```mojo
// WRONG:
var result = AnyTensor.from_scalar(value, dtype)

// CORRECT:
from projectodyssey.core.any_tensor import full
var result = full(tensor._shape, value, tensor._dtype)
```

**2. AnyTensor.sum() removed (4 errors)**

```mojo
// WRONG:
var total = tensor.sum()

// CORRECT:
from projectodyssey.core.reduction import sum as tensor_sum
var total = tensor_sum(tensor)
```

**3. AnyTensor.matmul() removed (8 errors)**

```mojo
// WRONG:
var result = a.matmul(b)

// CORRECT:
from projectodyssey.core.matrix import matmul
var result = matmul(a, b)
```

**4. AnyTensor() constructor requires arguments (1 error)**

```mojo
// WRONG:
self.tensor = AnyTensor()

// CORRECT:
self.tensor = AnyTensor(List[Float32](), DType.float32)
```

**Files Affected**: src/projectodyssey/core/activation.mojo, src/projectodyssey/testing/gradient_checker.mojo, tests/,
src/projectodyssey/training/optimizers/

---

#### 5.2 Float64 Constants Migration (8 errors)

**Pattern**: Float64 class methods removed, use arithmetic expressions.

**Error**:

```text
error: 'Float64' has no attribute 'nan'
```

**Fix**:

```mojo
// WRONG:
var nan_val = Float64.nan
var inf_val = Float64.inf
var neg_inf = Float64.neg_inf
var infinity = Float64.infinity

// CORRECT:
var nan_val = Float64(0.0) / Float64(0.0)  # NaN
var inf_val = Float64(1.0) / Float64(0.0)  # +Inf
var neg_inf = -Float64(1.0) / Float64(0.0)  # -Inf
var infinity = Float64(1.0) / Float64(0.0)  # +Inf
```

**Files Affected**: tests/projectodyssey/training/test_numerical_safety.mojo (7), notes/review/ (1)

---

#### 5.3 Missing dtype Parameters (14 errors)

**Pattern**: Initialization functions now require explicit dtype parameter.

**Error**:

```text
error: missing parameter 'dtype'
```

**Fix**:

```mojo
// WRONG:
var tensor = zeros(shape)
var tensor = ones(shape)
var tensor = full(shape, value)

// CORRECT:
var tensor = zeros(shape, DType.float32)
var tensor = ones(shape, DType.float32)
var tensor = full(shape, value, DType.float32)
```

**Files Affected**: 10 files in examples/ and tests/

**Agent Guidance**: Always provide explicit dtype for initialization functions.

---

#### 5.4 Property vs Method: .shape() → .shape (628 errors)

**Pattern**: `shape` changed from method to property in many files, but remains a method in
any_tensor.mojo.

**Error**:

```text
error: 'shape' expects 0 parameters, but 1 was specified
```

**Complexity**: Mixed usage - some files needed `.shape()`, others `.shape`

**Fix 1 (When shape is property)**:

```mojo
// WRONG:
var s = tensor.shape()
var dim0 = tensor.shape()[0]

// CORRECT:
var s = tensor.shape
var dim0 = tensor.shape[0]
```

**Fix 2 (When shape is method - matrix.mojo)**:

```mojo
// WRONG:
var s = tensor.shape
var dim0 = tensor.shape[0]

// CORRECT:
var s = tensor.shape()
var dim0 = tensor.shape()[0]
```

**Files Affected**: 99 files (628 occurrences automated via script)

**Agent Guidance**: Check whether shape is property or method in the specific file's context.

---

### 6. Closure & Type Inference (14 errors)

#### 6.1 Closure Emission Errors (13 errors)

**Pattern**: Cannot use method reference in closure context.

**Error**:

```text
error: cannot emit closure for method 'shape'
```

**Root Cause**: Trying to access `.shape` as property when it's defined as method.

**Fix**: Call method explicitly:

```mojo
// WRONG:
var s = tensor.shape  # Tries to create closure

// CORRECT:
var s = tensor.shape()  # Explicit call
```

**Files Affected**: src/projectodyssey/core/matrix.mojo (13 locations)

---

#### 6.2 SIMD Type Inference (1 error)

**Pattern**: Mixed generic and explicit types in SIMD operations.

**Error**:

```text
error: invalid call to '__add__': failed to infer parameter 'dtype',
       it inferred to two different values: 'T' and 'DType.float32'
```

**Fix**: Use consistent types or restructure:

```mojo
// WRONG:
return Scalar[T](1.0) / (Scalar[T](1.0) + exp(-Float32(x)))
# Mixes Scalar[T] with Float32

// CORRECT:
@parameter
if T == DType.float16:
    var x_f32 = Float32(x)
    var result_f32 = Float32(1.0) / (Float32(1.0) + exp(-x_f32))
    return Scalar[T](result_f32)
else:
    return Scalar[T](1.0) / (Scalar[T](1.0) + exp(-x))
```

**Files Affected**: src/projectodyssey/core/activation.mojo

**Agent Guidance**: Keep SIMD types consistent within operations.

---

### 7. Structural Limitations (5 errors)

#### 7.1 Struct Inheritance Not Allowed (1 error)

**Pattern**: Mojo does not support struct inheritance.

**Error**:

```text
error: inheriting from structs is not allowed
```

**Fix**: Use composition instead:

```mojo
// WRONG:
struct BatchLoader(BaseLoader, Copyable, Movable):
    pass

// CORRECT:
struct BatchLoader(Copyable, Movable):
    var dataset: Dataset
    var batch_size: Int
    var drop_last: Bool
    var _len: Int
    # Copy all fields from BaseLoader
```

**Files Affected**: src/projectodyssey/data/loaders.mojo

**Agent Guidance**: Use composition, not inheritance for structs.

---

#### 7.2 Dynamic Traits Not Supported (4 errors)

**Pattern**: Cannot store trait types in fields - must use compile-time generics.

**Error**:

```text
error: dynamic traits not supported yet, please use a compile time generic instead
```

**Fix**:

```mojo
// WRONG:
struct Container:
    var transform: Transform  # Runtime trait field

// CORRECT:
struct Container[T: Transform]:  # Compile-time generic
    var transform: T
```

**Files Affected**: src/projectodyssey/data/generic_transforms.mojo (4 structs)

**Agent Guidance**: Use parametric generics `[T: TraitName]` instead of trait-typed fields.

---

### 8. Compile-Time Import Limitation (#3731)

#### 8.0 Mojo Cannot Resolve Imports at Compile Time from Dynamic Paths

**Pattern**: Mojo v0.26.1 does not support importing symbols whose paths are
determined at runtime or through multi-hop re-exports across `__init__.mojo` files.

**Symptom**:

```text
error: cannot find 'LinearLayer' in module 'shared'
```

Even when `src/projectodyssey/__init__.mojo` imports from `shared.core` which imports from
`shared.core.layers`, a consumer doing `from shared import LinearLayer` may fail
because Mojo's compiler resolves imports at compile time and does not follow
re-export chains through multiple `__init__.mojo` levels.

**Root cause**: Mojo's import resolution is single-hop at compile time. A module
can import a symbol for internal use, but that symbol is not automatically re-exported
to the parent package's namespace in a way that downstream modules can discover.

**Fix**: Import directly from the defining module:

```mojo
# WRONG (may fail due to re-export chain):
from shared import LinearLayer

# CORRECT (import from the defining module):
from projectodyssey.core.layers import LinearLayer
```

**Workaround in `__init__.mojo`**: Keep commented import stubs with a note explaining
when they will be enabled (once the compiler limitation is resolved):

```mojo
# from .core.layers import Linear  # Blocked: re-export chain (#3754, #3731)
```

**Agent Guidance**: Always import from the most specific module path. Do not assume
that importing a symbol into `__init__.mojo` makes it available at the package level.

---

### 9. Test-Specific Errors (7 errors)

#### 8.1 Function Renames (6 errors)

**Pattern**: Loss functions renamed for consistency.

**Fixes**:

- `mse_loss` → `mean_squared_error`
- `bce_loss` → `binary_cross_entropy`

**Files Affected**: tests/test_core_operations.mojo

---

#### 8.2 DType Comparison (2 errors)

**Pattern**: DType doesn't implement Comparable, can't use assert_equal.

**Error**:

```text
error: no matching function in call to 'assert_equal'
```

**Fix**:

```mojo
// WRONG:
assert_equal(tensor.dtype, DType.float32, "message")

// CORRECT:
if tensor.dtype() != DType.float32:
    raise Error("message")
```

**Files Affected**: tests/test_core_operations.mojo

**Agent Guidance**: Manual comparison for DType using `dtype()` method.

---

#### 8.3 len() Type Resolution (7 errors)

**Pattern**: len() needs explicit type when called on method results.

**Error**:

```text
error: no matching function in call to 'len'
```

**Fix**:

```mojo
// WRONG:
if len(tensor.shape()) == 2:

// CORRECT:
var shape_vec = tensor.shape()
if len(shape_vec) == 2:
```

**Files Affected**: src/projectodyssey/training/metrics/accuracy.mojo, confusion_matrix.mojo

**Agent Guidance**: Assign method result to variable before calling len().

---

## Batch Summary

| Batch | Errors Fixed | Categories |
| --- | --- | --- |
| 1-6 | 950 | Syntax, imports, API, memory, types |
| 7 | 18 | UnsafePointer, List ownership, str() |
| 8-9 | 35 | Closure, inheritance, traits, init |
| 10 | 10 | SIMD inference, test imports |
| **Total** | **1,013** | **88.6% of 1,143** |

## Agent Update Priorities

### Critical (Always Apply)

1. ✅ **Use `out self` in `__init__`**, never `mut self`
2. ✅ **Use `String()` not `str()`**
3. ✅ **Use `mut` not `inout`** for parameters
4. ✅ **Add explicit traits**: `(Copyable, Movable)` to all structs
5. ✅ **No struct inheritance** - use composition
6. ✅ **Use parametric generics** `[T: Trait]`, not trait fields
7. ✅ **Ownership transfer**: Use `^` for AnyTensor, List assignments
8. ✅ **Import from correct modules**:
   - DType: `from memory import DType`
   - simdwidthof: `from sys.info import simdwidthof`
   - Pointer: `from memory import Pointer`

### High Priority (Common Patterns)

1. ✅ **AnyTensor API changes**:
   - Use `full()` not `AnyTensor.from_scalar()`
   - Use `tensor_sum()` not `.sum()`
   - Use `matmul()` not `.matmul()`
2. ✅ **Explicit dtype parameters** for zeros(), ones(), full()
3. ✅ **Don't import builtins**: Tuple, abs, round, max, min
4. ✅ **Use `@fieldwise_init`** with traits, not `@value`
5. ✅ **No mixed auto/manual init**: Remove `@fieldwise_init` if manual `__init__` exists

### Medium Priority (Context-Dependent)

1. ⚠️ **shape property vs method**: Check file context
2. ⚠️ **SIMD type consistency**: Don't mix generic and explicit types
3. ⚠️ **len() type resolution**: Assign to variable first
4. ⚠️ **Pointer API**: Use `Pointer.address_of()` and `[]` syntax

## Files Changed (By Category)

### Core Infrastructure (15 files)

- src/projectodyssey/core/any_tensor.mojo, activation.mojo, arithmetic_simd.mojo
- src/projectodyssey/core/matrix.mojo, broadcasting.mojo, bfloat16.mojo
- src/projectodyssey/core/types/ (7 files)
- src/projectodyssey/core/pooling.mojo, conv.mojo, dropout.mojo, etc.

### Training & Metrics (14 files)

- src/projectodyssey/training/metrics/ (4 files)
- src/projectodyssey/training/optimizers/ (3 files)
- src/projectodyssey/training/trainer, callbacks, mixed_precision

### Data Module (7 files)

- src/projectodyssey/data/datasets.mojo, loaders.mojo, samplers.mojo
- src/projectodyssey/data/transforms.mojo, text_transforms.mojo, generic_transforms.mojo
- src/projectodyssey/data/batch_utils.mojo

### Autograd (4 files)

- src/projectodyssey/autograd/variable.mojo, tape.mojo, optimizers.mojo, functional.mojo

### Utils (4 files)

- src/projectodyssey/utils/random.mojo, profiling.mojo, logging.mojo, io.mojo, visualization.mojo

### Tests (31 files)

- All test files updated for API changes

### Examples (41 files)

- All example files updated

## Lessons Learned

1. **Breaking changes are pervasive**: v0.26.1+ affected every major subsystem
2. **Ownership is explicit**: No more implicit copies - forces better design
3. **Type safety increased**: More explicit trait requirements, generics
4. **Stdlib consolidation**: Builtins reduce import boilerplate
5. **Pattern consistency**: Similar fixes across many files suggests systematic changes

## Next Steps

1. ✅ Fix remaining ~130 errors (11.4%)
2. ✅ Update agent configurations with error patterns
3. ✅ Add pre-commit checks for common errors
4. ✅ Document migration guide for future Mojo upgrades
