# Data Test Suite - Quick Fix Guide

## Overview

39 compilation errors blocking all data tests. All errors are fixable with systematic pattern replacements.

---

## Error Pattern 1: `__init__` Missing Return Type (17 errors)

### Symptom

```text
error: __init__ method must return Self type with 'out' argument
    fn __init__(mut self, size: Int):
       ^
```text

### Affected Files (5 test files + 2 implementation files)

- tests/shared/data/datasets/test_base_dataset.mojo:25
- tests/shared/data/datasets/test_tensor_dataset.mojo:32
- tests/shared/data/loaders/test_base_loader.mojo:20
- tests/shared/data/transforms/test_pipeline.mojo:25 & 35
- shared/data/transforms.mojo:621 & 400
- (Plus more in other implementation files)

### Current Code Pattern

```mojo
struct MyStruct:
    var field1: Type1
    var field2: Type2

    fn __init__(mut self, arg1: Type1, arg2: Type2):
        self.field1 = arg1
        self.field2 = arg2
```text

### Fixed Code Pattern

```mojo
struct MyStruct:
    var field1: Type1
    var field2: Type2

    fn __init__(mut self, arg1: Type1, arg2: Type2) -> Self:
        return Self(field1=arg1, field2=arg2)
```text

### Key Changes

1. Add `-> Self` return type annotation
2. Replace body with `return Self(...)`
3. Use named parameters in Self constructor

### Example Fix

**File**: tests/shared/data/datasets/test_base_dataset.mojo

**Before**:

```mojo
fn __init__(mut self, size: Int):
    """Create stub dataset with specified size."""
    self.size = size
    self.data = List[Float32](capacity=size)
    for i in range(size):
        self.data.append(Float32(i))
```text

**After**:

```mojo
fn __init__(mut self, size: Int) -> Self:
    """Create stub dataset with specified size."""
    var data = List[Float32](capacity=size)
    for i in range(size):
        data.append(Float32(i))
    return Self(size=size, data=data)
```text

---

## Error Pattern 2: AnyTensor Ownership Violations (11 errors)

### Symptom

```text
error: value of type 'AnyTensor' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'
            return data
                   ^~~~
note: consider transferring the value with '^'
```text

### Affected Files (1 file: shared/data/transforms.mojo)

- Line 523: RandomRotation.**call**
- Line 769: RandomErasing.**call** (path 1)
- Line 806: RandomErasing.**call** (path 2)
- Line 814: RandomErasing.**call** (path 3)

### Current Code Pattern

```mojo
fn __call__(self, data: AnyTensor) raises -> AnyTensor:
    # ... process data ...
    return data  # ERROR: implicit copy
```text

### Fixed Code Pattern

```mojo
fn __call__(self, data: AnyTensor) raises -> AnyTensor:
    # ... process data ...
    return data^  # Transfer ownership
```text

### Key Change

Add `^` caret operator after `data` to explicitly transfer ownership.

### Example Fix

**File**: shared/data/transforms.mojo, line 523

**Before**:

```mojo
fn __call__(self, data: AnyTensor) raises -> AnyTensor:
    # ... rotation logic ...
    return data
```text

**After**:

```mojo
fn __call__(self, data: AnyTensor) raises -> AnyTensor:
    # ... rotation logic ...
    return data^
```text

---

## Error Pattern 3: Missing Tensor Import (1 error, 17 usages)

### Symptom

```text
error: use of unknown declaration 'Tensor'
    var data = Tensor(data_list^)
               ^~~~~~
```text

### Affected File (1 file: test_augmentations.mojo)

- Lines: 37, 63, 67, 97, 116, 140, 164, 168, 187, 215, 242, 263, 288, 313, 346, 372

### Current Code

**File**: tests/shared/data/transforms/test_augmentations.mojo

**Current imports** (lines 7-19):

```mojo
from tests.shared.conftest import assert_true, assert_equal, assert_false, TestFixtures
from shared.data.transforms import (
    Transform,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomCrop,
    CenterCrop,
    RandomErasing,
    Pipeline,
    Compose,
)
from shared.core.any_tensor import AnyTensor
```text

### Solution

**Option A**: Add Tensor import

```mojo
# Change line 19 from:
from shared.core.any_tensor import AnyTensor

# To:
from shared.core.any_tensor import AnyTensor, Tensor
```text

**Option B**: Replace all Tensor usages (if Tensor is not available)

```bash
# Run in tests/shared/data/transforms/ directory
sed -i 's/Tensor(/AnyTensor(/g' test_augmentations.mojo
sed -i 's/List\[Tensor\]/List[AnyTensor]/g' test_augmentations.mojo
```text

### Verify Which Works

1. Check `/shared/core/any_tensor.mojo` for Tensor definition
2. If Tensor exists and is different from AnyTensor, use Option A (add import)
3. If Tensor is an alias for AnyTensor, use Option A (add import)
4. If Tensor doesn't exist, use Option B (replace with AnyTensor)

---

## Error Pattern 4: Invalid Optional Subscripting (1 error)

### Symptom

```text
error: 'Int' is not subscriptable
            var pad = self.padding.value()[]
                      ~~~~~~~~~~~~~~~~~~~~^
```text

### Affected File (1 file: shared/data/transforms.mojo)

- Line 458: RandomCrop.**call**

### Current Code Pattern

```mojo
if self.padding:
    var pad = self.padding.value()[]  # WRONG
    actual_top = top - pad
```text

### Fixed Code Pattern

```mojo
if self.padding:
    var pad = self.padding.value()  # CORRECT - no subscript
    actual_top = top - pad
```text

### Explanation

- `self.padding` is `Optional[Int]`
- `self.padding.value()` returns the contained `Int` value
- You cannot subscript an `Int` with `[]`
- Remove the `[]` subscript operator

### Example Fix

**File**: shared/data/transforms.mojo, line 458

**Before**:

```mojo
if self.padding:
    var pad = self.padding.value()[]
    actual_top = top - pad
    actual_left = left - pad
```text

**After**:

```mojo
if self.padding:
    var pad = self.padding.value()
    actual_top = top - pad
    actual_left = left - pad
```text

---

## Error Pattern 5: Missing Trait Conformances (9 cascading errors)

### Symptom (Cascading)

Various implicit copy errors in loops and variable assignments

### Affected Structs (Test files)

- `StubDataset` in test_base_dataset.mojo
- `TestTensorDataset` in test_tensor_dataset.mojo
- `SimpleDataLoader` in test_base_loader.mojo
- `SimpleTransform` in test_pipeline.mojo

### Current Code Pattern

```mojo
struct MyStruct:
    var field1: Type1
    var field2: Type2
    # Missing traits
```text

### Fixed Code Pattern

```mojo
@fieldwise_init
struct MyStruct(Copyable, Movable):
    var field1: Type1
    var field2: Type2
    # Now supports copy/move semantics
```text

### Key Changes

1. Add `@fieldwise_init` decorator (auto-generates constructor)
2. Add `(Copyable, Movable)` trait conformance
3. This enables implicit copying in loops and assignments

### Example Fix

**File**: tests/shared/data/datasets/test_base_dataset.mojo, line 15

**Before**:

```mojo
struct StubDataset:
    """Minimal stub dataset for testing Dataset interface requirements."""

    var size: Int
    var data: List[Float32]
```text

**After**:

```mojo
@fieldwise_init
struct StubDataset(Copyable, Movable):
    """Minimal stub dataset for testing Dataset interface requirements."""

    var size: Int
    var data: List[Float32]
```text

### Note

If struct has custom `__init__`, you may not be able to use `@fieldwise_init`. In that case:

1. Keep custom `__init__` with `-> Self` return type
2. Just add `(Copyable, Movable)` traits without decorator
3. Ensure implicit copies work in the code

---

## Fix Application Order

### Step 1: Fix **init** Return Types (30 min)

Fixes 17 errors - unblocks all test compilation

```bash
# Edit these files and add -> Self to all __init__ methods:
1. tests/shared/data/datasets/test_base_dataset.mojo (line 25)
2. tests/shared/data/datasets/test_tensor_dataset.mojo (line 32)
3. tests/shared/data/loaders/test_base_loader.mojo (line 20)
4. tests/shared/data/transforms/test_pipeline.mojo (lines 25, 35)
5. shared/data/transforms.mojo (lines 621, 400)
```text

**After Step 1**: Run `pixi run mojo -I . tests/shared/data/run_all_tests.mojo` to check progress

### Step 2: Fix AnyTensor Ownership (15 min)

Fixes 11 errors - enables transform tests

```bash
# Add ^ to return statements in shared/data/transforms.mojo:
1. Line 523: return data^ (RandomRotation)
2. Line 769: return data^ (RandomErasing path 1)
3. Line 806: return data^ (RandomErasing path 2)
4. Line 814: return data^ (RandomErasing path 3)
```text

**After Step 2**: Run `pixi run mojo -I . tests/shared/data/run_all_tests.mojo` to check progress

### Step 3: Fix Tensor Import (10 min)

Fixes 17 usage errors - enables augmentation tests

```bash
# Edit tests/shared/data/transforms/test_augmentations.mojo:
# Add Tensor to import on line 19:
from shared.core.any_tensor import AnyTensor, Tensor
```text

**After Step 3**: Run full test suite - should reach runtime

### Step 4: Fix Optional Syntax (5 min)

Fixes 1 error - enables RandomCrop

```bash
# Edit shared/data/transforms.mojo, line 458:
# Before: var pad = self.padding.value()[]
# After:  var pad = self.padding.value()
```text

### Step 5: Add Trait Conformances (20 min)

Fixes 9 cascading errors - proper type safety

```bash
# Add @fieldwise_init and (Copyable, Movable) to these structs:
1. StubDataset in test_base_dataset.mojo (line 15)
2. TestTensorDataset in test_tensor_dataset.mojo
3. SimpleDataLoader in test_base_loader.mojo
4. SimpleTransform in test_pipeline.mojo
```text

---

## Verification Checklist

After each fix step, run:

```bash
pixi run mojo -I . tests/shared/data/run_all_tests.mojo
```text

### After Step 1 (Fix **init**)

Expected: More compilation errors, but different ones
Should not see: `__init__ method must return Self`

### After Step 2 (Fix ownership)

Expected: Fewer compilation errors
Should not see: `AnyTensor' cannot be implicitly copied`

### After Step 3 (Fix Tensor import)

Expected: No compilation errors
Should see: Test execution starting

### After Step 4 (Fix Optional)

Expected: No `Int is not subscriptable` errors

### After Step 5 (Add traits)

Expected: All 43 tests run successfully
Pattern: `✓ test_name` for passes, `✗ test_name` for fails

---

## Reference: Key Mojo v0.26.1+ Syntax

### `__init__` Method

```mojo
# WRONG (old syntax)
fn __init__(mut self, arg: Type):
    self.field = arg

# CORRECT (v0.26.1+)
fn __init__(mut self, arg: Type) -> Self:
    return Self(field=arg)
```text

### Ownership Transfer

```mojo
# WRONG (implicit copy)
return data

# CORRECT (explicit transfer)
return data^
```text

### Optional Value Extraction

```mojo
# WRONG (subscript after value())
var x = optional.value()[]

# CORRECT (value() returns the contained value)
var x = optional.value()
```text

### Struct Definition

```mojo
# WRONG (missing traits)
struct MyStruct:
    var field: Type

# CORRECT (with traits)
@fieldwise_init
struct MyStruct(Copyable, Movable):
    var field: Type
```text

### Trait Conformance

```mojo
# WRONG (no traits)
struct MyStruct:
    var field: Type

# CORRECT (with traits)
struct MyStruct(Copyable, Movable):
    var field: Type
```text

---

## Files to Modify

### Test Files (5)

1. `/tests/shared/data/datasets/test_base_dataset.mojo`
2. `/tests/shared/data/datasets/test_tensor_dataset.mojo`
3. `/tests/shared/data/loaders/test_base_loader.mojo`
4. `/tests/shared/data/transforms/test_pipeline.mojo`
5. `/tests/shared/data/transforms/test_augmentations.mojo`

### Implementation Files (2)

1. `/shared/data/transforms.mojo`
2. `/shared/core/any_tensor.mojo` (check for Tensor definition)

---

## Success Criteria

All fixes complete when:

- `pixi run mojo -I . tests/shared/data/run_all_tests.mojo` executes without compilation errors
- Test output shows: `Total Tests: 43`, `Passed: X`, `Failed: Y`
- No errors of the 5 types listed above appear

---

## Notes

- These are systematic, low-risk fixes
- All changes follow Mojo v0.26.1+ best practices
- No logic changes required
- Estimated total time: 80 minutes for all 5 priority levels
