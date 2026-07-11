# Consolidated Root-Level Fixes

Detailed histories of critical bug fixes from root MDs (backed up in `notes/root-backup/`).
See [learnings.md](../learnings.md) for generalized lessons.

## AnyTensor Memory Leak (reshape/slice)

**Files**: `BUGFIX_MEMORY_LEAK.md`, `PHASE2_MEMORY_SAFETY_SUMMARY.md`

**Problem**: Dummy allocations in views orphaned; tcmalloc OOM during training.

**Root Cause**: `AnyTensor(dummy_shape, dtype)` alloc -> overwrite `_data` without free.

**Fix**:

- Free `result._data` before `result._data = self._data`.
- Added refcounting: `_refcount: UnsafePointer[Int]`, `__copyinit__` incr, `__del__` decr/free if 0.
- `reshape/slice`: Copy-based views, update shape/strides safely.

**Verification**: Stress tests (10k iters), full training no leaks.

**Modified**: `src/odyssey/core/any_tensor.mojo`.

## Broadcasting Crash

**File**: `BROADCAST_CRASH_FIX.md`

**Problem**: Segfault in FC bias add; 1D->2D broadcast.

**Root Cause**: Wrong index calc (right-to-left strides); out-of-bounds.

**Fix**: Precompute row-major strides; left-to-right coord extract (`// %`).

```mojo
# Strides e.g. (2,3) -> [3,1]
for dim in range(len(shape)):
    coord = temp_idx // strides[dim]
    temp_idx %= strides[dim]
```

**Verification**: 1D->2D, middle-dim, scalar broadcasts; inference success.

**Modified**: `src/odyssey/core/arithmetic.mojo:_broadcast_binary()`.

## Transpose Memory Corruption

**File**: `BUGFIX_TRANSPOSE_MEMORY_CORRUPTION.md`

**Problem**: Crash in FC1 transpose; segfault looked like OOM.

**Root Cause**: `List[Int](ndim)` wrong size; index uninit elems.

**Fix**: `List[Int]()` + append; temp lists for reverse.

**TDD Path**: Full train -> forward -> FC1 -> transpose minimal repro.

**Verification**: (120,256) transpose, batch=32 forward.

**Modified**: `src/odyssey/core/matrix.mojo:transpose()`.

## List Constructor Bugs (8x)

**File**: `LIST_CONSTRUCTOR_FIXES_SUMMARY.md`

**Problem**: `List[Int](n)` undefined size -> index crash.

**Instances**:

- `shape.mojo`: reshape(-1), squeeze, unsqueeze, concatenate (4x).
- `accuracy.mojo`: argmax, per_class_accuracy (2x).
- `confusion_matrix.mojo`: argmax (1x).
- `trainer_interface.mojo`: DataLoader.next() (1x).

**Fix**: Everywhere -> `List[Int]()` + append.

**Tests Created**: `test_*_bugs.mojo` suites.

## Other Fixes (To Merge)

- Blocker summaries, import fixes, security wave1, etc. (summarize from backups).

Updated: 2025-11-24
