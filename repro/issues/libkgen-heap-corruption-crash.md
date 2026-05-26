# [BUG] Deterministic heap corruption in libKGENCompilerRTShared.so after tensor alloc/free churn

AnyTensor variant: UnsafePointer.bitcast write triggers allocator crash after List[Int] churn.

## Environment

- **Mojo version**: 0.26.1.0 (156d3ac6)
- **OS**: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 (Ubuntu 24.04)
- **GLIBC**: 2.39
- **CPU**: x86_64

## Description

A **100% deterministic** runtime crash occurs in `libAsyncRTRuntimeGlobals.so`
(the Mojo allocator) when using `AnyTensor` from a shared library with conv2d
operations. The crash pattern is identical to modular/modular#6187 but triggered
through a higher-level API (AnyTensor + shared library conv2d) rather than an
inlined minimal struct.

The stack trace offsets are **identical across all runs**, indicating deterministic
heap metadata corruption rather than a race condition.

## Stack Trace (identical across all runs)

```text
Step A: conv2d heap churn... OK
Step B: bitcast write + conv2d...
#0 libKGENCompilerRTShared.so  +0x3cb78b
#1 libKGENCompilerRTShared.so  +0x3c93c6
#2 libKGENCompilerRTShared.so  +0x3cc397
#3 libc.so.6                   +0x45330   (sigaction)
#4 libAsyncRTRuntimeGlobals.so +0x416ba   (allocator — crash origin)
error: execution crashed
```

Same stack offsets as modular/modular#6187 — confirms same root cause.

## Reproducer

See `repro/repro_libkgen_crash.mojo` for the full reproducer. It requires
ProjectOdyssey's `shared` library (AnyTensor, conv2d, relu).

**Key sequence:**

```mojo
def step_a() raises:
    """2x conv2d creates ~20 intermediate AnyTensors, freed on return."""
    var x = ones([2, 3, 32, 32], DType.float32)
    x = conv2d(x, ones([16, 3, 3, 3], DType.float32), ...)
    x = relu(x)
    x = conv2d(x, ones([16, 16, 3, 3], DType.float32), ...)
    _ = relu(x)

def step_b() raises:
    """Bitcast write to small tensor then conv2d -> CRASH."""
    var target = zeros([2], DType.float32)
    # === TRIGGER ===
    var td = target._data.bitcast[Float32]()
    td[0] = 0.0
    td[1] = 1.0
    # === END TRIGGER ===
    var x = ones([2, 3, 32, 32], DType.float32)
    x = conv2d(x, ones([16, 3, 3, 3], DType.float32), ...)
    _ = relu(x)

def main() raises:
    step_a()   # heap churn
    step_b()   # crash
```

## Isolation Experiments

| Variant | Crashes? | Conclusion |
| --- | --- | --- |
| Full reproducer | **YES (100%)** | Baseline |
| Remove bitcast write (3 lines) | NO | Bitcast write is the trigger |
| step_b() alone (no prior step_a()) | NO | Prior heap churn required |
| Operations inline in main() (no function scope) | NO | Function-scoped destruction required |
| Smaller tensors (8x8 spatial) | NO | Insufficient allocation volume |
| Full code from repro_crash_standalone.mojo (zero dependencies) | **YES** | Same root cause, confirmed |

## Relationship to #6187

This is the **same heap corruption bug** as modular/modular#6187, reproduced through:

- `AnyTensor` (reference-counted, contains `List[Int]` shape field) instead of a
  minimal inline struct
- The `shared` library's conv2d (which creates intermediate `AnyTensor` objects
  with their `List[Int]` shape fields) instead of an inlined loop

The identical stack trace offsets confirm the same allocator corruption path.

## Root Cause (from #6187 investigation)

The crash requires all three conditions simultaneously:

1. A struct containing `List[Int]` field (`AnyTensor` has `_shape: List[Int]`)
2. Heavy alloc/free churn creating and destroying many such structs (step_a)
3. `UnsafePointer.bitcast` write that exposes corrupted heap state (step_b)

## Expected Behavior

Program completes without crashing. All operations are within bounds and
refcounts are balanced.

## Actual Behavior

100% deterministic crash in allocator during a subsequent `alloc` call inside
`step_b()`, after the bitcast write.
