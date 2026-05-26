# [BUG] Deterministic heap corruption crash during VGG16-style training loop

Repeated function-scoped forward passes + bitcast target write crashes libAsyncRTRuntimeGlobals.so.

## Environment

- **Mojo version**: 0.26.1.0 (156d3ac6)
- **OS**: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 (Ubuntu 24.04)
- **GLIBC**: 2.39
- **CPU**: x86_64

## Description

A **100% deterministic** runtime crash occurs in `libAsyncRTRuntimeGlobals.so`
(the Mojo allocator) during a VGG16-style training loop. After 2-3 function-scoped
VGG16 forward passes that free hundreds of intermediate tensors, the next
forward pass that also creates a small training target tensor with a bitcast write
triggers a crash in the allocator.

This is a more realistic variant of modular/modular#6187: same root cause (heap
corruption from `List[Int]`-containing struct churn + UnsafePointer.bitcast write),
but triggered through a full 13-conv-layer + 3-FC-layer deep learning architecture.

Stack trace offsets are **identical across all runs**, confirming deterministic heap
metadata corruption.

## Stack Trace (identical across all runs)

```text
VGG16 forward pass 1... OK
VGG16 forward pass 2... OK
VGG16 forward pass 3 with training target...
#0 libKGENCompilerRTShared.so  +0x3cb78b
#1 libKGENCompilerRTShared.so  +0x3c93c6
#2 libKGENCompilerRTShared.so  +0x3cc397
#3 libc.so.6                   +0x45330   (sigaction)
#4 libAsyncRTRuntimeGlobals.so +0x416ba   (allocator — crash origin)
error: execution crashed
```

Identical stack offsets to modular/modular#6187 — same allocator corruption path.

## Reproducer

See `repro/repro_libasyncrt_crash.mojo` for the full reproducer. It requires
ProjectOdyssey's `shared` library (AnyTensor, conv2d, linear, relu, maxpool2d).

**Key sequence:**

```mojo
def vgg16_forward(input: AnyTensor) raises -> AnyTensor:
    """13 conv layers + 3 FC layers = ~100+ intermediate AnyTensors per call."""
    var x = conv_block(input, 64, 2)
    var x = maxpool2d(x, 2, 2)
    # ... 5 conv blocks + 5 maxpools + 3 FC layers total
    return output

def train_step_with_target(epoch: Int) raises:
    """Forward pass + create training target with bitcast write -> CRASH."""
    var input = ones([2, 3, 224, 224], DType.float32)
    var output = vgg16_forward(input)
    # === TRIGGER ===
    var target = zeros([2], DType.float32)
    var td = target._data.bitcast[Float32]()
    td[0] = 0.0
    td[1] = Float32(epoch % 10)
    # === END TRIGGER ===
    return output

def main() raises:
    vgg16_forward(ones([2, 3, 224, 224], DType.float32))  # Pass 1: primes heap
    vgg16_forward(ones([2, 3, 224, 224], DType.float32))  # Pass 2: more churn
    train_step_with_target(1)                              # Pass 3: CRASH
```

## Isolation Experiments

| Variant | Crashes? | Conclusion |
| --- | --- | --- |
| Full reproducer (3 passes + bitcast) | **YES (100%)** | Baseline |
| Remove bitcast write from train_step | NO | Bitcast write is the trigger |
| Only 1 prior forward pass (not 2) | NO | Requires sufficient prior churn |
| train_step_with_target() alone (no priors) | NO | Prior heap state required |
| 20 forward passes in a loop in main() (no function scope) | NO | Function-scoped destruction required |
| Smaller input (32x32 instead of 224x224) | Flaky (less reliable) | Volume matters |
| Fewer conv blocks (2 blocks vs 5) | NO | Insufficient tensor churn |

## Relationship to #6187

This is the **same heap corruption bug** as modular/modular#6187, reproduced through:

- A realistic VGG16-style deep learning architecture with 13 conv layers
- Full end-to-end training step pattern (forward pass + target creation)
- Real-world tensor counts: ~100+ intermediate `AnyTensor` objects per forward
  pass, each containing `_shape: List[Int]`

The identical stack trace offsets confirm the same allocator corruption root cause.

## Root Cause (from #6187 investigation)

All three conditions must be present simultaneously:

1. Struct with `List[Int]` field — `AnyTensor._shape: List[Int]`
2. Heavy alloc/free churn of many such structs — VGG16 creates ~100 per pass,
   all freed when the function-scoped forward pass returns
3. `UnsafePointer.bitcast` write — creating the training target and writing
   through a bitcast pointer exposes the corrupted heap state

Removing any single condition prevents the crash.

## Impact

This crash blocks training loop implementation for deep architectures. Any
training loop that:

- Creates model predictions via function-scoped forward pass
- Creates training targets via separate tensor allocation with element assignment
- Runs for more than 2-3 steps

...will crash deterministically.

## Expected Behavior

Training loop runs for all epochs without crashing. All tensor operations are
within bounds and refcounts are balanced.

## Actual Behavior

Deterministic crash in the allocator on step 3, when the training target's
bitcast write is followed by another allocation inside the VGG16 forward pass.
