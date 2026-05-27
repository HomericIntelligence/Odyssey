"""Mojo v0.26.1 runtime crash: heap corruption after tensor alloc/free churn.

Environment: Mojo 0.26.1 (156d3ac6), GLIBC 2.39, Linux 6.6.87 x86_64 (WSL2)

This reproduces a deterministic crash in libKGENCompilerRTShared.so /
libAsyncRTRuntimeGlobals.so triggered by the following sequence:

1. step_a(): Run conv2d operations that allocate and free many intermediate
   tensors (~20+ alloc/free cycles). When the function returns, all local
   AnyTensor objects are destroyed, freeing their backing memory.

2. step_b(): Create a small tensor, obtain a bitcast pointer to its data,
   write through that pointer, then run another conv2d operation.
   The conv2d crashes during internal allocation.

Key observations:
- Crash is 100% deterministic (same stack offsets every run)
- Removing the bitcast write in step_b prevents the crash
- Creating the small tensor WITHOUT writing through bitcast does NOT crash
- Running the same sequence inline in main() (not via function calls)
  does NOT crash — function-scope destruction is part of the trigger
- Fewer conv2d ops (smaller tensors, e.g. 8x8) do NOT crash
- The crash occurs in the allocator (libAsyncRTRuntimeGlobals.so +0x416ba)
  during a subsequent allocation after the bitcast write

This suggests the heavy alloc/free churn in step_a() corrupts heap metadata,
and the bitcast write to a small tensor hits the corrupted region, causing
the next large allocation to crash.

Stack trace (constant across runs):
  #0 libKGENCompilerRTShared.so +0x3cb78b  (crash handler)
  #1 libKGENCompilerRTShared.so +0x3c93c6  (crash handler)
  #2 libKGENCompilerRTShared.so +0x3cc397  (crash handler)
  #3 libc.so.6                  +0x45330   (sigaction)
  #4 libAsyncRTRuntimeGlobals.so +0x416ba  (allocator — crash origin)

NOTE: This reproducer requires ProjectOdyssey's projectodyssey.core library (AnyTensor,
conv2d, relu). The crash is in the Mojo runtime allocator, not in our library
code. A fully self-contained reproducer would require reimplementing conv2d
(~200 lines) which defeats the purpose of minimality.
"""

from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones
from projectodyssey.core.conv import conv2d
from projectodyssey.core.activation import relu


def step_a() raises:
    """Heavy tensor alloc/free churn via conv2d.

    Two conv2d+relu operations create many intermediate tensors.
    All are freed when this function returns (AnyTensor destructor
    decrements refcount, frees when refcount reaches 0).
    """
    # Input: (batch=2, channels=3, height=32, width=32)
    var is_ = List[Int]()
    is_.append(2)
    is_.append(3)
    is_.append(32)
    is_.append(32)
    var x = ones(is_, DType.float32)

    # Kernel: (out_ch=16, in_ch=3, kH=3, kW=3)
    var ks = List[Int]()
    ks.append(16)
    ks.append(3)
    ks.append(3)
    ks.append(3)
    var bs = List[Int]()
    bs.append(16)

    # Conv 1: 3 -> 16 channels
    x = conv2d(x, ones(ks, DType.float32), zeros(bs, DType.float32), 1, 1)
    x = relu(x)

    # Conv 2: 16 -> 16 channels
    var ks2 = List[Int]()
    ks2.append(16)
    ks2.append(16)
    ks2.append(3)
    ks2.append(3)
    x = conv2d(x, ones(ks2, DType.float32), zeros(bs, DType.float32), 1, 1)
    _ = relu(x)
    # ~20+ tensors freed here


def step_b() raises:
    """Bitcast write on small tensor, then conv2d -> CRASH.

    The bitcast write is the trigger. Commenting out the 3 lines marked
    TRIGGER below prevents the crash.
    """
    # Small 2-element tensor
    var ts = List[Int]()
    ts.append(2)
    var target = zeros(ts, DType.float32)

    # === TRIGGER: comment out these 3 lines to prevent crash ===
    var td = target._data.bitcast[Float32]()
    td[0] = 0.0
    td[1] = 1.0
    # === END TRIGGER ===

    # This conv2d crashes during internal allocation
    var is_ = List[Int]()
    is_.append(2)
    is_.append(3)
    is_.append(32)
    is_.append(32)
    var x = ones(is_, DType.float32)
    var ks = List[Int]()
    ks.append(16)
    ks.append(3)
    ks.append(3)
    ks.append(3)
    var bs = List[Int]()
    bs.append(16)
    x = conv2d(x, ones(ks, DType.float32), zeros(bs, DType.float32), 1, 1)
    _ = relu(x)


def main() raises:
    print("Step A: conv2d heap churn...", end="")
    step_a()
    print(" OK")
    print("Step B: bitcast write + conv2d...", end="")
    step_b()
    print(" OK — no crash (bug may be environment-dependent)")
