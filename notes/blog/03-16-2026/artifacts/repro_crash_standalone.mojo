"""Self-contained Mojo v0.26.1 runtime crash reproducer.

NO external dependencies. Copy this single file to any Mojo 0.26.1 install.

Environment: Mojo 0.26.1 (156d3ac6), GLIBC 2.39, Linux 6.6.87 x86_64 (WSL2)

Reproduces a deterministic crash in the Mojo runtime allocator triggered by:
1. step_a(): Heavy alloc/free churn (conv2d creates many intermediate tensors,
   all freed when the function returns)
2. step_b(): Create a small tensor, write via UnsafePointer.bitcast, then do
   another conv2d -> CRASH in libAsyncRTRuntimeGlobals.so

Removing the bitcast write in step_b() prevents the crash.

Stack trace:
  #0 libKGENCompilerRTShared.so  +0x3cb78b
  #1 libKGENCompilerRTShared.so  +0x3c93c6
  #2 libKGENCompilerRTShared.so  +0x3cc397
  #3 libc.so.6                   +0x45330
  #4 libAsyncRTRuntimeGlobals.so +0x416ba  (allocator - crash origin)
"""

from memory import UnsafePointer, memset_zero, alloc, memcpy
from collections import List


# ============================================================================
# Minimal tensor struct (inlined from ExTensor)
# ============================================================================


struct Tensor(Movable, Copyable):
    var _data: UnsafePointer[UInt8, origin=MutAnyOrigin]
    var _shape: List[Int]
    var _numel: Int
    var _refcount: UnsafePointer[Int, origin=MutAnyOrigin]
    var _alloc_size: Int

    fn __init__(out self, shape: List[Int]) raises:
        self._shape = List[Int]()
        self._numel = 1
        for i in range(len(shape)):
            self._shape.append(shape[i])
            self._numel *= shape[i]
        self._alloc_size = self._numel * 4  # float32 = 4 bytes
        self._data = alloc[UInt8](self._alloc_size)
        memset_zero(self._data, self._alloc_size)
        self._refcount = alloc[Int](1)
        self._refcount[] = 1

    fn __copyinit__(out self, existing: Self):
        self._data = existing._data
        self._shape = existing._shape.copy()
        self._numel = existing._numel
        self._refcount = existing._refcount
        self._alloc_size = existing._alloc_size
        if self._refcount:
            self._refcount[] += 1

    fn __moveinit__(out self, deinit existing: Self):
        self._data = existing._data
        self._shape = existing._shape.copy()
        self._numel = existing._numel
        self._refcount = existing._refcount
        self._alloc_size = existing._alloc_size

    fn __del__(deinit self):
        if self._refcount:
            self._refcount[] -= 1
            if self._refcount[] == 0:
                self._data.free()
                self._refcount.free()

    fn fp(self) -> UnsafePointer[Float32, origin=MutAnyOrigin]:
        return self._data.bitcast[Float32]()


fn zeros(shape: List[Int]) raises -> Tensor:
    return Tensor(shape)


fn ones(shape: List[Int]) raises -> Tensor:
    var t = Tensor(shape)
    var p = t.fp()
    for i in range(t._numel):
        p[i] = 1.0
    return t^


fn shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


fn shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


# ============================================================================
# Inlined conv2d (sequential, float32 only, stride=1, padding=1)
# ============================================================================


fn conv2d(
    x: Tensor,
    kernel: Tensor,
    bias: Tensor,
) raises -> Tensor:
    var batch = x._shape[0]
    var in_ch = x._shape[1]
    var in_h = x._shape[2]
    var in_w = x._shape[3]
    var out_ch = kernel._shape[0]
    var kH = kernel._shape[2]
    var kW = kernel._shape[3]
    # padding=1, stride=1
    var out_h = in_h
    var out_w = in_w

    var output = zeros(shape4(batch, out_ch, out_h, out_w))
    var out_p = output.fp()
    var x_p = x.fp()
    var k_p = kernel.fp()
    var b_p = bias.fp()

    for b in range(batch):
        for oc in range(out_ch):
            for oh in range(out_h):
                for ow in range(out_w):
                    var sum_val = Float32(0.0)
                    var in_h_start = oh - 1  # padding=1
                    var in_w_start = ow - 1
                    for ic in range(in_ch):
                        for kh in range(kH):
                            for kw in range(kW):
                                var ih = in_h_start + kh
                                var iw = in_w_start + kw
                                if (
                                    ih >= 0
                                    and ih < in_h
                                    and iw >= 0
                                    and iw < in_w
                                ):
                                    var in_idx = (
                                        b * (in_ch * in_h * in_w)
                                        + ic * (in_h * in_w)
                                        + ih * in_w
                                        + iw
                                    )
                                    var k_idx = (
                                        oc * (in_ch * kH * kW)
                                        + ic * (kH * kW)
                                        + kh * kW
                                        + kw
                                    )
                                    sum_val += x_p[in_idx] * k_p[k_idx]
                    sum_val += b_p[oc]
                    var out_idx = (
                        b * (out_ch * out_h * out_w)
                        + oc * (out_h * out_w)
                        + oh * out_w
                        + ow
                    )
                    out_p[out_idx] = sum_val
    return output^


# ============================================================================
# Inlined ReLU
# ============================================================================


fn relu(t: Tensor) raises -> Tensor:
    var out = Tensor(t._shape)
    var ip = t.fp()
    var op = out.fp()
    for i in range(t._numel):
        op[i] = max(Float32(0), ip[i])
    return out^


# ============================================================================
# Crash reproducer
# ============================================================================


fn step_a() raises:
    """Heavy alloc/free via 2x conv+relu. ~20 tensors freed on return."""
    var x = ones(shape4(2, 3, 32, 32))
    # Conv 1: 3->16
    x = conv2d(x, ones(shape4(16, 3, 3, 3)), zeros(shape1(16)))
    x = relu(x)
    # Conv 2: 16->16
    x = conv2d(x, ones(shape4(16, 16, 3, 3)), zeros(shape1(16)))
    _ = relu(x)


fn step_b() raises:
    """Bitcast write then conv2d -> CRASH."""
    var target = zeros(shape1(2))
    # === TRIGGER: comment these 3 lines to prevent crash ===
    var td = target._data.bitcast[Float32]()
    td[0] = 0.0
    td[1] = 1.0
    # === END TRIGGER ===
    var x = ones(shape4(2, 3, 32, 32))
    x = conv2d(x, ones(shape4(16, 3, 3, 3)), zeros(shape1(16)))
    _ = relu(x)


fn main() raises:
    print("Step A: conv2d heap churn...", end="")
    step_a()
    print(" OK")
    print("Step B: bitcast write + conv2d...", end="")
    step_b()
    print(" OK — no crash (bug may be environment-dependent)")
