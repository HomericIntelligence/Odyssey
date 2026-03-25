"""Reproducer: debug_assert in @always_inline parametric method.

Investigate whether debug_assert causes JIT compilation crashes when used
in @always_inline parametric methods on a struct with UnsafePointer fields.

Context: Adding debug_assert to AnyTensor.load[dtype]/store[dtype]/data_ptr[dtype]
methods caused 140+ test files to crash with "execution crashed" in CI's JIT mode
(pixi run mojo --Werror), while passing locally in both JIT and AOT modes.

The crash manifests as libc.so.6+0x45330 (fortify_fail_abort) — identical to the
Day 53 bitcast UAF signature. The hypothesis is that debug_assert increases the JIT
compilation footprint enough to push more tests over the crash threshold in CI's
Docker container environment.

Test:
    # AOT:
    pixi run mojo build -o /tmp/repro repro_debug_assert_jit.mojo && /tmp/repro

    # JIT:
    pixi run mojo run repro_debug_assert_jit.mojo

    # JIT with --Werror (CI mode):
    pixi run mojo --Werror repro_debug_assert_jit.mojo

Local results (WSL2, GLIBC 2.39): ALL PASS
CI results (Docker, GLIBC 2.35): crashes when part of larger compilation (any_tensor.mojo)

Conclusion: debug_assert itself is not broken. The increased compilation footprint
from inlining parametric methods with debug_assert across 100+ call sites in the
training module pushes the JIT compiler past its buffer overflow threshold (same
root cause as Day 53 / modular#6187).
"""
from memory import UnsafePointer, alloc, memset_zero


struct Foo:
    var _data: UnsafePointer[UInt8, origin=MutAnyOrigin]
    var _dtype: DType

    fn __init__(out self):
        self._data = alloc[UInt8](16)
        memset_zero(self._data, 16)
        self._dtype = DType.float32

    fn __del__(deinit self):
        self._data.free()

    @always_inline
    fn load[dtype: DType](self, index: Int) -> Scalar[dtype]:
        debug_assert(
            self._dtype == dtype,
            "dtype mismatch",
        )
        return self._data.bitcast[Scalar[dtype]]()[index]

    @always_inline
    fn store[dtype: DType](self, index: Int, value: Scalar[dtype]):
        debug_assert(
            self._dtype == dtype,
            "dtype mismatch",
        )
        self._data.bitcast[Scalar[dtype]]()[index] = value


fn main():
    var f = Foo()

    # Store and load
    f.store[DType.float32](0, Float32(42.0))
    var val = f.load[DType.float32](0)
    print("load result:", val)

    # Multiple types
    f.store[DType.float32](1, Float32(3.14))
    print("load[1]:", f.load[DType.float32](1))

    print("PASS: debug_assert in @always_inline parametric methods works")
