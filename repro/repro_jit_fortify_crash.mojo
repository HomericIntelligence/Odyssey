"""Reproducer attempt for JIT __fortify_fail_abort in CI.

Environment: Mojo 0.26.3.0.dev2026040705 (69cac1bd), GLIBC 2.39, Ubuntu 24.04

This file attempts to reproduce the JIT buffer overflow (libKGENCompilerRTShared.so
offsets 0x6d4ab / 0x6a686 / 0x6e157) seen in CI by stressing the JIT compiler with:

  1. Many large structs with complex trait implementations
  2. Many generic functions with multiple type parameters
  3. Deep nesting of SIMD type instantiations
  4. Heavy compile-time integer arithmetic and conditional traits

The crash is CI-only (fresh pixi volumes + UID mismatch + no TTY). Running this
file locally will almost certainly succeed — that is expected. The file exists to:
  a) Document the stress patterns that correlate with the CI crash
  b) Provide a starting point if the JIT bug is ever made locally reproducible
  c) Confirm no user-code memory errors (ASAN reports clean on this file)

Local result: RUNS CLEAN (no crash, no ASAN errors)
CI result:    UNKNOWN — may trigger the crash if environment conditions align

See: repro/issues/jit-fortify-buffer-overflow.md
"""

from std.testing import assert_equal, assert_almost_equal, assert_true

# ============================================================================
# Stress Pattern 1: Many large structs with trait implementations
# Each struct forces the JIT to instantiate a separate vtable + method table.
# ============================================================================

trait Computable:
    def compute(self) -> Float32: ...
    def size(self) -> Int: ...
    def name(self) -> String: ...


struct Block0(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block0"


struct Block1(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block1"


struct Block2(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block2"


struct Block3(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block3"


struct Block4(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block4"


struct Block5(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block5"


struct Block6(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block6"


struct Block7(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block7"


struct Block8(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block8"


struct Block9(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block9"


struct Block10(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block10"


struct Block11(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block11"


struct Block12(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block12"


struct Block13(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block13"


struct Block14(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block14"


struct Block15(Computable):
    var data: SIMD[DType.float32, 16]

    def __init__(out self, val: Float32):
        self.data = SIMD[DType.float32, 16](val)

    def compute(self) -> Float32:
        return self.data.reduce_add()

    def size(self) -> Int:
        return 16

    def name(self) -> String:
        return "Block15"


# ============================================================================
# Stress Pattern 2: Generic functions with multiple SIMD type permutations
# Forces JIT to instantiate many specializations of the same function body.
# ============================================================================

def simd_sum[dtype: DType, width: Int](v: SIMD[dtype, width]) -> SIMD[dtype, 1]:
    return v.reduce_add()


def simd_max[dtype: DType, width: Int](v: SIMD[dtype, width]) -> SIMD[dtype, 1]:
    return v.reduce_max()


def simd_min[dtype: DType, width: Int](v: SIMD[dtype, width]) -> SIMD[dtype, 1]:
    return v.reduce_min()


def dot2[dtype: DType, width: Int](
    a: SIMD[dtype, width], b: SIMD[dtype, width]
) -> SIMD[dtype, 1]:
    return (a * b).reduce_add()


# ============================================================================
# Stress Pattern 3: Deep compile-time nesting via parametric wrapper structs
# ============================================================================

struct Scaled[dtype: DType, width: Int, factor: Int]:
    var v: SIMD[Self.dtype, Self.width]

    def __init__(out self, base: SIMD[Self.dtype, Self.width]):
        self.v = base * SIMD[Self.dtype, Self.width](Self.factor)

    def sum(self) -> SIMD[Self.dtype, 1]:
        return self.v.reduce_add()


struct NormedScaled[dtype: DType, width: Int, factor: Int, norm: Int]:
    var inner: Scaled[Self.dtype, Self.width, Self.factor]

    def __init__(out self, base: SIMD[Self.dtype, Self.width]):
        self.inner = Scaled[Self.dtype, Self.width, Self.factor](base)

    def value(self) -> SIMD[Self.dtype, 1]:
        return self.inner.sum() / SIMD[Self.dtype, 1](Self.norm)


# ============================================================================
# Stress Pattern 4: Many test functions (mimics CI-crashing test files)
# ============================================================================

def test_block0() raises:
    var b = Block0(1.0)
    assert_almost_equal(b.compute(), 16.0, atol=1e-5)
    assert_equal(b.size(), 16)


def test_block1() raises:
    var b = Block1(2.0)
    assert_almost_equal(b.compute(), 32.0, atol=1e-5)
    assert_equal(b.size(), 16)


def test_block2() raises:
    var b = Block2(0.5)
    assert_almost_equal(b.compute(), 8.0, atol=1e-5)


def test_block3() raises:
    var b = Block3(-1.0)
    assert_almost_equal(b.compute(), -16.0, atol=1e-5)


def test_block4() raises:
    var b = Block4(0.0)
    assert_almost_equal(b.compute(), 0.0, atol=1e-5)


def test_block5() raises:
    var b = Block5(1.5)
    assert_almost_equal(b.compute(), 24.0, atol=1e-5)


def test_block6() raises:
    var b = Block6(3.0)
    assert_almost_equal(b.compute(), 48.0, atol=1e-5)


def test_block7() raises:
    var b = Block7(0.25)
    assert_almost_equal(b.compute(), 4.0, atol=1e-5)


def test_block8() raises:
    var b = Block8(10.0)
    assert_almost_equal(b.compute(), 160.0, atol=1e-5)


def test_block9() raises:
    var b = Block9(-0.5)
    assert_almost_equal(b.compute(), -8.0, atol=1e-5)


def test_block10() raises:
    var b = Block10(100.0)
    assert_almost_equal(b.compute(), 1600.0, atol=1e-5)


def test_block11() raises:
    var b = Block11(0.1)
    assert_almost_equal(b.compute(), 1.6, atol=1e-4)


def test_block12() raises:
    var b = Block12(0.0)
    assert_almost_equal(b.compute(), 0.0, atol=1e-5)
    assert_equal(b.name(), "Block12")


def test_block13() raises:
    var b = Block13(1.0)
    assert_almost_equal(b.compute(), 16.0, atol=1e-5)
    assert_equal(b.name(), "Block13")


def test_block14() raises:
    var b = Block14(2.5)
    assert_almost_equal(b.compute(), 40.0, atol=1e-5)
    assert_equal(b.name(), "Block14")


def test_block15() raises:
    var b = Block15(-2.0)
    assert_almost_equal(b.compute(), -32.0, atol=1e-5)
    assert_equal(b.name(), "Block15")


def test_simd_sum_f32_4() raises:
    var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    assert_almost_equal(simd_sum(v)[0], 10.0, atol=1e-5)


def test_simd_sum_f32_8() raises:
    var v = SIMD[DType.float32, 8](1.0)
    assert_almost_equal(simd_sum(v)[0], 8.0, atol=1e-5)


def test_simd_sum_f32_16() raises:
    var v = SIMD[DType.float32, 16](1.0)
    assert_almost_equal(simd_sum(v)[0], 16.0, atol=1e-5)


def test_simd_max_f32_4() raises:
    var v = SIMD[DType.float32, 4](1.0, 4.0, 2.0, 3.0)
    assert_almost_equal(simd_max(v)[0], 4.0, atol=1e-5)


def test_simd_min_f32_4() raises:
    var v = SIMD[DType.float32, 4](1.0, 4.0, -2.0, 3.0)
    assert_almost_equal(simd_min(v)[0], -2.0, atol=1e-5)


def test_dot2_f32_4() raises:
    var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float32, 4](4.0, 3.0, 2.0, 1.0)
    assert_almost_equal(dot2(a, b)[0], 20.0, atol=1e-5)


def test_scaled_f32_8_factor2() raises:
    var base = SIMD[DType.float32, 8](1.0)
    var s = Scaled[DType.float32, 8, 2](base)
    assert_almost_equal(s.sum()[0], 16.0, atol=1e-5)


def test_scaled_f32_8_factor4() raises:
    var base = SIMD[DType.float32, 8](1.0)
    var s = Scaled[DType.float32, 8, 4](base)
    assert_almost_equal(s.sum()[0], 32.0, atol=1e-5)


def test_normed_scaled_f32_8_factor2_norm2() raises:
    var base = SIMD[DType.float32, 8](1.0)
    var ns = NormedScaled[DType.float32, 8, 2, 2](base)
    assert_almost_equal(ns.value()[0], 8.0, atol=1e-5)


def test_normed_scaled_f32_16_factor4_norm8() raises:
    var base = SIMD[DType.float32, 16](1.0)
    var ns = NormedScaled[DType.float32, 16, 4, 8](base)
    assert_almost_equal(ns.value()[0], 8.0, atol=1e-5)


def test_simd_sum_f64_4() raises:
    var v = SIMD[DType.float64, 4](1.0, 2.0, 3.0, 4.0)
    assert_almost_equal(simd_sum(v)[0], 10.0, atol=1e-10)


def test_dot2_f64_8() raises:
    var a = SIMD[DType.float64, 8](2.0)
    var b = SIMD[DType.float64, 8](3.0)
    assert_almost_equal(dot2(a, b)[0], 48.0, atol=1e-10)


def test_simd_max_f64_8() raises:
    var v = SIMD[DType.float64, 8](0.0, 5.0, -1.0, 3.0, 7.0, 2.0, -4.0, 1.0)
    assert_almost_equal(simd_max(v)[0], 7.0, atol=1e-10)


def test_simd_min_f64_8() raises:
    var v = SIMD[DType.float64, 8](0.0, 5.0, -1.0, 3.0, 7.0, 2.0, -4.0, 1.0)
    assert_almost_equal(simd_min(v)[0], -4.0, atol=1e-10)


def test_all_blocks_positive() raises:
    assert_true(Block0(1.0).compute() > 0.0)
    assert_true(Block1(1.0).compute() > 0.0)
    assert_true(Block2(1.0).compute() > 0.0)
    assert_true(Block3(1.0).compute() > 0.0)
    assert_true(Block4(1.0).compute() > 0.0)
    assert_true(Block5(1.0).compute() > 0.0)
    assert_true(Block6(1.0).compute() > 0.0)
    assert_true(Block7(1.0).compute() > 0.0)
    assert_true(Block8(1.0).compute() > 0.0)
    assert_true(Block9(1.0).compute() > 0.0)
    assert_true(Block10(1.0).compute() > 0.0)
    assert_true(Block11(1.0).compute() > 0.0)
    assert_true(Block12(1.0).compute() > 0.0)
    assert_true(Block13(1.0).compute() > 0.0)
    assert_true(Block14(1.0).compute() > 0.0)
    assert_true(Block15(1.0).compute() > 0.0)


def test_all_blocks_size_16() raises:
    assert_equal(Block0(0.0).size(), 16)
    assert_equal(Block1(0.0).size(), 16)
    assert_equal(Block2(0.0).size(), 16)
    assert_equal(Block3(0.0).size(), 16)
    assert_equal(Block4(0.0).size(), 16)
    assert_equal(Block5(0.0).size(), 16)
    assert_equal(Block6(0.0).size(), 16)
    assert_equal(Block7(0.0).size(), 16)
    assert_equal(Block8(0.0).size(), 16)
    assert_equal(Block9(0.0).size(), 16)
    assert_equal(Block10(0.0).size(), 16)
    assert_equal(Block11(0.0).size(), 16)
    assert_equal(Block12(0.0).size(), 16)
    assert_equal(Block13(0.0).size(), 16)
    assert_equal(Block14(0.0).size(), 16)
    assert_equal(Block15(0.0).size(), 16)


def test_scaled_width4_all_factors() raises:
    var base = SIMD[DType.float32, 4](1.0)
    assert_almost_equal(Scaled[DType.float32, 4, 1](base).sum()[0], 4.0, atol=1e-5)
    assert_almost_equal(Scaled[DType.float32, 4, 2](base).sum()[0], 8.0, atol=1e-5)
    assert_almost_equal(Scaled[DType.float32, 4, 3](base).sum()[0], 12.0, atol=1e-5)
    assert_almost_equal(Scaled[DType.float32, 4, 4](base).sum()[0], 16.0, atol=1e-5)
    assert_almost_equal(Scaled[DType.float32, 4, 8](base).sum()[0], 32.0, atol=1e-5)


def main():
    print("repro_jit_fortify_crash: stress-testing JIT compiler...")
    print("")

    try:
        test_block0()
        test_block1()
        test_block2()
        test_block3()
        test_block4()
        test_block5()
        test_block6()
        test_block7()
        test_block8()
        test_block9()
        test_block10()
        test_block11()
        test_block12()
        test_block13()
        test_block14()
        test_block15()
        print("  [PASS] Block struct tests (16 structs, 16 instantiations)")

        test_simd_sum_f32_4()
        test_simd_sum_f32_8()
        test_simd_sum_f32_16()
        test_simd_max_f32_4()
        test_simd_min_f32_4()
        test_dot2_f32_4()
        test_simd_sum_f64_4()
        test_dot2_f64_8()
        test_simd_max_f64_8()
        test_simd_min_f64_8()
        print("  [PASS] SIMD generic function tests (f32+f64, width 4/8/16)")

        test_scaled_f32_8_factor2()
        test_scaled_f32_8_factor4()
        test_scaled_width4_all_factors()
        test_normed_scaled_f32_8_factor2_norm2()
        test_normed_scaled_f32_16_factor4_norm8()
        print("  [PASS] Parametric wrapper struct tests (Scaled + NormedScaled)")

        test_all_blocks_positive()
        test_all_blocks_size_16()
        print("  [PASS] Cross-struct aggregate tests")

        print("")
        print("All tests passed — crash NOT reproduced locally (expected).")
        print("See repro/issues/jit-fortify-buffer-overflow.md for CI-only context.")
    except e:
        print("FAILED:", e)
        print("If this is 'error: execution crashed' with __fortify_fail_abort,")
        print("the CI-only JIT buffer overflow has been reproduced locally.")
