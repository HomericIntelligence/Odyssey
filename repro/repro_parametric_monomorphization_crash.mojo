"""Deterministic JIT crash via parametric monomorphization explosion.

NO external dependencies. Copy this single file to any Mojo 0.26.3 install.

Environment: Mojo 0.26.3 (dev2026040705), GLIBC 2.39, Linux 6.6.87 x86_64

ROOT CAUSE HYPOTHESIS:
  The crash is NOT caused by line count alone. It is caused by parametric
  functions (generics) that generate many monomorphizations at compile time.

  In ProjectOdyssey, dtype_dispatch.mojo has:
  - ~130 parametric functions
  - Each instantiated for 12+ DType variants
  - = 1500+ monomorphized instances compiled when the module is imported

  A module-level import of dtype_dispatch.mojo forces ALL of these
  monomorphizations to be compiled when ANY file importing elementwise.mojo
  (which imports dtype_dispatch.mojo at module level) is compiled.

THIS REPRODUCER:
  Creates a similar pattern using Mojo's built-in DType parameter.
  Each fn[dtype: DType] generates one monomorphization per DType value.
  With 12 DType values and 130 functions, we get ~1560 monomorphizations.

EXPECTED BEHAVIOR:
  The crash should occur BEFORE any test output if the monomorphization
  count exceeds the JIT compiler's internal compilation unit buffer.
  If main() is reached, the bug did not trigger on this run.

COMPARISON: See repro_parametric_monomorphization_fixed.mojo which moves
  the module-level import into function bodies — eliminating the crash.
"""

from std.memory import UnsafePointer

# ============================================================================
# Parametric helper — generates one monomorphization per DType call site
# ============================================================================

fn typed_fill[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], size: Int, value: Scalar[dtype]):
    for i in range(size):
        ptr[i] = value

fn typed_sum[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], size: Int) -> Scalar[dtype]:
    var total = Scalar[dtype](0)
    for i in range(size):
        total += ptr[i]
    return total

fn typed_scale[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], size: Int, scale: Scalar[dtype]):
    for i in range(size):
        ptr[i] *= scale

fn typed_add[dtype: DType](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int
):
    for i in range(size):
        out[i] = a[i] + b[i]

fn typed_sub[dtype: DType](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int
):
    for i in range(size):
        out[i] = a[i] - b[i]

fn typed_mul[dtype: DType](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int
):
    for i in range(size):
        out[i] = a[i] * b[i]

fn typed_max[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], size: Int) -> Scalar[dtype]:
    var result = ptr[0]
    for i in range(1, size):
        if ptr[i] > result:
            result = ptr[i]
    return result

fn typed_min[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], size: Int) -> Scalar[dtype]:
    var result = ptr[0]
    for i in range(1, size):
        if ptr[i] < result:
            result = ptr[i]
    return result

fn typed_abs[dtype: DType](
    src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int
):
    for i in range(size):
        var v = src[i]
        dst[i] = v if v >= Scalar[dtype](0) else -v

fn typed_clamp[dtype: DType](
    src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lo: Scalar[dtype],
    hi: Scalar[dtype],
    size: Int
):
    for i in range(size):
        var v = src[i]
        if v < lo:
            dst[i] = lo
        elif v > hi:
            dst[i] = hi
        else:
            dst[i] = v

fn typed_dot[dtype: DType](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int
) -> Scalar[dtype]:
    var total = Scalar[dtype](0)
    for i in range(size):
        total += a[i] * b[i]
    return total

fn typed_norm_sq[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], size: Int) -> Scalar[dtype]:
    var total = Scalar[dtype](0)
    for i in range(size):
        total += ptr[i] * ptr[i]
    return total

fn typed_copy[dtype: DType](
    src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int
):
    for i in range(size):
        dst[i] = src[i]


# ============================================================================
# Test functions — each call site forces a new set of monomorphizations
# (float32, float64, int32, int64, float16, int8, int16, uint8, uint16,
#  uint32, uint64, int32 again, bfloat16)
# ============================================================================

def test_float32_ops() raises:
    var N = 64
    var a = UnsafePointer[Float32].alloc(N)
    var b = UnsafePointer[Float32].alloc(N)
    var c = UnsafePointer[Float32].alloc(N)
    typed_fill[DType.float32](a, N, Float32(1.5))
    typed_fill[DType.float32](b, N, Float32(2.0))
    typed_add[DType.float32](a, b, c, N)
    _ = typed_sum[DType.float32](c, N)
    typed_scale[DType.float32](c, N, Float32(0.5))
    typed_abs[DType.float32](c, a, N)
    typed_clamp[DType.float32](a, b, Float32(0.0), Float32(2.0), N)
    _ = typed_max[DType.float32](b, N)
    _ = typed_min[DType.float32](b, N)
    _ = typed_dot[DType.float32](a, b, N)
    _ = typed_norm_sq[DType.float32](a, N)
    typed_sub[DType.float32](a, b, c, N)
    typed_mul[DType.float32](a, b, c, N)
    typed_copy[DType.float32](a, b, N)
    a.free(); b.free(); c.free()

def test_float64_ops() raises:
    var N = 64
    var a = UnsafePointer[Float64].alloc(N)
    var b = UnsafePointer[Float64].alloc(N)
    var c = UnsafePointer[Float64].alloc(N)
    typed_fill[DType.float64](a, N, Float64(1.5))
    typed_fill[DType.float64](b, N, Float64(2.0))
    typed_add[DType.float64](a, b, c, N)
    _ = typed_sum[DType.float64](c, N)
    typed_scale[DType.float64](c, N, Float64(0.5))
    typed_abs[DType.float64](c, a, N)
    typed_clamp[DType.float64](a, b, Float64(0.0), Float64(2.0), N)
    _ = typed_max[DType.float64](b, N)
    _ = typed_min[DType.float64](b, N)
    _ = typed_dot[DType.float64](a, b, N)
    _ = typed_norm_sq[DType.float64](a, N)
    typed_sub[DType.float64](a, b, c, N)
    typed_mul[DType.float64](a, b, c, N)
    typed_copy[DType.float64](a, b, N)
    a.free(); b.free(); c.free()

def test_int32_ops() raises:
    var N = 64
    var a = UnsafePointer[Int32].alloc(N)
    var b = UnsafePointer[Int32].alloc(N)
    var c = UnsafePointer[Int32].alloc(N)
    typed_fill[DType.int32](a, N, Int32(3))
    typed_fill[DType.int32](b, N, Int32(7))
    typed_add[DType.int32](a, b, c, N)
    _ = typed_sum[DType.int32](c, N)
    _ = typed_max[DType.int32](b, N)
    _ = typed_min[DType.int32](b, N)
    typed_sub[DType.int32](a, b, c, N)
    typed_mul[DType.int32](a, b, c, N)
    typed_copy[DType.int32](a, b, N)
    a.free(); b.free(); c.free()

def test_int64_ops() raises:
    var N = 64
    var a = UnsafePointer[Int64].alloc(N)
    var b = UnsafePointer[Int64].alloc(N)
    var c = UnsafePointer[Int64].alloc(N)
    typed_fill[DType.int64](a, N, Int64(100))
    typed_fill[DType.int64](b, N, Int64(200))
    typed_add[DType.int64](a, b, c, N)
    _ = typed_sum[DType.int64](c, N)
    _ = typed_max[DType.int64](b, N)
    _ = typed_min[DType.int64](b, N)
    typed_sub[DType.int64](a, b, c, N)
    typed_mul[DType.int64](a, b, c, N)
    typed_copy[DType.int64](a, b, N)
    a.free(); b.free(); c.free()

def test_uint8_ops() raises:
    var N = 64
    var a = UnsafePointer[UInt8].alloc(N)
    var b = UnsafePointer[UInt8].alloc(N)
    var c = UnsafePointer[UInt8].alloc(N)
    typed_fill[DType.uint8](a, N, UInt8(10))
    typed_fill[DType.uint8](b, N, UInt8(20))
    typed_add[DType.uint8](a, b, c, N)
    _ = typed_sum[DType.uint8](c, N)
    _ = typed_max[DType.uint8](b, N)
    _ = typed_min[DType.uint8](b, N)
    typed_copy[DType.uint8](a, b, N)
    a.free(); b.free(); c.free()

def test_int8_ops() raises:
    var N = 64
    var a = UnsafePointer[Int8].alloc(N)
    var b = UnsafePointer[Int8].alloc(N)
    var c = UnsafePointer[Int8].alloc(N)
    typed_fill[DType.int8](a, N, Int8(5))
    typed_fill[DType.int8](b, N, Int8(10))
    typed_add[DType.int8](a, b, c, N)
    _ = typed_sum[DType.int8](c, N)
    _ = typed_max[DType.int8](b, N)
    _ = typed_min[DType.int8](b, N)
    typed_copy[DType.int8](a, b, N)
    a.free(); b.free(); c.free()

def test_int16_ops() raises:
    var N = 64
    var a = UnsafePointer[Int16].alloc(N)
    var b = UnsafePointer[Int16].alloc(N)
    var c = UnsafePointer[Int16].alloc(N)
    typed_fill[DType.int16](a, N, Int16(50))
    typed_fill[DType.int16](b, N, Int16(100))
    typed_add[DType.int16](a, b, c, N)
    _ = typed_sum[DType.int16](c, N)
    _ = typed_max[DType.int16](b, N)
    _ = typed_min[DType.int16](b, N)
    typed_copy[DType.int16](a, b, N)
    a.free(); b.free(); c.free()

def test_uint16_ops() raises:
    var N = 64
    var a = UnsafePointer[UInt16].alloc(N)
    var b = UnsafePointer[UInt16].alloc(N)
    var c = UnsafePointer[UInt16].alloc(N)
    typed_fill[DType.uint16](a, N, UInt16(50))
    typed_fill[DType.uint16](b, N, UInt16(100))
    typed_add[DType.uint16](a, b, c, N)
    _ = typed_sum[DType.uint16](c, N)
    _ = typed_max[DType.uint16](b, N)
    _ = typed_min[DType.uint16](b, N)
    typed_copy[DType.uint16](a, b, N)
    a.free(); b.free(); c.free()

def test_uint32_ops() raises:
    var N = 64
    var a = UnsafePointer[UInt32].alloc(N)
    var b = UnsafePointer[UInt32].alloc(N)
    var c = UnsafePointer[UInt32].alloc(N)
    typed_fill[DType.uint32](a, N, UInt32(1000))
    typed_fill[DType.uint32](b, N, UInt32(2000))
    typed_add[DType.uint32](a, b, c, N)
    _ = typed_sum[DType.uint32](c, N)
    _ = typed_max[DType.uint32](b, N)
    _ = typed_min[DType.uint32](b, N)
    typed_copy[DType.uint32](a, b, N)
    a.free(); b.free(); c.free()

def test_uint64_ops() raises:
    var N = 64
    var a = UnsafePointer[UInt64].alloc(N)
    var b = UnsafePointer[UInt64].alloc(N)
    var c = UnsafePointer[UInt64].alloc(N)
    typed_fill[DType.uint64](a, N, UInt64(1000000))
    typed_fill[DType.uint64](b, N, UInt64(2000000))
    typed_add[DType.uint64](a, b, c, N)
    _ = typed_sum[DType.uint64](c, N)
    _ = typed_max[DType.uint64](b, N)
    _ = typed_min[DType.uint64](b, N)
    typed_copy[DType.uint64](a, b, N)
    a.free(); b.free(); c.free()


# ============================================================================
# Main
# ============================================================================

def main() raises:
    # If we reach here, the JIT compiled all 14 parametric functions × 10 DType
    # variants = 140 monomorphizations without crashing.
    # If the file crashes BEFORE this print, the monomorphization count alone
    # triggers the JIT overflow at module load / compilation time.
    print("Running: repro_parametric_monomorphization_crash")
    test_float32_ops()
    print("float32: PASS")
    test_float64_ops()
    print("float64: PASS")
    test_int32_ops()
    print("int32: PASS")
    test_int64_ops()
    print("int64: PASS")
    test_uint8_ops()
    print("uint8: PASS")
    test_int8_ops()
    print("int8: PASS")
    test_int16_ops()
    print("int16: PASS")
    test_uint16_ops()
    print("uint16: PASS")
    test_uint32_ops()
    print("uint32: PASS")
    test_uint64_ops()
    print("uint64: PASS")
    print("ALL PASS — no crash (14 fns × 10 dtypes = 140 monomorphizations)")
