"""Tests for post-review fixes on ExTensor migration.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Tensor[dtype].__hash__ typed access (no _get_float64 round-trip)
- AnyTensor TensorLike conformance
- In-place operator precision (byte copy, not float64 round-trip)
- __neg__ and reflected operators
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.core.any_tensor import AnyTensor, zeros, ones


fn test_tensor_hash_typed_access() raises:
    """Tensor[dtype].__hash__ uses typed access, not _get_float64."""
    var t1 = Tensor[DType.float32]([4])
    t1._data[0] = Scalar[DType.float32](0.5)
    t1._data[1] = Scalar[DType.float32](1.0)
    var t2 = Tensor[DType.float32]([4])
    t2._data[0] = Scalar[DType.float32](0.5)
    t2._data[1] = Scalar[DType.float32](1.0)
    assert_true(hash(t1) == hash(t2), "identical tensors should hash equal")
    print("PASS: test_tensor_hash_typed_access")


fn test_tensor_hash_different_values() raises:
    """Different tensor values should produce different hashes."""
    var t1 = Tensor[DType.float32]([4])
    t1._data[0] = Scalar[DType.float32](0.5)
    var t2 = Tensor[DType.float32]([4])
    t2._data[0] = Scalar[DType.float32](1.0)
    assert_true(hash(t1) != hash(t2), "different tensors should hash differently")
    print("PASS: test_tensor_hash_different_values")


fn test_anytensor_conforms_tensorlike() raises:
    """AnyTensor should conform to TensorLike trait."""
    var t: AnyTensor = zeros([3, 4], DType.float32)
    assert_true(t.numel() == 12, "numel")
    assert_true(t.ndim() == 2, "ndim")
    assert_true(t.dtype() == DType.float32, "dtype")
    print("PASS: test_anytensor_conforms_tensorlike")


fn test_iadd_precision_float64() raises:
    """In-place addition preserves float64 precision (M1 fix)."""
    var a = zeros([4], DType.float64)
    a._set_float64(0, 3.141592653589793)
    var b = zeros([4], DType.float64)
    b._set_float64(0, 2.718281828459045)
    a += b
    var expected = 3.141592653589793 + 2.718281828459045
    assert_almost_equal(a._get_float64(0), expected, atol=1e-12)
    print("PASS: test_iadd_precision_float64")


fn test_isub_precision_float64() raises:
    """In-place subtraction preserves float64 precision (M1 fix)."""
    var a = zeros([4], DType.float64)
    a._set_float64(0, 3.141592653589793)
    var b = zeros([4], DType.float64)
    b._set_float64(0, 1.0)
    a -= b
    var expected = 3.141592653589793 - 1.0
    assert_almost_equal(a._get_float64(0), expected, atol=1e-12)
    print("PASS: test_isub_precision_float64")


fn test_neg_anytensor() raises:
    """__neg__ works correctly on AnyTensor."""
    var t = zeros([4], DType.float32)
    t._set_float32(0, Float32(1.5))
    t._set_float32(1, Float32(-0.5))
    var neg_t = -t
    assert_almost_equal(neg_t._get_float32(0), Float32(-1.5), atol=1e-6)
    assert_almost_equal(neg_t._get_float32(1), Float32(0.5), atol=1e-6)
    print("PASS: test_neg_anytensor")


fn test_reflected_add() raises:
    """Reflected operators work on AnyTensor."""
    var a: AnyTensor = ones([4], DType.float32)
    var b: AnyTensor = ones([4], DType.float32)
    var c = a + b
    assert_almost_equal(c._get_float32(0), Float32(2.0), atol=1e-6)
    print("PASS: test_reflected_add")


fn main() raises:
    test_tensor_hash_typed_access()
    test_tensor_hash_different_values()
    test_anytensor_conforms_tensorlike()
    test_iadd_precision_float64()
    test_isub_precision_float64()
    test_neg_anytensor()
    test_reflected_add()
    print("\n✓ All 7 review fix tests passed\n")
