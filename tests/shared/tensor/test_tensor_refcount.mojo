"""Tests for Tensor[dtype] <-> AnyTensor shared refcount safety (B4).

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- as_any() shares refcount (same data)
- as_tensor() shares refcount (same data)
- B4 regression: source destroyed, converted tensor still valid
- Reverse B4: Tensor[dtype] destroyed, AnyTensor still valid
- Copy increments refcount
- Multiple conversions all share refcount
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.any_tensor import AnyTensor, zeros


fn test_refcount_shared_on_as_any() raises:
    """As_any() shares refcount -- both tensors access same data."""
    var t = Tensor[DType.float32]([4])
    t._data[0] = Scalar[DType.float32](1.5)

    var any_t = t.as_any()

    # Both should see the same value
    assert_almost_equal(any_t._get_float32(0), Float32(1.5), atol=1e-6)
    assert_almost_equal(Float32(t[0]), Float32(1.5), atol=1e-6)
    print("PASS: test_refcount_shared_on_as_any")


fn test_refcount_shared_on_as_tensor() raises:
    """As_tensor() shares refcount -- both tensors access same data."""
    var any_t = zeros([4], DType.float32)
    any_t._set_float32(0, Float32(0.25))

    var t = any_t.as_tensor[DType.float32]()

    assert_almost_equal(Float32(t[0]), Float32(0.25), atol=1e-6)
    print("PASS: test_refcount_shared_on_as_tensor")


fn _make_tensor_from_anytensor() raises -> Tensor[DType.float32]:
    """Helper: create AnyTensor, convert to Tensor, return Tensor (AnyTensor destroyed on return)."""
    var any_t = zeros([4], DType.float32)
    any_t._set_float32(0, Float32(1.5))
    any_t._set_float32(1, Float32(0.5))
    return any_t.as_tensor[DType.float32]()
    # any_t is destroyed here — ASAP destruction


fn test_refcount_survives_source_destruction() raises:
    """B4 regression: AnyTensor destroyed, Tensor[dtype] still valid."""
    var t = _make_tensor_from_anytensor()
    # any_t is definitively gone
    assert_almost_equal(Float32(t[0]), Float32(1.5), atol=1e-6)
    assert_almost_equal(Float32(t[1]), Float32(0.5), atol=1e-6)
    print("PASS: test_refcount_survives_source_destruction")


fn _make_anytensor_from_tensor() raises -> AnyTensor:
    """Helper: create Tensor, convert to AnyTensor, return AnyTensor (Tensor destroyed on return)."""
    var t = Tensor[DType.float32]([4])
    t._data[0] = Scalar[DType.float32](0.125)
    return t.as_any()
    # t is destroyed here — ASAP destruction


fn test_refcount_survives_tensor_destruction() raises:
    """Reverse B4: Tensor[dtype] destroyed, AnyTensor still valid."""
    var any_t = _make_anytensor_from_tensor()
    assert_almost_equal(any_t._get_float32(0), Float32(0.125), atol=1e-6)
    print("PASS: test_refcount_survives_tensor_destruction")


fn test_tensor_copy_increments_refcount() raises:
    """Copying a Tensor[dtype] increments shared refcount."""
    var t1 = Tensor[DType.float32]([4])
    t1._data[0] = Scalar[DType.float32](1.0)

    var t2 = t1  # __copyinit__
    assert_almost_equal(Float32(t2[0]), Float32(1.0), atol=1e-6)
    assert_almost_equal(Float32(t1[0]), Float32(1.0), atol=1e-6)
    print("PASS: test_tensor_copy_increments_refcount")


fn test_tensor_multiple_conversions() raises:
    """Multiple conversions from same source all share refcount."""
    var any_t = zeros([4], DType.float32)
    any_t._set_float32(0, Float32(0.5))

    var t1 = any_t.as_tensor[DType.float32]()
    var t2 = any_t.as_tensor[DType.float32]()

    assert_almost_equal(Float32(t1[0]), Float32(0.5), atol=1e-6)
    assert_almost_equal(Float32(t2[0]), Float32(0.5), atol=1e-6)
    print("PASS: test_tensor_multiple_conversions")


fn main() raises:
    test_refcount_shared_on_as_any()
    test_refcount_shared_on_as_tensor()
    test_refcount_survives_source_destruction()
    test_refcount_survives_tensor_destruction()
    test_tensor_copy_increments_refcount()
    test_tensor_multiple_conversions()
    print("All 6 tensor refcount tests passed!")
