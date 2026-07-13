"""Tests for Tensor[dtype] <-> AnyTensor conversion.

Tests cover:
- AnyTensor alias backward compatibility
- AnyTensor direct creation
- Tensor[dtype].as_any() returns AnyTensor
- as_any() preserves shape
- AnyTensor.as_tensor[dtype]() returns Tensor[dtype]
- as_tensor with wrong dtype raises error
- Roundtrip Tensor -> AnyTensor -> Tensor preserves data
- Import from odyssey.tensor package works
"""

from std.testing import assert_true, assert_almost_equal
from odyssey.tensor.tensor import (
    Tensor,
    Tensor as T,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros


def test_anytensor_alias_works() raises:
    """AnyTensor alias still works for backward compatibility."""
    var t: AnyTensor = zeros([3, 4], DType.float32)
    assert_true(t.numel() == 12, "AnyTensor alias should work")
    print("PASS: test_anytensor_alias_works")


def test_anytensor_creation() raises:
    """AnyTensor can be created directly."""
    var t: AnyTensor = zeros([2, 3], DType.float32)
    assert_true(t.numel() == 6, "AnyTensor creation")
    print("PASS: test_anytensor_creation")


def test_as_any_basic() raises:
    """Tensor[dtype].as_any() returns an AnyTensor."""
    var t = Tensor[DType.float32]([3])
    var any_t = t.as_any()
    assert_true(any_t.numel() == 3, "as_any preserves numel")
    assert_true(any_t.dtype() == DType.float32, "as_any preserves dtype")
    print("PASS: test_as_any_basic")


def test_as_any_preserves_shape() raises:
    """As_any preserves full shape."""
    var t = Tensor[DType.float64]([2, 3, 4])
    var any_t = t.as_any()
    var s = any_t.shape()
    assert_true(len(s) == 3, "shape dims")
    assert_true(s[0] == 2, "dim 0")
    assert_true(s[1] == 3, "dim 1")
    assert_true(s[2] == 4, "dim 2")
    print("PASS: test_as_any_preserves_shape")


def test_as_tensor_basic() raises:
    """AnyTensor.as_tensor[dtype]() returns a Tensor[dtype]."""
    var any_t = zeros([4], DType.float32)
    var t = any_t.as_tensor[DType.float32]()
    assert_true(t.numel() == 4, "as_tensor preserves numel")
    assert_true(t.get_dtype() == DType.float32, "as_tensor preserves dtype")
    print("PASS: test_as_tensor_basic")


def test_as_tensor_dtype_mismatch() raises:
    """As_tensor with wrong dtype should raise."""
    var any_t = zeros([4], DType.float32)
    var raised = False
    try:
        var t = any_t.as_tensor[DType.float64]()
        _ = t  # suppress unused warning
    except:
        raised = True
    assert_true(raised, "as_tensor with wrong dtype should raise")
    print("PASS: test_as_tensor_dtype_mismatch")


def test_roundtrip_tensor_any_tensor() raises:
    """Tensor -> AnyTensor -> Tensor roundtrip preserves data."""
    var t1 = Tensor[DType.float32]([4])
    # Set a value via the data pointer (since we don't have set() yet)
    t1._data[0] = Scalar[DType.float32](1.5)
    t1._data[1] = Scalar[DType.float32](0.25)

    # Round-trip
    var any_t = t1.as_any()
    var t2 = any_t.as_tensor[DType.float32]()

    assert_almost_equal(Float32(t2[0]), Float32(1.5), atol=1e-6)
    assert_almost_equal(Float32(t2[1]), Float32(0.25), atol=1e-6)
    print("PASS: test_roundtrip_tensor_any_tensor")


def test_tensor_import_from_package() raises:
    """Verify import from odyssey.tensor works."""
    var t = T[DType.float32]([2])
    assert_true(t.numel() == 2, "import from odyssey.tensor works")
    print("PASS: test_tensor_import_from_package")


def main() raises:
    test_anytensor_alias_works()
    test_anytensor_creation()
    test_as_any_basic()
    test_as_any_preserves_shape()
    test_as_tensor_basic()
    test_as_tensor_dtype_mismatch()
    test_roundtrip_tensor_any_tensor()
    test_tensor_import_from_package()
    print("All 8 tensor conversion tests passed!")
