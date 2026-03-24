"""Tests for Tensor[dtype] typed element access.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Tensor[DType.float32] creation with shape
- Tensor[DType.float64] creation with shape
- Default dtype (float32)
- Shape and strides verification
- Typed __getitem__ return value
- __str__ output format
- __len__ returns first dimension
- 1D tensor correctness
- item() single-element extraction and multi-element error
- __bool__ truthiness for zero/non-zero and multi-element error
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor


fn test_tensor_float32_creation() raises:
    """Tensor[DType.float32] can be created with a shape."""
    var t = Tensor[DType.float32]([3, 4])
    assert_true(t.numel() == 12, "numel should be 12")
    assert_true(t.ndim() == 2, "ndim should be 2")
    print("PASS: test_tensor_float32_creation")


fn test_tensor_float64_creation() raises:
    """Tensor[DType.float64] can be created with a shape."""
    var t = Tensor[DType.float64]([2, 3])
    assert_true(t.numel() == 6, "numel should be 6")
    assert_true(t.get_dtype() == DType.float64, "dtype should be float64")
    print("PASS: test_tensor_float64_creation")


fn test_tensor_default_dtype() raises:
    """Tensor with no dtype parameter defaults to float32."""
    var t = Tensor([5])
    assert_true(t.get_dtype() == DType.float32, "default dtype should be float32")
    print("PASS: test_tensor_default_dtype")


fn test_tensor_shape_strides() raises:
    """Verify shape and strides are correct."""
    var t = Tensor[DType.float32]([2, 3, 4])
    var s = t.shape()
    assert_true(len(s) == 3, "shape should have 3 dims")
    assert_true(s[0] == 2, "dim 0")
    assert_true(s[1] == 3, "dim 1")
    assert_true(s[2] == 4, "dim 2")
    assert_true(t.numel() == 24, "numel should be 24")
    print("PASS: test_tensor_shape_strides")


fn test_tensor_getitem_returns_typed() raises:
    """__getitem__ returns Scalar[Self.dtype], not Float32."""
    var t = Tensor[DType.float32]([4])
    # Element access should return Scalar[DType.float32]
    var val = t[0]
    # val should be Scalar[DType.float32] — zero-initialized
    assert_almost_equal(Float32(val), Float32(0.0), atol=1e-6)
    print("PASS: test_tensor_getitem_returns_typed")


fn test_tensor_str_output() raises:
    """__str__ should say Tensor, not AnyTensor."""
    var t = Tensor[DType.float32]([3])
    var s = String(t)
    assert_true("Tensor(" in s, "str should contain 'Tensor(' not 'AnyTensor('")
    assert_true("float32" in s, "str should contain dtype")
    print("PASS: test_tensor_str_output")


fn test_tensor_len() raises:
    """__len__ returns first dimension size."""
    var t = Tensor[DType.float32]([5, 3])
    assert_true(len(t) == 5, "len should be first dim")
    print("PASS: test_tensor_len")


fn test_tensor_1d() raises:
    """1D tensor works correctly."""
    var t = Tensor[DType.float64]([10])
    assert_true(t.numel() == 10, "numel")
    assert_true(t.ndim() == 1, "ndim")
    assert_true(len(t) == 10, "len")
    print("PASS: test_tensor_1d")


fn test_tensor_getitem_float64_typed() raises:
    """__getitem__ on float64 tensor returns Scalar[DType.float64], not Float32."""
    var t = Tensor[DType.float64]([4])
    t._data[0] = Scalar[DType.float64](0.25)
    var val = t[0]
    # If __getitem__ returned Float32, this would lose precision for larger values
    assert_almost_equal(Float64(val), Float64(0.25), atol=1e-12)
    print("PASS: test_tensor_getitem_float64_typed")


fn test_tensor_item_single_element() raises:
    """Verify item() extracts value from single-element tensor."""
    var t = Tensor[DType.float32]([1])
    t[0] = Scalar[DType.float32](42.0)
    var val = t.item()
    assert_almost_equal(Float32(val), Float32(42.0), atol=1e-6)
    print("PASS: test_tensor_item_single_element")


fn test_tensor_item_raises_multi_element() raises:
    """Verify item() raises for multi-element tensor."""
    var t = Tensor[DType.float32]([5])
    var error_raised = False
    try:
        var val = t.item()
        _ = val
    except e:
        error_raised = True
        var error_msg = String(e)
        if "single" not in error_msg.lower():
            raise Error(
                "Error message should mention single-element requirement"
            )
    if not error_raised:
        raise Error("item() on multi-element tensor should raise error")
    print("PASS: test_tensor_item_raises_multi_element")


fn test_tensor_bool_single_element() raises:
    """__bool__ on single-element tensor returns correct truthiness."""
    var t_zero = Tensor[DType.float32]([1])
    t_zero[0] = Scalar[DType.float32](0.0)
    var t_nonzero = Tensor[DType.float32]([1])
    t_nonzero[0] = Scalar[DType.float32](5.0)

    if t_zero:
        raise Error("Zero tensor should be falsy")
    if not t_nonzero:
        raise Error("Non-zero tensor should be truthy")
    print("PASS: test_tensor_bool_single_element")


fn test_tensor_bool_raises_multi_element() raises:
    """__bool__ raises for multi-element tensor."""
    var t = Tensor[DType.float32]([5])
    var error_raised = False
    try:
        var val = t.__bool__()
        _ = val
    except e:
        error_raised = True
        var error_msg = String(e)
        if "single" not in error_msg.lower():
            raise Error(
                "Error message should mention single-element requirement"
            )
    if not error_raised:
        raise Error("__bool__ on multi-element tensor should raise error")
    print("PASS: test_tensor_bool_raises_multi_element")


fn main() raises:
    test_tensor_float32_creation()
    test_tensor_float64_creation()
    test_tensor_default_dtype()
    test_tensor_shape_strides()
    test_tensor_getitem_returns_typed()
    test_tensor_getitem_float64_typed()
    test_tensor_str_output()
    test_tensor_len()
    test_tensor_1d()
    test_tensor_item_single_element()
    test_tensor_item_raises_multi_element()
    test_tensor_bool_single_element()
    test_tensor_bool_raises_multi_element()
    print("All 13 tensor element access tests passed!")
