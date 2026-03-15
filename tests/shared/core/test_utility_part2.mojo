"""Tests for ExTensor utility operations - Part 2: contiguity, item(), and tolist().

Tests contiguity checks, as_contiguous(), item() scalar extraction,
and tolist() conversion methods.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_utility.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full, arange, clone, item, diff
from shared.core.shape import as_contiguous
from shared.core.matrix import transpose_view

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_equal,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_equal_int,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
)


# ============================================================================
# Test contiguity
# ============================================================================


fn test_is_contiguous_true() raises:
    """Test that newly created tensors are contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_true(t.is_contiguous(), "Newly created tensor should be contiguous")


fn test_is_contiguous_after_transpose() raises:
    """Test that a transposed tensor is not contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = transpose_view(a)
    assert_false(
        b.is_contiguous(), "Transposed tensor should not be contiguous"
    )


# ============================================================================
# Test contiguous() - make contiguous copy
# ============================================================================


fn test_contiguous_on_noncontiguous() raises:
    """Test making non-contiguous tensor contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var b = a.reshape(shape)
    var t = transpose_view(b)
    assert_false(
        t.is_contiguous(), "Transposed tensor should not be contiguous"
    )

    # as_contiguous() should produce a contiguous copy
    var c = as_contiguous(t)
    assert_true(
        c.is_contiguous(), "as_contiguous() result should be contiguous"
    )

    # Result should have row-major strides [4, 1] for shape (4, 3) after transpose
    assert_equal_int(c._strides[0], 3, "Stride for dim 0 should be 3 (rows)")
    assert_equal_int(c._strides[1], 1, "Stride for dim 1 should be 1")


fn test_contiguous_stride_correct_values() raises:
    """Regression test: as_contiguous() must use stride-based indexing.

    Constructs a 2x3 tensor with column-major strides [1, 2] (simulating a
    non-contiguous view), calls as_contiguous(), and verifies that each output
    element appears at the correct row-major (C-order) position.

    For shape [2, 3] with strides [1, 2], the stride-based source offsets are:
      Output[0,0] = src[0*1 + 0*2] = src[0] = 0.0
      Output[0,1] = src[0*1 + 1*2] = src[2] = 2.0
      Output[0,2] = src[0*1 + 2*2] = src[4] = 4.0
      Output[1,0] = src[1*1 + 0*2] = src[1] = 1.0
      Output[1,1] = src[1*1 + 1*2] = src[3] = 3.0
      Output[1,2] = src[1*1 + 2*2] = src[5] = 5.0

    Flat output: [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]

    This is a regression test for the bug where _get_float64(i) was used
    instead of stride-based indexing (fix in shared/core/shape.mojo).
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = arange(0.0, 6.0, 1.0, DType.float32)
    var b = a.reshape(shape)

    # Manually set column-major strides [1, 2] to simulate a non-contiguous view
    b._strides[0] = 1
    b._strides[1] = 2

    assert_false(b.is_contiguous(), "Column-major tensor should not be contiguous")

    var c = as_contiguous(b)

    assert_true(c.is_contiguous(), "as_contiguous() result should be contiguous")

    # Verify element values using stride-based source offsets
    assert_almost_equal(c._get_float64(0), 0.0, 1e-6, "c[0,0] should be 0.0")
    assert_almost_equal(c._get_float64(1), 2.0, 1e-6, "c[0,1] should be 2.0")
    assert_almost_equal(c._get_float64(2), 4.0, 1e-6, "c[0,2] should be 4.0")
    assert_almost_equal(c._get_float64(3), 1.0, 1e-6, "c[1,0] should be 1.0")
    assert_almost_equal(c._get_float64(4), 3.0, 1e-6, "c[1,1] should be 3.0")
    assert_almost_equal(c._get_float64(5), 5.0, 1e-6, "c[1,2] should be 5.0")


# ============================================================================
# Test item() - scalar extraction
# ============================================================================


fn test_item_single_element() raises:
    """Test extracting value from single-element tensor."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)
    var val = item(t)

    assert_almost_equal(val, 42.0, 1e-6, "item() should extract scalar value")


fn test_item_requires_single_element() raises:
    """Test that item() requires single-element tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    # Should raise error for multi-element tensor
    var raised = False
    try:
        var val = item(t)
        _ = val
    except e:
        raised = True

    if not raised:
        raise Error("item() should raise error for multi-element tensor")


# ============================================================================
# Test tolist() - convert to nested list
# ============================================================================


fn test_tolist_1d() raises:
    """Test converting 1D tensor to list."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var lst = t.tolist()

    # Should return [0, 1, 2, 3, 4]
    assert_equal_int(len(lst), 5, "List should have 5 elements")
    for i in range(5):
        assert_almost_equal(
            lst[i], Float64(i), 1e-6, "List value should match tensor"
        )


fn test_tolist_nested() raises:
    """Test converting multi-dimensional tensor to nested list."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var lst = t.tolist()

    # tolist() returns flat list, not nested
    assert_equal_int(len(lst), 6, "List should have 6 elements")
    for i in range(6):
        assert_almost_equal(
            lst[i], Float64(i), 1e-6, "List value should match tensor"
        )


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run utility operation tests - Part 2: contiguity, item(), and tolist().
    """
    print("Running ExTensor utility operation tests (Part 2)...")

    # Contiguity
    print("  Testing contiguity...")
    test_is_contiguous_true()
    test_is_contiguous_after_transpose()
    test_contiguous_on_noncontiguous()
    test_contiguous_stride_correct_values()

    # item() extraction
    print("  Testing item()...")
    test_item_single_element()
    test_item_requires_single_element()

    # tolist() conversion
    print("  Testing tolist()...")
    test_tolist_1d()
    test_tolist_nested()

    print("All utility operation tests (Part 2) completed!")
