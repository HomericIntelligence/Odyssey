"""Tests for AnyTensor utility operations - Part 3: __len__, __setitem__, __bool__, and partial __hash__.

Tests length accessor, item assignment, boolean conversion via __bool__,
and hash consistency for large and small float values.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_utility.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange, clone, item, diff
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
# Test __len__
# ============================================================================


fn test_len_first_dim() raises:
    """Test __len__ returns size of first dimension."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(3)
    var t = ones(shape, DType.float32)

    var length = len(t)
    assert_equal_int(length, 5, "__len__ should return first dimension")


fn test_len_1d() raises:
    """Test __len__ on 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    var length = len(t)
    assert_equal_int(length, 10, "__len__ should return size for 1D")


# ============================================================================
# Test __setitem__
# ============================================================================


fn test_setitem_valid_index() raises:
    """Test setting value at valid flat index, verified with __getitem__."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)
    t[1] = 9.5
    assert_value_at(t, 1, 9.5, 1e-6, "__setitem__ should set value at index 1")
    # Other elements unchanged
    assert_value_at(t, 0, 0.0, 1e-6, "Element 0 should remain 0.0")
    assert_value_at(t, 2, 0.0, 1e-6, "Element 2 should remain 0.0")


fn test_setitem_integer_dtype() raises:
    """Test setting value on integer dtype tensor."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.int32)
    t[2] = 7.0
    assert_value_at(
        t, 2, 7.0, 1e-6, "__setitem__ should set value on int32 tensor"
    )
    assert_value_at(t, 0, 0.0, 1e-6, "Element 0 should remain 0")


fn test_setitem_out_of_bounds() raises:
    """Test that __setitem__ raises error for out-of-bounds index."""
    var shape = List[Int]()
    shape.append(3)
    var t = zeros(shape, DType.float32)

    var raised = False
    try:
        t[5] = 1.0
    except e:
        raised = True

    if not raised:
        raise Error("__setitem__ should raise error for out-of-bounds index")


# ============================================================================
# Test __bool__
# ============================================================================


fn test_bool_single_element() raises:
    """Test __bool__ on single-element tensor."""
    var shape = List[Int]()
    var t_zero = full(shape, 0.0, DType.float32)
    var t_nonzero = full(shape, 5.0, DType.float32)

    if t_zero:
        raise Error("Zero tensor should be falsy")
    if not t_nonzero:
        raise Error("Non-zero tensor should be truthy")


fn test_bool_requires_single_element() raises:
    """Test that __bool__() raises for multi-element tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    var error_raised = False
    try:
        var val = t.__bool__()  # Should raise error for multi-element tensor
        _ = val  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions single-element requirement
        if (
            "single" not in error_msg.lower()
            and "element" not in error_msg.lower()
        ):
            raise Error(
                "Error message should mention single-element requirement"
            )

    if not error_raised:
        raise Error("__bool__() on multi-element tensor should raise error")


# ============================================================================
# Test __hash__ (large/small value edge cases)
# ============================================================================


fn test_hash_large_values() raises:
    """Test that large float values hash consistently without Int overflow."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1e15, DType.float64)
    var b = full(shape, 1e15, DType.float64)

    var hash_a = hash(a)
    var hash_b = hash(b)
    assert_equal_int(
        Int(hash_a), Int(hash_b), "Large values must hash consistently"
    )


fn test_hash_small_values_distinguish() raises:
    """Test that small but distinct float values produce different hashes."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1e-7, DType.float64)
    var b = full(shape, 2e-7, DType.float64)

    var hash_a = hash(a)
    var hash_b = hash(b)
    if hash_a == hash_b:
        raise Error(
            "Distinct small values should have different hashes with bitcast"
        )


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run utility operation tests - Part 3: __len__, __setitem__, __bool__, and partial __hash__.
    """
    print("Running AnyTensor utility operation tests (Part 3)...")

    # __len__
    print("  Testing __len__...")
    test_len_first_dim()
    test_len_1d()

    # __setitem__
    print("  Testing __setitem__...")
    test_setitem_valid_index()
    test_setitem_integer_dtype()
    test_setitem_out_of_bounds()

    # __bool__
    print("  Testing __bool__...")
    test_bool_single_element()
    test_bool_requires_single_element()

    # __hash__ (edge cases)
    print("  Testing __hash__ edge cases...")
    test_hash_large_values()
    test_hash_small_values_distinguish()

    print("All utility operation tests (Part 3) completed!")
