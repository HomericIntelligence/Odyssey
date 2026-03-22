"""Tests for AnyTensor arange/eye patterns, is_view, and dtype size (Part 5 of 5).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_properties.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full, arange, eye

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_equal_int,
    assert_value_at,
)


# ============================================================================
# Test special tensor creation patterns (continued)
# ============================================================================


fn test_arange_sequential_pattern() raises:
    """Test that arange creates sequential values."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    for i in range(10):
        assert_value_at(t, i, Float64(i), 1e-6, "Sequential values")


fn test_eye_identity_pattern() raises:
    """Test that eye creates proper identity pattern."""
    var t = eye(4, 4, 0, DType.float32)

    for i in range(4):
        for j in range(4):
            var idx = i * 4 + j
            if i == j:
                assert_value_at(t, idx, 1.0, 1e-6, "Diagonal should be 1")
            else:
                assert_value_at(t, idx, 0.0, 1e-6, "Off-diagonal should be 0")


# ============================================================================
# Test is_view property
# ============================================================================


fn test_is_view_false_for_new_tensors() raises:
    """Test that newly created tensors are not views."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_false(t._is_view, "Newly created tensor should not be a view")


# ============================================================================
# Test dtype size calculations
# ============================================================================


fn test_dtype_size_float32() raises:
    """Test dtype size for float32."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float32)

    var size = t._get_dtype_size()
    assert_equal_int(size, 4, "float32 should be 4 bytes")


fn test_dtype_size_float64() raises:
    """Test dtype size for float64."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float64)

    var size = t._get_dtype_size()
    assert_equal_int(size, 8, "float64 should be 8 bytes")


fn test_dtype_size_int32() raises:
    """Test dtype size for int32."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.int32)

    var size = t._get_dtype_size()
    assert_equal_int(size, 4, "int32 should be 4 bytes")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run arange/eye patterns, is_view, and dtype size tests (Part 5)."""
    print(
        "Running AnyTensor arange/eye patterns, is_view, and dtype size tests"
        " (Part 5)..."
    )

    # Pattern tests
    print("  Testing creation patterns...")
    test_arange_sequential_pattern()
    test_eye_identity_pattern()

    # View property tests
    print("  Testing is_view property...")
    test_is_view_false_for_new_tensors()

    # DType size tests
    print("  Testing dtype size calculations...")
    test_dtype_size_float32()
    test_dtype_size_float64()
    test_dtype_size_int32()

    print("All Part 5 tests completed!")
