"""Tests for shape preservation and error handling in arithmetic operations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Shape preservation: 1D and 3D tensors
- Error handling: mismatched shapes, mismatched dtypes
"""

from tests.shared.conftest import (
    assert_dim,
    assert_numel,
)
from shared.core.extensor import ExTensor, ones
from shared.core.arithmetic import (
    add,
    multiply,
)


fn test_add_preserves_shape_1d() raises:
    """Test that add preserves 1D shape."""
    var shape = List[Int]()
    shape.append(10)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    assert_dim(c, 1, "Result should be 1D")
    assert_numel(c, 10, "Result should have 10 elements")


fn test_add_preserves_shape_3d() raises:
    """Test that add preserves 3D shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    assert_dim(c, 3, "Result should be 3D")
    assert_numel(c, 24, "Result should have 24 elements")


fn test_add_mismatched_shapes_raises_error() raises:
    """Test that add with mismatched shapes raises error."""
    var shape_a = List[Int]()
    shape_a.append(5)
    var shape_b = List[Int]()
    shape_b.append(10)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        var c = add(a, b)
        _ = c  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions shape/broadcast incompatibility
        if (
            "broadcast" not in error_msg.lower()
            and "shape" not in error_msg.lower()
        ):
            raise Error(
                "Error message should mention shape or broadcast compatibility"
            )

    if not error_raised:
        raise Error(
            "add with mismatched non-broadcastable shapes should raise error"
        )


fn test_multiply_mismatched_shapes_raises_error() raises:
    """Test that multiply with mismatched shapes raises error."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(5)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        var c = multiply(a, b)
        _ = c  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions shape/broadcast incompatibility
        if (
            "broadcast" not in error_msg.lower()
            and "shape" not in error_msg.lower()
        ):
            raise Error(
                "Error message should mention shape or broadcast compatibility"
            )

    if not error_raised:
        raise Error(
            "multiply with mismatched non-broadcastable shapes should raise"
            " error"
        )


fn test_add_mismatched_dtypes_raises_error() raises:
    """Test that add with mismatched dtypes raises error."""
    var shape = List[Int]()
    shape.append(5)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)

    var error_raised = False
    try:
        var c = add(a, b)
        _ = c  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions dtype mismatch
        if "dtype" not in error_msg.lower():
            raise Error("Error message should mention dtype mismatch")

    if not error_raised:
        raise Error("add with mismatched dtypes should raise error")


fn main() raises:
    """Run shape preservation and error handling tests."""
    print("Running shape preservation and error handling tests (part 8)...")

    test_add_preserves_shape_1d()
    print("    ✓ test_add_preserves_shape_1d")
    test_add_preserves_shape_3d()
    print("    ✓ test_add_preserves_shape_3d")
    test_add_mismatched_shapes_raises_error()
    print("    ✓ test_add_mismatched_shapes_raises_error")
    test_multiply_mismatched_shapes_raises_error()
    print("    ✓ test_multiply_mismatched_shapes_raises_error")
    test_add_mismatched_dtypes_raises_error()
    print("    ✓ test_add_mismatched_dtypes_raises_error")

    print("\nAll arithmetic part 8 tests passed! (5 tests)")
