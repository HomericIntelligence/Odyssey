"""Tests for ExTensor broadcasting operations - Part 2: Size-1 dims, missing dims, complex multi-dim.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_broadcasting.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests NumPy-style broadcasting rules for trailing size-1 dimensions, missing dimensions, and complex cases.
"""

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.arithmetic import add, multiply
from testing import assert_true

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test dimension size 1 broadcasting (continued)
# ============================================================================


fn test_broadcast_size_one_dim_trailing() raises:
    """Test broadcasting with trailing dimension of size 1."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)
    shape_a.append(1)
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(3)
    shape_b.append(4)

    var a = full(shape_a, 3.0, DType.float32)  # 2x3x1
    var b = full(shape_b, 2.0, DType.float32)  # 2x3x4
    var c = add(a, b)  # Expected: 2x3x4, all 5s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "Broadcasting 2x3x1 to 2x3x4")


# ============================================================================
# Test missing dimensions broadcasting
# ============================================================================


fn test_broadcast_missing_leading_dims() raises:
    """Test broadcasting when tensor has fewer dimensions (aligned to right)."""
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    var shape_1d = List[Int]()
    shape_1d.append(4)

    var a = ones(shape_3d, DType.float32)  # 2x3x4
    var b = full(
        shape_1d, 2.0, DType.float32
    )  # (4,) -> broadcasts to (1,1,4) -> (2,3,4)
    var c = multiply(a, b)  # Expected: 2x3x4, all 2s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 2.0, 1e-6, "Broadcasting (4,) to (2,3,4)")


fn test_broadcast_2d_to_3d() raises:
    """Test broadcasting 2D to 3D."""
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)

    var a = ones(shape_3d, DType.float32)  # 2x3x4
    var b = full(
        shape_2d, 3.0, DType.float32
    )  # 3x4 -> broadcasts to (1,3,4) -> (2,3,4)
    var c = add(a, b)  # Expected: 2x3x4, all 4s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 4.0, 1e-6, "Broadcasting (3,4) to (2,3,4)")


# ============================================================================
# Test complex multi-dimensional broadcasting
# ============================================================================


fn test_broadcast_3d_complex() raises:
    """Test complex 3D broadcasting with multiple size-1 dimensions."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(1)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(1)
    shape_b.append(3)
    shape_b.append(4)

    var a = full(shape_a, 2.0, DType.float32)  # 2x1x4
    var b = full(shape_b, 3.0, DType.float32)  # 1x3x4
    var c = add(a, b)  # Expected: 2x3x4, all 5s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "Broadcasting (2,1,4) + (1,3,4) to (2,3,4)")


fn test_broadcast_4d() raises:
    """Test 4D broadcasting."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(1)
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(1)
    shape_b.append(5)
    shape_b.append(3)
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)  # 2x1x3x4
    var b = full(shape_b, 2.0, DType.float32)  # 1x5x3x4
    var c = multiply(a, b)  # Expected: 2x5x3x4, all 2s

    assert_numel(c, 120, "Result should have 120 elements (2*5*3*4)")
    assert_all_values(
        c, 2.0, 1e-6, "Broadcasting (2,1,3,4) * (1,5,3,4) to (2,5,3,4)"
    )


# ============================================================================
# Test incompatible shapes (should error)
# ============================================================================


fn test_broadcast_incompatible_shapes_different_sizes() raises:
    """Test that incompatible shapes raise error."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(5)  # Incompatible: 4 != 5 and neither is 1

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    # Verify this raises an error
    var error_raised = False
    try:
        var c = add(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error(
            "Should have raised error for incompatible broadcast shapes (3,4)"
            " and (3,5)"
        )


fn test_broadcast_incompatible_inner_dims() raises:
    """Test that incompatible inner dimensions raise error."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(5)  # Incompatible: 3 != 5 and neither is 1)
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    # Verify this raises an error
    var error_raised = False
    try:
        var c = add(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error(
            "Should have raised error for incompatible broadcast shapes (2,3,4)"
            " and (2,5,4)"
        )


# ============================================================================
# Test broadcast output shape
# ============================================================================


fn test_broadcast_output_shape_scalar_1d() raises:
    """Test broadcast output shape for scalar + 1D."""
    var shape_vec = List[Int]()
    shape_vec.append(5)
    var shape_scalar = List[Int]()

    var a = ones(shape_vec, DType.float32)
    var b = ones(shape_scalar, DType.float32)
    var c = add(a, b)

    assert_dim(c, 1, "Output should be 1D")
    assert_numel(c, 5, "Output should have 5 elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run broadcasting part 2 tests."""
    print("Running ExTensor broadcasting tests - Part 2...")

    # Size-1 dimension broadcasting (trailing)
    print("  Testing size-1 dimension broadcasting (trailing)...")
    test_broadcast_size_one_dim_trailing()

    # Missing dimensions
    print("  Testing missing dimensions broadcasting...")
    test_broadcast_missing_leading_dims()
    test_broadcast_2d_to_3d()

    # Complex multi-dimensional
    print("  Testing complex multi-dimensional broadcasting...")
    test_broadcast_3d_complex()
    test_broadcast_4d()

    # Incompatible shapes
    print("  Testing incompatible shapes...")
    test_broadcast_incompatible_shapes_different_sizes()
    test_broadcast_incompatible_inner_dims()

    # Output shape verification
    print("  Testing broadcast output shapes...")
    test_broadcast_output_shape_scalar_1d()

    print("All broadcasting part 2 tests completed!")
