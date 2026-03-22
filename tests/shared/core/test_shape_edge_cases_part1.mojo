# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for reshape edge cases.

Tests edge cases for reshape operations including:
- Reshape to/from scalars
- Reshape with empty tensors
- Reshape between dimensions
"""

# Import AnyTensor and operations
from shared.core.extensor import AnyTensor, zeros, ones, full, arange
from shared.core.shape import reshape, squeeze, unsqueeze, concatenate, stack

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_shape,
)


# ============================================================================
# Test reshape edge cases
# ============================================================================


fn test_reshape_to_scalar() raises:
    """Reshape [1] to [] (scalar)."""
    var shape_from = List[Int]()
    shape_from.append(1)
    var t = full(shape_from, 42.0, DType.float32)

    var shape_to = List[Int]()
    var result = reshape(t, shape_to)

    assert_dim(result, 0, "Result should be 0D scalar")
    assert_numel(result, 1, "Scalar should have 1 element")
    assert_value_at(result, 0, 42.0, 1e-6, "Value should be preserved")


fn test_reshape_from_scalar() raises:
    """Reshape [] to [1, 1, 1]."""
    var shape_from = List[Int]()
    var t = full(shape_from, 42.0, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(1)
    shape_to.append(1)
    shape_to.append(1)
    var result = reshape(t, shape_to)

    assert_dim(result, 3, "Result should be 3D")
    assert_numel(result, 1, "Should have 1 element")
    assert_value_at(result, 0, 42.0, 1e-6, "Value should be preserved")


fn test_reshape_1d_to_2d() raises:
    """Reshape [6] to [2, 3]."""
    var shape_from = List[Int]()
    shape_from.append(6)
    var t = arange(0, 6, 1, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(2)
    shape_to.append(3)
    var result = reshape(t, shape_to)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 6, "Should have 6 elements")


fn test_reshape_2d_to_1d() raises:
    """Reshape [2, 3] to [6]."""
    var shape_from = List[Int]()
    shape_from.append(2)
    shape_from.append(3)
    var t = ones(shape_from, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(6)
    var result = reshape(t, shape_to)

    assert_dim(result, 1, "Result should be 1D")
    assert_numel(result, 6, "Should have 6 elements")


fn test_reshape_preserve_size() raises:
    """Reshape [2, 3, 4] to [8, 3]."""
    var shape_from = List[Int]()
    shape_from.append(2)
    shape_from.append(3)
    shape_from.append(4)
    var t = ones(shape_from, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(8)
    shape_to.append(3)
    var result = reshape(t, shape_to)

    assert_numel(result, 24, "Should preserve element count (2*3*4=24)")


fn test_reshape_empty_tensor() raises:
    """Reshape [0] to [0, 3] (empty tensor)."""
    var shape_from = List[Int]()
    shape_from.append(0)
    var t = zeros(shape_from, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(0)
    shape_to.append(3)
    var result = reshape(t, shape_to)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 0, "Should have 0 elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run reshape edge case tests."""
    print("Running reshape edge case tests...")

    test_reshape_to_scalar()
    test_reshape_from_scalar()
    test_reshape_1d_to_2d()
    test_reshape_2d_to_1d()
    test_reshape_preserve_size()
    test_reshape_empty_tensor()

    print("All reshape edge case tests completed!")
