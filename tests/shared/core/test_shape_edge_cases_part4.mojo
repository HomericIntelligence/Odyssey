# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for stack and dimension preservation edge cases.

Tests edge cases for stack operations and dimension preservation including:
- Stack single tensor
- Stack 2D tensors
- Stack along different axes
- Reshape/squeeze/unsqueeze preserve element count
"""

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full, arange
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
# Test stack edge cases
# ============================================================================


fn test_stack_single_tensor() raises:
    """Stack single tensor adds dimension."""
    var shape = List[Int]()
    shape.append(3)
    var t = ones(shape, DType.float32)

    var tensors = List[ExTensor]()
    tensors.append(t)
    var result = stack(tensors, 0)

    # Stack should add dimension: [3] -> [1, 3]
    assert_dim(result, 2, "Result should be 2D after stacking")


fn test_stack_2d_tensors() raises:
    """Stack 2D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var t1 = ones(shape, DType.float32)
    var t2 = full(shape, 2.0, DType.float32)

    var tensors = List[ExTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = stack(tensors, 0)

    # [2,3] stacked at axis 0 -> [2, 2, 3]
    assert_dim(result, 3, "Result should be 3D")


fn test_stack_along_different_axis() raises:
    """Stack along different axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var t1 = ones(shape, DType.float32)
    var t2 = full(shape, 2.0, DType.float32)

    var tensors = List[ExTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = stack(tensors, 1)

    # [2,3] stacked at axis 1 -> [2, 2, 3]
    assert_dim(result, 3, "Result should be 3D")


# ============================================================================
# Test dimension preservation
# ============================================================================


fn test_reshape_preserves_numel() raises:
    """Reshape must preserve total number of elements."""
    var shape_from = List[Int]()
    shape_from.append(3)
    shape_from.append(4)
    shape_from.append(5)
    var t = ones(shape_from, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(6)
    shape_to.append(10)
    var result = reshape(t, shape_to)

    assert_numel(result, 60, "Should preserve 3*4*5=60 elements")


fn test_squeeze_preserves_numel() raises:
    """Squeeze must preserve total number of elements."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(5)
    shape.append(1)
    var t = ones(shape, DType.float32)

    var result = squeeze(t)

    assert_numel(result, 5, "Should preserve 5 elements")


fn test_unsqueeze_preserves_numel() raises:
    """Unsqueeze must preserve total number of elements."""
    var shape = List[Int]()
    shape.append(3)
    var t = ones(shape, DType.float32)

    var result = unsqueeze(t, 0)

    assert_numel(result, 3, "Should preserve 3 elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run stack and dimension preservation edge case tests."""
    print("Running stack and dimension preservation edge case tests...")

    # Stack edge cases
    print("  Testing stack edge cases...")
    test_stack_single_tensor()
    test_stack_2d_tensors()
    test_stack_along_different_axis()

    # Dimension preservation
    print("  Testing dimension preservation...")
    test_reshape_preserves_numel()
    test_squeeze_preserves_numel()
    test_unsqueeze_preserves_numel()

    print("All stack and dimension preservation edge case tests completed!")
