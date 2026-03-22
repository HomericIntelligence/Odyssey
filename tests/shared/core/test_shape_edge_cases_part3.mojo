# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for concatenate edge cases.

Tests edge cases for concatenate operations including:
- Concatenate single tensor
- Concatenate along different axes
- Concatenate with empty tensors
- Concatenate 1D tensors
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full, arange
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
# Test concatenate edge cases
# ============================================================================


fn test_concatenate_single_tensor() raises:
    """Concatenate single tensor returns copy."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t)
    var result = concatenate(tensors, 0)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 12, "Should have 12 elements")


fn test_concatenate_along_axis_0() raises:
    """Concatenate [2,3] + [2,3] along axis 0 -> [4,3]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var t1 = ones(shape, DType.float32)
    var t2 = full(shape, 2.0, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = concatenate(tensors, 0)

    assert_dim(result, 2, "Result should be 2D")
    # Result should be [4, 3]


fn test_concatenate_along_axis_1() raises:
    """Concatenate [2,3] + [2,4] along axis 1 -> [2,7]."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(4)

    var t1 = ones(shape_a, DType.float32)
    var t2 = full(shape_b, 2.0, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = concatenate(tensors, 1)

    assert_dim(result, 2, "Result should be 2D")
    # Result should be [2, 7]


fn test_concatenate_with_empty() raises:
    """Concatenate [3, 0] + [3, 4] along axis 1 -> [3, 4]."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(0)

    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(4)

    var t1 = zeros(shape_a, DType.float32)
    var t2 = ones(shape_b, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = concatenate(tensors, 1)

    assert_dim(result, 2, "Result should be 2D")


fn test_concatenate_1d_tensors() raises:
    """Concatenate 1D tensors."""
    var shape = List[Int]()
    shape.append(3)

    var t1 = full(shape, 1.0, DType.float32)
    var t2 = full(shape, 2.0, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = concatenate(tensors, 0)

    assert_dim(result, 1, "Result should be 1D")
    assert_numel(result, 6, "Should have 6 elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run concatenate edge case tests."""
    print("Running concatenate edge case tests...")

    test_concatenate_single_tensor()
    test_concatenate_along_axis_0()
    test_concatenate_along_axis_1()
    test_concatenate_with_empty()
    test_concatenate_1d_tensors()

    print("All concatenate edge case tests completed!")
