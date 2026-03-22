"""Tests for reduction operation edge cases (Part 2).

Tests edge cases for sum, mean, max_reduce, min_reduce operations including:
- 2D tensor reductions
- 3D tensor reductions
- 4D+ (high-dimensional) tensor reductions

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.extensor import AnyTensor, zeros, ones, full, arange
from shared.core.reduction import sum, mean, max_reduce, min_reduce

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_true,
)


# ============================================================================
# Test 2D reductions
# ============================================================================


fn test_sum_2d() raises:
    """Sum of 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = full(shape, 2.0, DType.float32)
    var result = sum(t)

    # Sum of all 2.0 in 3x4 = 24.0
    assert_value_at(result, 0, 24.0, 1e-5, "Sum of 3x4 tensor of 2.0")


fn test_mean_2d() raises:
    """Mean of 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = full(shape, 8.0, DType.float32)
    var result = mean(t)

    # Mean of all 8.0 = 8.0
    assert_value_at(result, 0, 8.0, 1e-5, "Mean of constant 2D tensor")


# ============================================================================
# Test 3D reductions
# ============================================================================


fn test_sum_3d_tensor() raises:
    """Sum reduction on 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    # Sum of all ones in 2x3x4 = 24
    assert_value_at(result, 0, 24.0, 1e-5, "Sum of 3D ones")


fn test_mean_3d_tensor() raises:
    """Mean reduction on 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = full(shape, 5.0, DType.float32)
    var result = mean(t)

    # Mean of all 5.0 = 5.0
    assert_value_at(result, 0, 5.0, 1e-5, "Mean of 3D constant tensor")


# ============================================================================
# Test 4D+ reductions
# ============================================================================


fn test_sum_4d_tensor() raises:
    """Sum reduction on 4D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    # Sum of all ones = 2^4 = 16
    assert_value_at(result, 0, 16.0, 1e-5, "Sum of 4D ones")


fn test_sum_5d_tensor() raises:
    """Sum reduction on 5D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    # Sum of all ones = 2^5 = 32
    assert_value_at(result, 0, 32.0, 1e-5, "Sum of 5D ones")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run reduction edge case tests (Part 2)."""
    print("Running reduction edge case tests (Part 2)...")

    # 2D reductions
    print("  Testing 2D reductions...")
    test_sum_2d()
    test_mean_2d()

    # 3D reductions
    print("  Testing 3D reductions...")
    test_sum_3d_tensor()
    test_mean_3d_tensor()

    # 4D+ reductions
    print("  Testing 4D+ reductions...")
    test_sum_4d_tensor()
    test_sum_5d_tensor()

    print("All reduction edge case tests (Part 2) completed!")
