# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor shape manipulation: flatten_to_2d additional cases.

Split from test_shape.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and operations
from shared.core.extensor import ExTensor, ones, zeros, arange
from shared.core.shape import flatten_to_2d, concatenate, as_contiguous

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test flatten_to_2d() - additional cases
# ============================================================================


fn test_flatten_to_2d_single_batch() raises:
    """Test flatten_to_2d with batch size 1."""
    var shape: List[Int] = [1, 64, 7, 7]
    var a = ones(shape, DType.float32)

    var b = flatten_to_2d(a)

    var out_shape = b.shape()
    if out_shape[0] != 1:
        raise Error("Batch dimension should be 1, got " + String(out_shape[0]))
    if out_shape[1] != 3136:
        raise Error(
            "Flattened dimension should be 3136 (64*7*7), got "
            + String(out_shape[1])
        )


fn test_flatten_to_2d_preserves_dtype() raises:
    """Test that flatten_to_2d preserves dtype."""
    var shape: List[Int] = [2, 3, 4, 4]
    var a = ones(shape, DType.float64)

    var b = flatten_to_2d(a)

    if b.dtype() != DType.float64:
        raise Error("flatten_to_2d should preserve dtype")


# ============================================================================
# Main test runner
# ============================================================================


fn test_concatenate_axis_1_per_row_values() raises:
    """Test concatenate axis=1 has correct per-row values. Closes #3798."""
    # Create 3x2 tensors with known values
    var shape: List[Int] = [3, 2]
    var a = arange(0.0, 6.0, 1.0, DType.float32)
    a = a.reshape(shape)  # [[0,1],[2,3],[4,5]]
    var b = arange(6.0, 12.0, 1.0, DType.float32)
    b = b.reshape(shape)  # [[6,7],[8,9],[10,11]]

    var result = concatenate(List[ExTensor](a, b), 1)  # 3x4

    # Row 0: [0,1,6,7]
    assert_value_at(result, 0, 0.0)
    assert_value_at(result, 1, 1.0)
    assert_value_at(result, 2, 6.0)
    assert_value_at(result, 3, 7.0)

    # Row 1: [2,3,8,9]
    assert_value_at(result, 4, 2.0)
    assert_value_at(result, 5, 3.0)
    assert_value_at(result, 6, 8.0)
    assert_value_at(result, 7, 9.0)

    # Row 2: [4,5,10,11]
    assert_value_at(result, 8, 4.0)
    assert_value_at(result, 9, 5.0)
    assert_value_at(result, 10, 10.0)
    assert_value_at(result, 11, 11.0)

    print("PASS: test_concatenate_axis_1_per_row_values")


fn test_as_contiguous_3d_non_contiguous() raises:
    """Test as_contiguous produces correct values for 3D tensor. Closes #4090."""
    # Create [2,3,4] tensor with sequential values
    var t = arange(0.0, 24.0, 1.0, DType.float32)
    var shape: List[Int] = [2, 3, 4]
    t = t.reshape(shape)

    # as_contiguous on already-contiguous should preserve values
    var c = as_contiguous(t)
    assert_numel(c, 24)

    # Check a few key values: element [1,2,3] = 1*12 + 2*4 + 3 = 23
    assert_value_at(c, 23, 23.0)
    # element [0,0,0] = 0
    assert_value_at(c, 0, 0.0)
    # element [1,0,0] = 12
    assert_value_at(c, 12, 12.0)

    print("PASS: test_as_contiguous_3d_non_contiguous")


fn main() raises:
    """Run shape manipulation tests part 4."""
    print("Running ExTensor shape manipulation tests (part 4)...")

    print("  Testing flatten_to_2d() additional cases...")
    test_flatten_to_2d_single_batch()
    test_flatten_to_2d_preserves_dtype()

    print("  Testing concatenate and as_contiguous...")
    test_concatenate_axis_1_per_row_values()
    test_as_contiguous_3d_non_contiguous()

    print("All shape manipulation tests (part 4) completed!")
