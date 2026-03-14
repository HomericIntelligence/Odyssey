# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor shape manipulation: concatenate, stack, split, tile.

Split from test_shape.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    concatenate,
    stack,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test concatenate()
# ============================================================================


fn test_concatenate_axis_0() raises:
    """Test concatenating along axis 0."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(3)

    var a = ones(shape_a, DType.float32)  # 2x3
    var b = full(shape_b, 2.0, DType.float32)  # 3x3

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)
    var c = concatenate(tensors, axis=0)

    # Result should be 5x3 (2+3 rows, 3 cols)
    assert_dim(c, 2, "Concatenated tensor should be 2D")
    assert_numel(c, 15, "Should have 15 elements (5*3)")


fn test_concatenate_axis_1() raises:
    """Test concatenating along axis 1."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(2)
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)  # 3x2
    var b = full(shape_b, 2.0, DType.float32)  # 3x4

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)
    var c = concatenate(tensors, axis=1)

    # Result should be 3x6 (3 rows, 2+4 cols)
    assert_numel(c, 18, "Should have 18 elements (3*6)")


# ============================================================================
# Test stack()
# ============================================================================


fn test_stack_new_axis() raises:
    """Test stacking tensors along new axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var a = ones(shape, DType.float32)  # 2x3
    var b = full(shape, 2.0, DType.float32)  # 2x3

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)
    var c = stack(tensors, axis=0)

    # Result should be 2x2x3 (stacked along new axis 0)
    assert_dim(c, 3, "Stacked tensor should be 3D")
    assert_numel(c, 12, "Should have 12 elements (2*2*3)")


fn test_stack_axis_1() raises:
    """Test stacking along axis 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)
    var c = stack(tensors, axis=1)

    # Result should be 2x2x3 (stacked along axis 1)
    assert_dim(c, 3, "Should be 3D")


# ============================================================================
# Test split()
# ============================================================================


fn test_split_equal() raises:
    """Test splitting into equal parts."""
    from shared.core import split

    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var parts = split(a, 3)

    # Should give 3 tensors of size 4 each
    if len(parts) != 3:
        raise Error("Should split into 3 parts")
    for i in range(3):
        assert_numel(parts[i], 4, "Each part should have 4 elements")
    # Verify actual values: [0,1,2,3], [4,5,6,7], [8,9,10,11]
    assert_value_at(parts[0], 0, 0.0, "Part 0, index 0 should be 0.0")
    assert_value_at(parts[0], 3, 3.0, "Part 0, index 3 should be 3.0")
    assert_value_at(parts[1], 0, 4.0, "Part 1, index 0 should be 4.0")
    assert_value_at(parts[1], 3, 7.0, "Part 1, index 3 should be 7.0")
    assert_value_at(parts[2], 0, 8.0, "Part 2, index 0 should be 8.0")
    assert_value_at(parts[2], 3, 11.0, "Part 2, index 3 should be 11.0")


fn test_split_unequal() raises:
    """Test splitting into unequal parts."""
    from shared.core import split_with_indices

    var a = arange(0.0, 10.0, 1.0, DType.float32)
    var indices = List[Int]()
    indices.append(3)
    indices.append(7)
    var parts = split_with_indices(a, indices)

    # Should give 3 tensors of sizes 3, 4, 3
    if len(parts) != 3:
        raise Error("Should split into 3 parts")
    assert_numel(parts[0], 3, "First part should have 3 elements")
    assert_numel(parts[1], 4, "Second part should have 4 elements")
    assert_numel(parts[2], 3, "Third part should have 3 elements")
    # Verify actual values: [0,1,2], [3,4,5,6], [7,8,9]
    assert_value_at(parts[0], 0, 0.0, "Part 0, index 0 should be 0.0")
    assert_value_at(parts[0], 2, 2.0, "Part 0, index 2 should be 2.0")
    assert_value_at(parts[1], 0, 3.0, "Part 1, index 0 should be 3.0")
    assert_value_at(parts[1], 3, 6.0, "Part 1, index 3 should be 6.0")
    assert_value_at(parts[2], 0, 7.0, "Part 2, index 0 should be 7.0")
    assert_value_at(parts[2], 2, 9.0, "Part 2, index 2 should be 9.0")


# ============================================================================
# Test tile()
# ============================================================================


fn test_tile_1d() raises:
    """Test tiling 1D tensor."""
    from shared.core import tile

    var a = arange(0.0, 3.0, 1.0, DType.float32)  # [0, 1, 2]
    var reps = List[Int]()
    reps.append(3)
    var b = tile(a, reps)

    # Result: [0, 1, 2, 0, 1, 2, 0, 1, 2] (9 elements)
    assert_numel(b, 9, "Tiled tensor should have 9 elements")


fn test_tile_multidim() raises:
    """Test tiling with multi-dimensional repetitions."""
    from shared.core import tile

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)  # 2x3
    var reps = List[Int]()
    reps.append(2)
    reps.append(3)
    var b = tile(a, reps)

    # Result should be 4x9 (2*2 rows, 3*3 cols)
    assert_numel(b, 36, "Should have 36 elements (4*9)")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run shape manipulation tests part 2 (concatenate, stack, split, tile)."""
    print("Running ExTensor shape manipulation tests (part 2)...")

    # concatenate() tests
    print("  Testing concatenate()...")
    test_concatenate_axis_0()
    test_concatenate_axis_1()

    # stack() tests
    print("  Testing stack()...")
    test_stack_new_axis()
    test_stack_axis_1()

    # split() tests
    print("  Testing split()...")
    test_split_equal()
    test_split_unequal()

    # tile() tests
    print("  Testing tile()...")
    test_tile_1d()
    test_tile_multidim()

    print("All shape manipulation tests (part 2) completed!")
