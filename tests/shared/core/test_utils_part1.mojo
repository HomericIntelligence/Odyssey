# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for argmax utility functions in shared.core.utils (Part 1 of 3).

This module tests argmax functions including:
- argmax (scalar)
- argmax (axis-based)
- top_k_indices_simple
"""

from tests.shared.conftest import (
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
    assert_close_float,
)
from shared.core.extensor import ExTensor, zeros, ones, arange
from shared.core.utils import (
    argmax,
    top_k_indices,
)


# ============================================================================
# Argmax Tests (Scalar)
# ============================================================================


fn test_argmax_scalar_simple() raises:
    """Test argmax on a simple 1D tensor."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var idx = argmax(t)
    assert_equal_int(idx, 9)


fn test_argmax_scalar_negative_values() raises:
    """Test argmax with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = -5.0
    t._data.bitcast[Float32]()[1] = -2.0
    t._data.bitcast[Float32]()[2] = -10.0
    t._data.bitcast[Float32]()[3] = -1.0
    t._data.bitcast[Float32]()[4] = -3.0
    var idx = argmax(t)
    assert_equal_int(idx, 3)


fn test_argmax_scalar_multi_dimensional() raises:
    """Test argmax on multi-dimensional tensor flattens correctly."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max at linear index 5 (row 1, col 1)
    t._data.bitcast[Float32]()[5] = 42.0
    var idx = argmax(t)
    assert_equal_int(idx, 5)


# ============================================================================
# Argmax Tests (Axis)
# ============================================================================


fn test_argmax_axis_1d() raises:
    """Test argmax along axis on 1D tensor."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var result = argmax(t, axis=0)
    assert_equal_int(result.numel(), 1)
    assert_equal_int(Int(result._get_int64(0)), 4)


fn test_argmax_axis_2d_axis0() raises:
    """Test argmax along axis 0 on 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set values such that row 2 has the max in each column
    # Column 0: indices 0, 4, 8 -> max at 8
    t._data.bitcast[Float32]()[8] = 100.0
    t._data.bitcast[Float32]()[9] = 100.0
    t._data.bitcast[Float32]()[10] = 100.0
    t._data.bitcast[Float32]()[11] = 100.0

    var result = argmax(t, axis=0)
    assert_shape(result, [4])

    # All should be 2 (row index 2, which is linear indices 8-11)
    for i in range(4):
        assert_equal_int(Int(result._get_int64(i)), 2)


fn test_argmax_axis_2d_axis1() raises:
    """Test argmax along axis 1 on 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max in last column for each row
    # Row 0, col 3: linear index 3
    # Row 1, col 3: linear index 7
    # Row 2, col 3: linear index 11
    t._data.bitcast[Float32]()[3] = 10.0
    t._data.bitcast[Float32]()[7] = 20.0
    t._data.bitcast[Float32]()[11] = 30.0

    var result = argmax(t, axis=1)
    assert_shape(result, [3])
    assert_equal_int(Int(result._get_int64(0)), 3)
    assert_equal_int(Int(result._get_int64(1)), 3)
    assert_equal_int(Int(result._get_int64(2)), 3)


fn test_argmax_axis_3d() raises:
    """Test argmax on 3D tensor along axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max values
    t._data.bitcast[Float32]()[5] = 50.0

    var result = argmax(t, axis=2)
    assert_shape(result, [2, 3])


# ============================================================================
# Top K Tests (partial)
# ============================================================================


fn test_top_k_indices_simple() raises:
    """Test top_k_indices on a simple tensor."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 3)
    assert_equal_int(indices[0], 9)
    assert_equal_int(indices[1], 8)
    assert_equal_int(indices[2], 7)


fn main() raises:
    """Run all tests."""
    print("=" * 60)
    print("Running shared.core.utils tests (Part 1)")
    print("=" * 60)

    # Argmax scalar tests
    print("\n=== Argmax (Scalar) ===")
    test_argmax_scalar_simple()
    print("✓ test_argmax_scalar_simple")
    test_argmax_scalar_negative_values()
    print("✓ test_argmax_scalar_negative_values")
    test_argmax_scalar_multi_dimensional()
    print("✓ test_argmax_scalar_multi_dimensional")

    # Argmax axis tests
    print("\n=== Argmax (Axis) ===")
    test_argmax_axis_1d()
    print("✓ test_argmax_axis_1d")
    test_argmax_axis_2d_axis0()
    print("✓ test_argmax_axis_2d_axis0")
    test_argmax_axis_2d_axis1()
    print("✓ test_argmax_axis_2d_axis1")
    test_argmax_axis_3d()
    print("✓ test_argmax_axis_3d")

    # Top K tests (partial)
    print("\n=== Top K (partial) ===")
    test_top_k_indices_simple()
    print("✓ test_top_k_indices_simple")

    print("\n" + "=" * 60)
    print("All 8 utils tests (Part 1) passed!")
    print("=" * 60)
