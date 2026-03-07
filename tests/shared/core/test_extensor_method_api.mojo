"""Tests for ExTensor method-style API: tile, repeat, permute, split.

Verifies that the thin wrapper methods on ExTensor produce identical results
to the functional implementations in shared.core.shape. Follows #3243.
"""

from shared.core import ExTensor, arange, ones, tile, repeat, permute, split
from tests.shared.conftest import assert_equal, assert_almost_equal, assert_true


# ============================================================================
# tile() method tests
# ============================================================================


fn test_tile_method_1d() raises:
    """tile() method on 1D tensor matches functional tile()."""
    var a = arange(0.0, 3.0, 1.0, DType.float32)
    var reps = List[Int]()
    reps.append(3)

    var expected = tile(a, reps)
    var actual = a.tile(reps)

    assert_equal(actual.numel(), expected.numel())
    for i in range(actual.numel()):
        assert_almost_equal(
            Float64(actual[i]), Float64(expected[i]), tolerance=1e-6
        )


fn test_tile_method_multidim() raises:
    """tile() method on 2D tensor matches functional tile()."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)

    var reps = List[Int]()
    reps.append(2)
    reps.append(3)

    var expected = tile(a, reps)
    var actual = a.tile(reps)

    assert_equal(actual.numel(), expected.numel())
    assert_equal(actual.numel(), 36)


fn test_tile_method_returns_copy() raises:
    """tile() method returns a new tensor, not a view."""
    var a = arange(0.0, 3.0, 1.0, DType.float32)
    var reps = List[Int]()
    reps.append(2)
    var b = a.tile(reps)

    assert_equal(b.numel(), 6)
    assert_equal(a.numel(), 3)


# ============================================================================
# repeat() method tests
# ============================================================================


fn test_repeat_method_flatten() raises:
    """repeat() method (no axis) matches functional repeat()."""
    var a = arange(0.0, 3.0, 1.0, DType.float32)

    var expected = repeat(a, 2)
    var actual = a.repeat(2)

    assert_equal(actual.numel(), expected.numel())
    for i in range(actual.numel()):
        assert_almost_equal(
            Float64(actual[i]), Float64(expected[i]), tolerance=1e-6
        )


fn test_repeat_method_axis() raises:
    """repeat() method with explicit axis matches functional repeat()."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)

    var expected = repeat(a, 2, axis=0)
    var actual = a.repeat(2, axis=0)

    assert_equal(actual.numel(), expected.numel())
    assert_equal(actual.numel(), 12)


fn test_repeat_method_values() raises:
    """repeat() method produces correct element values."""
    var a = arange(0.0, 3.0, 1.0, DType.float32)  # [0, 1, 2]
    var b = a.repeat(2)  # Expected: [0, 0, 1, 1, 2, 2]

    assert_equal(b.numel(), 6)
    assert_almost_equal(Float64(b[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(b[1]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(b[2]), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(b[3]), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(b[4]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(b[5]), 2.0, tolerance=1e-6)


# ============================================================================
# permute() method tests
# ============================================================================


fn test_permute_method_3d() raises:
    """permute() method matches functional permute()."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)

    var dims = List[Int]()
    dims.append(2)
    dims.append(0)
    dims.append(1)

    var expected = permute(a, dims)
    var actual = a.permute(dims)

    assert_equal(actual.numel(), expected.numel())
    assert_equal(actual.numel(), 24)


fn test_permute_method_shape() raises:
    """permute() method produces correct output shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)

    var dims = List[Int]()
    dims.append(2)
    dims.append(0)
    dims.append(1)
    var b = a.permute(dims)

    # Result should be (4, 2, 3)
    var b_shape = b.shape()
    assert_equal(b_shape[0], 4)
    assert_equal(b_shape[1], 2)
    assert_equal(b_shape[2], 3)


fn test_permute_method_2d_transpose() raises:
    """permute() method can perform 2D transpose."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(5)
    var a = ones(shape, DType.float32)

    var dims = List[Int]()
    dims.append(1)
    dims.append(0)
    var b = a.permute(dims)

    var b_shape = b.shape()
    assert_equal(b_shape[0], 5)
    assert_equal(b_shape[1], 3)


# ============================================================================
# split() method tests
# ============================================================================


fn test_split_method_equal_parts() raises:
    """split() method matches functional split()."""
    var a = arange(0.0, 12.0, 1.0, DType.float32)

    var expected = split(a, 3)
    var actual = a.split(3)

    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i].numel(), expected[i].numel())


fn test_split_method_sizes() raises:
    """split() method produces correct part sizes."""
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var parts = a.split(3)

    assert_equal(len(parts), 3)
    for i in range(3):
        assert_equal(parts[i].numel(), 4)


fn test_split_method_axis() raises:
    """split() method with axis=1 splits along correct dimension."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(6)
    var a = ones(shape, DType.float32)

    var parts = a.split(3, axis=1)

    assert_equal(len(parts), 3)
    for i in range(3):
        var part_shape = parts[i].shape()
        assert_equal(part_shape[0], 4)
        assert_equal(part_shape[1], 2)


fn test_split_method_values() raises:
    """split() method produces correct element values."""
    var a = arange(0.0, 6.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4, 5]
    var parts = a.split(2)  # [0, 1, 2] and [3, 4, 5]

    assert_almost_equal(Float64(parts[0][0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(parts[0][1]), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(parts[0][2]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(parts[1][0]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(parts[1][1]), 4.0, tolerance=1e-6)
    assert_almost_equal(Float64(parts[1][2]), 5.0, tolerance=1e-6)


# ============================================================================
# Main
# ============================================================================


fn main() raises:
    print("Testing ExTensor method API (tile, repeat, permute, split)...")

    print("  Testing tile() method...")
    test_tile_method_1d()
    test_tile_method_multidim()
    test_tile_method_returns_copy()
    print("  tile(): PASSED")

    print("  Testing repeat() method...")
    test_repeat_method_flatten()
    test_repeat_method_axis()
    test_repeat_method_values()
    print("  repeat(): PASSED")

    print("  Testing permute() method...")
    test_permute_method_3d()
    test_permute_method_shape()
    test_permute_method_2d_transpose()
    print("  permute(): PASSED")

    print("  Testing split() method...")
    test_split_method_equal_parts()
    test_split_method_sizes()
    test_split_method_axis()
    test_split_method_values()
    print("  split(): PASSED")

    print("\nAll ExTensor method API tests PASSED!")
