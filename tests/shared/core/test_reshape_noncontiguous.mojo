"""Tests for non-contiguous reshape bug fix (Issue #4084).

Verifies that reshape() correctly handles non-contiguous tensors by using
stride-based element access instead of flat-index access.
"""

from shared.tensor.any_tensor import AnyTensor, arange
from shared.core import reshape
from shared.core.shape import is_contiguous

from tests.shared.conftest import (
    assert_numel,
    assert_dim,
    assert_dtype,
    assert_value_at,
)


def test_reshape_noncontiguous_column_major() raises:
    """Regression test: reshape on transposed (non-contiguous) tensor produces correct order.

    Simulates transpose_view(arange(12).reshape(3,4)) by setting strides to
    match a transposed (4,3) view of row-major (3,4) data.
    Reshaping to [12] should give [0,4,8,1,5,9,2,6,10,3,7,11].
    """
    var t = arange(0.0, 12.0, 1.0, DType.float64)
    var t2 = reshape(t, [3, 4])

    # Simulate transpose: shape (4,3) with strides (1,4)
    # Original (3,4) row-major strides are (4,1); transposed shape is (4,3) with strides (1,4)
    t2._shape[0] = 4
    t2._shape[1] = 3
    t2._strides[0] = 1
    t2._strides[1] = 4

    var result = reshape(t2, [12])

    assert_numel(result, 12, "result should have 12 elements")
    assert_dim(result, 1, "result should be 1D")

    # Transposed read: (row,col) -> offset = row*1 + col*4
    # i=0: (0,0)->0, i=1: (0,1)->4, i=2: (0,2)->8
    # i=3: (1,0)->1, i=4: (1,1)->5, i=5: (1,2)->9
    # i=6: (2,0)->2, i=7: (2,1)->6, i=8: (2,2)->10
    # i=9: (3,0)->3, i=10: (3,1)->7, i=11: (3,2)->11
    var expected: List[Float64] = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    for i in range(12):
        assert_value_at(
            result,
            i,
            expected[i],
            message="transposed reshape index " + String(i),
        )


def test_reshape_contiguous_unchanged() raises:
    """Contiguous tensors still produce correct order after the fix."""
    var t = arange(0.0, 12.0, 1.0, DType.float64)
    var t2 = reshape(t, [3, 4])

    var result = reshape(t2, [12])

    assert_numel(result, 12, "result should have 12 elements")
    for i in range(12):
        assert_value_at(
            result,
            i,
            Float64(i),
            message="contiguous reshape index " + String(i),
        )


def test_reshape_noncontiguous_2d_to_2d() raises:
    """Non-contiguous (4,3) transposed view -> (2,6) reshape produces correct values."""
    var t = arange(0.0, 12.0, 1.0, DType.float64)
    var t2 = reshape(t, [3, 4])

    # Simulate transpose: shape (4,3), strides (1,4)
    t2._shape[0] = 4
    t2._shape[1] = 3
    t2._strides[0] = 1
    t2._strides[1] = 4

    var result = reshape(t2, [2, 6])

    assert_dim(result, 2, "result should be 2D")
    assert_numel(result, 12, "result should have 12 elements")

    # Transposed read order: [0,4,8,1,5,9,2,6,10,3,7,11]
    var expected: List[Float64] = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    for i in range(12):
        assert_value_at(
            result,
            i,
            expected[i],
            message="noncontiguous 2D->2D reshape index " + String(i),
        )


def test_reshape_noncontiguous_preserves_dtype() raises:
    """Non-contiguous path preserves the source dtype."""
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var t2 = reshape(t, [2, 3])

    # Simulate transpose: shape (3,2), strides (1,2)
    t2._shape[0] = 3
    t2._shape[1] = 2
    t2._strides[0] = 1
    t2._strides[1] = 3

    var result = reshape(t2, [6])

    assert_dtype(result, DType.float32, "dtype should be preserved as float32")
    assert_numel(result, 6, "result should have 6 elements")


def main() raises:
    print("Testing reshape() non-contiguous fix (Issue #4084)...")

    print("  test_reshape_noncontiguous_column_major...")
    test_reshape_noncontiguous_column_major()
    print("    ✓ passed")

    print("  test_reshape_contiguous_unchanged...")
    test_reshape_contiguous_unchanged()
    print("    ✓ passed")

    print("  test_reshape_noncontiguous_2d_to_2d...")
    test_reshape_noncontiguous_2d_to_2d()
    print("    ✓ passed")

    print("  test_reshape_noncontiguous_preserves_dtype...")
    test_reshape_noncontiguous_preserves_dtype()
    print("    ✓ passed")

    print("All reshape non-contiguous tests passed!")
