"""Tests for AnyTensor broadcasting operations

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_broadcasting.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests NumPy-style broadcasting rules for scalar and vector-to-matrix cases.
"""


from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.arithmetic import add, multiply, subtract, divide
from shared.core.comparison import greater, less_equal
from testing import assert_true
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)
from shared.base.broadcasting import are_shapes_broadcastable


def test_broadcast_scalar_to_1d() raises:
    """Test broadcasting scalar to 1D tensor."""
    var shape_vec = List[Int]()
    shape_vec.append(5)
    var shape_scalar = List[Int]()

    var a = full(shape_vec, 3.0, DType.float32)  # [3, 3, 3, 3, 3]
    var b = full(shape_scalar, 2.0, DType.float32)  # scalar 2
    var c = add(a, b)  # Expected: [5, 5, 5, 5, 5]

    assert_numel(c, 5, "Result should have 5 elements")
    assert_all_values(c, 5.0, 1e-6, "3 + 2 should broadcast to [5, 5, 5, 5, 5]")


def test_broadcast_scalar_to_2d() raises:
    """Test broadcasting scalar to 2D tensor."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_scalar = List[Int]()

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix of ones
    var b = full(shape_scalar, 5.0, DType.float32)  # scalar 5
    var c = multiply(a, b)  # Expected: 3x4 matrix of fives

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 5.0, 1e-6, "1 * 5 should broadcast to all 5s")


def test_broadcast_scalar_to_3d() raises:
    """Test broadcasting scalar to 3D tensor."""
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    var shape_scalar = List[Int]()

    var a = full(shape_3d, 2.0, DType.float32)  # 2x3x4 tensor
    var b = full(shape_scalar, 3.0, DType.float32)  # scalar 3
    var c = add(a, b)  # Expected: 2x3x4 tensor of fives

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "2 + 3 should broadcast to all 5s")


def test_broadcast_vector_to_matrix_row() raises:
    """Test broadcasting row vector to matrix."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_vec = List[Int]()
    shape_vec.append(1)
    shape_vec.append(4)

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix
    var b = full(shape_vec, 2.0, DType.float32)  # 1x4 vector
    var c = add(a, b)  # Expected: 3x4 matrix, each row is [3, 3, 3, 3]

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 3.0, 1e-6, "Broadcasting 1x4 vector to 3x4 matrix")


def test_broadcast_vector_to_matrix_column() raises:
    """Test broadcasting column vector to matrix."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_vec = List[Int]()
    shape_vec.append(3)
    shape_vec.append(1)

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix
    var b = full(shape_vec, 2.0, DType.float32)  # 3x1 vector
    var c = multiply(a, b)  # Expected: 3x4 matrix, each column multiplied by 2

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 2.0, 1e-6, "Broadcasting 3x1 vector to 3x4 matrix")


def test_broadcast_1d_to_2d() raises:
    """Test broadcasting 1D vector to 2D matrix."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_vec = List[Int]()
    shape_vec.append(4)

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix
    var b = full(shape_vec, 3.0, DType.float32)  # 4-element vector
    var c = add(a, b)  # Expected: 3x4 matrix, each row is [4, 4, 4, 4]

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 4.0, 1e-6, "Broadcasting 1D(4) to 2D(3,4)")


def test_broadcast_size_one_dim_leading() raises:
    """Test broadcasting with leading dimension of size 1."""
    var shape_a = List[Int]()
    shape_a.append(1)
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(3)
    shape_b.append(4)

    var a = full(shape_a, 2.0, DType.float32)  # 1x3x4
    var b = ones(shape_b, DType.float32)  # 2x3x4
    var c = add(a, b)  # Expected: 2x3x4, all 3s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 3.0, 1e-6, "Broadcasting 1x3x4 to 2x3x4")


def test_broadcast_size_one_dim_middle() raises:
    """Test broadcasting with middle dimension of size 1."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(1)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(3)
    shape_b.append(4)

    var a = full(shape_a, 5.0, DType.float32)  # 2x1x4
    var b = ones(shape_b, DType.float32)  # 2x3x4
    var c = multiply(a, b)  # Expected: 2x3x4, all 5s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "Broadcasting 2x1x4 to 2x3x4")


def test_broadcast_size_one_dim_trailing() raises:
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


def test_broadcast_missing_leading_dims() raises:
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


def test_broadcast_2d_to_3d() raises:
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


def test_broadcast_3d_complex() raises:
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


def test_broadcast_4d() raises:
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


def test_broadcast_incompatible_shapes_different_sizes() raises:
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


def test_broadcast_incompatible_inner_dims() raises:
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


def test_broadcast_output_shape_scalar_1d() raises:
    """Test broadcast output shape for scalar + 1D."""
    var shape_vec = List[Int]()
    shape_vec.append(5)
    var shape_scalar = List[Int]()

    var a = ones(shape_vec, DType.float32)
    var b = ones(shape_scalar, DType.float32)
    var c = add(a, b)

    assert_dim(c, 1, "Output should be 1D")
    assert_numel(c, 5, "Output should have 5 elements")


def test_broadcast_output_shape_1d_2d() raises:
    """Test broadcast output shape for 1D + 2D."""
    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)
    var shape_1d = List[Int]()
    shape_1d.append(4)

    var a = ones(shape_2d, DType.float32)
    var b = ones(shape_1d, DType.float32)
    var c = add(a, b)

    assert_dim(c, 2, "Output should be 2D")
    assert_numel(c, 12, "Output should have 12 elements")


def test_broadcast_output_shape_3d_complex() raises:
    """Test broadcast output shape for complex 3D case."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(1)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(1)
    shape_b.append(3)
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)
    var c = add(a, b)

    assert_dim(c, 3, "Output should be 3D")
    assert_numel(c, 24, "Output should have 24 elements (2*3*4)")


def test_broadcast_preserves_dtype() raises:
    """Test that broadcasting preserves dtype."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(4)

    var a = ones(shape_a, DType.float64)
    var b = ones(shape_b, DType.float64)
    var c = add(a, b)

    assert_dtype(c, DType.float64, "Broadcast should preserve float64 dtype")


def test_broadcast_with_comparison_scalar() raises:
    """Test broadcasting scalar with comparison operations."""
    var shape_vec = List[Int]()
    shape_vec.append(5)
    var shape_scalar = List[Int]()

    var a = full(shape_vec, 3.0, DType.float32)  # [3, 3, 3, 3, 3]
    var b = full(shape_scalar, 2.0, DType.float32)  # scalar 2
    var c = greater(a, b)  # Should broadcast: [True, True, True, True, True]

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.bool, "Comparison should return bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "3 > 2 should be True")


def test_broadcast_with_comparison_vector_matrix() raises:
    """Test broadcasting vector to matrix with comparison."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_vec = List[Int]()
    shape_vec.append(4)

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix of ones
    var b = full(shape_vec, 2.0, DType.float32)  # vector [2, 2, 2, 2]
    var c = less_equal(a, b)  # 1 <= 2 broadcasts to 3x4

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.bool, "Comparison should return bool dtype")
    for i in range(12):
        assert_value_at(c, i, 1.0, 1e-6, "1 <= 2 should be True")


def test_broadcast_chained_operations() raises:
    """Test chained operations with broadcasting."""
    var shape_mat = List[Int]()
    shape_mat.append(2)
    shape_mat.append(3)
    var shape_scalar = List[Int]()

    var a = full(shape_mat, 5.0, DType.float32)  # 2x3 matrix
    var b = full(shape_scalar, 2.0, DType.float32)  # scalar
    var c = full(shape_scalar, 3.0, DType.float32)  # scalar

    # (a + b) * c = (5 + 2) * 3 = 7 * 3 = 21
    var result = multiply(add(a, b), c)

    assert_numel(result, 6, "Result should have 6 elements")
    assert_all_values(result, 21.0, 1e-6, "(5 + 2) * 3 should be 21")


def test_broadcast_with_subtract() raises:
    """Test broadcasting with subtraction."""
    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)
    var shape_1d = List[Int]()
    shape_1d.append(4)

    var a = full(shape_2d, 10.0, DType.float32)  # 3x4 matrix of 10s
    var b = full(shape_1d, 3.0, DType.float32)  # vector [3, 3, 3, 3]
    var c = subtract(a, b)  # 10 - 3 = 7, broadcast to 3x4

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 7.0, 1e-6, "10 - 3 should broadcast to all 7s")


def test_broadcast_with_divide() raises:
    """Test broadcasting with division."""
    var shape_mat = List[Int]()
    shape_mat.append(2)
    shape_mat.append(5)
    var shape_scalar = List[Int]()

    var a = full(shape_mat, 20.0, DType.float32)  # 2x5 matrix of 20s
    var b = full(shape_scalar, 4.0, DType.float32)  # scalar 4
    var c = divide(a, b)  # 20 / 4 = 5, broadcast

    assert_numel(c, 10, "Result should have 10 elements")
    assert_all_values(c, 5.0, 1e-6, "20 / 4 should broadcast to all 5s")


def test_broadcast_complex_3d_with_multiply() raises:
    """Test complex 3D broadcasting with multiply."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(1)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(1)
    shape_b.append(3)
    shape_b.append(4)

    var a = full(shape_a, 3.0, DType.float32)  # 2x1x4
    var b = full(shape_b, 4.0, DType.float32)  # 1x3x4
    var c = multiply(a, b)  # 3 * 4 = 12, broadcast to 2x3x4

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 12.0, 1e-6, "3 * 4 should broadcast to all 12s")


def test_broadcast_iterator_1d() raises:
    """Test BroadcastIterator with 1D tensors."""
    from shared.base.broadcasting import BroadcastIterator

    # Shape: [3]
    # Strides: [1] for both (no broadcasting)
    var shape = List[Int]()
    shape.append(3)

    var strides1 = List[Int]()
    strides1.append(1)

    var strides2 = List[Int]()
    strides2.append(1)

    var iterator = BroadcastIterator(shape^, strides1^, strides2^)

    # Expected sequence: (0,0), (1,1), (2,2)
    var count = 0
    while iterator.has_next():
        var result = iterator.__next__()
        var idx1 = result[0]
        var idx2 = result[1]
        assert_true(
            idx1 == count, "Index 1 mismatch at iteration " + String(count)
        )
        assert_true(
            idx2 == count, "Index 2 mismatch at iteration " + String(count)
        )
        count += 1

    assert_true(count == 3, "Should iterate exactly 3 times")


def test_broadcast_iterator_2d_no_broadcast() raises:
    """Test BroadcastIterator with 2D tensors (no broadcasting)."""
    from shared.base.broadcasting import BroadcastIterator

    # Shape: [2, 3]
    # Strides: [3, 1] for both (row-major, no broadcasting)
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var strides1 = List[Int]()
    strides1.append(3)
    strides1.append(1)

    var strides2 = List[Int]()
    strides2.append(3)
    strides2.append(1)

    var iterator = BroadcastIterator(shape^, strides1^, strides2^)

    # Expected sequence (row-major):
    # (0,0), (1,1), (2,2), (3,3), (4,4), (5,5)
    var expected_pairs = List[Tuple[Int, Int]]()
    for i in range(6):
        expected_pairs.append((i, i))

    var count = 0
    while iterator.has_next():
        var result2 = iterator.__next__()
        var idx1 = result2[0]
        var idx2 = result2[1]
        var expected_pair = expected_pairs[count]
        var exp1 = expected_pair[0]
        var exp2 = expected_pair[1]
        assert_true(
            idx1 == exp1 and idx2 == exp2,
            "Mismatch at iteration "
            + String(count)
            + ": got ("
            + String(idx1)
            + ","
            + String(idx2)
            + ") expected ("
            + String(exp1)
            + ","
            + String(exp2)
            + ")",
        )
        count += 1

    assert_true(count == 6, "Should iterate 6 times")


def test_broadcast_iterator_2d_broadcast_second() raises:
    """Test BroadcastIterator with 2D broadcast (second tensor is [1,3])."""
    from shared.base.broadcasting import BroadcastIterator

    # Shape: [2, 3] (broadcast result)
    # Tensor A: [2, 3], strides [3, 1]
    # Tensor B: [1, 3], strides [0, 1] (first dim broadcasted)
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var strides1 = List[Int]()
    strides1.append(3)
    strides1.append(1)

    var strides2 = List[Int]()
    strides2.append(0)  # Broadcasted dimension
    strides2.append(1)

    var iterator = BroadcastIterator(shape^, strides1^, strides2^)

    # Expected: tensor A accesses [0,1,2,3,4,5], tensor B accesses [0,1,2,0,1,2]
    var count = 0
    while iterator.has_next():
        var result3 = iterator.__next__()
        var idx1 = result3[0]
        var idx2 = result3[1]
        var expected_idx1 = count
        var expected_idx2 = count % 3  # Only 3 elements in tensor B
        assert_true(
            idx1 == expected_idx1,
            "Index 1 mismatch at iteration "
            + String(count)
            + ": got "
            + String(idx1)
            + " expected "
            + String(expected_idx1),
        )
        assert_true(
            idx2 == expected_idx2,
            "Index 2 mismatch at iteration "
            + String(count)
            + ": got "
            + String(idx2)
            + " expected "
            + String(expected_idx2),
        )
        count += 1

    assert_true(count == 6, "Should iterate 6 times")


def test_broadcast_iterator_3d_complex() raises:
    """Test BroadcastIterator with complex 3D case."""
    from shared.base.broadcasting import BroadcastIterator

    # Shape: [2, 3, 4] (broadcast result)
    # Both tensors have same shape, strides [12, 4, 1]
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var strides1 = List[Int]()
    strides1.append(12)
    strides1.append(4)
    strides1.append(1)

    var strides2 = List[Int]()
    strides2.append(12)
    strides2.append(4)
    strides2.append(1)

    var iterator = BroadcastIterator(shape^, strides1^, strides2^)

    # Should iterate through all 2*3*4 = 24 elements
    var count = 0
    while iterator.has_next():
        var result4 = iterator.__next__()
        var idx1 = result4[0]
        var idx2 = result4[1]
        # Both should have same index
        assert_true(idx1 == idx2, "Indices should match for non-broadcast case")
        assert_true(idx1 == count, "Index should match position")
        count += 1

    assert_true(count == 24, "Should iterate 24 times for 2x3x4")


def test_broadcast_iterator_scalar_broadcast() raises:
    """Test BroadcastIterator broadcasting scalar to 1D."""
    from shared.base.broadcasting import BroadcastIterator

    # Shape: [5] (broadcast result)
    # Tensor A: [5], strides [1]
    # Tensor B: scalar [0], strides [0] (entire dimension is broadcast)
    var shape = List[Int]()
    shape.append(5)

    var strides1 = List[Int]()
    strides1.append(1)

    var strides2 = List[Int]()
    strides2.append(0)  # Scalar: stride 0

    var iterator = BroadcastIterator(shape^, strides1^, strides2^)

    # Expected: A accesses [0,1,2,3,4], B always accesses [0]
    var count = 0
    while iterator.has_next():
        var result5 = iterator.__next__()
        var idx1 = result5[0]
        var idx2 = result5[1]
        assert_true(idx1 == count, "Index 1 should match position")
        assert_true(idx2 == 0, "Index 2 should always be 0 for scalar")
        count += 1

    assert_true(count == 5, "Should iterate 5 times")


def test_broadcast_iterator_exhaustion() raises:
    """Test that BroadcastIterator properly signals exhaustion."""
    from shared.base.broadcasting import BroadcastIterator

    var shape = List[Int]()
    shape.append(2)

    var strides1 = List[Int]()
    strides1.append(1)

    var strides2 = List[Int]()
    strides2.append(1)

    var iterator = BroadcastIterator(shape^, strides1^, strides2^)

    # Consume all elements
    while iterator.has_next():
        var _ = iterator.__next__()

    # Try to get another element
    var error_raised = False
    try:
        var _ = iterator.__next__()
    except:
        error_raised = True

    assert_true(error_raised, "Iterator should raise error when exhausted")


def test_are_shapes_broadcastable_ndim_reduction_returns_false() raises:
    """Verify are_shapes_broadcastable([3,4,5], [4,5]) returns False (ndim reduction)."""
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    shape1.append(5)
    var shape2 = List[Int]()
    shape2.append(4)
    shape2.append(5)
    assert_true(
        not are_shapes_broadcastable(shape1, shape2),
        "3D->2D ndim reduction must return False",
    )


def test_are_shapes_broadcastable_1d_vs_2d_reduction() raises:
    """Verify are_shapes_broadcastable([3,4], [4]) returns False (ndim reduction)."""
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(4)
    assert_true(
        not are_shapes_broadcastable(shape1, shape2),
        "2D->1D ndim reduction must return False",
    )


def test_are_shapes_broadcastable_empty_target_returns_false() raises:
    """Verify are_shapes_broadcastable([3], []) returns False (empty target)."""
    var shape1 = List[Int]()
    shape1.append(3)
    var shape2 = List[Int]()
    assert_true(
        not are_shapes_broadcastable(shape1, shape2),
        "Non-empty source vs empty target must return False",
    )


def test_are_shapes_broadcastable_expanding_ndim_ok() raises:
    """Verify are_shapes_broadcastable([4,5], [3,4,5]) returns True (expanding dims)."""
    var shape1 = List[Int]()
    shape1.append(4)
    shape1.append(5)
    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    shape2.append(5)
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "2D->3D expansion must return True",
    )


def test_are_shapes_broadcastable_same_ndim_compatible() raises:
    """Verify are_shapes_broadcastable([3,4], [3,4]) returns True (same shape)."""
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "Identical shapes must return True",
    )


def test_are_shapes_broadcastable_broadcast_1_dim_ok() raises:
    """Verify are_shapes_broadcastable([1,4], [3,4]) returns True (dim-1 broadcast)."""
    var shape1 = List[Int]()
    shape1.append(1)
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "Dim-1 broadcast [1,4]->[3,4] must return True",
    )


def test_are_shapes_broadcastable_incompatible_dims_unchanged() raises:
    """Verify are_shapes_broadcastable([3,4], [5,4]) returns False (incompatible dims)."""
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(5)
    shape2.append(4)
    assert_true(
        not are_shapes_broadcastable(shape1, shape2),
        "Incompatible dims [3,4] vs [5,4] must return False",
    )


def test_are_shapes_broadcastable_scalar_source_empty_target() raises:
    """Verify are_shapes_broadcastable([], []) returns True (both empty/scalar)."""
    var shape1 = List[Int]()
    var shape2 = List[Int]()
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "Both empty shapes (scalar) must return True",
    )


def test_are_shapes_broadcastable_scalar_source_to_1d() raises:
    """Verify are_shapes_broadcastable([], [3]) returns True (scalar to 1D)."""
    var shape1 = List[Int]()
    var shape2 = List[Int]()
    shape2.append(3)
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "Scalar source to 1D target must return True",
    )


def main() raises:
    """Run all test_broadcasting tests."""
    print("Running test_broadcasting tests...")

    test_broadcast_scalar_to_1d()
    print("✓ test_broadcast_scalar_to_1d")

    test_broadcast_scalar_to_2d()
    print("✓ test_broadcast_scalar_to_2d")

    test_broadcast_scalar_to_3d()
    print("✓ test_broadcast_scalar_to_3d")

    test_broadcast_vector_to_matrix_row()
    print("✓ test_broadcast_vector_to_matrix_row")

    test_broadcast_vector_to_matrix_column()
    print("✓ test_broadcast_vector_to_matrix_column")

    test_broadcast_1d_to_2d()
    print("✓ test_broadcast_1d_to_2d")

    test_broadcast_size_one_dim_leading()
    print("✓ test_broadcast_size_one_dim_leading")

    test_broadcast_size_one_dim_middle()
    print("✓ test_broadcast_size_one_dim_middle")

    test_broadcast_size_one_dim_trailing()
    print("✓ test_broadcast_size_one_dim_trailing")

    test_broadcast_missing_leading_dims()
    print("✓ test_broadcast_missing_leading_dims")

    test_broadcast_2d_to_3d()
    print("✓ test_broadcast_2d_to_3d")

    test_broadcast_3d_complex()
    print("✓ test_broadcast_3d_complex")

    test_broadcast_4d()
    print("✓ test_broadcast_4d")

    test_broadcast_incompatible_shapes_different_sizes()
    print("✓ test_broadcast_incompatible_shapes_different_sizes")

    test_broadcast_incompatible_inner_dims()
    print("✓ test_broadcast_incompatible_inner_dims")

    test_broadcast_output_shape_scalar_1d()
    print("✓ test_broadcast_output_shape_scalar_1d")

    test_broadcast_output_shape_1d_2d()
    print("✓ test_broadcast_output_shape_1d_2d")

    test_broadcast_output_shape_3d_complex()
    print("✓ test_broadcast_output_shape_3d_complex")

    test_broadcast_preserves_dtype()
    print("✓ test_broadcast_preserves_dtype")

    test_broadcast_with_comparison_scalar()
    print("✓ test_broadcast_with_comparison_scalar")

    test_broadcast_with_comparison_vector_matrix()
    print("✓ test_broadcast_with_comparison_vector_matrix")

    test_broadcast_chained_operations()
    print("✓ test_broadcast_chained_operations")

    test_broadcast_with_subtract()
    print("✓ test_broadcast_with_subtract")

    test_broadcast_with_divide()
    print("✓ test_broadcast_with_divide")

    test_broadcast_complex_3d_with_multiply()
    print("✓ test_broadcast_complex_3d_with_multiply")

    test_broadcast_iterator_1d()
    print("✓ test_broadcast_iterator_1d")

    test_broadcast_iterator_2d_no_broadcast()
    print("✓ test_broadcast_iterator_2d_no_broadcast")

    test_broadcast_iterator_2d_broadcast_second()
    print("✓ test_broadcast_iterator_2d_broadcast_second")

    test_broadcast_iterator_3d_complex()
    print("✓ test_broadcast_iterator_3d_complex")

    test_broadcast_iterator_scalar_broadcast()
    print("✓ test_broadcast_iterator_scalar_broadcast")

    test_broadcast_iterator_exhaustion()
    print("✓ test_broadcast_iterator_exhaustion")

    test_are_shapes_broadcastable_ndim_reduction_returns_false()
    print("✓ test_are_shapes_broadcastable_ndim_reduction_returns_false")

    test_are_shapes_broadcastable_1d_vs_2d_reduction()
    print("✓ test_are_shapes_broadcastable_1d_vs_2d_reduction")

    test_are_shapes_broadcastable_empty_target_returns_false()
    print("✓ test_are_shapes_broadcastable_empty_target_returns_false")

    test_are_shapes_broadcastable_expanding_ndim_ok()
    print("✓ test_are_shapes_broadcastable_expanding_ndim_ok")

    test_are_shapes_broadcastable_same_ndim_compatible()
    print("✓ test_are_shapes_broadcastable_same_ndim_compatible")

    test_are_shapes_broadcastable_broadcast_1_dim_ok()
    print("✓ test_are_shapes_broadcastable_broadcast_1_dim_ok")

    test_are_shapes_broadcastable_incompatible_dims_unchanged()
    print("✓ test_are_shapes_broadcastable_incompatible_dims_unchanged")

    test_are_shapes_broadcastable_scalar_source_empty_target()
    print("✓ test_are_shapes_broadcastable_scalar_source_empty_target")

    test_are_shapes_broadcastable_scalar_source_to_1d()
    print("✓ test_are_shapes_broadcastable_scalar_source_to_1d")

    print("\nAll test_broadcasting tests passed!")
