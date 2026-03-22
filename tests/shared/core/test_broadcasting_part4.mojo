"""Tests for AnyTensor broadcasting operations - Part 4: BroadcastIterator implementation.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_broadcasting.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests the BroadcastIterator implementation for correctness across 1D, 2D, 3D cases.
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.arithmetic import add, multiply
from testing import assert_true

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test complex 3D broadcasting with arithmetic
# ============================================================================


fn test_broadcast_complex_3d_with_multiply() raises:
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


# ============================================================================
# Test BroadcastIterator implementation
# ============================================================================


fn test_broadcast_iterator_1d() raises:
    """Test BroadcastIterator with 1D tensors."""
    from shared.core.broadcasting import BroadcastIterator

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


fn test_broadcast_iterator_2d_no_broadcast() raises:
    """Test BroadcastIterator with 2D tensors (no broadcasting)."""
    from shared.core.broadcasting import BroadcastIterator

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


fn test_broadcast_iterator_2d_broadcast_second() raises:
    """Test BroadcastIterator with 2D broadcast (second tensor is [1,3])."""
    from shared.core.broadcasting import BroadcastIterator

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


fn test_broadcast_iterator_3d_complex() raises:
    """Test BroadcastIterator with complex 3D case."""
    from shared.core.broadcasting import BroadcastIterator

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


fn test_broadcast_iterator_scalar_broadcast() raises:
    """Test BroadcastIterator broadcasting scalar to 1D."""
    from shared.core.broadcasting import BroadcastIterator

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


fn test_broadcast_iterator_exhaustion() raises:
    """Test that BroadcastIterator properly signals exhaustion."""
    from shared.core.broadcasting import BroadcastIterator

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


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run broadcasting part 4 tests."""
    print("Running AnyTensor broadcasting tests - Part 4...")

    # Complex 3D broadcasting
    print("  Testing complex 3D broadcasting with multiply...")
    test_broadcast_complex_3d_with_multiply()

    # BroadcastIterator tests
    print("  Testing BroadcastIterator implementation...")
    test_broadcast_iterator_1d()
    test_broadcast_iterator_2d_no_broadcast()
    test_broadcast_iterator_2d_broadcast_second()
    test_broadcast_iterator_3d_complex()
    test_broadcast_iterator_scalar_broadcast()
    test_broadcast_iterator_exhaustion()

    print("All broadcasting part 4 tests completed!")
