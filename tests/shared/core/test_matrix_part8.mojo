"""Tests for matrix operations - Part 8: Outer Product & Operator Overloading.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Outer product shapes
- Outer product values
- Outer product of two vectors
- Outer non-1D error
- Outer dtype mismatch error
- Outer with zeros
- Outer dtype preservation
- __matmul__ operator overloading (a @ b)
"""

from tests.shared.conftest import (
    assert_all_close,
    assert_all_values,
    assert_almost_equal,
    assert_close_float,
    assert_dim,
    assert_dtype,
    assert_equal,
    assert_equal_int,
    assert_numel,
    assert_shape,
    assert_true,
    assert_value_at,
)
from tests.shared.conftest import TestFixtures
from shared.core.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
    full,
    arange,
    eye,
)
from shared.core.matrix import (
    matmul,
    transpose,
    dot,
    outer,
    matmul_backward,
    transpose_backward,
)
from shared.testing import (
    check_gradient,
    compute_numerical_gradient,
    assert_gradients_close,
)


# ============================================================================
# Outer Product Tests
# ============================================================================


fn test_outer_shapes() raises:
    """Test that outer returns correct output shape."""
    var shape_a = List[Int]()
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = outer(a, b)

    # (3,) outer (4,) = (3, 4)
    assert_equal(result.shape()[0], 3)
    assert_equal(result.shape()[1], 4)


fn test_outer_values() raises:
    """Test that outer computes correct values."""
    var shape_a = List[Int]()
    shape_a.append(2)

    var shape_b = List[Int]()
    shape_b.append(3)

    var a = zeros(shape_a, DType.float32)
    var b = zeros(shape_b, DType.float32)

    # a = [2, 3]
    a._data.bitcast[Float32]()[0] = 2.0
    a._data.bitcast[Float32]()[1] = 3.0

    # b = [4, 5, 6]
    b._data.bitcast[Float32]()[0] = 4.0
    b._data.bitcast[Float32]()[1] = 5.0
    b._data.bitcast[Float32]()[2] = 6.0

    var result = outer(a, b)

    # Outer product = [[2*4, 2*5, 2*6], [3*4, 3*5, 3*6]]
    #                = [[8, 10, 12], [12, 15, 18]]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(8.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(10.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(12.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(12.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(15.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[5], Float32(18.0), tolerance=1e-5
    )


fn test_outer_vectors() raises:
    """Test outer product of two vectors."""
    var a = arange(1.0, 4.0, 1.0, DType.float32)  # [1, 2, 3]
    var b = arange(1.0, 3.0, 1.0, DType.float32)  # [1, 2]
    var c = outer(a, b)

    # Expected 3x2 matrix:
    # [[1, 2],
    #  [2, 4],
    #  [3, 6]]
    assert_dim(c, 2, "Outer product should be 2D")
    assert_numel(c, 6, "Outer product should be 3x2 (6 elements)")
    assert_value_at(c, 0, 1.0, 1e-6, "c[0,0] = 1*1 = 1")
    assert_value_at(c, 1, 2.0, 1e-6, "c[0,1] = 1*2 = 2")
    assert_value_at(c, 2, 2.0, 1e-6, "c[1,0] = 2*1 = 2")
    assert_value_at(c, 3, 4.0, 1e-6, "c[1,1] = 2*2 = 4")
    assert_value_at(c, 4, 3.0, 1e-6, "c[2,0] = 3*1 = 3")
    assert_value_at(c, 5, 6.0, 1e-6, "c[2,1] = 3*2 = 6")


fn test_outer_not_1d_error() raises:
    """Test that non-1D inputs raise error."""
    var shape_2d = List[Int]()
    shape_2d.append(2)
    shape_2d.append(3)
    var shape_1d = List[Int]()
    shape_1d.append(3)

    var a = ones(shape_2d, DType.float32)  # 2D tensor
    var b = ones(shape_1d, DType.float32)  # 1D vector

    var error_raised = False
    try:
        var c = outer(a, b)  # Should error: outer requires 1D
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for non-1D input to outer")


fn test_outer_dtype_mismatch() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)

    var error_raised = False
    try:
        var c = outer(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in outer")


fn test_outer_with_zeros() raises:
    """Test outer product with zero vector."""
    var shape_a = List[Int]()
    shape_a.append(3)
    var shape_b = List[Int]()
    shape_b.append(4)

    var a = zeros(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)
    var c = outer(a, b)

    assert_dim(c, 2, "Outer product should be 2D")
    assert_numel(c, 12, "Outer product should be 3x4 (12 elements)")
    assert_all_values(
        c, 0.0, 1e-6, "Outer with zero vector should be all zeros"
    )


fn test_outer_preserves_dtype() raises:
    """Test that outer preserves dtype."""
    var shape_a = List[Int]()
    shape_a.append(2)
    var shape_b = List[Int]()
    shape_b.append(3)

    var a = ones(shape_a, DType.float64)
    var b = ones(shape_b, DType.float64)
    var c = outer(a, b)

    assert_dtype(c, DType.float64, "outer should preserve float64")


# ============================================================================
# Operator Overloading Tests
# ============================================================================


fn test_dunder_matmul() raises:
    """Test __matmul__ operator overloading (a @ b)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = a @ b

    # Each element should be 1*2 + 1*2 = 4
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4, "Result should be 2x2 (4 elements)")
    assert_all_values(c, 4.0, 1e-6, "a @ b should work via __matmul__")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run outer product and operator overloading tests."""
    print(
        "Running matrix operation tests - Part 8: Outer Product & Operators..."
    )

    print("\n=== Outer Product ===")
    test_outer_shapes()
    print("✓ test_outer_shapes")
    test_outer_values()
    print("✓ test_outer_values")
    test_outer_vectors()
    print("✓ test_outer_vectors")
    test_outer_not_1d_error()
    print("✓ test_outer_not_1d_error")
    test_outer_dtype_mismatch()
    print("✓ test_outer_dtype_mismatch")
    test_outer_with_zeros()
    print("✓ test_outer_with_zeros")
    test_outer_preserves_dtype()
    print("✓ test_outer_preserves_dtype")

    print("\n=== Operator Overloading ===")
    test_dunder_matmul()
    print("✓ test_dunder_matmul")

    print("\n" + "=" * 60)
    print("All 8 tests passed! (Part 8)")
