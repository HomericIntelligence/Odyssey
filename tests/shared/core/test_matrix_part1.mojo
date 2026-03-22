"""Tests for matrix operations - Part 1: Matmul Basic 2D & Batched Operations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Matrix multiplication shapes (2D)
- Matrix multiplication values (2D)
- Matrix multiplication with identity
- Matrix multiplication 2D square/rectangular
- Matrix multiplication with zeros
- Batched matrix multiplication (3D, 4D)
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
from shared.core.extensor import (
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
# Matrix Multiplication Tests - Basic 2D Operations
# ============================================================================


fn test_matmul_shapes() raises:
    """Test that matmul returns correct output shape."""
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(5)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = matmul(a, b)

    # (4, 3) @ (3, 5) = (4, 5)
    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 5)


fn test_matmul_values() raises:
    """Test that matmul computes correct values."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(2)

    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(2)

    var a = zeros(shape_a, DType.float32)
    var b = zeros(shape_b, DType.float32)

    # A = [[1, 2], [3, 4]]
    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0
    a._data.bitcast[Float32]()[3] = 4.0

    # B = [[5, 6], [7, 8]]
    b._data.bitcast[Float32]()[0] = 5.0
    b._data.bitcast[Float32]()[1] = 6.0
    b._data.bitcast[Float32]()[2] = 7.0
    b._data.bitcast[Float32]()[3] = 8.0

    var result = matmul(a, b)

    # Result = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    #        = [[19, 22], [43, 50]]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(19.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(22.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(43.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(50.0), tolerance=1e-5
    )


fn test_matmul_identity() raises:
    """Test matmul with identity matrix."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)

    var a = zeros(shape, DType.float32)
    var identity = zeros(shape, DType.float32)

    # A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i in range(9):
        a._data.bitcast[Float32]()[i] = Float32(i + 1)

    # Identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    identity._data.bitcast[Float32]()[0] = 1.0  # (0, 0)
    identity._data.bitcast[Float32]()[4] = 1.0  # (1, 1)
    identity._data.bitcast[Float32]()[8] = 1.0  # (2, 2)

    var result = matmul(a, identity)

    # A @ I = A
    for i in range(9):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            a._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


fn test_matmul_2d_square() raises:
    """Test 2D matrix multiplication with square matrices."""
    var shape_3x3 = List[Int]()
    shape_3x3.append(3)
    shape_3x3.append(3)

    var a = eye(3, 3, 0, DType.float32)  # 3x3 identity
    var b = full(shape_3x3, 2.0, DType.float32)  # 3x3 matrix of 2s
    var c = matmul(a, b)

    # Identity @ B = B, so result should be all 2s
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 9, "Result should be 3x3 (9 elements)")
    assert_all_values(c, 2.0, 1e-6, "Identity @ B should equal B")


fn test_matmul_2d_rectangular() raises:
    """Test 2D matrix multiplication with rectangular matrices."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(4)
    shape_b.append(2)

    var a = ones(shape_a, DType.float32)  # 3x4
    var b = full(shape_b, 2.0, DType.float32)  # 4x2
    var c = matmul(a, b)

    # Result should be 3x2, each element = 1*2 + 1*2 + 1*2 + 1*2 = 8
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 6, "Result should be 3x2 (6 elements)")
    assert_all_values(c, 8.0, 1e-6, "Each element should be 8")


fn test_matmul_with_zeros() raises:
    """Test matmul with zero matrices."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 9, "Result should be 3x3")
    assert_all_values(c, 0.0, 1e-6, "Zero matrix @ anything = zero matrix")


# ============================================================================
# Matrix Multiplication Tests - Batched Operations
# ============================================================================


fn test_matmul_batched_3d() raises:
    """Test batched matrix multiplication (3D)."""
    var shape_a = List[Int]()
    shape_a.append(2)  # batch size
    shape_a.append(3)  # rows
    shape_a.append(4)  # cols
    var shape_b = List[Int]()
    shape_b.append(2)  # batch size
    shape_b.append(4)  # rows
    shape_b.append(2)  # cols

    var a = ones(shape_a, DType.float32)  # 2x3x4
    var b = full(shape_b, 0.5, DType.float32)  # 2x4x2
    var c = matmul(a, b)

    # Result should be 2x3x2 (batch_size x a_rows x b_cols)
    # Each element = 1*0.5 + 1*0.5 + 1*0.5 + 1*0.5 = 2
    assert_dim(c, 3, "Result should be 3D")
    assert_numel(c, 12, "Result should be 2x3x2 (12 elements)")
    assert_all_values(c, 2.0, 1e-6, "Each element should be 2")


fn test_matmul_batched_4d() raises:
    """Test batched matrix multiplication (4D)."""
    var shape_a = List[Int]()
    shape_a.append(2)  # batch dim 1
    shape_a.append(3)  # batch dim 2
    shape_a.append(4)  # rows
    shape_a.append(5)  # cols
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(3)
    shape_b.append(5)  # rows
    shape_b.append(2)  # cols

    var a = ones(shape_a, DType.float32)  # 2x3x4x5
    var b = ones(shape_b, DType.float32)  # 2x3x5x2
    var c = matmul(a, b)

    # Result should be 2x3x4x2
    assert_dim(c, 4, "Result should be 4D")
    assert_numel(c, 48, "Result should be 2x3x4x2 (48 elements)")
    # Each element = 1*1 + ... (5 times) = 5
    assert_all_values(c, 5.0, 1e-6, "Each element should be 5")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run matmul basic 2D and batched tests."""
    print("Running matrix operation tests - Part 1: Matmul Basic & Batched...")

    print("\n=== Matrix Multiplication: Basic 2D ===")
    test_matmul_shapes()
    print("✓ test_matmul_shapes")
    test_matmul_values()
    print("✓ test_matmul_values")
    test_matmul_identity()
    print("✓ test_matmul_identity")
    test_matmul_2d_square()
    print("✓ test_matmul_2d_square")
    test_matmul_2d_rectangular()
    print("✓ test_matmul_2d_rectangular")
    test_matmul_with_zeros()
    print("✓ test_matmul_with_zeros")

    print("\n=== Matrix Multiplication: Batched ===")
    test_matmul_batched_3d()
    print("✓ test_matmul_batched_3d")
    test_matmul_batched_4d()
    print("✓ test_matmul_batched_4d")

    print("\n" + "=" * 60)
    print("All 8 tests passed! (Part 1)")
