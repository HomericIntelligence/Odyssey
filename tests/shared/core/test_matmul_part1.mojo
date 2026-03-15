# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matmul.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Matrix multiplication tests - Part 1: Utilities and Baseline Correctness.

Tests cover:
- Correctness: Baseline (Stage 0) tests
- Edge Cases: Small sizes, trivial cases
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
    assert_matrices_equal,
    assert_numel,
    assert_shape,
    assert_true,
    assert_value_at,
)
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
    full,
    arange,
    eye,
)
from shared.core.matrix import matmul


# ============================================================================
# Correctness Tests - Stage 0 (Baseline)
# ============================================================================


fn test_matmul_baseline_2x2() raises:
    """Test baseline matmul with simple 2x2 matrices (reference test)."""
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


fn test_matmul_baseline_identity() raises:
    """Test baseline matmul with identity matrix."""
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


fn test_matmul_baseline_zero_matrix() raises:
    """Test baseline matmul with zero matrix."""
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
# Edge Case Tests - Test Shapes from Plan
# ============================================================================


fn test_matmul_trivial_1x1() raises:
    """Test 1x1 matrix multiplication (trivial case)."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(1)

    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 4.0, DType.float32)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 1, "Result should be 1x1 (1 element)")
    assert_value_at(c, 0, 12.0, 1e-6, "1x1 @ 1x1 = 3*4")


fn test_matmul_vector_matrix_1x64x1() raises:
    """Test vector-matrix multiplication (1x64) @ (64x1)."""
    var shape_a = List[Int]()
    shape_a.append(1)
    shape_a.append(64)

    var shape_b = List[Int]()
    shape_b.append(64)
    shape_b.append(1)

    var a = ones(shape_a, DType.float32)
    var b = full(shape_b, 2.0, DType.float32)
    var c = matmul(a, b)

    # Result: 1x1 matrix with value 64*2 = 128
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 1, "Result should be 1x1")
    assert_value_at(c, 0, 128.0, 1e-5, "Sum of 64 products of 1*2")


fn test_matmul_matrix_vector_64x1x64() raises:
    """Test matrix-vector multiplication (64x1) @ (1x64)."""
    var shape_a = List[Int]()
    shape_a.append(64)
    shape_a.append(1)

    var shape_b = List[Int]()
    shape_b.append(1)
    shape_b.append(64)

    var a = full(shape_a, 2.0, DType.float32)
    var b = ones(shape_b, DType.float32)
    var c = matmul(a, b)

    # Result: 64x64 matrix with all elements = 2*1 = 2
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4096, "Result should be 64x64")
    assert_all_values(c, 2.0, 1e-6, "All elements should be 2.0")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run matrix multiplication tests - Part 1."""
    print("Running matrix multiplication tests - Part 1...")
    print("=" * 70)

    # Baseline correctness tests
    print("\n=== Baseline Correctness (Stage 0) ===")
    test_matmul_baseline_2x2()
    print("✓ test_matmul_baseline_2x2")
    test_matmul_baseline_identity()
    print("✓ test_matmul_baseline_identity")
    test_matmul_baseline_zero_matrix()
    print("✓ test_matmul_baseline_zero_matrix")

    # Edge case tests
    print("\n=== Edge Cases (Test Shapes from Plan) ===")
    test_matmul_trivial_1x1()
    print("✓ test_matmul_trivial_1x1 (1x1)")
    test_matmul_vector_matrix_1x64x1()
    print("✓ test_matmul_vector_matrix_1x64x1 (1x64 @ 64x1)")
    test_matmul_matrix_vector_64x1x64()
    print("✓ test_matmul_matrix_vector_64x1x64 (64x1 @ 1x64)")

    print("\n" + "=" * 70)
    print("All 6 matrix multiplication Part 1 tests passed!")
    print("=" * 70)
    print("\n=== Test Coverage Summary ===")
    print("✓ Baseline Correctness (Stage 0):     3 tests")
    print("✓ Edge Cases (partial):               3 tests")
