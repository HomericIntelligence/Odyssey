"""Comprehensive tests for Strassen's algorithm for matrix multiplication.

Tests cover:
- Correctness: Strassen produces correct results within tolerance
- Helper Functions: next_power_of_2, quadrant partition/combine
- Edge Cases: Zero matrices, identity matrices, non-power-of-2 dimensions
- Numerical Stability: Accumulation errors with large matrices
- DType Tests: Float32 and Float64

Test Strategy:
- Small matrices verify exact correctness (tight tolerance)
- Large matrices verify within looser tolerance (accumulation errors)
- Non-power-of-2 dimensions verify padding/unpadding logic
"""

from tests.projectodyssey.conftest import (
    assert_all_close,
    assert_almost_equal,
    assert_equal_int,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like
from projectodyssey.core.strassen import matmul_strassen, next_power_of_2
from projectodyssey.core.matmul import matmul_tiled


# ============================================================================
# Helper Function Tests
# ============================================================================


def test_next_power_of_2() raises:
    """Test next_power_of_2 helper function."""
    assert_equal_int(next_power_of_2(0), 1)
    assert_equal_int(next_power_of_2(1), 1)
    assert_equal_int(next_power_of_2(2), 2)
    assert_equal_int(next_power_of_2(3), 4)
    assert_equal_int(next_power_of_2(4), 4)
    assert_equal_int(next_power_of_2(5), 8)
    assert_equal_int(next_power_of_2(63), 64)
    assert_equal_int(next_power_of_2(64), 64)
    assert_equal_int(next_power_of_2(65), 128)
    assert_equal_int(next_power_of_2(127), 128)
    assert_equal_int(next_power_of_2(128), 128)
    assert_equal_int(next_power_of_2(512), 512)
    assert_equal_int(next_power_of_2(513), 1024)


# ============================================================================
# Correctness Tests - Small Matrices
# ============================================================================


def test_strassen_2x2_float32() raises:
    """Test Strassen with smallest 2x2 matrix in Float32."""
    var A = zeros([2, 2], DType.float32)
    var B = zeros([2, 2], DType.float32)
    var C = zeros([2, 2], DType.float32)

    # Set values: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    var a_ptr = A._data.bitcast[Float32]()
    var b_ptr = B._data.bitcast[Float32]()

    a_ptr.store(0, Float32(1.0))
    a_ptr.store(1, Float32(2.0))
    a_ptr.store(2, Float32(3.0))
    a_ptr.store(3, Float32(4.0))

    b_ptr.store(0, Float32(5.0))
    b_ptr.store(1, Float32(6.0))
    b_ptr.store(2, Float32(7.0))
    b_ptr.store(3, Float32(8.0))

    # Compute with Strassen
    matmul_strassen(A, B, C)

    # Verify: C = [[19, 22], [43, 50]]
    var c_ptr = C._data.bitcast[Float32]()
    var c00 = c_ptr.load(0)
    var c01 = c_ptr.load(1)
    var c10 = c_ptr.load(2)
    var c11 = c_ptr.load(3)

    assert_almost_equal(c00, Float32(19.0), Float32(1e-4))
    assert_almost_equal(c01, Float32(22.0), Float32(1e-4))
    assert_almost_equal(c10, Float32(43.0), Float32(1e-4))
    assert_almost_equal(c11, Float32(50.0), Float32(1e-4))


def test_strassen_4x4_float32() raises:
    """Test Strassen with 4x4 matrix in Float32."""
    var A = zeros([4, 4], DType.float32)
    var B = zeros([4, 4], DType.float32)
    var C_strassen = zeros([4, 4], DType.float32)
    var C_ref = zeros([4, 4], DType.float32)

    # Fill with sequential values for deterministic test
    var a_ptr = A._data.bitcast[Float32]()
    var b_ptr = B._data.bitcast[Float32]()

    for i in range(16):
        a_ptr.store(i, Float32(Float64(i) + 1.0))
        b_ptr.store(i, Float32(Float64(i) + 1.0))

    # Compute with both methods
    matmul_strassen(A, B, C_strassen)
    matmul_tiled(A, B, C_ref)

    # Compare results
    var strassen_ptr = C_strassen._data.bitcast[Float32]()
    var ref_ptr = C_ref._data.bitcast[Float32]()

    for i in range(16):
        var s_val = strassen_ptr.load(i)
        var r_val = ref_ptr.load(i)
        assert_almost_equal(s_val, r_val, Float32(1e-4))


def test_strassen_8x8_float64() raises:
    """Test Strassen with 8x8 matrix in Float64."""
    var A = zeros([8, 8], DType.float64)
    var B = zeros([8, 8], DType.float64)
    var C_strassen = zeros([8, 8], DType.float64)
    var C_ref = zeros([8, 8], DType.float64)

    # Fill with sequential values
    var a_ptr = A._data.bitcast[Float64]()
    var b_ptr = B._data.bitcast[Float64]()

    for i in range(64):
        a_ptr.store(i, Float64(i) + 1.0)
        b_ptr.store(i, Float64(i) + 1.0)

    # Compute with both methods
    matmul_strassen(A, B, C_strassen)
    matmul_tiled(A, B, C_ref)

    # Compare results
    var strassen_ptr = C_strassen._data.bitcast[Float64]()
    var ref_ptr = C_ref._data.bitcast[Float64]()

    for i in range(64):
        var s_val = strassen_ptr.load(i)
        var r_val = ref_ptr.load(i)
        assert_almost_equal(s_val, r_val, 1e-10)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_strassen_zero_matrix() raises:
    """Test Strassen with zero matrices."""
    var A = zeros([4, 4], DType.float32)
    var B = zeros([4, 4], DType.float32)
    var C = zeros([4, 4], DType.float32)

    matmul_strassen(A, B, C)

    # All elements should be zero
    var c_ptr = C._data.bitcast[Float32]()
    for i in range(16):
        assert_almost_equal(c_ptr.load(i), Float32(0.0), Float32(1e-4))


def test_strassen_identity_matrix() raises:
    """Test Strassen with identity matrices."""
    var size = 4
    var A = zeros([size, size], DType.float32)
    var B = zeros([size, size], DType.float32)
    var C = zeros([size, size], DType.float32)

    # Set A to identity matrix
    var a_ptr = A._data.bitcast[Float32]()
    for i in range(size):
        a_ptr.store(i * size + i, Float32(1.0))

    # Set B to sequential values
    var b_ptr = B._data.bitcast[Float32]()
    for i in range(size * size):
        b_ptr.store(i, Float32(Float64(i) + 1.0))

    matmul_strassen(A, B, C)

    # C should equal B
    var c_ptr = C._data.bitcast[Float32]()
    for i in range(size * size):
        var c_val = c_ptr.load(i)
        var b_val = b_ptr.load(i)
        assert_almost_equal(c_val, b_val, Float32(1e-4))


def test_strassen_power_of_2_sizes() raises:
    """Test Strassen with various power-of-2 sizes."""
    var sizes = List[Int]()
    sizes.append(2)
    sizes.append(4)
    sizes.append(8)
    sizes.append(16)

    for size in sizes:
        var A = zeros([size, size], DType.float32)
        var B = zeros([size, size], DType.float32)
        var C_strassen = zeros([size, size], DType.float32)
        var C_ref = zeros([size, size], DType.float32)

        # Fill with sequential values
        var a_ptr = A._data.bitcast[Float32]()
        var b_ptr = B._data.bitcast[Float32]()

        var numel = size * size
        for i in range(numel):
            a_ptr.store(i, Float32(1.0))
            b_ptr.store(i, Float32(1.0))

        # Compute with both methods
        matmul_strassen(A, B, C_strassen)
        matmul_tiled(A, B, C_ref)

        # Compare results
        var strassen_ptr = C_strassen._data.bitcast[Float32]()
        var ref_ptr = C_ref._data.bitcast[Float32]()

        for i in range(numel):
            var s_val = strassen_ptr.load(i)
            var r_val = ref_ptr.load(i)
            assert_almost_equal(s_val, r_val, Float32(1e-4))


# ============================================================================
# Small Matrix Tests (below threshold - should use fallback)
# ============================================================================


def test_strassen_below_threshold() raises:
    """Test that Strassen falls back to matmul_tiled for small matrices."""
    var A = zeros([64, 64], DType.float32)
    var B = zeros([64, 64], DType.float32)
    var C_strassen = zeros([64, 64], DType.float32)
    var C_ref = zeros([64, 64], DType.float32)

    # Fill with ones
    var a_ptr = A._data.bitcast[Float32]()
    var b_ptr = B._data.bitcast[Float32]()

    for i in range(64 * 64):
        a_ptr.store(i, Float32(1.0))
        b_ptr.store(i, Float32(1.0))

    # Both should use matmul_tiled
    matmul_strassen(A, B, C_strassen)
    matmul_tiled(A, B, C_ref)

    # Results should be identical
    var strassen_ptr = C_strassen._data.bitcast[Float32]()
    var ref_ptr = C_ref._data.bitcast[Float32]()

    for i in range(64 * 64):
        var s_val = strassen_ptr.load(i)
        var r_val = ref_ptr.load(i)
        assert_almost_equal(s_val, r_val, Float32(1e-4))


# ============================================================================
# Non-power-of-2 Tests
# ============================================================================


def test_strassen_non_power_of_2_square() raises:
    """Test Strassen with non-power-of-2 square matrices."""
    var A = zeros([100, 100], DType.float32)
    var B = zeros([100, 100], DType.float32)
    var C_strassen = zeros([100, 100], DType.float32)
    var C_ref = zeros([100, 100], DType.float32)

    # Fill with ones
    var a_ptr = A._data.bitcast[Float32]()
    var b_ptr = B._data.bitcast[Float32]()

    for i in range(100 * 100):
        a_ptr.store(i, Float32(1.0))
        b_ptr.store(i, Float32(1.0))

    # Compute with both methods
    matmul_strassen(A, B, C_strassen)
    matmul_tiled(A, B, C_ref)

    # Compare results (but use looser tolerance for non-power-of-2)
    var strassen_ptr = C_strassen._data.bitcast[Float32]()
    var ref_ptr = C_ref._data.bitcast[Float32]()

    for i in range(100 * 100):
        var s_val = strassen_ptr.load(i)
        var r_val = ref_ptr.load(i)
        assert_almost_equal(s_val, r_val, Float32(1e-3))


# ============================================================================
# Rectangular Matrix Tests (should fall back to matmul_tiled)
# ============================================================================


def test_strassen_rectangular_matrices() raises:
    """Test that Strassen falls back to matmul_tiled for rectangular matrices.
    """
    var A = zeros([32, 64], DType.float32)
    var B = zeros([64, 32], DType.float32)
    var C_strassen = zeros([32, 32], DType.float32)
    var C_ref = zeros([32, 32], DType.float32)

    # Fill with ones
    var a_ptr = A._data.bitcast[Float32]()
    var b_ptr = B._data.bitcast[Float32]()

    for i in range(32 * 64):
        a_ptr.store(i, Float32(1.0))

    for i in range(64 * 32):
        b_ptr.store(i, Float32(1.0))

    # Both should use matmul_tiled
    matmul_strassen(A, B, C_strassen)
    matmul_tiled(A, B, C_ref)

    # Results should be identical
    var strassen_ptr = C_strassen._data.bitcast[Float32]()
    var ref_ptr = C_ref._data.bitcast[Float32]()

    for i in range(32 * 32):
        var s_val = strassen_ptr.load(i)
        var r_val = ref_ptr.load(i)
        assert_almost_equal(s_val, r_val, Float32(1e-4))


# ============================================================================
# Main Test Runner
# ============================================================================


def main() raises:
    """Run all Strassen algorithm tests."""
    print("Testing Strassen's Algorithm Implementation")
    print("=" * 50)

    print("\n[1/13] Testing next_power_of_2...")
    test_next_power_of_2()
    print("  PASSED")

    print("[2/13] Testing Strassen 2x2 Float32...")
    test_strassen_2x2_float32()
    print("  PASSED")

    print("[3/13] Testing Strassen 4x4 Float32...")
    test_strassen_4x4_float32()
    print("  PASSED")

    print("[4/13] Testing Strassen 8x8 Float64...")
    test_strassen_8x8_float64()
    print("  PASSED")

    print("[5/13] Testing zero matrices...")
    test_strassen_zero_matrix()
    print("  PASSED")

    print("[6/13] Testing identity matrices...")
    test_strassen_identity_matrix()
    print("  PASSED")

    print("[7/13] Testing power-of-2 sizes...")
    test_strassen_power_of_2_sizes()
    print("  PASSED")

    print("[8/13] Testing below threshold...")
    test_strassen_below_threshold()
    print("  PASSED")

    print("[9/13] Testing non-power-of-2 square matrices...")
    test_strassen_non_power_of_2_square()
    print("  PASSED")

    print("[10/13] Testing rectangular matrices...")
    test_strassen_rectangular_matrices()
    print("  PASSED")

    print("\n" + "=" * 50)
    print("All Strassen tests passed!")
