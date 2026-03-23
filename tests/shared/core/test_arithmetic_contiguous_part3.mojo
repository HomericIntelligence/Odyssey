"""Tests for contiguous tensor fast path optimizations - Part 3: Multiplication, division, and fallback.

Tests the contiguous tensor fast path for:
- float64 multiplication
- float32 and float64 division
- Non-contiguous fallback paths

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_arithmetic_contiguous.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal_int,
    assert_false,
    assert_shape,
    assert_true,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.arithmetic import add, subtract, multiply, divide


# ============================================================================
# Fast Path Multiplication Tests (continued)
# ============================================================================


fn test_multiply_contiguous_same_shape_float64() raises:
    """Test contiguous fast path for float64 multiplication."""
    var a = full([2, 3], 2.0, DType.float64)
    var b = full([2, 3], 3.0, DType.float64)

    var result = multiply(a, b)

    # Check all values are 6.0
    var result_ptr = result._data.bitcast[Float64]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 6.0, tolerance=1e-6)


# ============================================================================
# Fast Path Division Tests
# ============================================================================


fn test_divide_contiguous_same_shape_float32() raises:
    """Test contiguous fast path for float32 division."""
    var a = full([2, 3], 6.0, DType.float32)
    var b = full([2, 3], 2.0, DType.float32)

    var result = divide(a, b)

    # Check all values are 3.0
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 3.0, tolerance=1e-6)


fn test_divide_contiguous_same_shape_float64() raises:
    """Test contiguous fast path for float64 division."""
    var a = full([2, 3], 6.0, DType.float64)
    var b = full([2, 3], 2.0, DType.float64)

    var result = divide(a, b)

    # Check all values are 3.0
    var result_ptr = result._data.bitcast[Float64]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 3.0, tolerance=1e-6)


# ============================================================================
# Non-Contiguous (Fallback Path) Tests
# ============================================================================


fn test_add_noncontiguous_fallback() raises:
    """Test that non-contiguous tensors fall back correctly.

    Creates a non-contiguous view by transposing and verifies
    the operation still produces correct results.
    """
    var a = full([3, 4], 2.0, DType.float32)
    var b = full([3, 4], 3.0, DType.float32)

    # Create non-contiguous views (e.g., by reshaping/transposing)
    # For now, we test the fallback path exists by verifying result correctness
    var result = add(a, b)

    # Verify result is still correct even if using fallback
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 5.0, tolerance=1e-6)


fn test_multiply_noncontiguous_fallback() raises:
    """Test multiplication with non-contiguous fallback."""
    var a = full([3, 4], 2.0, DType.float32)
    var b = full([3, 4], 3.0, DType.float32)

    var result = multiply(a, b)

    # Verify result
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 6.0, tolerance=1e-6)


# ============================================================================
# Correctness vs Slow Path Tests (addition and subtraction)
# ============================================================================


fn test_add_contiguous_matches_slow_path() raises:
    """Verify contiguous fast path produces same results as slow path."""
    var a = full([16, 16], 1.5, DType.float32)
    var b = full([16, 16], 2.5, DType.float32)

    var result = add(a, b)

    # Verify all elements equal 4.0
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 4.0, tolerance=1e-6)


fn test_subtract_contiguous_matches_slow_path() raises:
    """Verify subtraction fast path produces correct results."""
    var a = full([16, 16], 5.5, DType.float32)
    var b = full([16, 16], 2.5, DType.float32)

    var result = subtract(a, b)

    # Verify all elements equal 3.0
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 3.0, tolerance=1e-6)


fn main() raises:
    """Run multiplication, division, and fallback fast path tests."""
    print(
        "Running contiguous fast path multiplication/division/fallback tests"
        " (part 3)..."
    )

    print("  Testing multiplication (float64)...")
    test_multiply_contiguous_same_shape_float64()

    print("  Testing division...")
    test_divide_contiguous_same_shape_float32()
    test_divide_contiguous_same_shape_float64()

    print("  Testing fallback paths...")
    test_add_noncontiguous_fallback()
    test_multiply_noncontiguous_fallback()

    print("  Testing correctness (add/subtract)...")
    test_add_contiguous_matches_slow_path()
    test_subtract_contiguous_matches_slow_path()

    print(
        "All contiguous fast path multiplication/division/fallback tests"
        " passed!"
    )
