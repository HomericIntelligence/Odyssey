"""Tests for contiguous tensor fast path optimizations - Part 2: Addition and subtraction.

Tests the contiguous tensor fast path for addition and subtraction operations:
- float32 and float64 addition with contiguous same-shape tensors
- Large and small tensor addition
- float32 and float64 subtraction

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
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.arithmetic import add, subtract, multiply, divide


# ============================================================================
# Fast Path Addition Tests
# ============================================================================


fn test_add_contiguous_same_shape_float32() raises:
    """Test contiguous fast path for float32 addition."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)

    # Verify contiguity
    assert_true(a.is_contiguous(), "a should be contiguous")
    assert_true(b.is_contiguous(), "b should be contiguous")

    # Perform addition
    var result = add(a, b)

    # Verify result
    assert_equal_int(result.shape()[0], 2)
    assert_equal_int(result.shape()[1], 3)

    # Check all values are 5.0
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 5.0, tolerance=1e-6)


fn test_add_contiguous_same_shape_float64() raises:
    """Test contiguous fast path for float64 addition."""
    var a = full([2, 3], 2.0, DType.float64)
    var b = full([2, 3], 3.0, DType.float64)

    # Verify contiguity
    assert_true(a.is_contiguous(), "a should be contiguous")
    assert_true(b.is_contiguous(), "b should be contiguous")

    # Perform addition
    var result = add(a, b)

    # Verify result
    assert_equal_int(result.shape()[0], 2)
    assert_equal_int(result.shape()[1], 3)

    # Check all values are 5.0
    var result_ptr = result._data.bitcast[Float64]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 5.0, tolerance=1e-6)


fn test_add_contiguous_large_tensor() raises:
    """Test contiguous fast path with large tensor (1024x1024)."""
    var a = full([1024, 1024], 2.0, DType.float32)
    var b = full([1024, 1024], 3.0, DType.float32)

    # Verify contiguity
    assert_true(a.is_contiguous(), "a should be contiguous")
    assert_true(b.is_contiguous(), "b should be contiguous")

    # Perform addition
    var result = add(a, b)

    # Spot check a few values
    var result_ptr = result._data.bitcast[Float32]()
    assert_almost_equal(result_ptr[0], 5.0, tolerance=1e-6)
    assert_almost_equal(result_ptr[1000000], 5.0, tolerance=1e-6)


fn test_add_contiguous_small_tensor() raises:
    """Test contiguous fast path with small tensor."""
    var a = full([2], 2.0, DType.float32)
    var b = full([2], 3.0, DType.float32)

    # Verify contiguity
    assert_true(a.is_contiguous(), "a should be contiguous")
    assert_true(b.is_contiguous(), "b should be contiguous")

    var result = add(a, b)

    # Check values
    var result_ptr = result._data.bitcast[Float32]()
    assert_almost_equal(result_ptr[0], 5.0, tolerance=1e-6)
    assert_almost_equal(result_ptr[1], 5.0, tolerance=1e-6)


# ============================================================================
# Fast Path Subtraction Tests
# ============================================================================


fn test_subtract_contiguous_same_shape_float32() raises:
    """Test contiguous fast path for float32 subtraction."""
    var a = full([2, 3], 5.0, DType.float32)
    var b = full([2, 3], 2.0, DType.float32)

    # Verify contiguity
    assert_true(a.is_contiguous(), "a should be contiguous")
    assert_true(b.is_contiguous(), "b should be contiguous")

    var result = subtract(a, b)

    # Check all values are 3.0
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 3.0, tolerance=1e-6)


fn test_subtract_contiguous_same_shape_float64() raises:
    """Test contiguous fast path for float64 subtraction."""
    var a = full([2, 3], 5.0, DType.float64)
    var b = full([2, 3], 2.0, DType.float64)

    var result = subtract(a, b)

    # Check all values are 3.0
    var result_ptr = result._data.bitcast[Float64]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 3.0, tolerance=1e-6)


fn test_multiply_contiguous_same_shape_float32() raises:
    """Test contiguous fast path for float32 multiplication."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)

    var result = multiply(a, b)

    # Check all values are 6.0
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 6.0, tolerance=1e-6)


fn main() raises:
    """Run all addition and subtraction fast path tests."""
    print("Running contiguous fast path addition/subtraction tests (part 2)...")

    print("  Testing addition...")
    test_add_contiguous_same_shape_float32()
    test_add_contiguous_same_shape_float64()
    test_add_contiguous_large_tensor()
    test_add_contiguous_small_tensor()

    print("  Testing subtraction...")
    test_subtract_contiguous_same_shape_float32()
    test_subtract_contiguous_same_shape_float64()

    print("  Testing multiplication (float32)...")
    test_multiply_contiguous_same_shape_float32()

    print("All contiguous fast path addition/subtraction tests passed!")
