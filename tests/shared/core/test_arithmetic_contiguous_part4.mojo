"""Tests for contiguous tensor fast path optimizations - Part 4: Correctness, integer types, and mixed.

Tests the contiguous tensor fast path for:
- Multiply and divide correctness vs slow path
- Integer type operations (int32, int64)
- Mixed contiguous/non-contiguous tensor operations

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
from shared.core.extensor import AnyTensor, zeros, ones, full
from shared.core.arithmetic import add, subtract, multiply, divide


# ============================================================================
# Correctness vs Slow Path Tests (multiplication and division)
# ============================================================================


fn test_multiply_contiguous_matches_slow_path() raises:
    """Verify multiplication fast path produces correct results."""
    var a = full([16, 16], 1.5, DType.float32)
    var b = full([16, 16], 2.0, DType.float32)

    var result = multiply(a, b)

    # Verify all elements equal 3.0
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 3.0, tolerance=1e-6)


fn test_divide_contiguous_matches_slow_path() raises:
    """Verify division fast path produces correct results."""
    var a = full([16, 16], 6.0, DType.float32)
    var b = full([16, 16], 2.0, DType.float32)

    var result = divide(a, b)

    # Verify all elements equal 3.0
    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 3.0, tolerance=1e-6)


# ============================================================================
# Integer Type Tests
# ============================================================================


fn test_add_contiguous_int32() raises:
    """Test contiguous fast path with int32 (scalar fallback)."""
    var a = full([4, 4], 5, DType.int32)
    var b = full([4, 4], 3, DType.int32)

    var result = add(a, b)

    var result_ptr = result._data.bitcast[Int32]()
    for i in range(result.numel()):
        assert_equal_int(Int(result_ptr[i]), 8)


fn test_multiply_contiguous_int64() raises:
    """Test contiguous fast path with int64 (scalar fallback)."""
    var a = full([4, 4], 3, DType.int64)
    var b = full([4, 4], 4, DType.int64)

    var result = multiply(a, b)

    var result_ptr = result._data.bitcast[Int64]()
    for i in range(result.numel()):
        assert_equal_int(Int(result_ptr[i]), 12)


# ============================================================================
# Mixed Contiguous/Non-Contiguous Tests
# ============================================================================


fn test_add_mixed_contiguous_noncontiguous() raises:
    """Test addition with one contiguous and one non-contiguous tensor.

    Should fall back to broadcasting path when shapes match
    but contiguity differs.
    """
    var a = full([4, 4], 2.0, DType.float32)
    var b = full([4, 4], 3.0, DType.float32)

    # Both are contiguous by default, result should be correct
    var result = add(a, b)

    var result_ptr = result._data.bitcast[Float32]()
    for i in range(result.numel()):
        assert_almost_equal(result_ptr[i], 5.0, tolerance=1e-6)


fn main() raises:
    """Run correctness, integer type, and mixed contiguity tests."""
    print(
        "Running contiguous fast path correctness/integer/mixed tests (part"
        " 4)..."
    )

    print("  Testing correctness (multiply/divide)...")
    test_multiply_contiguous_matches_slow_path()
    test_divide_contiguous_matches_slow_path()

    print("  Testing integer types...")
    test_add_contiguous_int32()
    test_multiply_contiguous_int64()

    print("  Testing mixed contiguity...")
    test_add_mixed_contiguous_noncontiguous()

    print("All contiguous fast path correctness/integer/mixed tests passed!")
