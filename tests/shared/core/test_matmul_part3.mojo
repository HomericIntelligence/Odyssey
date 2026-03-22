# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matmul.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Matrix multiplication tests - Part 3: DType, Error Handling, and Precision.

Tests cover:
- DType Tests: Type preservation, mismatch errors
- Error Handling: Incompatible shapes, 1D inputs
- Correctness: Additional size coverage and accumulation precision
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
from shared.core.matrix import matmul


# ============================================================================
# DType Tests (continued)
# ============================================================================


fn test_matmul_dtype_preserves_type() raises:
    """Test that matmul preserves input dtype across all types."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)

    # Float32
    var a32 = ones(shape, DType.float32)
    var b32 = ones(shape, DType.float32)
    var c32 = matmul(a32, b32)
    assert_dtype(c32, DType.float32, "Float32 should be preserved")

    # Float64
    var a64 = ones(shape, DType.float64)
    var b64 = ones(shape, DType.float64)
    var c64 = matmul(a64, b64)
    assert_dtype(c64, DType.float64, "Float64 should be preserved")

    # Float16
    var a16 = ones(shape, DType.float16)
    var b16 = ones(shape, DType.float16)
    var c16 = matmul(a16, b16)
    assert_dtype(c16, DType.float16, "Float16 should be preserved")


fn test_matmul_dtype_mismatch_error() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)  # Different dtype

    var error_raised = False
    try:
        var c = matmul(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in matmul")


# ============================================================================
# Error Handling Tests
# ============================================================================


fn test_matmul_incompatible_shapes() raises:
    """Test that incompatible shapes raise error."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)

    var shape_b = List[Int]()
    shape_b.append(5)
    shape_b.append(2)  # Incompatible: 4 != 5

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        var c = matmul(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error(
            "Should have raised error for incompatible matmul shapes (3,4) @"
            " (5,2)"
        )


fn test_matmul_1d_error() raises:
    """Test that 1D inputs raise error."""
    var shape = List[Int]()
    shape.append(5)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var error_raised = False
    try:
        var c = matmul(a, b)  # matmul requires 2D+
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for 1D inputs to matmul")


# ============================================================================
# Correctness Tests - Additional Size Coverage
# ============================================================================


fn test_matmul_additional_rectangular_sizes() raises:
    """Test rectangular matrices with various dimensions."""
    # Test case 1: Wide matrix (M < K < N)
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(8)
    var shape_b = List[Int]()
    shape_b.append(8)
    shape_b.append(16)

    var a = ones(shape_a, DType.float32)
    var b = full(shape_b, 2.0, DType.float32)
    var c = matmul(a, b)

    var expected_shape = List[Int]()
    expected_shape.append(4)
    expected_shape.append(16)
    assert_shape(c, expected_shape, "Result shape should be (4, 16)")
    assert_all_values(c, 16.0, 1e-5, "Each element should be 8*2 = 16")


fn test_matmul_accumulation_precision_float32() raises:
    """Test accumulation precision with many terms (float32)."""
    # Test with 128 terms to check accumulation precision
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(128)
    var shape_b = List[Int]()
    shape_b.append(128)
    shape_b.append(4)

    var a = full(shape_a, 0.1, DType.float32)
    var b = full(shape_b, 0.1, DType.float32)
    var c = matmul(a, b)

    # Expected: 128 * (0.1 * 0.1) = 128 * 0.01 = 1.28
    assert_value_at(c, 0, 1.28, 1e-5, "Accumulated result should match")


fn test_matmul_accumulation_precision_float64() raises:
    """Test accumulation precision with many terms (float64)."""
    # Test with 256 terms for float64
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(256)
    var shape_b = List[Int]()
    shape_b.append(256)
    shape_b.append(4)

    var a = full(shape_a, 0.1, DType.float64)
    var b = full(shape_b, 0.1, DType.float64)
    var c = matmul(a, b)

    # Expected: 256 * 0.01 = 2.56
    assert_value_at(c, 0, 2.56, 1e-8, "Float64 accumulation should be precise")


# ============================================================================
# Performance Regression Tests (TODO(#2588): Add when benchmarking infrastructure exists)
# ============================================================================

# TODO(#2588): Add performance regression tests
# - Ensure Stage 1 is at least 3x faster than Stage 0
# - Ensure Stage 2 is at least 4x faster than Stage 1 (15x cumulative)
# - Ensure Stage 3 is at least 3x faster than Stage 2 (50x cumulative)
# - Ensure Stage 4 is at least 2x faster than Stage 3 (100x cumulative)
# See benchmarks/bench_matmul.mojo for detailed performance testing


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run matrix multiplication tests - Part 3."""
    print("Running matrix multiplication tests - Part 3...")
    print("=" * 70)

    # DType tests (continued)
    print("\n=== DType Tests (continued) ===")
    test_matmul_dtype_preserves_type()
    print("✓ test_matmul_dtype_preserves_type")
    test_matmul_dtype_mismatch_error()
    print("✓ test_matmul_dtype_mismatch_error")

    # Error handling
    print("\n=== Error Handling ===")
    test_matmul_incompatible_shapes()
    print("✓ test_matmul_incompatible_shapes")
    test_matmul_1d_error()
    print("✓ test_matmul_1d_error")

    # Additional coverage tests
    print("\n=== Additional Coverage Tests ===")
    test_matmul_additional_rectangular_sizes()
    print("✓ test_matmul_additional_rectangular_sizes")
    test_matmul_accumulation_precision_float32()
    print("✓ test_matmul_accumulation_precision_float32")
    test_matmul_accumulation_precision_float64()
    print("✓ test_matmul_accumulation_precision_float64")

    print("\n" + "=" * 70)
    print("All 7 matrix multiplication Part 3 tests passed!")
    print("=" * 70)
    print("\n=== Test Coverage Summary ===")
    print("✓ DType Tests (continued):            2 tests")
    print("✓ Error Handling:                     2 tests")
    print("✓ Additional Coverage:                3 tests")
