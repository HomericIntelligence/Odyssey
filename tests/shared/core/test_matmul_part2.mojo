# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matmul.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Matrix multiplication tests - Part 2: Edge Cases and DType Tests.

Tests cover:
- Edge Cases: Non-power-of-2, block size, large rectangular
- DType Tests: Float32, Float64, Float16
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
# Edge Case Tests (continued)
# ============================================================================


fn test_matmul_smaller_than_simd_7x7x7() raises:
    """Test matrices smaller than SIMD width (7x7)."""
    var shape = List[Int]()
    shape.append(7)
    shape.append(7)

    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = matmul(a, b)

    # Each element = 1*2 + ... (7 times) = 14
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 49, "Result should be 7x7")
    assert_all_values(c, 14.0, 1e-5, "Each element should be 14")


fn test_matmul_non_power_of_2() raises:
    """Test non-power-of-2 sizes (63x65) @ (65x67)."""
    var shape_a = List[Int]()
    shape_a.append(63)
    shape_a.append(65)

    var shape_b = List[Int]()
    shape_b.append(65)
    shape_b.append(67)

    var a = ones(shape_a, DType.float32)
    var b = full(shape_b, 0.5, DType.float32)
    var c = matmul(a, b)

    # Each element = 1*0.5 + ... (65 times) = 32.5
    assert_dim(c, 2, "Result should be 2D")
    assert_equal(c.shape()[0], 63, "First dimension should be 63")
    assert_equal(c.shape()[1], 67, "Second dimension should be 67")
    assert_numel(c, 4221, "Result should be 63x67")

    # Check first and last elements
    assert_value_at(c, 0, 32.5, 1e-4, "First element should be 32.5")
    assert_value_at(c, 4220, 32.5, 1e-4, "Last element should be 32.5")


fn test_matmul_exact_block_size_64x64x64() raises:
    """Test matrices matching typical cache block size (64x64)."""
    var shape = List[Int]()
    shape.append(64)
    shape.append(64)

    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = matmul(a, b)

    # Each element = 1*2 + ... (64 times) = 128
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4096, "Result should be 64x64")
    assert_all_values(c, 128.0, 1e-4, "Each element should be 128")


fn test_matmul_large_rectangular() raises:
    """Test large rectangular matrices (1024x512) @ (512x2048)."""
    var shape_a = List[Int]()
    shape_a.append(1024)
    shape_a.append(512)

    var shape_b = List[Int]()
    shape_b.append(512)
    shape_b.append(2048)

    var a = full(shape_a, 0.1, DType.float32)
    var b = full(shape_b, 0.2, DType.float32)
    var c = matmul(a, b)

    # Each element = 0.1*0.2 + ... (512 times) = 10.24
    assert_dim(c, 2, "Result should be 2D")
    assert_equal(c.shape()[0], 1024, "First dimension should be 1024")
    assert_equal(c.shape()[1], 2048, "Second dimension should be 2048")

    # Check spot values (avoid checking all 2M+ elements)
    assert_value_at(c, 0, 10.24, 1e-3, "First element should be 10.24")
    assert_value_at(c, 2048, 10.24, 1e-3, "Second row first element")
    # Note: Using lower tolerance for accumulated floating-point errors


# ============================================================================
# DType Tests - Float32, Float64, Float16
# ============================================================================


fn test_matmul_dtype_float32() raises:
    """Test matmul with Float32 dtype."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)

    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = matmul(a, b)

    assert_dtype(c, DType.float32, "Result should be Float32")
    assert_all_values(c, 8.0, 1e-5, "Each element should be 8.0")


fn test_matmul_dtype_float64() raises:
    """Test matmul with Float64 dtype."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)

    var a = ones(shape, DType.float64)
    var b = full(shape, 2.0, DType.float64)
    var c = matmul(a, b)

    assert_dtype(c, DType.float64, "Result should be Float64")
    assert_all_values(c, 8.0, 1e-8, "Each element should be 8.0")


fn test_matmul_dtype_float16() raises:
    """Test matmul with Float16 dtype."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)

    var a = ones(shape, DType.float16)
    var b = full(shape, 2.0, DType.float16)
    var c = matmul(a, b)

    assert_dtype(c, DType.float16, "Result should be Float16")
    # Note: Float16 has lower precision, use looser tolerance
    assert_all_values(c, 8.0, 1e-2, "Each element should be ~8.0")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run matrix multiplication tests - Part 2."""
    print("Running matrix multiplication tests - Part 2...")
    print("=" * 70)

    # Edge case tests (continued)
    print("\n=== Edge Cases (continued) ===")
    test_matmul_smaller_than_simd_7x7x7()
    print("✓ test_matmul_smaller_than_simd_7x7x7 (smaller than SIMD width)")
    test_matmul_non_power_of_2()
    print("✓ test_matmul_non_power_of_2 (63x65 @ 65x67)")
    test_matmul_exact_block_size_64x64x64()
    print("✓ test_matmul_exact_block_size_64x64x64 (exact cache block size)")
    test_matmul_large_rectangular()
    print("✓ test_matmul_large_rectangular (1024x512 @ 512x2048)")

    # DType tests
    print("\n=== DType Tests ===")
    test_matmul_dtype_float32()
    print("✓ test_matmul_dtype_float32")
    test_matmul_dtype_float64()
    print("✓ test_matmul_dtype_float64")
    test_matmul_dtype_float16()
    print("✓ test_matmul_dtype_float16")

    print("\n" + "=" * 70)
    print("All 7 matrix multiplication Part 2 tests passed!")
    print("=" * 70)
    print("\n=== Test Coverage Summary ===")
    print("✓ Edge Cases (continued):             4 tests")
    print("✓ DType Tests (partial):              3 tests")
