"""Tests for matrix operations - Part 7: Dot Product.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Dot product shapes (scalar output)
- Dot product values
- Dot product of orthogonal vectors
- Dot product 1D vectors
- Dot product 2D (matmul equivalent)
- Dot incompatible shapes error
- Dot dtype mismatch error
- Dot dtype preservation
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
# Dot Product Tests
# ============================================================================


fn test_dot_shapes() raises:
    """Test that dot returns scalar output."""
    var shape = List[Int]()
    shape.append(5)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = dot(a, b)

    # Dot product returns scalar (0D tensor with empty shape)
    assert_dim(result, 0, "Dot product should be 0D scalar")
    assert_numel(result, 1, "Dot product should have 1 element")


fn test_dot_values() raises:
    """Test that dot computes correct values."""
    var shape = List[Int]()
    shape.append(3)

    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0

    b._data.bitcast[Float32]()[0] = 4.0
    b._data.bitcast[Float32]()[1] = 5.0
    b._data.bitcast[Float32]()[2] = 6.0

    var result = dot(a, b)

    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(32.0), tolerance=1e-5
    )


fn test_dot_orthogonal() raises:
    """Test dot product of orthogonal vectors."""
    var shape = List[Int]()
    shape.append(2)

    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # a = [1, 0], b = [0, 1]
    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 0.0

    b._data.bitcast[Float32]()[0] = 0.0
    b._data.bitcast[Float32]()[1] = 1.0

    var result = dot(a, b)

    # Orthogonal vectors have dot product = 0
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


fn test_dot_1d() raises:
    """Test dot product of two 1D vectors."""
    var a = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    var b = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    var c = dot(a, b)

    # Expected: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
    assert_dim(c, 0, "Dot product should be scalar (0D)")
    assert_numel(c, 1, "Dot product should have 1 element")
    assert_value_at(c, 0, 55.0, 1e-4, "Dot product result")


fn test_dot_2d() raises:
    """Test dot product (equivalent to matmul for 2D)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)  # 2x3
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(2)
    var b = ones(shape_b, DType.float32)  # 3x2
    var c = dot(a, b)

    # Should behave like matmul for 2D
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4, "Result should be 2x2 (4 elements)")
    assert_all_values(c, 3.0, 1e-6, "Each element should be 3")


fn test_dot_incompatible_shapes() raises:
    """Test that incompatible 1D shapes raise error."""
    var shape_a = List[Int]()
    shape_a.append(5)
    var shape_b = List[Int]()
    shape_b.append(3)  # Different size

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        var c = dot(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error(
            "Should have raised error for incompatible dot shapes (5,) and (3,)"
        )


fn test_dot_dtype_mismatch() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)

    var error_raised = False
    try:
        var c = dot(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in dot")


fn test_dot_preserves_dtype() raises:
    """Test that dot preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float64)
    var b = ones(shape, DType.float64)
    var c = dot(a, b)

    assert_dtype(c, DType.float64, "dot should preserve float64")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run dot product tests."""
    print("Running matrix operation tests - Part 7: Dot Product...")

    print("\n=== Dot Product ===")
    test_dot_shapes()
    print("✓ test_dot_shapes")
    test_dot_values()
    print("✓ test_dot_values")
    test_dot_orthogonal()
    print("✓ test_dot_orthogonal")
    test_dot_1d()
    print("✓ test_dot_1d")
    test_dot_2d()
    print("✓ test_dot_2d")
    test_dot_incompatible_shapes()
    print("✓ test_dot_incompatible_shapes")
    test_dot_dtype_mismatch()
    print("✓ test_dot_dtype_mismatch")
    test_dot_preserves_dtype()
    print("✓ test_dot_preserves_dtype")

    print("\n" + "=" * 60)
    print("All 8 tests passed! (Part 7)")
