"""Tests for matrix operations - Part 6: Transpose Axes Advanced.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Transpose axes default None (reverse all)
- Transpose axes 4D permutation
- Transpose axes invalid duplicate
- Transpose axes invalid out of bounds
- Transpose axes invalid length
- Transpose axes backward 3D
- Transpose axes double permutation (recovers original)
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
    ExTensor,
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
# Transpose Tests - Custom Axes Permutation (Issue #2389) continued
# ============================================================================


fn test_transpose_axes_default_none() raises:
    """Test transpose with axes=None uses default (reverse all)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)  # 2x3x4

    # Explicit default: axes=None
    var result = transpose(t)

    # Should reverse all axes: (2, 3, 4) -> (4, 3, 2)
    assert_dim(result, 3, "Result should be 3D")
    assert_equal(result.shape()[0], 4, "First dimension should be 4")
    assert_equal(result.shape()[1], 3, "Second dimension should be 3")
    assert_equal(result.shape()[2], 2, "Third dimension should be 2")


fn test_transpose_axes_4d_permutation() raises:
    """Test 4D transpose with custom permutation [3, 1, 2, 0]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    shape.append(5)

    var t = ones(shape, DType.float32)  # 2x3x4x5

    # Permutation [3, 1, 2, 0]: (2, 3, 4, 5) -> (5, 3, 4, 2)
    var axes = List[Int]()
    axes.append(3)
    axes.append(1)
    axes.append(2)
    axes.append(0)

    var result = transpose(t, axes^)

    # Result shape should be (5, 3, 4, 2)
    assert_dim(result, 4, "Result should be 4D")
    assert_equal(result.shape()[0], 5, "First dimension should be 5")
    assert_equal(result.shape()[1], 3, "Second dimension should be 3")
    assert_equal(result.shape()[2], 4, "Third dimension should be 4")
    assert_equal(result.shape()[3], 2, "Fourth dimension should be 2")

    # Verify element count
    assert_numel(result, 120, "Result should have 120 elements")


fn test_transpose_axes_invalid_duplicate() raises:
    """Test that duplicate axes raise error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)

    # Invalid: duplicate axis 0
    var axes = List[Int]()
    axes.append(0)
    axes.append(0)
    axes.append(1)

    var error_raised = False
    try:
        var result = transpose(t, axes^)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for duplicate axes")


fn test_transpose_axes_invalid_out_of_bounds() raises:
    """Test that out-of-bounds axes raise error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)

    # Invalid: axis 5 is out of bounds (only 3 dimensions)
    var axes = List[Int]()
    axes.append(0)
    axes.append(1)
    axes.append(5)

    var error_raised = False
    try:
        var result = transpose(t, axes^)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for out-of-bounds axis")


fn test_transpose_axes_invalid_length() raises:
    """Test that wrong-length axes raise error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)

    # Invalid: only 2 axes provided for 3D tensor
    var axes = List[Int]()
    axes.append(0)
    axes.append(1)

    var error_raised = False
    try:
        var result = transpose(t, axes^)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for wrong-length axes")


fn test_transpose_axes_backward_3d() raises:
    """Test transpose_backward with custom axes."""
    var m = 2
    var n = 3
    var p = 4

    # Create 3D input with shape (m, n, p)
    var shape = List[Int]()
    shape.append(m)
    shape.append(n)
    shape.append(p)
    var x = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(m * n * p):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 2.0

    # Forward with axes [2, 0, 1]: (2, 3, 4) -> (4, 2, 3)
    var axes = List[Int]()
    axes.append(2)
    axes.append(0)
    axes.append(1)

    var output = transpose(x, axes^)
    var grad_output = ones_like(output)

    # Compute backward - need new axes list since original was transferred
    var backward_axes = List[Int]()
    backward_axes.append(2)
    backward_axes.append(0)
    backward_axes.append(1)
    var grad_input = transpose_backward(grad_output, backward_axes^)

    # Gradient should have same shape as input
    assert_equal(grad_input.shape()[0], m, "Gradient dim 0 should match input")
    assert_equal(grad_input.shape()[1], n, "Gradient dim 1 should match input")
    assert_equal(grad_input.shape()[2], p, "Gradient dim 2 should match input")


fn test_transpose_axes_double_permutation() raises:
    """Test that transpose(transpose(x, axes), inverse_axes) recovers original.
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = zeros(shape, DType.float32)

    # Fill with sequential values
    for i in range(24):
        t._data.bitcast[Float32]()[i] = Float32(i)

    # Forward permutation [2, 0, 1]
    var axes = List[Int]()
    axes.append(2)
    axes.append(0)
    axes.append(1)

    var t_perm = transpose(t, axes^)

    # Compute inverse permutation
    # For axes [2, 0, 1], inverse is [1, 2, 0]
    var inverse_axes = List[Int]()
    inverse_axes.append(1)
    inverse_axes.append(2)
    inverse_axes.append(0)

    var t_recovered = transpose(t_perm, inverse_axes^)

    # Should recover original shape
    assert_equal(t_recovered.shape()[0], 2, "Recovered dim 0 should be 2")
    assert_equal(t_recovered.shape()[1], 3, "Recovered dim 1 should be 3")
    assert_equal(t_recovered.shape()[2], 4, "Recovered dim 2 should be 4")

    # Check values are recovered
    for i in range(24):
        assert_almost_equal(
            t_recovered._data.bitcast[Float32]()[i],
            t._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run advanced transpose axes tests."""
    print("Running matrix operation tests - Part 6: Transpose Axes Advanced...")

    print("\n=== Transpose: Custom Axes Permutation (continued) ===")
    test_transpose_axes_default_none()
    print("✓ test_transpose_axes_default_none")
    test_transpose_axes_4d_permutation()
    print("✓ test_transpose_axes_4d_permutation")
    test_transpose_axes_invalid_duplicate()
    print("✓ test_transpose_axes_invalid_duplicate")
    test_transpose_axes_invalid_out_of_bounds()
    print("✓ test_transpose_axes_invalid_out_of_bounds")
    test_transpose_axes_invalid_length()
    print("✓ test_transpose_axes_invalid_length")
    test_transpose_axes_backward_3d()
    print("✓ test_transpose_axes_backward_3d")
    test_transpose_axes_double_permutation()
    print("✓ test_transpose_axes_double_permutation")

    print("\n" + "=" * 60)
    print("All 7 tests passed! (Part 6)")
