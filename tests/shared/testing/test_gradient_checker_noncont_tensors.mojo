"""Tests for gradient checking with non-contiguous tensor inputs.

This module validates that the gradient_checker module correctly handles
non-contiguous tensor views (e.g., transposed weight matrices). These tests
ensure the gradient checker's _get_float64/_set_float64 stride-aware
operations work correctly on non-contiguous memory layouts.

Issue #3801: Add non-contiguous tensor support to gradient checker
"""

from std.testing import assert_true, assert_false
from shared.testing import (
    NumericalForward,
    NumericalBackward,
    check_gradients,
    check_gradients_verbose,
    compute_numerical_gradient,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, randn
from shared.core.shape import as_contiguous
from shared.core.matrix import transpose_view


# ============================================================================
# Shared function structs for gradient checking
# ============================================================================


@fieldwise_init
struct _SimpleSquareFwd(NumericalForward):
    """Forward pass: f(x) = x^2."""

    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        var result = zeros(x.shape(), x.dtype())
        for i in range(x.numel()):
            var val = x._get_float64(i)
            result._set_float64(i, val * val)
        return result^


@fieldwise_init
struct _SimpleSquareBwd(NumericalBackward):
    """Backward pass: f'(x) = 2*x."""

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var result = zeros(x.shape(), x.dtype())
        for i in range(x.numel()):
            var grad = grad_out._get_float64(i)
            var x_val = x._get_float64(i)
            result._set_float64(i, grad * 2.0 * x_val)
        return result^


@fieldwise_init
struct _ReluFwd(NumericalForward):
    """Forward pass: ReLU."""

    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        var result = zeros(x.shape(), x.dtype())
        for i in range(x.numel()):
            var val = x._get_float64(i)
            result._set_float64(i, max(val, 0.0))
        return result^


@fieldwise_init
struct _ReluBwd(NumericalBackward):
    """Backward pass: ReLU."""

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var result = zeros(x.shape(), x.dtype())
        for i in range(x.numel()):
            var grad = grad_out._get_float64(i)
            var x_val = x._get_float64(i)
            result._set_float64(i, grad if x_val > 0.0 else 0.0)
        return result^


# ============================================================================
# Non-Contiguous Tensor Tests
# ============================================================================


def test_gradient_check_transposed_input() raises:
    """Test gradient checking on transposed (non-contiguous) input.

    Creates a 2D tensor and transposes it, creating a non-contiguous view.
    Validates that check_gradients correctly handles stride-aware access.
    """
    print("Testing gradient checking with transposed input...")

    # Create a small 2x3 tensor
    var shape_2d = [2, 3]
    var x = full(shape_2d, 2.0, DType.float32)

    # Transpose to create non-contiguous view: 3x2 with modified strides
    var x_transposed = transpose_view(x)

    # Verify it's non-contiguous
    assert_false(
        x_transposed.is_contiguous(),
        "Transposed tensor should be non-contiguous",
    )

    # Check gradients on the non-contiguous tensor
    # For f(x) = x*x, f'(x) = 2*x
    var passed = check_gradients(
        _SimpleSquareFwd(),
        _SimpleSquareBwd(),
        x_transposed,
        epsilon=1e-4,
        tolerance=1e-2,
    )
    assert_true(passed, "Gradient check should pass on transposed tensor")
    print("  ✓ Gradient check passes on transposed input")


def test_gradient_check_transposed_relu() raises:
    """Test ReLU gradient checking on non-contiguous input.

    ReLU is piecewise linear, so numerical gradients should match analytical
    exactly (within floating-point precision) even on non-contiguous tensors.
    """
    print("Testing ReLU gradient checking with transposed input...")

    # Create a 3x2 tensor with mixed positive/negative values
    var shape_2d = [3, 2]
    var x = full(shape_2d, 0.5, DType.float32)
    # Set some negative values to test piecewise behavior
    x._set_float64(0, -1.0)
    x._set_float64(3, -0.5)

    # Transpose to create non-contiguous view
    var x_transposed = transpose_view(x)
    assert_false(
        x_transposed.is_contiguous(),
        "Transposed tensor should be non-contiguous",
    )

    # ReLU gradient should be exact on transposed input
    var passed = check_gradients(
        _ReluFwd(), _ReluBwd(), x_transposed, epsilon=3e-4, tolerance=1e-3
    )
    assert_true(passed, "ReLU gradient check should pass on transposed tensor")
    print("  ✓ ReLU gradient check passes on transposed input")


def test_gradient_check_partial_transpose() raises:
    """Test gradient checking with partial axis permutation.

    Creates a 3D tensor and permutes axes to create a non-contiguous layout,
    validating that gradient checking handles complex stride patterns.
    """
    print("Testing gradient checking with partial axis permutation...")

    # Create a 2x3x2 tensor
    var shape_3d = [2, 3, 2]
    var x = randn(shape_3d, DType.float32)

    # Permute axes (0, 2, 1) -> shape becomes (2, 2, 3)
    # This creates a non-contiguous layout
    var axes: List[Int] = [0, 2, 1]
    var x_permuted = transpose_view(x, axes^)

    assert_false(
        x_permuted.is_contiguous(),
        "Permuted tensor should be non-contiguous",
    )

    # Test gradient checking on the permuted tensor
    var passed = check_gradients(
        _SimpleSquareFwd(),
        _SimpleSquareBwd(),
        x_permuted,
        epsilon=1e-4,
        tolerance=1e-2,
    )
    assert_true(passed, "Gradient check should pass on permuted tensor")
    print("  ✓ Gradient check passes on permuted 3D tensor")


def test_gradient_check_contiguous_copy() raises:
    """Test that as_contiguous() works correctly before gradient checking.

    Verifies that converting a non-contiguous tensor to contiguous layout
    preserves gradients for subsequent checking.
    """
    print("Testing gradient check after as_contiguous() conversion...")

    # Create transposed tensor (non-contiguous)
    var shape_2d = [2, 3]
    var x = full(shape_2d, 1.5, DType.float32)
    var x_transposed = transpose_view(x)

    # Convert to contiguous
    var x_contiguous = as_contiguous(x_transposed)
    assert_true(
        x_contiguous.is_contiguous(),
        "Tensor should be contiguous after as_contiguous()",
    )

    # Gradient check on contiguous version should pass
    var passed = check_gradients(
        _SimpleSquareFwd(),
        _SimpleSquareBwd(),
        x_contiguous,
        epsilon=1e-4,
        tolerance=1e-2,
    )
    assert_true(passed, "Gradient check should pass on contiguous tensor")
    print("  ✓ Gradient check passes after as_contiguous() conversion")


def test_numerical_gradient_noncont() raises:
    """Test compute_numerical_gradient on non-contiguous input.

    Validates that numerical gradient computation correctly handles
    stride-aware access on non-contiguous tensors.
    """
    print("Testing numerical gradient computation on non-contiguous tensor...")

    # Create a small transposed tensor
    var shape_2d = [2, 3]
    var x = full(shape_2d, 2.0, DType.float32)
    var x_transposed = transpose_view(x)

    assert_false(
        x_transposed.is_contiguous(), "Tensor should be non-contiguous"
    )

    # Compute numerical gradient
    var num_grad = compute_numerical_gradient(
        _SimpleSquareFwd(), x_transposed, epsilon=1e-4
    )

    # For f(x) = x*x, numerical gradient at x=2.0 should be ~4.0
    # Check a few elements
    var val0 = num_grad._get_float64(0)
    var val1 = num_grad._get_float64(1)

    assert_true(
        val0 > 3.5 and val0 < 4.5,
        "Numerical gradient should be ~4.0 for x=2.0 on square function",
    )
    assert_true(
        val1 > 3.5 and val1 < 4.5,
        "Numerical gradient should be ~4.0 for x=2.0 on square function",
    )
    print("  ✓ Numerical gradient computed correctly on non-contiguous tensor")


def test_gradient_check_verbose_noncont() raises:
    """Test verbose gradient checking on non-contiguous input.

    Validates that the verbose output works correctly with
    stride-aware access patterns.
    """
    print("Testing verbose gradient checking with non-contiguous tensor...")

    # Create small tensor
    var shape_2d = [2, 2]
    var x = full(shape_2d, 1.0, DType.float32)
    var x_transposed = transpose_view(x)

    assert_false(
        x_transposed.is_contiguous(), "Tensor should be non-contiguous"
    )

    # Run verbose gradient check
    var passed = check_gradients_verbose(
        _SimpleSquareFwd(),
        _SimpleSquareBwd(),
        x_transposed,
        epsilon=1e-4,
        tolerance=1e-2,
        print_all=False,
    )
    assert_true(
        passed, "Verbose gradient check should pass on non-contiguous tensor"
    )
    print("  ✓ Verbose gradient check works on non-contiguous tensor")


# ============================================================================
# Main Test Entry Point
# ============================================================================


def main() raises:
    """Run all non-contiguous tensor gradient checking tests."""
    print("\n" + "=" * 70)
    print("Testing Gradient Checking with Non-Contiguous Tensors (#3801)")
    print("=" * 70 + "\n")

    test_gradient_check_transposed_input()
    test_gradient_check_transposed_relu()
    test_gradient_check_partial_transpose()
    test_gradient_check_contiguous_copy()
    test_numerical_gradient_noncont()
    test_gradient_check_verbose_noncont()

    print("\n" + "=" * 70)
    print("All non-contiguous tensor gradient checking tests passed!")
    print("=" * 70 + "\n")
