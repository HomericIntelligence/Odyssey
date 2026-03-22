# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic_backward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for arithmetic backward passes — Part 2: broadcasting and gradient checking (A operand).

Tests cover:
- Broadcasting behavior with different tensor shapes
- Numerical gradient checking for A operand (add, subtract, multiply, divide)
- Tests 9–16 of the original test_arithmetic_backward.mojo
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import (
    AnyTensor,
    zeros,
    ones,
    ones_like,
    zeros_like,
    full,
)
from shared.core.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)
from shared.testing import (
    check_gradient,
    compute_numerical_gradient,
)


# ============================================================================
# Test Helpers
# ============================================================================


fn create_shape_vec(*dims: Int) -> List[Int]:
    """Create a List[Int] from variadic arguments.

    Args:
        dims: Variable number of dimension sizes.

    Returns:
        List[Int] with specified dimensions.
    """
    var shape = List[Int]()
    for i in range(len(dims)):
        shape.append(dims[i])
    return shape^


# ============================================================================
# Test 9: Addition with Broadcasting (different shape)
# ============================================================================


fn test_add_broadcast() raises:
    """Test add_backward with broadcasting.

    Broadcasting case: [2, 3] + [3] -> [2, 3]
    The gradient for the [3] tensor must be summed over the broadcast dimension.
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = add_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)

    # grad_b should contain sum over first dimension (2 ones = 2.0)
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )


# ============================================================================
# Test 10: Subtraction with Broadcasting
# ============================================================================


fn test_subtract_broadcast() raises:
    """Test subtract_backward with broadcasting.

    Broadcasting case: [2, 3] - [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = subtract_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3] with value 1.0
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3] with value -2.0 (sum of -ones)
    assert_equal(grad_b.shape()[0], 3)
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-2.0), tolerance=1e-5
        )


# ============================================================================
# Test 11: Multiplication with Broadcasting
# ============================================================================


fn test_multiply_broadcast() raises:
    """Test multiply_backward with broadcasting.

    Broadcasting case: [2, 3] * [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = ones(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    for i in range(3):
        b._data.bitcast[Float32]()[i] = Float32(i + 1)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = multiply_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)
    # grad_b[i] = sum(grad_output * a) = sum(1.0 * 1.0) over first dim = 2.0
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )


# ============================================================================
# Test 12: Division with Broadcasting
# ============================================================================


fn test_divide_broadcast() raises:
    """Test divide_backward with broadcasting.

    Broadcasting case: [2, 3] / [3] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Fill a with 2.0, b with 2.0
    for i in range(6):
        a._data.bitcast[Float32]()[i] = 2.0
    for i in range(3):
        b._data.bitcast[Float32]()[i] = 2.0

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = divide_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [3]
    assert_equal(grad_b.shape()[0], 3)
    # grad_b[i] = sum(-grad_output * a / b²) = sum(-1.0 * 2.0 / 4.0) over first dim
    #           = 2 * (-0.5) = -1.0
    for i in range(3):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-4
        )


# ============================================================================
# Test 13: Addition Backward with Numerical Gradient Checking
# ============================================================================


fn test_add_backward_gradient() raises:
    """Test add_backward with numerical gradient checking.

    Validates that analytical gradients match numerical gradients computed
    via central differences, confirming correct implementation of the
    backward pass for addition.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.2
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 - 0.8

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return add(inp, b)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = add_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, a, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 14: Subtraction Backward with Numerical Gradient Checking
# ============================================================================


fn test_subtract_backward_gradient() raises:
    """Test subtract_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for subtraction operation.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.5
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.2 - 1.5

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return subtract(inp, b)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = subtract_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, a, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 15: Multiplication Backward with Numerical Gradient Checking
# ============================================================================


fn test_multiply_backward_gradient() raises:
    """Test multiply_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for multiplication operation.
    Tests product rule: ∂(A*B)/∂A = B.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero to test product properly)
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.1
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 + 0.2

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return multiply(inp, b)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = multiply_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, a, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 16: Division Backward with Numerical Gradient Checking
# ============================================================================


fn test_divide_backward_gradient() raises:
    """Test divide_backward with numerical gradient checking.

    Validates analytical vs numerical gradients for division operation.
    Tests quotient rule: ∂(A/B)/∂A = 1/B.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero denominator)
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.5
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 1.0  # Ensure b > 0

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return divide(inp, b)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = divide_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, a, grad_output, rtol=1e-2, atol=1e-5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_arithmetic_backward_part2.mojo")
    print("=" * 70 + "\n")

    # test_add_broadcast
    total += 1
    try:
        test_add_broadcast()
        passed += 1
        print("  ✓ test_add_broadcast")
    except e:
        failed += 1
        print("  ✗ test_add_broadcast:", e)

    # test_subtract_broadcast
    total += 1
    try:
        test_subtract_broadcast()
        passed += 1
        print("  ✓ test_subtract_broadcast")
    except e:
        failed += 1
        print("  ✗ test_subtract_broadcast:", e)

    # test_multiply_broadcast
    total += 1
    try:
        test_multiply_broadcast()
        passed += 1
        print("  ✓ test_multiply_broadcast")
    except e:
        failed += 1
        print("  ✗ test_multiply_broadcast:", e)

    # test_divide_broadcast
    total += 1
    try:
        test_divide_broadcast()
        passed += 1
        print("  ✓ test_divide_broadcast")
    except e:
        failed += 1
        print("  ✗ test_divide_broadcast:", e)

    # test_add_backward_gradient
    total += 1
    try:
        test_add_backward_gradient()
        passed += 1
        print("  ✓ test_add_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_add_backward_gradient:", e)

    # test_subtract_backward_gradient
    total += 1
    try:
        test_subtract_backward_gradient()
        passed += 1
        print("  ✓ test_subtract_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_subtract_backward_gradient:", e)

    # test_multiply_backward_gradient
    total += 1
    try:
        test_multiply_backward_gradient()
        passed += 1
        print("  ✓ test_multiply_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_multiply_backward_gradient:", e)

    # test_divide_backward_gradient
    total += 1
    try:
        test_divide_backward_gradient()
        passed += 1
        print("  ✓ test_divide_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_divide_backward_gradient:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
