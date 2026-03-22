# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic_backward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for arithmetic backward passes — Part 3: gradient checking for B operand and broadcast.

Tests cover:
- Numerical gradient checking for B (second) operand: add, subtract, multiply, divide
- Broadcast numerical gradient checking for add, multiply, divide
- Tests 17–23 of the original test_arithmetic_backward.mojo
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.any_tensor import (
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
# Test 17: Add Backward (B operand) with Numerical Gradient Checking
# ============================================================================


fn test_add_backward_b_gradient() raises:
    """Test add_backward gradient w.r.t. second operand (B).

    Validates gradient computation for the second input of addition.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 0.5
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.12 + 0.3

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return add(a, inp)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = add_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 18: Subtract Backward (B operand) with Numerical Gradient Checking
# ============================================================================


fn test_subtract_backward_b_gradient() raises:
    """Test subtract_backward gradient w.r.t. second operand (B).

    Validates that gradient for B is negated: ∂(A-B)/∂B = -1
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.15 + 0.2
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.0

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return subtract(a, inp)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = subtract_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 19: Multiply Backward (B operand) with Numerical Gradient Checking
# ============================================================================


fn test_multiply_backward_b_gradient() raises:
    """Test multiply_backward gradient w.r.t. second operand (B).

    Validates product rule for second operand: ∂(A*B)/∂B = A.
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (avoid zero for product)
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.1
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 + 0.15

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return multiply(a, inp)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = multiply_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=1e-2, atol=1e-5)


# ============================================================================
# Test 20: Divide Backward (B operand) with Numerical Gradient Checking
# ============================================================================


fn test_divide_backward_b_gradient() raises:
    """Test divide_backward gradient w.r.t. second operand (B).

    Validates quotient rule for denominator: ∂(A/B)/∂B = -A/B²
    """
    var shape = create_shape_vec(3, 4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Initialize with non-uniform values (ensure b > 0 to avoid division by zero)
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.5
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 1.5  # b > 0

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return divide(a, inp)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = divide_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=1e-2, atol=1e-5)


# ============================================================================
# Test 21: Add Backward Broadcast with Numerical Gradient Checking
# ============================================================================


fn test_add_backward_broadcast_gradient() raises:
    """Test add_backward with broadcasting and numerical gradient checking.

    Validates gradient computation when one operand broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(6):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.2
    for i in range(3):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.15 - 0.3

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return add(a, inp)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = add_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 22: Multiply Backward Broadcast with Numerical Gradient Checking
# ============================================================================


fn test_multiply_backward_broadcast_gradient() raises:
    """Test multiply_backward with broadcasting and numerical gradient checking.

    Validates product rule when one operand broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values (avoid zero for product)
    for i in range(6):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.1
    for i in range(3):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.2

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return multiply(a, inp)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = multiply_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Test 23: Divide Backward Broadcast with Numerical Gradient Checking
# ============================================================================


fn test_divide_backward_broadcast_gradient() raises:
    """Test divide_backward with broadcasting and numerical gradient checking.

    Validates quotient rule when denominator broadcasts.
    Broadcasting case: [3] broadcast to [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(3)

    var a = zeros(a_shape, DType.float32)
    var b = zeros(b_shape, DType.float32)

    # Initialize with non-uniform values (ensure b > 0)
    for i in range(6):
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.5
    for i in range(3):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 1.0  # b > 0

    fn forward(inp: AnyTensor) raises -> AnyTensor:
        return divide(a, inp)

    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = divide_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    check_gradient(forward, backward, b, grad_output, rtol=5e-3, atol=1e-5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_arithmetic_backward_part3.mojo")
    print("=" * 70 + "\n")

    # test_add_backward_b_gradient
    total += 1
    try:
        test_add_backward_b_gradient()
        passed += 1
        print("  ✓ test_add_backward_b_gradient")
    except e:
        failed += 1
        print("  ✗ test_add_backward_b_gradient:", e)

    # test_subtract_backward_b_gradient
    total += 1
    try:
        test_subtract_backward_b_gradient()
        passed += 1
        print("  ✓ test_subtract_backward_b_gradient")
    except e:
        failed += 1
        print("  ✗ test_subtract_backward_b_gradient:", e)

    # test_multiply_backward_b_gradient
    total += 1
    try:
        test_multiply_backward_b_gradient()
        passed += 1
        print("  ✓ test_multiply_backward_b_gradient")
    except e:
        failed += 1
        print("  ✗ test_multiply_backward_b_gradient:", e)

    # test_divide_backward_b_gradient
    total += 1
    try:
        test_divide_backward_b_gradient()
        passed += 1
        print("  ✓ test_divide_backward_b_gradient")
    except e:
        failed += 1
        print("  ✗ test_divide_backward_b_gradient:", e)

    # test_add_backward_broadcast_gradient
    total += 1
    try:
        test_add_backward_broadcast_gradient()
        passed += 1
        print("  ✓ test_add_backward_broadcast_gradient")
    except e:
        failed += 1
        print("  ✗ test_add_backward_broadcast_gradient:", e)

    # test_multiply_backward_broadcast_gradient
    total += 1
    try:
        test_multiply_backward_broadcast_gradient()
        passed += 1
        print("  ✓ test_multiply_backward_broadcast_gradient")
    except e:
        failed += 1
        print("  ✗ test_multiply_backward_broadcast_gradient:", e)

    # test_divide_backward_broadcast_gradient
    total += 1
    try:
        test_divide_backward_broadcast_gradient()
        passed += 1
        print("  ✓ test_divide_backward_broadcast_gradient")
    except e:
        failed += 1
        print("  ✗ test_divide_backward_broadcast_gradient:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
