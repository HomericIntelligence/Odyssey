# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic_backward.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for arithmetic backward passes — Part 1: element-wise and scalar operations.

Tests cover:
- Element-wise operations: add, subtract, multiply, divide backward
- Scalar (broadcast) variants of each operation
- Tests 1–8 of the original test_arithmetic_backward.mojo
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


fn fill_tensor_sequential(tensor: AnyTensor, start_val: Float32 = 1.0) -> None:
    """Fill tensor with sequential values starting from start_val.

    Args:
        tensor: AnyTensor to fill.
        start_val: Starting value for sequence.
    """
    for i in range(tensor.numel()):
        tensor._data.bitcast[Float32]()[i] = start_val + Float32(i)


# ============================================================================
# Test 1: Element-wise Addition Backward
# ============================================================================


fn test_add_backward() raises:
    """Test add_backward with same-shaped tensors.

    Tests that ∂L/∂A and ∂L/∂B equal 1.0 for C = A + B when upstream
    gradient is 1.0. This is because ∂(A+B)/∂A = 1 and ∂(A+B)/∂B = 1.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    # Call add_backward - passing shapes as per function signature
    var grads = add_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # For addition, gradients should just pass through (equal to grad_output)
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


# ============================================================================
# Test 2: Scalar Addition Backward
# ============================================================================


fn test_add_scalar_backward() raises:
    """Test add_backward with scalar (broadcast) addition.

    Tests gradient computation when one operand broadcasts to the other.
    Broadcasting case: [2, 3] + [1] -> [2, 3]
    The gradient for [1] should sum over the broadcast dimensions.
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = ones(b_shape, DType.float32)

    # Create gradient matching output shape [2, 3]
    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    # Call add_backward
    var grads = add_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)

    # grad_b should contain sum of 6 ones = 6.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-5
    )


# ============================================================================
# Test 3: Element-wise Subtraction Backward
# ============================================================================


fn test_subtract_backward() raises:
    """Test subtract_backward with same-shaped tensors.

    Tests that ∂L/∂A = 1.0 and ∂L/∂B = -1.0 for C = A - B when upstream
    gradient is 1.0. This is because ∂(A-B)/∂A = 1 and ∂(A-B)/∂B = -1.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var grads = subtract_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should be positive (1.0)
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )

    # grad_b should be negative (-1.0)
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-5
        )


# ============================================================================
# Test 4: Scalar Subtraction Backward
# ============================================================================


fn test_subtract_scalar_backward() raises:
    """Test subtract_backward with scalar (broadcast) subtraction.

    Broadcasting case: [2, 3] - [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = ones(b_shape, DType.float32)

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = subtract_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3] with value 1.0
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should be reduced to shape [1] with value -6.0 (sum of -ones)
    assert_equal(grad_b.shape()[0], 1)
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(-6.0), tolerance=1e-5
    )


# ============================================================================
# Test 5: Element-wise Multiplication Backward
# ============================================================================


fn test_multiply_backward() raises:
    """Test multiply_backward with same-shaped tensors.

    Tests that ∂L/∂A = ∂L/∂C * B and ∂L/∂B = ∂L/∂C * A for C = A * B.
    Uses product rule of differentiation.
    """
    var shape = create_shape_vec(2, 3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    # Fill b with value 2.0
    for i in range(b.numel()):
        b._data.bitcast[Float32]()[i] = 2.0

    var grad_output = ones(shape, DType.float32)

    var grads = multiply_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a = grad_output * b = 1.0 * 2.0 = 2.0
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )

    # grad_b = grad_output * a = 1.0 * 1.0 = 1.0
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


# ============================================================================
# Test 6: Scalar Multiplication Backward
# ============================================================================


fn test_multiply_scalar_backward() raises:
    """Test multiply_backward with scalar (broadcast) multiplication.

    Broadcasting case: [2, 3] * [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = ones(a_shape, DType.float32)
    var b_scalar = zeros(b_shape, DType.float32)
    b_scalar._data.bitcast[Float32]()[0] = 2.0

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = multiply_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)
    # grad_a = grad_output * b = 1.0 * 2.0 = 2.0
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(2.0), tolerance=1e-5
        )

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)
    # grad_b = sum(grad_output * a) = sum(1.0 * 1.0) = 6.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-5
    )


# ============================================================================
# Test 7: Element-wise Division Backward
# ============================================================================


fn test_divide_backward() raises:
    """Test divide_backward with same-shaped tensors.

    Tests that ∂L/∂A = ∂L/∂C / B and ∂L/∂B = -∂L/∂C * A / B² for C = A / B.
    Uses quotient rule of differentiation.
    """
    var shape = create_shape_vec(2, 3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Fill a with 2.0, b with 2.0
    for i in range(6):
        a._data.bitcast[Float32]()[i] = 2.0
        b._data.bitcast[Float32]()[i] = 2.0

    var grad_output = ones(shape, DType.float32)

    var grads = divide_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a = grad_output / b = 1.0 / 2.0 = 0.5
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(0.5), tolerance=1e-4
        )

    # grad_b = -grad_output * a / b² = -1.0 * 2.0 / 4.0 = -0.5
    for i in range(6):
        assert_almost_equal(
            grad_b._data.bitcast[Float32]()[i], Float32(-0.5), tolerance=1e-4
        )


# ============================================================================
# Test 8: Scalar Division Backward
# ============================================================================


fn test_divide_scalar_backward() raises:
    """Test divide_backward with scalar (broadcast) division.

    Broadcasting case: [2, 3] / [1] -> [2, 3]
    """
    var a_shape = create_shape_vec(2, 3)
    var b_shape = create_shape_vec(1)

    var a = zeros(a_shape, DType.float32)
    var b_scalar = zeros(b_shape, DType.float32)

    for i in range(6):
        a._data.bitcast[Float32]()[i] = 2.0
    b_scalar._data.bitcast[Float32]()[0] = 2.0

    var grad_output_shape = create_shape_vec(2, 3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = divide_backward(grad_output, a, b_scalar)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should match shape [2, 3]
    assert_equal(grad_a.shape()[0], 2)
    assert_equal(grad_a.shape()[1], 3)
    # grad_a = grad_output / b = 1.0 / 2.0 = 0.5
    for i in range(6):
        assert_almost_equal(
            grad_a._data.bitcast[Float32]()[i], Float32(0.5), tolerance=1e-4
        )

    # grad_b should be reduced to shape [1]
    assert_equal(grad_b.shape()[0], 1)
    # grad_b = sum(-grad_output * a / b²) = sum(-1.0 * 2.0 / 4.0) = 6 * (-0.5) = -3.0
    assert_almost_equal(
        grad_b._data.bitcast[Float32]()[0], Float32(-3.0), tolerance=1e-4
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_arithmetic_backward_part1.mojo")
    print("=" * 70 + "\n")

    # test_add_backward
    total += 1
    try:
        test_add_backward()
        passed += 1
        print("  ✓ test_add_backward")
    except e:
        failed += 1
        print("  ✗ test_add_backward:", e)

    # test_add_scalar_backward
    total += 1
    try:
        test_add_scalar_backward()
        passed += 1
        print("  ✓ test_add_scalar_backward")
    except e:
        failed += 1
        print("  ✗ test_add_scalar_backward:", e)

    # test_subtract_backward
    total += 1
    try:
        test_subtract_backward()
        passed += 1
        print("  ✓ test_subtract_backward")
    except e:
        failed += 1
        print("  ✗ test_subtract_backward:", e)

    # test_subtract_scalar_backward
    total += 1
    try:
        test_subtract_scalar_backward()
        passed += 1
        print("  ✓ test_subtract_scalar_backward")
    except e:
        failed += 1
        print("  ✗ test_subtract_scalar_backward:", e)

    # test_multiply_backward
    total += 1
    try:
        test_multiply_backward()
        passed += 1
        print("  ✓ test_multiply_backward")
    except e:
        failed += 1
        print("  ✗ test_multiply_backward:", e)

    # test_multiply_scalar_backward
    total += 1
    try:
        test_multiply_scalar_backward()
        passed += 1
        print("  ✓ test_multiply_scalar_backward")
    except e:
        failed += 1
        print("  ✗ test_multiply_scalar_backward:", e)

    # test_divide_backward
    total += 1
    try:
        test_divide_backward()
        passed += 1
        print("  ✓ test_divide_backward")
    except e:
        failed += 1
        print("  ✗ test_divide_backward:", e)

    # test_divide_scalar_backward
    total += 1
    try:
        test_divide_scalar_backward()
        passed += 1
        print("  ✓ test_divide_scalar_backward")
    except e:
        failed += 1
        print("  ✗ test_divide_scalar_backward:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
