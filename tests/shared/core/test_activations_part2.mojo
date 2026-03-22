"""Tests for activation functions - Part 2: Leaky ReLU (backward), PReLU, Sigmoid (basic).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests: test_leaky_relu_backward, test_prelu_basic, test_prelu_scalar_alpha,
       test_prelu_elementwise_alpha, test_prelu_backward,
       test_sigmoid_basic, test_sigmoid_backward, test_sigmoid_range
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_true,
)
from shared.core.extensor import (
    AnyTensor,
    zeros,
    full,
    ones_like,
)
from shared.core.activation import (
    leaky_relu,
    prelu,
    sigmoid,
    leaky_relu_backward,
    prelu_backward,
    sigmoid_backward,
)
from shared.testing import (
    check_gradient,
)


# ============================================================================
# Leaky ReLU Tests (continued)
# ============================================================================


fn test_leaky_relu_backward() raises:
    """Test Leaky ReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 1.0

    # Forward function wrapper
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return leaky_relu(x, alpha=0.1)

    var y = leaky_relu(x, alpha=0.1)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    fn backward_wrapper(
        grad: AnyTensor, x: AnyTensor
    ) raises escaping -> AnyTensor:
        return leaky_relu_backward(grad, x, alpha=0.1)

    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_wrapper, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# PReLU Tests
# ============================================================================


fn test_prelu_basic() raises:
    """Test PReLU with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    var alpha = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 2.0

    alpha._data.bitcast[Float32]()[0] = 0.25
    alpha._data.bitcast[Float32]()[1] = 0.25
    alpha._data.bitcast[Float32]()[2] = 0.25

    var y = prelu(x, alpha)

    # Expected: [-0.5, 0, 2.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.5), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_prelu_scalar_alpha() raises:
    """Test PReLU with scalar alpha parameter."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    # Scalar alpha = 0.2
    var alpha_shape = List[Int]()
    alpha_shape.append(1)
    var alpha = full(alpha_shape, 0.2, DType.float32)

    var y = prelu(x, alpha)

    # Expected with alpha=0.2: [-0.4, -0.2, 0, 1, 2]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.4), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(-0.2), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


fn test_prelu_elementwise_alpha() raises:
    """Test PReLU with element-wise alpha parameters."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 2.0

    # Element-wise alpha = [0.1, 0.2, 0.3]
    var alpha = zeros(shape, DType.float32)
    alpha._data.bitcast[Float32]()[0] = 0.1
    alpha._data.bitcast[Float32]()[1] = 0.2
    alpha._data.bitcast[Float32]()[2] = 0.3

    var y = prelu(x, alpha)

    # Expected: [-0.2, -0.2, 2.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.2), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(-0.2), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_prelu_backward() raises:
    """Test PReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var alpha = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 1.0

    alpha._data.bitcast[Float32]()[0] = 0.5
    alpha._data.bitcast[Float32]()[1] = 0.5

    # Forward function wrapper
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return prelu(x, alpha)

    var y = prelu(x, alpha)
    var grad_out = ones_like(y)

    # Validate gradient w.r.t. input using numerical checking
    fn backward_input(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var result = prelu_backward(grad, x, alpha)
        return result.grad_a

    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_input, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Sigmoid Tests
# ============================================================================


fn test_sigmoid_basic() raises:
    """Test sigmoid with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -100.0  # Should be ~0
    x._data.bitcast[Float32]()[1] = 0.0  # Should be 0.5
    x._data.bitcast[Float32]()[2] = 100.0  # Should be ~1

    var y = sigmoid(x)

    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-3
    )


fn test_sigmoid_backward() raises:
    """Test sigmoid gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Use multiple test points for better coverage
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return sigmoid(x)

    var y = sigmoid(x)
    var grad_out = ones_like(y)

    # Note: sigmoid_backward takes output y, not input x
    fn backward_fn(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var out = sigmoid(x)  # Recompute output inside wrapper
        return sigmoid_backward(grad, out)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


fn test_sigmoid_range() raises:
    """Test sigmoid output is in (0, 1)."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -10.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 10.0

    var y = sigmoid(x)

    # All values should be in (0, 1)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val > 0.0)
        assert_true(val < 1.0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_activations_part2.mojo")
    print("=" * 70 + "\n")

    # test_leaky_relu_backward
    total += 1
    try:
        test_leaky_relu_backward()
        passed += 1
        print("  ✓ test_leaky_relu_backward")
    except e:
        failed += 1
        print("  ✗ test_leaky_relu_backward:", e)

    # test_prelu_basic
    total += 1
    try:
        test_prelu_basic()
        passed += 1
        print("  ✓ test_prelu_basic")
    except e:
        failed += 1
        print("  ✗ test_prelu_basic:", e)

    # test_prelu_scalar_alpha
    total += 1
    try:
        test_prelu_scalar_alpha()
        passed += 1
        print("  ✓ test_prelu_scalar_alpha")
    except e:
        failed += 1
        print("  ✗ test_prelu_scalar_alpha:", e)

    # test_prelu_elementwise_alpha
    total += 1
    try:
        test_prelu_elementwise_alpha()
        passed += 1
        print("  ✓ test_prelu_elementwise_alpha")
    except e:
        failed += 1
        print("  ✗ test_prelu_elementwise_alpha:", e)

    # test_prelu_backward
    total += 1
    try:
        test_prelu_backward()
        passed += 1
        print("  ✓ test_prelu_backward")
    except e:
        failed += 1
        print("  ✗ test_prelu_backward:", e)

    # test_sigmoid_basic
    total += 1
    try:
        test_sigmoid_basic()
        passed += 1
        print("  ✓ test_sigmoid_basic")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_basic:", e)

    # test_sigmoid_backward
    total += 1
    try:
        test_sigmoid_backward()
        passed += 1
        print("  ✓ test_sigmoid_backward")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_backward:", e)

    # test_sigmoid_range
    total += 1
    try:
        test_sigmoid_range()
        passed += 1
        print("  ✓ test_sigmoid_range")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_range:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
