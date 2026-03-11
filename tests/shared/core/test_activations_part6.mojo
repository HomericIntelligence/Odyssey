"""Tests for activation functions - Part 6: Mish (continued), ELU, Integration.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests: test_mish_shape, test_mish_backward_gradient,
       test_elu_basic, test_elu_backward,
       test_integration_forward_backward
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    ones_like,
)
from shared.core.activation import (
    relu,
    sigmoid,
    mish,
    elu,
    relu_backward,
    sigmoid_backward,
    mish_backward,
    elu_backward,
)
from shared.testing import (
    check_gradient,
)


# ============================================================================
# Mish Tests (continued)
# ============================================================================


fn test_mish_shape() raises:
    """Test mish preserves shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var y = mish(x)

    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)
    assert_equal(y.shape()[2], 4)


fn test_mish_backward_gradient() raises:
    """Test Mish backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return mish(x)

    var y = mish(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return mish_backward(grad, x)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# ELU Tests
# ============================================================================


fn test_elu_basic() raises:
    """Test ELU with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    var y = elu(x, alpha=1.0)

    # ELU(-1) = 1.0 * (exp(-1) - 1) ≈ -0.632
    # ELU(0) = 0
    # ELU(1) = 1
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.632), tolerance=0.01
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )


fn test_elu_backward() raises:
    """Test ELU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return elu(x, alpha=1.0)

    var y = elu(x, alpha=1.0)
    var grad_out = ones_like(y)

    # Note: elu_backward takes x, y, and alpha
    fn backward_fn(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return elu_backward(grad, x, alpha=1.0)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Integration Tests
# ============================================================================


fn test_integration_forward_backward() raises:
    """Integration test: Complete forward and backward pass through activations.

    Simulates a simple neural network layer with:
    - Input -> ReLU -> Sigmoid -> Output
    - Loss gradient flows back through the network.
    """
    # Input data
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.5
    x._data.bitcast[Float32]()[2] = 2.0

    # Forward pass: x -> ReLU -> Sigmoid
    var relu_out = relu(x)
    var sigmoid_out = sigmoid(relu_out)

    # Check forward pass values
    # After ReLU: [0, 0.5, 2.0]
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.001
    )
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[1], Float32(0.5), tolerance=0.001
    )
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[2], Float32(2.0), tolerance=0.001
    )

    # After Sigmoid: [0.5, sigmoid(0.5), sigmoid(2.0)]
    var sig_0_5 = sigmoid_out._data.bitcast[Float32]()[1]
    var sig_2_0 = sigmoid_out._data.bitcast[Float32]()[2]
    assert_true(sig_0_5 > 0.6 and sig_0_5 < 0.7)
    assert_true(sig_2_0 > 0.8 and sig_2_0 < 0.9)

    # Simulate loss gradient (all ones)
    var grad_loss = ones(shape, DType.float32)

    # Backward pass: Sigmoid <- ReLU <- x
    var grad_sigmoid = sigmoid_backward(grad_loss, sigmoid_out)
    var grad_x = relu_backward(grad_sigmoid, x)

    # Check backward pass values
    # Gradient through ReLU should be 0 at x=-1 (negative input)
    assert_almost_equal(
        grad_x._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.001
    )

    # Gradients at positive inputs should be non-zero
    assert_true(grad_x._data.bitcast[Float32]()[1] > 0.0)
    assert_true(grad_x._data.bitcast[Float32]()[2] > 0.0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_activations_part6.mojo")
    print("=" * 70 + "\n")

    # test_mish_shape
    total += 1
    try:
        test_mish_shape()
        passed += 1
        print("  ✓ test_mish_shape")
    except e:
        failed += 1
        print("  ✗ test_mish_shape:", e)

    # test_mish_backward_gradient
    total += 1
    try:
        test_mish_backward_gradient()
        passed += 1
        print("  ✓ test_mish_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_mish_backward_gradient:", e)

    # test_elu_basic
    total += 1
    try:
        test_elu_basic()
        passed += 1
        print("  ✓ test_elu_basic")
    except e:
        failed += 1
        print("  ✗ test_elu_basic:", e)

    # test_elu_backward
    total += 1
    try:
        test_elu_backward()
        passed += 1
        print("  ✓ test_elu_backward")
    except e:
        failed += 1
        print("  ✗ test_elu_backward:", e)

    # test_integration_forward_backward
    total += 1
    try:
        test_integration_forward_backward()
        passed += 1
        print("  ✓ test_integration_forward_backward")
    except e:
        failed += 1
        print("  ✗ test_integration_forward_backward:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
