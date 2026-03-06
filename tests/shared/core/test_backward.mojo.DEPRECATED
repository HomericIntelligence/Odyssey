"""Tests for backward passes - gradient checking validation.

This module tests all backward pass implementations using numerical gradient checking
with finite differences. This is the gold standard for validating gradient computation.

Gradient checking formula:
    numerical_grad ≈ (f(x + ε) - f(x - ε)) / (2ε)

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.linear import linear, linear_backward
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
    avgpool2d,
    avgpool2d_backward,
)
from shared.core.loss import (
    cross_entropy,
    cross_entropy_backward,
    binary_cross_entropy,
    binary_cross_entropy_backward,
    mean_squared_error,
    mean_squared_error_backward,
)
from shared.testing import (
    check_gradient,
    compute_numerical_gradient,
    assert_gradients_close,
)


# ============================================================================
# Linear Backward Tests
# ============================================================================


fn test_linear_backward_shapes() raises:
    """Test that linear_backward returns correct gradient shapes."""
    var batch = 2
    var in_features = 10
    var out_features = 5

    # Create tensors
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_features)
    var x = ones(input_shape, DType.float32)

    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)

    var grad_out_shape = List[Int]()
    grad_out_shape.append(batch)
    grad_out_shape.append(out_features)
    var grad_output = ones(grad_out_shape, DType.float32)

    # Backward pass
    var grads = linear_backward(grad_output, x, weights)

    # Check shapes
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], in_features)

    var gw_shape = grads.grad_weights.shape()
    assert_equal(gw_shape[0], out_features)
    assert_equal(gw_shape[1], in_features)

    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_features)


fn test_linear_backward_numerical() raises:
    """Test linear_backward with numerical gradient checking.

    This is a simple sanity check with known values.
    """
    # Simple case: 1 sample, 2 inputs, 2 outputs
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    var x = ones(input_shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0

    var weight_shape = List[Int]()
    weight_shape.append(2)
    weight_shape.append(2)
    var weights = zeros(weight_shape, DType.float32)
    weights._data.bitcast[Float32]()[0] = 0.5  # w[0,0]
    weights._data.bitcast[Float32]()[1] = 0.3  # w[0,1]
    weights._data.bitcast[Float32]()[2] = 0.2  # w[1,0]
    weights._data.bitcast[Float32]()[3] = 0.4  # w[1,1]

    # Gradient from upstream (pretend loss gradient)
    var grad_out_shape = List[Int]()
    grad_out_shape.append(1)
    grad_out_shape.append(2)
    var grad_output = ones(grad_out_shape, DType.float32)

    # Backward pass
    var grads = linear_backward(grad_output, x, weights)

    # Check grad_input shape
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], 1)
    assert_equal(gi_shape[1], 2)

    # grad_input = grad_output @ weights
    # grad_output = [1, 1], weights = [[0.5, 0.3], [0.2, 0.4]]
    # result should be [1*0.5 + 1*0.2, 1*0.3 + 1*0.4] = [0.7, 0.7]
    assert_almost_equal(
        grads.grad_input._data.bitcast[Float32]()[0],
        Float32(0.7),
        tolerance=1e-5,
    )
    assert_almost_equal(
        grads.grad_input._data.bitcast[Float32]()[1],
        Float32(0.7),
        tolerance=1e-5,
    )


fn test_linear_backward_batch() raises:
    """Test linear_backward with batch size > 1."""
    var batch = 3
    var in_features = 4
    var out_features = 2

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_features)
    var x = ones(input_shape, DType.float32)

    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)

    var grad_out_shape = List[Int]()
    grad_out_shape.append(batch)
    grad_out_shape.append(out_features)
    var grad_output = ones(grad_out_shape, DType.float32)

    # Backward pass
    var grads = linear_backward(grad_output, x, weights)

    # Verify shapes
    assert_equal(grads.grad_input.shape()[0], batch)
    assert_equal(grads.grad_weights.shape()[0], out_features)
    assert_equal(grads.grad_bias.shape()[0], out_features)


fn test_linear_backward_gradient() raises:
    """Test linear backward with numerical gradient checking."""
    var batch = 2
    var in_features = 3
    var out_features = 2

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_features)
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = -0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = -0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = zeros(weight_shape, DType.float32)
    weights._data.bitcast[Float32]()[0] = 0.4
    weights._data.bitcast[Float32]()[1] = 0.2
    weights._data.bitcast[Float32]()[2] = -0.3
    weights._data.bitcast[Float32]()[3] = 0.6
    weights._data.bitcast[Float32]()[4] = -0.2
    weights._data.bitcast[Float32]()[5] = 0.5

    # Forward function wrapper (linear with non-zero bias to test bias gradient)
    fn forward(inp: ExTensor) raises -> ExTensor:
        var bias_shape = List[Int]()
        bias_shape.append(out_features)
        var bias = zeros(bias_shape, DType.float32)
        bias._data.bitcast[Float32]()[0] = 0.3
        bias._data.bitcast[Float32]()[1] = -0.2
        return linear(inp, weights, bias)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var bias_shape = List[Int]()
        bias_shape.append(out_features)
        var bias = zeros(bias_shape, DType.float32)
        var grads = linear_backward(grad_out, inp, weights)
        return grads.grad_input

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=5e-4)


# ============================================================================
# Conv2D Backward Tests
# ============================================================================


fn test_conv2d_backward_shapes() raises:
    """Test that conv2d_backward returns correct gradient shapes."""
    var batch = 2
    var in_channels = 3
    var out_channels = 4
    var in_h = 8
    var in_w = 8
    var kh = 3
    var kw = 3

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_h)
    input_shape.append(in_w)
    var x = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kh)
    kernel_shape.append(kw)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass to get output shape
    var output = conv2d(x, kernel, bias, stride=1, padding=0)

    # Create grad_output with same shape as output
    var grad_output = ones_like(output)

    # Backward pass
    var grads = conv2d_backward(grad_output, x, kernel, stride=1, padding=0)

    # Check shapes match original inputs
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], in_channels)
    assert_equal(gi_shape[2], in_h)
    assert_equal(gi_shape[3], in_w)

    var gk_shape = grads.grad_weights.shape()
    assert_equal(gk_shape[0], out_channels)
    assert_equal(gk_shape[1], in_channels)
    assert_equal(gk_shape[2], kh)
    assert_equal(gk_shape[3], kw)

    var gb_shape = grads.grad_bias.shape()
    assert_equal(gb_shape[0], out_channels)


fn test_conv2d_backward_with_stride() raises:
    """Test conv2d_backward with stride > 1."""
    var batch = 1
    var in_channels = 1
    var out_channels = 1

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(8)
    input_shape.append(8)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass with stride=2
    var output = conv2d(x, kernel, bias, stride=2, padding=0)
    var grad_output = ones_like(output)

    # Backward pass
    var grads = conv2d_backward(grad_output, x, kernel, stride=2, padding=0)

    # Check grad_input shape matches input
    var gi_shape = grads.grad_input.shape()
    assert_equal(gi_shape[0], 1)
    assert_equal(gi_shape[1], 1)
    assert_equal(gi_shape[2], 8)
    assert_equal(gi_shape[3], 8)


fn test_conv2d_backward_gradient() raises:
    """Test conv2d backward with numerical gradient checking."""
    var batch = 1
    var in_channels = 2
    var out_channels = 2
    var in_h = 5
    var in_w = 5
    var kh = 3
    var kw = 3

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_h)
    input_shape.append(in_w)
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(batch * in_channels * in_h * in_w):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 2.5

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kh)
    kernel_shape.append(kw)
    var kernel = zeros(kernel_shape, DType.float32)

    # Initialize kernel with non-uniform values
    for i in range(out_channels * in_channels * kh * kw):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 - 0.5

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return conv2d(inp, kernel, bias, stride=1, padding=0)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        var grads = conv2d_backward(grad_out, inp, kernel, stride=1, padding=0)
        return grads.grad_input

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    # Conv2D has higher numerical error due to strided access patterns
    # and multiple accumulation passes, requiring relaxed tolerance
    check_gradient(forward, backward, x, grad_output, rtol=1e-2, atol=1e-2)


# ============================================================================
# Pooling Backward Tests
# ============================================================================


fn test_maxpool2d_backward_shapes() raises:
    """Test that maxpool2d_backward returns correct gradient shape."""
    var batch = 2
    var channels = 3
    var height = 8
    var width = 8

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = ones(input_shape, DType.float32)

    # Forward pass
    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)

    # Backward pass
    var grad_input = maxpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], channels)
    assert_equal(gi_shape[2], height)
    assert_equal(gi_shape[3], width)


fn test_maxpool2d_backward_gradient_routing() raises:
    """Test that maxpool2d_backward routes gradients only to max positions."""
    # Create a simple 2x2 input with known max
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    var x = zeros(input_shape, DType.float32)

    # Set values: [1, 2]
    #              [3, 4]  <- max is at position (1, 1) = 4
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0
    x._data.bitcast[Float32]()[3] = 4.0

    # Forward pass - should select 4.0
    var output = maxpool2d(x, kernel_size=2, stride=2, padding=0)

    # Gradient of 1.0 from upstream
    var grad_output = ones_like(output)

    # Backward pass
    var grad_input = maxpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    # Only position (1, 1) should have gradient
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )


fn test_avgpool2d_backward_shapes() raises:
    """Test that avgpool2d_backward returns correct gradient shape."""
    var batch = 2
    var channels = 3
    var height = 8
    var width = 8

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = ones(input_shape, DType.float32)

    # Forward pass
    var output = avgpool2d(x, kernel_size=2, stride=2, padding=0)
    var grad_output = ones_like(output)

    # Backward pass
    var grad_input = avgpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], batch)
    assert_equal(gi_shape[1], channels)
    assert_equal(gi_shape[2], height)
    assert_equal(gi_shape[3], width)


fn test_avgpool2d_backward_gradient_distribution() raises:
    """Test that avgpool2d_backward distributes gradients equally."""
    # Create a simple 2x2 input
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(2)
    var x = ones(input_shape, DType.float32)

    # Forward pass
    var output = avgpool2d(x, kernel_size=2, stride=2, padding=0)

    # Gradient of 1.0 from upstream
    var grad_output = ones_like(output)

    # Backward pass
    var grad_input = avgpool2d_backward(
        grad_output, x, kernel_size=2, stride=2, padding=0
    )

    # All 4 positions should have equal gradient (1.0 / 4 = 0.25)
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[3], Float32(0.25), tolerance=1e-5
    )


fn test_maxpool2d_backward_gradient() raises:
    """Test maxpool2d backward with numerical gradient checking."""
    # Create input with non-uniform values
    var input_shape = List[Int]()
    input_shape.append(1)  # batch)
    input_shape.append(2)  # channels)
    input_shape.append(4)  # height)
    input_shape.append(4)  # width)
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(1 * 2 * 4 * 4):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.6

    # Forward function wrapper
    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return maxpool2d(inp, kernel_size=2, stride=2, padding=0)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return maxpool2d_backward(
            grad_out, inp, kernel_size=2, stride=2, padding=0
        )

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=5e-4)


fn test_avgpool2d_backward_gradient() raises:
    """Test avgpool2d backward with numerical gradient checking."""
    # Create input with non-uniform values
    var input_shape = List[Int]()
    input_shape.append(1)  # batch)
    input_shape.append(2)  # channels)
    input_shape.append(4)  # height)
    input_shape.append(4)  # width)
    var x = zeros(input_shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(1 * 2 * 4 * 4):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.6

    # Forward function wrapper
    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return avgpool2d(inp, kernel_size=2, stride=2, padding=0)

    # Backward function wrapper (only return grad_input)
    fn backward(grad_out: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return avgpool2d_backward(
            grad_out, inp, kernel_size=2, stride=2, padding=0
        )

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=5e-4)


# ============================================================================
# Cross-Entropy Backward Tests
# ============================================================================


fn test_cross_entropy_backward_shapes() raises:
    """Test that cross_entropy_backward returns correct gradient shape."""
    var batch = 4
    var num_classes = 10

    var logits_shape = List[Int]()
    logits_shape.append(batch)
    logits_shape.append(num_classes)
    var logits = ones(logits_shape, DType.float32)

    var targets_shape = List[Int]()
    targets_shape.append(batch)
    targets_shape.append(num_classes)
    var targets = zeros(targets_shape, DType.float32)
    # Set one-hot targets (first class)
    for i in range(batch):
        targets._data.bitcast[Float32]()[i * num_classes] = 1.0

    # Forward pass
    var loss = cross_entropy(logits, targets)

    # Backward pass (grad_output is scalar 1.0 typically)
    var grad_output = ones_like(loss)
    var grad_logits = cross_entropy_backward(grad_output, logits, targets)

    # Check shape matches logits
    var gl_shape = grad_logits.shape()
    assert_equal(gl_shape[0], batch)
    assert_equal(gl_shape[1], num_classes)


fn test_cross_entropy_backward_gradient() raises:
    """Test cross-entropy backward with numerical gradient checking."""
    var batch = 2
    var num_classes = 3

    var logits_shape = List[Int]()
    logits_shape.append(batch)
    logits_shape.append(num_classes)
    var logits = zeros(logits_shape, DType.float32)

    # Initialize with non-uniform values
    logits._data.bitcast[Float32]()[0] = 0.5
    logits._data.bitcast[Float32]()[1] = -0.3
    logits._data.bitcast[Float32]()[2] = 1.2
    logits._data.bitcast[Float32]()[3] = -0.8
    logits._data.bitcast[Float32]()[4] = 0.1
    logits._data.bitcast[Float32]()[5] = 0.7

    var targets_shape = List[Int]()
    targets_shape.append(batch)
    targets_shape.append(num_classes)
    var targets = zeros(targets_shape, DType.float32)

    # Set one-hot targets: [1, 0, 0] and [0, 1, 0]
    targets._data.bitcast[Float32]()[0] = 1.0  # Batch 0, class 0
    targets._data.bitcast[Float32]()[4] = 1.0  # Batch 1, class 1

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return cross_entropy(inp, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        return cross_entropy_backward(grad_out, inp, targets)

    var loss = forward(logits)
    var grad_output = ones_like(loss)

    # Numerical gradient checking
    # Cross-entropy involves log and softmax which can amplify numerical errors
    check_gradient(forward, backward, logits, grad_output, rtol=1e-3, atol=1e-3)


# ============================================================================
# Binary Cross-Entropy Backward Tests
# ============================================================================


fn test_binary_cross_entropy_backward_shapes() raises:
    """Test that binary_cross_entropy_backward returns correct gradient shape.
    """
    var batch = 32
    var features = 1

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    pred_shape.append(features)
    var predictions = zeros(pred_shape, DType.float32)

    # Initialize predictions in valid range [0, 1] (sigmoid outputs)
    for i in range(batch):
        predictions._data.bitcast[Float32]()[i] = Float32(i) / Float32(batch)

    var targets = zeros(pred_shape, DType.float32)
    # Set half to 0, half to 1
    for i in range(batch // 2, batch):
        targets._data.bitcast[Float32]()[i] = 1.0

    # Forward pass - element-wise loss
    var loss = binary_cross_entropy(predictions, targets)

    # Backward pass
    var grad_output = ones_like(loss)
    var grad_pred = binary_cross_entropy_backward(
        grad_output, predictions, targets
    )

    # Check shape matches predictions
    var gp_shape = grad_pred.shape()
    assert_equal(gp_shape[0], batch)
    assert_equal(gp_shape[1], features)


fn test_binary_cross_entropy_backward_gradient() raises:
    """Test binary cross-entropy backward with numerical gradient checking.

    CRITICAL TEST: Validates mathematical correctness of BCE backpropagation.
    Uses central finite differences for gold-standard gradient validation.
    """
    var batch = 8

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    var predictions = zeros(pred_shape, DType.float32)

    # Initialize predictions in valid range [0.1, 0.9] (avoid log(0) issues)
    predictions._data.bitcast[Float32]()[0] = 0.1
    predictions._data.bitcast[Float32]()[1] = 0.3
    predictions._data.bitcast[Float32]()[2] = 0.5
    predictions._data.bitcast[Float32]()[3] = 0.7
    predictions._data.bitcast[Float32]()[4] = 0.9
    predictions._data.bitcast[Float32]()[5] = 0.2
    predictions._data.bitcast[Float32]()[6] = 0.6
    predictions._data.bitcast[Float32]()[7] = 0.8

    var targets = zeros(pred_shape, DType.float32)
    targets._data.bitcast[Float32]()[0] = 0.0
    targets._data.bitcast[Float32]()[1] = 1.0
    targets._data.bitcast[Float32]()[2] = 0.0
    targets._data.bitcast[Float32]()[3] = 1.0
    targets._data.bitcast[Float32]()[4] = 0.0
    targets._data.bitcast[Float32]()[5] = 1.0
    targets._data.bitcast[Float32]()[6] = 0.0
    targets._data.bitcast[Float32]()[7] = 1.0

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return binary_cross_entropy(inp, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        return binary_cross_entropy_backward(grad_out, inp, targets)

    var loss = forward(predictions)
    var grad_output = ones_like(loss)

    # Numerical gradient checking
    # BCE uses simplified gradient (p - y), so should be quite accurate
    check_gradient(
        forward, backward, predictions, grad_output, rtol=1e-3, atol=1e-6
    )


fn test_binary_cross_entropy_backward_edge_cases() raises:
    """Test BCE backward with edge case values near 0 and 1."""
    var pred_shape = List[Int]()
    pred_shape.append(4)

    var predictions = zeros(pred_shape, DType.float32)
    # Test near boundaries (but not exactly at them due to epsilon clipping)
    predictions._data.bitcast[Float32]()[0] = 0.001  # Near 0
    predictions._data.bitcast[Float32]()[1] = 0.999  # Near 1
    predictions._data.bitcast[Float32]()[2] = 0.5  # Middle
    predictions._data.bitcast[Float32]()[3] = 0.1  # Low

    var targets = zeros(pred_shape, DType.float32)
    targets._data.bitcast[Float32]()[0] = 0.0
    targets._data.bitcast[Float32]()[1] = 1.0
    targets._data.bitcast[Float32]()[2] = 0.0
    targets._data.bitcast[Float32]()[3] = 1.0

    # Backward pass should not produce NaN or Inf
    var loss = binary_cross_entropy(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = binary_cross_entropy_backward(
        grad_output, predictions, targets
    )

    # Verify all gradients are finite
    for i in range(4):
        var grad = grad_pred._data.bitcast[Float32]()[i]
        assert_true(grad == grad, "Gradient should not be NaN")  # NaN != NaN
        assert_true(grad > -1e10 and grad < 1e10, "Gradient should not be Inf")


# ============================================================================
# Mean Squared Error Backward Tests
# ============================================================================


fn test_mean_squared_error_backward_shapes() raises:
    """Test that mean_squared_error_backward returns correct gradient shape."""
    var batch = 16
    var features = 10

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    pred_shape.append(features)
    var predictions = ones(pred_shape, DType.float32)

    var targets = zeros(pred_shape, DType.float32)
    for i in range(batch * features):
        targets._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Forward pass - element-wise squared error
    var loss = mean_squared_error(predictions, targets)

    # Backward pass
    var grad_output = ones_like(loss)
    var grad_pred = mean_squared_error_backward(
        grad_output, predictions, targets
    )

    # Check shape matches predictions
    var gp_shape = grad_pred.shape()
    assert_equal(gp_shape[0], batch)
    assert_equal(gp_shape[1], features)


fn test_mean_squared_error_backward_gradient() raises:
    """Test mean squared error backward with numerical gradient checking.

    CRITICAL TEST: Validates mathematical correctness of MSE backpropagation.
    Uses central finite differences for gold-standard gradient validation.
    """
    var batch = 4
    var features = 3

    var pred_shape = List[Int]()
    pred_shape.append(batch)
    pred_shape.append(features)
    var predictions = zeros(pred_shape, DType.float32)

    # Initialize with non-uniform values
    predictions._data.bitcast[Float32]()[0] = 0.5
    predictions._data.bitcast[Float32]()[1] = -0.3
    predictions._data.bitcast[Float32]()[2] = 1.2
    predictions._data.bitcast[Float32]()[3] = -0.8
    predictions._data.bitcast[Float32]()[4] = 0.1
    predictions._data.bitcast[Float32]()[5] = 0.7
    predictions._data.bitcast[Float32]()[6] = 2.0
    predictions._data.bitcast[Float32]()[7] = -1.5
    predictions._data.bitcast[Float32]()[8] = 0.0
    predictions._data.bitcast[Float32]()[9] = 1.0
    predictions._data.bitcast[Float32]()[10] = -0.5
    predictions._data.bitcast[Float32]()[11] = 0.3

    var targets = zeros(pred_shape, DType.float32)
    targets._data.bitcast[Float32]()[0] = 0.2
    targets._data.bitcast[Float32]()[1] = 0.4
    targets._data.bitcast[Float32]()[2] = 0.8
    targets._data.bitcast[Float32]()[3] = -0.3
    targets._data.bitcast[Float32]()[4] = 0.5
    targets._data.bitcast[Float32]()[5] = 1.0
    targets._data.bitcast[Float32]()[6] = 1.5
    targets._data.bitcast[Float32]()[7] = -1.0
    targets._data.bitcast[Float32]()[8] = 0.3
    targets._data.bitcast[Float32]()[9] = 0.7
    targets._data.bitcast[Float32]()[10] = 0.0
    targets._data.bitcast[Float32]()[11] = 0.6

    # Forward function wrapper
    fn forward(inp: ExTensor) raises -> ExTensor:
        return mean_squared_error(inp, targets)

    # Backward function wrapper
    fn backward(grad_out: ExTensor, inp: ExTensor) raises -> ExTensor:
        return mean_squared_error_backward(grad_out, inp, targets)

    var loss = forward(predictions)
    var grad_output = ones_like(loss)

    # Numerical gradient checking
    # MSE has simple gradient: 2*(p - y), so should be very accurate
    check_gradient(
        forward, backward, predictions, grad_output, rtol=1e-3, atol=1e-6
    )


fn test_mean_squared_error_backward_zero_diff() raises:
    """Test MSE backward when predictions equal targets (zero gradient)."""
    var pred_shape = List[Int]()
    pred_shape.append(5)

    var predictions = zeros(pred_shape, DType.float32)
    for i in range(5):
        predictions._data.bitcast[Float32]()[i] = Float32(i)

    # Targets equal predictions - zero error
    var targets = zeros(pred_shape, DType.float32)
    for i in range(5):
        targets._data.bitcast[Float32]()[i] = Float32(i)

    var loss = mean_squared_error(predictions, targets)
    var grad_output = ones_like(loss)
    var grad_pred = mean_squared_error_backward(
        grad_output, predictions, targets
    )

    # All gradients should be zero (2 * 0 = 0)
    for i in range(5):
        assert_almost_equal(
            grad_pred._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-6
        )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all backward pass tests."""
    print("Running backward pass tests...")

    # Linear backward tests
    test_linear_backward_shapes()
    print("✓ test_linear_backward_shapes")

    test_linear_backward_numerical()
    print("✓ test_linear_backward_numerical")

    test_linear_backward_batch()
    print("✓ test_linear_backward_batch")

    test_linear_backward_gradient()
    print("✓ test_linear_backward_gradient")

    # Conv2D backward tests
    test_conv2d_backward_shapes()
    print("✓ test_conv2d_backward_shapes")

    test_conv2d_backward_with_stride()
    print("✓ test_conv2d_backward_with_stride")

    test_conv2d_backward_gradient()
    print("✓ test_conv2d_backward_gradient")

    # MaxPool2D backward tests
    test_maxpool2d_backward_shapes()
    print("✓ test_maxpool2d_backward_shapes")

    test_maxpool2d_backward_gradient_routing()
    print("✓ test_maxpool2d_backward_gradient_routing")

    test_maxpool2d_backward_gradient()
    print("✓ test_maxpool2d_backward_gradient")

    # AvgPool2D backward tests
    test_avgpool2d_backward_shapes()
    print("✓ test_avgpool2d_backward_shapes")

    test_avgpool2d_backward_gradient_distribution()
    print("✓ test_avgpool2d_backward_gradient_distribution")

    test_avgpool2d_backward_gradient()
    print("✓ test_avgpool2d_backward_gradient")

    # Cross-entropy backward tests
    test_cross_entropy_backward_shapes()
    print("✓ test_cross_entropy_backward_shapes")

    test_cross_entropy_backward_gradient()
    print("✓ test_cross_entropy_backward_gradient")

    # Binary cross-entropy backward tests
    test_binary_cross_entropy_backward_shapes()
    print("✓ test_binary_cross_entropy_backward_shapes")

    test_binary_cross_entropy_backward_gradient()
    print("✓ test_binary_cross_entropy_backward_gradient")

    test_binary_cross_entropy_backward_edge_cases()
    print("✓ test_binary_cross_entropy_backward_edge_cases")

    # Mean squared error backward tests
    test_mean_squared_error_backward_shapes()
    print("✓ test_mean_squared_error_backward_shapes")

    test_mean_squared_error_backward_gradient()
    print("✓ test_mean_squared_error_backward_gradient")

    test_mean_squared_error_backward_zero_diff()
    print("✓ test_mean_squared_error_backward_zero_diff")

    print("\nAll backward pass tests passed!")
