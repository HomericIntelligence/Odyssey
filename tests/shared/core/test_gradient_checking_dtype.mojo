"""Gradient checking tests for dtype-specific precision (FP32, FP16).

Note: Split from test_gradient_checking.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from tests.shared.conftest import assert_true
from shared.testing import check_gradients
from shared.core import ExTensor, zeros, ones, full
from shared.core.activation import (
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
)
from shared.core.arithmetic import multiply, multiply_backward
from shared.core.linear import linear, linear_backward
from shared.core.conv import conv2d, conv2d_backward
from shared.core.loss import cross_entropy, cross_entropy_backward
from shared.training.precision_config import PrecisionConfig


fn test_composite_relu_multiply() raises:
    """Test gradient through composite operation: multiply -> relu."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var input_a = full(shape, 2.0, DType.float32)
    var input_b = full(shape, 3.0, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        var mul_result = multiply(x, input_b)
        return relu(mul_result)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var mul_result = multiply(x, input_b)
        var grad_relu = relu_backward(grad_out, mul_result)
        var grads = multiply_backward(grad_relu, x, input_b)
        return grads.grad_a

    var passed = check_gradients(forward, backward, input_a)
    assert_true(passed, "Composite gradient check failed")


fn test_linear_gradient_fp32() raises:
    """Test linear layer gradient in FP32 precision."""
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(4)
    var input = full(input_shape, 0.5, DType.float32)

    var weight_shape = List[Int]()
    weight_shape.append(3)
    weight_shape.append(4)
    var weights = full(weight_shape, 0.1, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return linear(x, weights, bias)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var grads = linear_backward(grad_out, x, weights)
        return grads.grad_input

    var passed = check_gradients(
        forward, backward, input, epsilon=1e-5, tolerance=3e-3
    )
    assert_true(passed, "Linear FP32 gradient check failed")


fn test_linear_gradient_fp16() raises:
    """Test linear layer gradient in FP16 precision."""
    var config = PrecisionConfig.fp16()

    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(4)
    var input = config.cast_to_compute(full(input_shape, 0.5, DType.float32))

    var weight_shape = List[Int]()
    weight_shape.append(3)
    weight_shape.append(4)
    var weights = config.cast_to_compute(full(weight_shape, 0.1, DType.float32))

    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = config.cast_to_compute(zeros(bias_shape, DType.float32))

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return linear(x, weights, bias)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var grads = linear_backward(grad_out, x, weights)
        return grads.grad_input

    var passed = check_gradients(
        forward, backward, input, epsilon=1e-2, tolerance=2e-1
    )
    assert_true(passed, "Linear FP16 gradient check failed")


fn test_conv2d_gradient_fp32() raises:
    """Test Conv2D gradient in FP32 precision."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(5)
    input_shape.append(5)
    var input = full(input_shape, 0.5, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = full(kernel_shape, 0.1, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return conv2d(x, kernel, bias, stride=1, padding=0)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        var grads = conv2d_backward(grad_out, x, kernel, stride=1, padding=0)
        return grads.grad_input

    var passed = check_gradients(
        forward, backward, input, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "Conv2D FP32 gradient check failed")


fn test_cross_entropy_gradient_fp32() raises:
    """Test cross-entropy loss gradient in FP32 precision."""
    var logits_shape = List[Int]()
    logits_shape.append(1)
    logits_shape.append(4)
    var logits = zeros(logits_shape, DType.float32)
    logits._set_float64(0, 1.0)
    logits._set_float64(1, 0.5)
    logits._set_float64(2, -0.5)
    logits._set_float64(3, 0.2)

    var labels_shape = List[Int]()
    labels_shape.append(1)
    labels_shape.append(4)
    var labels = zeros(labels_shape, DType.float32)
    labels._set_float64(0, 1.0)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return cross_entropy(x, labels)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return cross_entropy_backward(grad_out, x, labels)

    var passed = check_gradients(
        forward, backward, logits, epsilon=1e-5, tolerance=1e-2
    )
    assert_true(passed, "CrossEntropy FP32 gradient check failed")


fn main() raises:
    """Run dtype-specific gradient checking tests."""
    print("Running composite gradient tests...")
    test_composite_relu_multiply()

    print("Running dtype-specific gradient tests...")
    test_linear_gradient_fp32()
    test_linear_gradient_fp16()
    test_conv2d_gradient_fp32()
    test_cross_entropy_gradient_fp32()

    print("All dtype gradient checking tests passed!")
