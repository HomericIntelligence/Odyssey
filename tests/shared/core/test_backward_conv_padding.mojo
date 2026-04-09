# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_backward_conv_pool.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Numerical gradient tests for conv2d_backward with padding > 0.

Follow-up from #3248: parametrized coverage for padding=1 and padding=2,
exercising the boundary-handling path in the transposed convolution used
to compute grad_input.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.conv import conv2d, conv2d_backward
from shared.testing import check_gradient


def test_conv2d_backward_grad_input_padding1() raises:
    """Numerical gradient check for grad_input with padding=1.

    padding=1 with a (1,1,3,3) kernel produces same-size output (1,1,4,4).
    Every input position lies adjacent to a padded boundary, fully exercising
    the boundary-handling path in the transposed convolution for grad_input.
    Non-uniform grad_output avoids gradient cancellation at boundaries.
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)

    # Non-uniform input for meaningful gradient signal
    for i in range(16):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    # Non-uniform kernel weights
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    def forward_input(inp: AnyTensor) raises -> AnyTensor:
        return conv2d(inp, kernel, bias, stride=1, padding=1)

    def backward_input(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = conv2d_backward(grad_out, inp, kernel, stride=1, padding=1)
        return grads.grad_input

    # Non-uniform grad_output: avoids pathological cancellation at boundaries
    var output = forward_input(x)
    var grad_output = zeros_like(output)
    for i in range(16):
        grad_output.set(i, Float32(i % 4) * Float32(0.25) - Float32(0.3))

    check_gradient(
        forward_input, backward_input, x, grad_output, rtol=1e-2, atol=1e-2
    )


def test_conv2d_backward_grad_weights_padding1() raises:
    """Numerical gradient check for grad_weights with padding=1.

    Treats the kernel as the variable being perturbed; x is held fixed.
    padding=1 with input (1,1,4,4) and kernel (1,1,3,3) produces (1,1,4,4)
    output, exercising all kernel-weight gradient accumulation paths including
    the padded boundary rows/columns.
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)

    # Non-uniform input
    for i in range(16):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    # Non-uniform kernel weights
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    def forward_weights(k: AnyTensor) raises -> AnyTensor:
        return conv2d(x, k, bias, stride=1, padding=1)

    def backward_weights(grad_out: AnyTensor, k: AnyTensor) raises -> AnyTensor:
        var grads = conv2d_backward(grad_out, x, k, stride=1, padding=1)
        return grads.grad_weights

    # Non-uniform grad_output
    var output = forward_weights(kernel)
    var grad_output = zeros_like(output)
    for i in range(16):
        grad_output.set(i, Float32(i % 4) * Float32(0.25) - Float32(0.3))

    check_gradient(
        forward_weights,
        backward_weights,
        kernel,
        grad_output,
        rtol=1e-2,
        atol=1e-2,
    )


def test_conv2d_backward_grad_input_padding2() raises:
    """Numerical gradient check for grad_input with padding=2.

    padding=2 with a (1,1,3,3) kernel and input (1,1,5,5) produces (1,1,7,7)
    output. Double-padded boundaries require the transposed convolution path to
    handle positions where the kernel extends entirely into the padding region,
    which is not exercised by padding=0 or padding=1 tests.
    Non-uniform grad_output avoids gradient cancellation at boundaries.
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)

    # Non-uniform input
    for i in range(25):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    # Non-uniform kernel weights
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    def forward_input(inp: AnyTensor) raises -> AnyTensor:
        return conv2d(inp, kernel, bias, stride=1, padding=2)

    def backward_input(grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = conv2d_backward(grad_out, inp, kernel, stride=1, padding=2)
        return grads.grad_input

    # Non-uniform grad_output over (1,1,7,7) = 49 elements
    var output = forward_input(x)
    var grad_output = zeros_like(output)
    for i in range(49):
        grad_output.set(i, Float32(i % 4) * Float32(0.25) - Float32(0.3))

    check_gradient(
        forward_input, backward_input, x, grad_output, rtol=1e-2, atol=1e-2
    )


def test_conv2d_backward_grad_weights_padding2() raises:
    """Numerical gradient check for grad_weights with padding=2.

    Treats the kernel as the variable being perturbed; x is held fixed.
    padding=2 with input (1,1,5,5) and kernel (1,1,3,3) produces (1,1,7,7)
    output. Double-padded boundaries exercise gradient accumulation paths
    where kernel positions overlap exclusively with the padding region,
    which is not covered by the padding=0 or padding=1 weight gradient tests.
    """
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)

    # Non-uniform input
    for i in range(25):
        x.set(i, Float32(Float32(i) * 0.1))

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)

    # Non-uniform kernel weights
    for i in range(9):
        kernel.set(i, Float32(Float32(i) * 0.05 + 0.1))

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    def forward_weights(k: AnyTensor) raises -> AnyTensor:
        return conv2d(x, k, bias, stride=1, padding=2)

    def backward_weights(grad_out: AnyTensor, k: AnyTensor) raises -> AnyTensor:
        var grads = conv2d_backward(grad_out, x, k, stride=1, padding=2)
        return grads.grad_weights

    # Non-uniform grad_output over (1,1,7,7) = 49 elements
    var output = forward_weights(kernel)
    var grad_output = zeros_like(output)
    for i in range(49):
        grad_output.set(i, Float32(i % 4) * Float32(0.25) - Float32(0.3))

    check_gradient(
        forward_weights,
        backward_weights,
        kernel,
        grad_output,
        rtol=1e-2,
        atol=1e-2,
    )


def main() raises:
    """Run numerical gradient tests for conv2d_backward with padding > 0."""
    print("Running conv2d_backward numerical gradient tests with padding > 0...")
    test_conv2d_backward_grad_input_padding1()
    print("✓ test_conv2d_backward_grad_input_padding1")
    test_conv2d_backward_grad_weights_padding1()
    print("✓ test_conv2d_backward_grad_weights_padding1")
    test_conv2d_backward_grad_input_padding2()
    print("✓ test_conv2d_backward_grad_input_padding2")
    test_conv2d_backward_grad_weights_padding2()
    print("✓ test_conv2d_backward_grad_weights_padding2")
    print("All conv2d_backward padding gradient tests passed!")
