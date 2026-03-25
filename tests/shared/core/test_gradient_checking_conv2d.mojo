# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Gradient checking tests for conv2d backward: grad_input, grad_weights, grad_bias.

Verifies all three conv2d backward outputs via finite-difference gradient
checking across three configurations: same-padding, strided, and multi-channel.

Uses default epsilon=3e-4 (GRADIENT_CHECK_EPSILON_FLOAT32) instead of 1e-4.
With epsilon=1e-4 and multi-channel configs (75 output elements, values ~44.5),
accumulated float32 precision error exceeds 1e-2 absolute tolerance. The default
3e-4 gives ~1.2% error, within tolerance. See issue #2704.

Test Coverage:
- Config A (same-padding): stride=1, padding=1, input (1,1,4,4), kernel (1,1,3,3)
- Config B (strided): stride=2, padding=0, input (1,1,7,7), kernel (1,1,3,3)
- Config C (multi-channel): stride=1, padding=1, in_channels=2, out_channels=3,
    input (1,2,5,5), kernel (3,2,3,3)

References:
    - Issue #3774: Add gradient checking for grad_kernel and grad_bias in conv2d
    - Follow-up from #3233
"""

from shared.core.conv import conv2d, conv2d_backward
from shared.tensor.any_tensor import AnyTensor, zeros
from shared.testing.gradient_checker import check_gradients
from shared.testing.assertions import assert_true


fn test_conv2d_same_padding_grad_input() raises:
    """Test conv2d grad_input with same-padding (stride=1, padding=1)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(inp, kernel, bias, stride=1, padding=1)

    fn backward_fn(grad_out: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, inp, kernel, stride=1, padding=1)
        return result.grad_input

    var passed = check_gradients(forward, backward_fn, x, tolerance=1e-2)
    assert_true(passed, "Conv2D same-padding grad_input check failed")


fn test_conv2d_same_padding_grad_weights() raises:
    """Test conv2d grad_weights with same-padding (stride=1, padding=1)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(k: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(x, k, bias, stride=1, padding=1)

    fn backward_fn(grad_out: AnyTensor, k: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, x, k, stride=1, padding=1)
        return result.grad_weights

    var passed = check_gradients(forward, backward_fn, kernel, tolerance=1e-2)
    assert_true(passed, "Conv2D same-padding grad_weights check failed")


fn test_conv2d_same_padding_grad_bias() raises:
    """Test conv2d grad_bias with same-padding (stride=1, padding=1)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var x = zeros(input_shape, DType.float32)
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(b: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(x, kernel, b, stride=1, padding=1)

    fn backward_fn(grad_out: AnyTensor, b: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, x, kernel, stride=1, padding=1)
        return result.grad_bias

    var passed = check_gradients(forward, backward_fn, bias, tolerance=1e-2)
    assert_true(passed, "Conv2D same-padding grad_bias check failed")


fn test_conv2d_strided_grad_input() raises:
    """Test conv2d grad_input with stride=2, padding=0, input (1,1,7,7)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(7)
    input_shape.append(7)
    var x = zeros(input_shape, DType.float32)
    for i in range(49):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(inp, kernel, bias, stride=2, padding=0)

    fn backward_fn(grad_out: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, inp, kernel, stride=2, padding=0)
        return result.grad_input

    var passed = check_gradients(forward, backward_fn, x, tolerance=1e-2)
    assert_true(passed, "Conv2D strided grad_input check failed")


fn test_conv2d_strided_grad_weights() raises:
    """Test conv2d grad_weights with stride=2, padding=0, input (1,1,7,7)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(7)
    input_shape.append(7)
    var x = zeros(input_shape, DType.float32)
    for i in range(49):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(k: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(x, k, bias, stride=2, padding=0)

    fn backward_fn(grad_out: AnyTensor, k: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, x, k, stride=2, padding=0)
        return result.grad_weights

    var passed = check_gradients(forward, backward_fn, kernel, tolerance=1e-2)
    assert_true(passed, "Conv2D strided grad_weights check failed")


fn test_conv2d_strided_grad_bias() raises:
    """Test conv2d grad_bias with stride=2, padding=0, input (1,1,7,7)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(7)
    input_shape.append(7)
    var x = zeros(input_shape, DType.float32)
    for i in range(49):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(1)
    kernel_shape.append(1)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(9):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(1)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(b: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(x, kernel, b, stride=2, padding=0)

    fn backward_fn(grad_out: AnyTensor, b: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, x, kernel, stride=2, padding=0)
        return result.grad_bias

    var passed = check_gradients(forward, backward_fn, bias, tolerance=1e-2)
    assert_true(passed, "Conv2D strided grad_bias check failed")


fn test_conv2d_multichannel_grad_input() raises:
    """Test conv2d grad_input with in_channels=2, out_channels=3, input (1,2,5,5)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)
    for i in range(50):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(3)
    kernel_shape.append(2)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(54):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(inp, kernel, bias, stride=1, padding=1)

    fn backward_fn(grad_out: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, inp, kernel, stride=1, padding=1)
        return result.grad_input

    var passed = check_gradients(forward, backward_fn, x, tolerance=1e-2)
    assert_true(passed, "Conv2D multi-channel grad_input check failed")


fn test_conv2d_multichannel_grad_weights() raises:
    """Test conv2d grad_weights with in_channels=2, out_channels=3, kernel (3,2,3,3)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)
    for i in range(50):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(3)
    kernel_shape.append(2)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(54):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(k: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(x, k, bias, stride=1, padding=1)

    fn backward_fn(grad_out: AnyTensor, k: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, x, k, stride=1, padding=1)
        return result.grad_weights

    var passed = check_gradients(forward, backward_fn, kernel, tolerance=1e-2)
    assert_true(passed, "Conv2D multi-channel grad_weights check failed")


fn test_conv2d_multichannel_grad_bias() raises:
    """Test conv2d grad_bias with out_channels=3, bias shape (3,)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(2)
    input_shape.append(5)
    input_shape.append(5)
    var x = zeros(input_shape, DType.float32)
    for i in range(50):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var kernel_shape = List[Int]()
    kernel_shape.append(3)
    kernel_shape.append(2)
    kernel_shape.append(3)
    kernel_shape.append(3)
    var kernel = zeros(kernel_shape, DType.float32)
    for i in range(54):
        kernel._data.bitcast[Float32]()[i] = Float32(i) * 0.05 + 0.1

    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)

    fn forward(b: AnyTensor) raises escaping -> AnyTensor:
        return conv2d(x, kernel, b, stride=1, padding=1)

    fn backward_fn(grad_out: AnyTensor, b: AnyTensor) raises escaping -> AnyTensor:
        var result = conv2d_backward(grad_out, x, kernel, stride=1, padding=1)
        return result.grad_bias

    var passed = check_gradients(forward, backward_fn, bias, tolerance=1e-2)
    assert_true(passed, "Conv2D multi-channel grad_bias check failed")


fn main() raises:
    """Run all 9 conv2d gradient checking tests."""
    print("Running Conv2D Gradient Checking Tests...")
    print("=" * 60)

    print("[1/9] Config A — same-padding grad_input...")
    test_conv2d_same_padding_grad_input()
    print("✓ PASSED")

    print("[2/9] Config A — same-padding grad_weights...")
    test_conv2d_same_padding_grad_weights()
    print("✓ PASSED")

    print("[3/9] Config A — same-padding grad_bias...")
    test_conv2d_same_padding_grad_bias()
    print("✓ PASSED")

    print("[4/9] Config B — strided grad_input...")
    test_conv2d_strided_grad_input()
    print("✓ PASSED")

    print("[5/9] Config B — strided grad_weights...")
    test_conv2d_strided_grad_weights()
    print("✓ PASSED")

    print("[6/9] Config B — strided grad_bias...")
    test_conv2d_strided_grad_bias()
    print("✓ PASSED")

    print("[7/9] Config C — multi-channel grad_input...")
    test_conv2d_multichannel_grad_input()
    print("✓ PASSED")

    print("[8/9] Config C — multi-channel grad_weights...")
    test_conv2d_multichannel_grad_weights()
    print("✓ PASSED")

    print("[9/9] Config C — multi-channel grad_bias...")
    test_conv2d_multichannel_grad_bias()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All 9 conv2d gradient checking tests PASSED! ✓")
    print("grad_input, grad_weights, grad_bias verified for 3 configs.")
