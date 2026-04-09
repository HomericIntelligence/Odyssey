"""Unit tests for ReLULayer.

Tests cover:
- Forward pass: verifies max(0, x) behavior
- Backward pass: verifies gradient computation
- Various input shapes and dtypes
"""

from testing import assert_true, assert_false
from shared.core.layers import ReLULayer
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full


def test_relu_forward_basic() raises:
    """Test ReLU forward pass with basic inputs."""
    var layer = ReLULayer()

    # Create input with mixed positive and negative values
    var input = AnyTensor([5], DType.float32)
    var input_values: List[Float32] = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for i in range(5):
        input.set(i, Float32(input_values[i]))

    var output = layer.forward(input)

    # Expected: [0, 0, 0, 1, 2]
    var expected: List[Float32] = [0.0, 0.0, 0.0, 1.0, 2.0]
    for i in range(5):
        var out_val = output._data.bitcast[Float32]()[i]
        assert_true(
            out_val == expected[i], "ReLU output mismatch at index " + String(i)
        )

    print("✓ test_relu_forward_basic passed")


def test_relu_forward_all_negative() raises:
    """Test ReLU forward pass with all negative inputs."""
    var layer = ReLULayer()

    var input = AnyTensor([4], DType.float32)
    var input_values: List[Float32] = [-5.0, -2.0, -0.1, -10.0]
    for i in range(4):
        input.set(i, Float32(input_values[i]))

    var output = layer.forward(input)

    # Expected: [0, 0, 0, 0]
    for i in range(4):
        var out_val = output._data.bitcast[Float32]()[i]
        assert_true(out_val == 0.0, "ReLU should zero negative values")

    print("✓ test_relu_forward_all_negative passed")


def test_relu_forward_all_positive() raises:
    """Test ReLU forward pass with all positive inputs."""
    var layer = ReLULayer()

    var input = AnyTensor([4], DType.float32)
    var input_values: List[Float32] = [0.5, 1.0, 5.0, 10.0]
    for i in range(4):
        input.set(i, Float32(input_values[i]))

    var output = layer.forward(input)

    # Expected: [0.5, 1.0, 5.0, 10.0]
    for i in range(4):
        var out_val = output._data.bitcast[Float32]()[i]
        assert_true(
            out_val == input_values[i], "ReLU should preserve positive values"
        )

    print("✓ test_relu_forward_all_positive passed")


def test_relu_forward_batch() raises:
    """Test ReLU forward pass with 2D batch input."""
    var layer = ReLULayer()

    # Create 2x3 batch
    var input = AnyTensor([2, 3], DType.float32)
    var input_values: List[Float32] = [-1.0, 0.5, -0.5, 2.0, -2.0, 1.5]
    for i in range(6):
        input.set(i, Float32(input_values[i]))

    var output = layer.forward(input)

    var expected: List[Float32] = [0.0, 0.5, 0.0, 2.0, 0.0, 1.5]
    for i in range(6):
        var out_val = output._data.bitcast[Float32]()[i]
        assert_true(out_val == expected[i], "Batch ReLU mismatch")

    print("✓ test_relu_forward_batch passed")


def test_relu_backward_basic() raises:
    """Test ReLU backward pass with basic inputs."""
    var layer = ReLULayer()

    # Input: [-2, -1, 0, 1, 2]
    var input = AnyTensor([5], DType.float32)
    var input_values: List[Float32] = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for i in range(5):
        input.set(i, Float32(input_values[i]))

    # Gradient from upstream: [0.1, 0.2, 0.3, 0.4, 0.5]
    var grad_output = AnyTensor([5], DType.float32)
    var grad_values: List[Float32] = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range(5):
        grad_output.set(i, Float32(grad_values[i]))

    var grad_input = layer.backward(grad_output, input)

    # Expected: [0, 0, 0, 0.4, 0.5] (pass through only where input > 0)
    var expected: List[Float32] = [0.0, 0.0, 0.0, 0.4, 0.5]
    for i in range(5):
        var grad_val = grad_input._data.bitcast[Float32]()[i]
        assert_true(
            grad_val == expected[i],
            "ReLU backward mismatch at index " + String(i),
        )

    print("✓ test_relu_backward_basic passed")


def test_relu_backward_all_positive() raises:
    """Test ReLU backward pass with all positive inputs."""
    var layer = ReLULayer()

    var input = AnyTensor([4], DType.float32)
    var input_values: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    for i in range(4):
        input.set(i, Float32(input_values[i]))

    var grad_output = AnyTensor([4], DType.float32)
    var grad_values: List[Float32] = [0.1, 0.2, 0.3, 0.4]
    for i in range(4):
        grad_output.set(i, Float32(grad_values[i]))

    var grad_input = layer.backward(grad_output, input)

    # Expected: [0.1, 0.2, 0.3, 0.4] (all pass through)
    for i in range(4):
        var grad_val = grad_input._data.bitcast[Float32]()[i]
        assert_true(
            grad_val == grad_values[i],
            "ReLU backward should pass through all positive",
        )

    print("✓ test_relu_backward_all_positive passed")


def test_relu_backward_all_negative() raises:
    """Test ReLU backward pass with all negative inputs."""
    var layer = ReLULayer()

    var input = AnyTensor([4], DType.float32)
    var input_values: List[Float32] = [-1.0, -2.0, -3.0, -4.0]
    for i in range(4):
        input.set(i, Float32(input_values[i]))

    var grad_output = AnyTensor([4], DType.float32)
    var grad_values: List[Float32] = [0.1, 0.2, 0.3, 0.4]
    for i in range(4):
        grad_output.set(i, Float32(grad_values[i]))

    var grad_input = layer.backward(grad_output, input)

    # Expected: [0, 0, 0, 0] (all blocked)
    for i in range(4):
        var grad_val = grad_input._data.bitcast[Float32]()[i]
        assert_true(grad_val == 0.0, "ReLU backward should block all negative")

    print("✓ test_relu_backward_all_negative passed")


def test_relu_parameters() raises:
    """Test that ReLU has no learnable parameters."""
    var layer = ReLULayer()
    var params = layer.parameters()
    assert_true(len(params) == 0, "ReLU should have no parameters")

    print("✓ test_relu_parameters passed")


def test_relu_forward_float64() raises:
    """Test ReLU forward pass with float64 dtype."""
    var layer = ReLULayer()

    var input = AnyTensor([4], DType.float64)
    var input_values: List[Float64] = [-1.5, 0.5, -2.5, 3.5]
    for i in range(4):
        input.set(i, Float64(input_values[i]))

    var output = layer.forward(input)

    var expected: List[Float64] = [0.0, 0.5, 0.0, 3.5]
    for i in range(4):
        var out_val = output._data.bitcast[Float64]()[i]
        assert_true(
            out_val == expected[i],
            "ReLU float64 mismatch at index " + String(i),
        )

    print("✓ test_relu_forward_float64 passed")


def test_relu_backward_float64() raises:
    """Test ReLU backward pass with float64 dtype."""
    var layer = ReLULayer()

    var input = AnyTensor([4], DType.float64)
    var input_values: List[Float64] = [-1.0, 0.0, 1.0, 2.0]
    for i in range(4):
        input.set(i, Float64(input_values[i]))

    var grad_output = AnyTensor([4], DType.float64)
    var grad_values: List[Float64] = [0.1, 0.2, 0.3, 0.4]
    for i in range(4):
        grad_output.set(i, Float64(grad_values[i]))

    var grad_input = layer.backward(grad_output, input)

    var expected: List[Float64] = [0.0, 0.0, 0.3, 0.4]
    for i in range(4):
        var grad_val = grad_input._data.bitcast[Float64]()[i]
        assert_true(
            grad_val == expected[i],
            "ReLU float64 backward mismatch at index " + String(i),
        )

    print("✓ test_relu_backward_float64 passed")


def main():
    """Run all ReLU tests."""
    print("Running ReLULayer tests...")
    try:
        test_relu_forward_basic()
        test_relu_forward_all_negative()
        test_relu_forward_all_positive()
        test_relu_forward_batch()
        test_relu_backward_basic()
        test_relu_backward_all_positive()
        test_relu_backward_all_negative()
        test_relu_parameters()
        test_relu_forward_float64()
        test_relu_backward_float64()
        print("\nAll ReLULayer tests passed!")
    except e:
        print("Test failed:", e)
