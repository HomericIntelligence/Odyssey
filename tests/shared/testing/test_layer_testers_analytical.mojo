"""Test gradient validation in layer testers.

Demonstrates the new validate_analytical parameter introduced in #3011.
This test file shows how to enable analytical gradient validation in layer tests.

Example usage:
    pixi run mojo tests/shared/testing/test_layer_testers_analytical.mojo
"""

from shared.testing import LayerTester
from shared.core.extensor import ExTensor, zeros
from shared.core.initializers import kaiming_uniform


fn test_conv_backward_with_analytical_validation() raises:
    """Test conv layer backward with analytical gradient validation enabled."""
    # Create small conv layer parameters: 3→8 channels, 3x3 kernel
    var in_channels = 3
    var out_channels = 8
    var kernel_size = 3
    var dtype = DType.float32

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, dtype)

    # Test backward pass WITH analytical gradient validation
    LayerTester.test_conv_layer_backward(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        input_h=8,  # Small size to avoid timeout
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=1,
        validate_analytical=True,  # NEW: Enable analytical validation
    )


fn test_linear_backward_with_analytical_validation() raises:
    """Test linear layer backward with analytical gradient validation enabled.
    """
    # Create small linear layer: 16→10
    var in_features = 16
    var out_features = 10
    var dtype = DType.float32

    # Create weights
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = kaiming_uniform(
        in_features, out_features, weights_shape, dtype=dtype
    )

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, dtype)

    # Test backward pass WITH analytical gradient validation
    LayerTester.test_linear_layer_backward(
        in_features=in_features,
        out_features=out_features,
        weights=weights,
        bias=bias,
        dtype=dtype,
        validate_analytical=True,  # NEW: Enable analytical validation
    )


fn test_relu_backward_with_analytical_validation() raises:
    """Test ReLU activation backward with analytical gradient validation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    shape.append(4)
    var dtype = DType.float32

    # Test backward pass WITH analytical gradient validation
    LayerTester.test_activation_layer_backward(
        shape=shape,
        dtype=dtype,
        activation="relu",
        validate_analytical=True,  # NEW: Enable analytical validation
    )


fn test_sigmoid_backward_with_analytical_validation() raises:
    """Test Sigmoid activation backward with analytical gradient validation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    shape.append(4)
    var dtype = DType.float32

    # Test backward pass WITH analytical gradient validation
    LayerTester.test_activation_layer_backward(
        shape=shape,
        dtype=dtype,
        activation="sigmoid",
        validate_analytical=True,  # NEW: Enable analytical validation
    )


fn test_tanh_backward_with_analytical_validation() raises:
    """Test Tanh activation backward with analytical gradient validation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    shape.append(4)
    var dtype = DType.float32

    # Test backward pass WITH analytical gradient validation
    LayerTester.test_activation_layer_backward(
        shape=shape,
        dtype=dtype,
        activation="tanh",
        validate_analytical=True,  # NEW: Enable analytical validation
    )


fn main() raises:
    """Run all analytical gradient validation tests."""
    print("Testing Layer Testers with Analytical Gradient Validation...")
    print("=" * 70)

    print("\n[1/5] Testing Conv2D backward with analytical validation...")
    test_conv_backward_with_analytical_validation()
    print("✓ PASSED")

    print("[2/5] Testing Linear backward with analytical validation...")
    test_linear_backward_with_analytical_validation()
    print("✓ PASSED")

    print("[3/5] Testing ReLU backward with analytical validation...")
    test_relu_backward_with_analytical_validation()
    print("✓ PASSED")

    print("[4/5] Testing Sigmoid backward with analytical validation...")
    test_sigmoid_backward_with_analytical_validation()
    print("✓ PASSED")

    print("[5/5] Testing Tanh backward with analytical validation...")
    test_tanh_backward_with_analytical_validation()
    print("✓ PASSED")

    print("\n" + "=" * 70)
    print("All 5 analytical gradient validation tests PASSED! ✓")
    print("validate_analytical parameter is working correctly.")
