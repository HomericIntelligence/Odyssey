# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_layer_testers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for layer_testers module (Part 2 of 2)

Tests LayerTester backward pass functions:
- test_activation_layer_backward (tanh)
- test_linear_layer_backward
- test_conv_layer_backward
- test_batchnorm_layer
- test_batchnorm_layer_backward

Split from test_layer_testers.mojo per ADR-009 (≤10 fn test_ per file).
See test_layer_testers_part1.mojo for remaining tests.
"""

from math import isnan, isinf
from shared.testing.layer_testers import LayerTester
from shared.testing.special_values import (
    create_ones_tensor,
    create_special_value_tensor,
    create_seeded_random_tensor,
)
from shared.testing.assertions import assert_true
from shared.testing.tensor_factory import zeros_tensor
from shared.core.extensor import ExTensor


fn test_activation_layer_backward_tanh() raises:
    """Test Tanh activation backward pass with gradient checking."""
    # Use small shape to avoid timeout
    LayerTester.test_activation_layer_backward(
        shape=[2, 3, 4, 4], dtype=DType.float32, activation="tanh"
    )


fn test_linear_layer_backward_fp32() raises:
    """Test linear layer backward pass with FP32 gradient checking."""
    # Create small weight and bias tensors
    var weights = create_ones_tensor([10, 8], DType.float32)
    var bias = create_ones_tensor([10], DType.float32)

    # Test backward pass
    LayerTester.test_linear_layer_backward(
        in_features=8,
        out_features=10,
        weights=weights,
        bias=bias,
        dtype=DType.float32,
    )


fn test_conv_layer_backward_fp32() raises:
    """Test conv layer backward pass with FP32 gradient checking."""
    # Create small kernel and bias tensors (small spatial dimensions to avoid timeout)
    var weights = create_ones_tensor(
        [8, 3, 3, 3], DType.float32
    )  # 8 filters, 3x3
    var bias = create_ones_tensor([8], DType.float32)

    # Test backward pass with small input size
    LayerTester.test_conv_layer_backward(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        input_h=8,
        input_w=8,
        weights=weights,
        bias=bias,
        dtype=DType.float32,
        stride=1,
        padding=1,
    )


fn test_batchnorm_layer_training_mode() raises:
    """Test BatchNorm layer in training mode."""
    var num_features = 16
    var input_shape = List[Int]()
    input_shape.append(2)  # batch size
    input_shape.append(16)  # channels
    input_shape.append(4)  # height
    input_shape.append(4)  # width

    var gamma = create_ones_tensor([num_features], DType.float32)
    var beta = create_ones_tensor([num_features], DType.float32)
    var running_mean = create_ones_tensor([num_features], DType.float32)
    var running_var = create_ones_tensor([num_features], DType.float32)

    # Test training mode
    LayerTester.test_batchnorm_layer(
        num_features=num_features,
        input_shape=input_shape,
        gamma=gamma,
        beta=beta,
        running_mean=running_mean,
        running_var=running_var,
        dtype=DType.float32,
        training_mode=True,
    )


fn test_batchnorm_layer_inference_mode() raises:
    """Test BatchNorm layer in inference mode."""
    var num_features = 16
    var input_shape = List[Int]()
    input_shape.append(2)  # batch size
    input_shape.append(16)  # channels
    input_shape.append(4)  # height
    input_shape.append(4)  # width

    var gamma = create_ones_tensor([num_features], DType.float32)
    var beta = create_ones_tensor([num_features], DType.float32)
    var running_mean = create_ones_tensor([num_features], DType.float32)
    var running_var = create_ones_tensor([num_features], DType.float32)

    # Test inference mode
    LayerTester.test_batchnorm_layer(
        num_features=num_features,
        input_shape=input_shape,
        gamma=gamma,
        beta=beta,
        running_mean=running_mean,
        running_var=running_var,
        dtype=DType.float32,
        training_mode=False,
    )


fn test_batchnorm_layer_backward_fp32() raises:
    """Test BatchNorm backward pass with FP32."""
    var num_features = 16
    var input_shape = List[Int]()
    input_shape.append(2)  # batch size
    input_shape.append(16)  # channels
    input_shape.append(4)  # height
    input_shape.append(4)  # width

    var gamma = create_ones_tensor([num_features], DType.float32)
    var beta = create_ones_tensor([num_features], DType.float32)
    var running_mean = create_ones_tensor([num_features], DType.float32)
    var running_var = create_ones_tensor([num_features], DType.float32)

    # Test backward pass
    LayerTester.test_batchnorm_layer_backward(
        num_features=num_features,
        input_shape=input_shape,
        gamma=gamma,
        beta=beta,
        running_mean=running_mean,
        running_var=running_var,
        dtype=DType.float32,
    )


fn main() raises:
    print("Testing layer_testers module (Part 2)...")

    # Test backward pass methods
    test_activation_layer_backward_tanh()
    print("✓ test_activation_layer_backward_tanh")

    test_linear_layer_backward_fp32()
    print("✓ test_linear_layer_backward_fp32")

    test_conv_layer_backward_fp32()
    print("✓ test_conv_layer_backward_fp32")

    test_batchnorm_layer_training_mode()
    print("✓ test_batchnorm_layer_training_mode")

    test_batchnorm_layer_inference_mode()
    print("✓ test_batchnorm_layer_inference_mode")

    test_batchnorm_layer_backward_fp32()
    print("✓ test_batchnorm_layer_backward_fp32")

    print("\n✅ All layer_testers Part 2 tests passed!")
    print("\nBackward pass testing:")
    print("- Uses seeded random tensors for reproducibility")
    print("- Validates numerical gradients via finite differences")
    print("- Checks for NaN/Inf in gradient outputs")
    print("- Supports ReLU, Sigmoid, Tanh activations")
