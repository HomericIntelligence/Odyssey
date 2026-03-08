"""End-to-end tests for MobileNetV1 model (Part 2 of 2).

Tests cover edge cases and robustness:
- Inference mode with BatchNorm eval mode
- Different batch sizes
- Gradient flow verification

MobileNetV1 Architecture (Simplified for Testing):
Input: (batch, 3, 224, 224) - but tests use CIFAR-10 (32x32)
  Conv 3x3, stride 2 -> (batch, 32, 16, 16)
  Block 1: 32->64, stride 1
  Block 2: 64->128, stride 2
  Block 3: 128->128, stride 1
  Block 4: 128->256, stride 2
  Block 5: 256->512, stride 1
  Block 6: 512->512, stride 2 (repeated 5x)
  GlobalAvgPool -> (batch, 1024)
  FC -> (batch, num_classes)

Testing Strategy:
- Use small images (8x8 or 16x16) to keep tests fast
- Use small batch sizes (2-4)
- Test selective blocks rather than full model
- Verify loss computation and gradient flow

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_mobilenetv1_e2e.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones, full, randn
from shared.core.conv import (
    conv2d,
    depthwise_conv2d,
    depthwise_separable_conv2d,
)
from shared.core.activation import relu
from shared.core.layers.batchnorm import BatchNorm2dLayer
from shared.core.pooling import global_avgpool2d
from shared.core.loss import cross_entropy_loss
from shared.core.linear import Linear


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


fn test_mobilenetv1_inference_mode() raises:
    """Test inference mode with BatchNorm in eval mode.

    In inference mode, BatchNorm uses running statistics instead of batch stats.
    """
    var batch_size = 1
    var num_channels = 32
    var height = 8
    var width = 8

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(num_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # BatchNorm in inference mode
    var bn = BatchNorm2dLayer(num_channels)
    var output = bn.forward(input, training=False)

    # Verify output shape unchanged
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], num_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


fn test_mobilenetv1_different_batch_sizes() raises:
    """Test forward pass with different batch sizes.

    Batch sizes: 1, 2, 4
    """
    var in_channels = 32
    var height = 8
    var width = 8

    for batch_size in [1, 2, 4]:
        # Create input
        var input_shape = List[Int]()
        input_shape.append(batch_size)
        input_shape.append(in_channels)
        input_shape.append(height)
        input_shape.append(width)
        var input = ones(input_shape, DType.float32)

        # Depthwise conv
        var dw_kernel_shape = List[Int]()
        dw_kernel_shape.append(in_channels)
        dw_kernel_shape.append(1)
        dw_kernel_shape.append(3)
        dw_kernel_shape.append(3)
        var dw_kernel = ones(dw_kernel_shape, DType.float32)
        var dw_bias = zeros([in_channels], DType.float32)

        var output = depthwise_conv2d(
            input, dw_kernel, dw_bias, stride=1, padding=1
        )

        # Verify output batch size matches
        var out_shape = output.shape()
        assert_equal(out_shape[0], batch_size)


fn test_mobilenetv1_gradient_flow_through_convs() raises:
    """Test that gradients flow correctly through conv operations.

    Verifies gradient propagation through depthwise -> pointwise convolutions.
    """
    var batch_size = 1
    var channels = 8
    var height = 4
    var width = 4

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Forward: depthwise -> pointwise
    var dw_kernel_shape = List[Int]()
    dw_kernel_shape.append(channels)
    dw_kernel_shape.append(1)
    dw_kernel_shape.append(3)
    dw_kernel_shape.append(3)
    var dw_kernel = ones(dw_kernel_shape, DType.float32)
    var dw_bias = zeros([channels], DType.float32)

    var dw_out = depthwise_conv2d(
        input, dw_kernel, dw_bias, stride=1, padding=1
    )

    var pw_kernel_shape = List[Int]()
    pw_kernel_shape.append(channels)
    pw_kernel_shape.append(channels)
    pw_kernel_shape.append(1)
    pw_kernel_shape.append(1)
    var pw_kernel = ones(pw_kernel_shape, DType.float32)
    var pw_bias = zeros([channels], DType.float32)

    var output = conv2d(dw_out, pw_kernel, pw_bias, stride=1, padding=0)

    # Create grad_output
    var grad_output = ones(output.shape(), DType.float32)

    # Backward: verify shapes propagate correctly
    from shared.core.conv import conv2d_backward, depthwise_conv2d_backward

    # Backward through pointwise conv
    var pw_grad = conv2d_backward(
        grad_output, dw_out, pw_kernel, stride=1, padding=0
    )
    var grad_pw_in = pw_grad.grad_input
    var grad_pw_in_shape = grad_pw_in.shape()
    assert_equal(grad_pw_in_shape[0], batch_size)
    assert_equal(grad_pw_in_shape[1], channels)

    # Backward through depthwise conv
    var dw_grad = depthwise_conv2d_backward(
        grad_pw_in, input, dw_kernel, stride=1, padding=1
    )
    var grad_dw_in = dw_grad.grad_input
    var grad_dw_in_shape = grad_dw_in.shape()
    assert_equal(grad_dw_in_shape[0], batch_size)
    assert_equal(grad_dw_in_shape[1], channels)
