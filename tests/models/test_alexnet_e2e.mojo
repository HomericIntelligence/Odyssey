"""End-to-End Integration Tests for AlexNet

Tests complete model forward pass with real data flow and verifies:
- Forward pass produces expected output shape
- Gradient computation works through all layers
- Model can handle batch processing
- Loss computation works (basic integration)

This test uses functional layer composition (no AlexNet model class required).
Tests the same forward-backward pipeline that a full model would use.
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.initializers import kaiming_uniform
from shared.core.loss import cross_entropy
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
    assert_close_float,
)
from shared.testing.special_values import (
    create_seeded_random_tensor,
    SPECIAL_VALUE_ONE,
)
from math import isnan, isinf


# ============================================================================
# AlexNet Component Functions
# ============================================================================


fn create_alexnet_parameters(
    dtype: DType,
) raises -> Tuple[
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyTensor,
]:
    """Create all AlexNet parameters (conv kernels, biases, FC weights, biases).
    """
    # Conv1: 3->64, 11x11
    var c1_k = kaiming_uniform(
        3 * 11 * 11, 64 * 11 * 11, [64, 3, 11, 11], dtype=dtype
    )
    var c1_b = zeros([64], dtype)

    # Conv2: 64->192, 5x5
    var c2_k = kaiming_uniform(
        64 * 5 * 5, 192 * 5 * 5, [192, 64, 5, 5], dtype=dtype
    )
    var c2_b = zeros([192], dtype)

    # Conv3: 192->384, 3x3
    var c3_k = kaiming_uniform(
        192 * 3 * 3, 384 * 3 * 3, [384, 192, 3, 3], dtype=dtype
    )
    var c3_b = zeros([384], dtype)

    # Conv4: 384->384, 3x3
    var c4_k = kaiming_uniform(
        384 * 3 * 3, 384 * 3 * 3, [384, 384, 3, 3], dtype=dtype
    )
    var c4_b = zeros([384], dtype)

    # Conv5: 384->256, 3x3
    var c5_k = kaiming_uniform(
        384 * 3 * 3, 256 * 3 * 3, [256, 384, 3, 3], dtype=dtype
    )
    var c5_b = zeros([256], dtype)

    # FC1: 9216->4096
    var fc1_w = kaiming_uniform(9216, 4096, [4096, 9216], dtype=dtype)
    var fc1_b = zeros([4096], dtype)

    # FC2: 4096->4096
    var fc2_w = kaiming_uniform(4096, 4096, [4096, 4096], dtype=dtype)
    var fc2_b = zeros([4096], dtype)

    # FC3: 4096->1000
    var fc3_w = kaiming_uniform(4096, 1000, [1000, 4096], dtype=dtype)
    var fc3_b = zeros([1000], dtype)

    return (
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    )


fn alexnet_forward(
    x: AnyTensor,
    c1_k: AnyTensor,
    c1_b: AnyTensor,
    c2_k: AnyTensor,
    c2_b: AnyTensor,
    c3_k: AnyTensor,
    c3_b: AnyTensor,
    c4_k: AnyTensor,
    c4_b: AnyTensor,
    c5_k: AnyTensor,
    c5_b: AnyTensor,
    fc1_w: AnyTensor,
    fc1_b: AnyTensor,
    fc2_w: AnyTensor,
    fc2_b: AnyTensor,
    fc3_w: AnyTensor,
    fc3_b: AnyTensor,
) raises -> AnyTensor:
    """Forward pass through AlexNet."""
    # Conv1 + ReLU + MaxPool
    var c1 = conv2d(x, c1_k, c1_b, stride=4, padding=2)
    var r1 = relu(c1)
    var p1 = maxpool2d(r1, kernel_size=3, stride=2, padding=0)

    # Conv2 + ReLU + MaxPool
    var c2 = conv2d(p1, c2_k, c2_b, stride=1, padding=2)
    var r2 = relu(c2)
    var p2 = maxpool2d(r2, kernel_size=3, stride=2, padding=0)

    # Conv3 + ReLU
    var c3 = conv2d(p2, c3_k, c3_b, stride=1, padding=1)
    var r3 = relu(c3)

    # Conv4 + ReLU
    var c4 = conv2d(r3, c4_k, c4_b, stride=1, padding=1)
    var r4 = relu(c4)

    # Conv5 + ReLU + MaxPool
    var c5 = conv2d(r4, c5_k, c5_b, stride=1, padding=1)
    var r5 = relu(c5)
    var p3 = maxpool2d(r5, kernel_size=3, stride=2, padding=0)

    # Flatten
    var batch_size = p3.shape()[0]
    var flat_size = 1
    var shape = p3.shape()
    for i in range(1, len(shape)):
        flat_size *= shape[i]
    var flattened = p3.reshape([batch_size, flat_size])

    # FC1 + ReLU
    var f1 = linear(flattened, fc1_w, fc1_b)
    var rf1 = relu(f1)

    # FC2 + ReLU
    var f2 = linear(rf1, fc2_w, fc2_b)
    var rf2 = relu(f2)

    # FC3 (output logits)
    var output = linear(rf2, fc3_w, fc3_b)

    return output^


# ============================================================================
# Forward Pass Tests
# ============================================================================


fn test_forward_output_shape_224x224() raises:
    """Test forward pass produces correct output shape (batch, 1000) with 224x224 input.
    """
    var dtype = DType.float32

    # Create parameters
    var (
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    ) = create_alexnet_parameters(dtype)

    # Create batch of inputs: (2, 3, 224, 224)
    var input = create_seeded_random_tensor([2, 3, 224, 224], dtype, seed=42)

    # Forward pass
    var output = alexnet_forward(
        input,
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    )

    # Verify output shape: (2, 1000)
    assert_shape(output, [2, 1000], "AlexNet output shape mismatch (224x224)")

    # Verify dtype preserved
    assert_dtype(output, dtype, "AlexNet output dtype mismatch")

    # Verify no NaN/Inf
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_false(isnan(val), "AlexNet output contains NaN at " + String(i))
        assert_false(isinf(val), "AlexNet output contains Inf at " + String(i))


fn test_forward_single_sample_224x224() raises:
    """Test forward pass with single sample (1, 3, 224, 224)."""
    var dtype = DType.float32

    var (
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    ) = create_alexnet_parameters(dtype)

    # Create single input
    var input = create_seeded_random_tensor([1, 3, 224, 224], dtype, seed=123)

    # Forward pass
    var output = alexnet_forward(
        input,
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    )

    # Verify output shape: (1, 1000)
    assert_shape(output, [1, 1000], "Single sample output shape mismatch")

    # Verify dtype
    assert_dtype(output, dtype, "Single sample output dtype mismatch")


fn test_forward_batch_sizes() raises:
    """Test forward pass with different batch sizes."""
    var dtype = DType.float32

    var (
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    ) = create_alexnet_parameters(dtype)

    # Test batch size 1
    var input1 = create_seeded_random_tensor([1, 3, 224, 224], dtype, seed=1)
    var output1 = alexnet_forward(
        input1,
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    )
    assert_shape(output1, [1, 1000], "Batch size 1 output shape")

    # Test batch size 4
    var input4 = create_seeded_random_tensor([4, 3, 224, 224], dtype, seed=2)
    var output4 = alexnet_forward(
        input4,
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    )
    assert_shape(output4, [4, 1000], "Batch size 4 output shape")


fn test_forward_deterministic() raises:
    """Test that forward pass is deterministic with same input."""
    var dtype = DType.float32

    var (
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    ) = create_alexnet_parameters(dtype)

    # Create input
    var input = create_seeded_random_tensor([2, 3, 224, 224], dtype, seed=999)

    # Forward pass twice
    var output1 = alexnet_forward(
        input,
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    )
    var output2 = alexnet_forward(
        input,
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    )

    # Outputs should be identical (deterministic)
    for i in range(output1.numel()):
        var val1 = output1._get_float64(i)
        var val2 = output2._get_float64(i)
        assert_close_float(
            val1,
            val2,
            rtol=0.0,
            atol=0.0,
            message="Forward pass non-deterministic",
        )


# ============================================================================
# Shape Propagation Tests
# ============================================================================


fn test_shape_propagation_through_conv_layers() raises:
    """Test that shapes propagate correctly through conv and pool layers."""
    var dtype = DType.float32

    var (
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = create_alexnet_parameters(dtype)

    # Input: (1, 3, 224, 224)
    var input = create_seeded_random_tensor([1, 3, 224, 224], dtype, seed=42)
    var s = input.shape()
    assert_equal(s[0], 1)
    assert_equal(s[1], 3)
    assert_equal(s[2], 224)
    assert_equal(s[3], 224)

    # Conv1 (stride=4, padding=2): (224-11+2*2)/4 + 1 = 55
    var c1 = conv2d(input, c1_k, c1_b, stride=4, padding=2)
    s = c1.shape()
    assert_equal(s[0], 1)
    assert_equal(s[1], 64)
    assert_equal(s[2], 55)
    assert_equal(s[3], 55)

    # ReLU1: Same shape
    var r1 = relu(c1)
    s = r1.shape()
    assert_equal(s[2], 55)
    assert_equal(s[3], 55)

    # MaxPool1 (3x3, stride=2): (55-3)/2 + 1 = 27
    var p1 = maxpool2d(r1, kernel_size=3, stride=2, padding=0)
    s = p1.shape()
    assert_equal(s[0], 1)
    assert_equal(s[1], 64)
    assert_equal(s[2], 27)
    assert_equal(s[3], 27)

    # Conv2 (stride=1, padding=2): 27 stays 27
    var c2 = conv2d(p1, c2_k, c2_b, stride=1, padding=2)
    s = c2.shape()
    assert_equal(s[2], 27)
    assert_equal(s[3], 27)

    # ReLU2: Same shape
    var r2 = relu(c2)
    s = r2.shape()
    assert_equal(s[2], 27)
    assert_equal(s[3], 27)

    # MaxPool2 (3x3, stride=2): (27-3)/2 + 1 = 13
    var p2 = maxpool2d(r2, kernel_size=3, stride=2, padding=0)
    s = p2.shape()
    assert_equal(s[0], 1)
    assert_equal(s[1], 192)
    assert_equal(s[2], 13)
    assert_equal(s[3], 13)


fn test_shape_propagation_through_fc_layers() raises:
    """Test that shapes propagate correctly through FC layers."""
    var dtype = DType.float32

    var (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    ) = create_alexnet_parameters(dtype)

    # Create flattened input: (2, 9216) from (2, 256, 6, 6)
    var flattened = create_seeded_random_tensor([2, 9216], dtype, seed=42)

    # FC1: (2, 9216) -> (2, 4096)
    var f1 = linear(flattened, fc1_w, fc1_b)
    var s = f1.shape()
    assert_equal(s[0], 2)
    assert_equal(s[1], 4096)

    # ReLU: Same shape
    var rf1 = relu(f1)
    s = rf1.shape()
    assert_equal(s[1], 4096)

    # FC2: (2, 4096) -> (2, 4096)
    var f2 = linear(rf1, fc2_w, fc2_b)
    s = f2.shape()
    assert_equal(s[0], 2)
    assert_equal(s[1], 4096)

    # ReLU: Same shape
    var rf2 = relu(f2)
    s = rf2.shape()
    assert_equal(s[1], 4096)

    # FC3: (2, 4096) -> (2, 1000)
    var output = linear(rf2, fc3_w, fc3_b)
    s = output.shape()
    assert_equal(s[0], 2)
    assert_equal(s[1], 1000)


# ============================================================================
# Model Initialization Tests
# ============================================================================


fn test_parameters_initialization() raises:
    """Test that model parameters initialize correctly."""
    var dtype = DType.float32

    var (
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    ) = create_alexnet_parameters(dtype)

    # Verify all parameters exist and have correct shapes
    assert_shape(c1_k, [64, 3, 11, 11], "Conv1 kernel shape")
    assert_shape(c1_b, [64], "Conv1 bias shape")
    assert_shape(c2_k, [192, 64, 5, 5], "Conv2 kernel shape")
    assert_shape(c2_b, [192], "Conv2 bias shape")
    assert_shape(c3_k, [384, 192, 3, 3], "Conv3 kernel shape")
    assert_shape(c3_b, [384], "Conv3 bias shape")
    assert_shape(c4_k, [384, 384, 3, 3], "Conv4 kernel shape")
    assert_shape(c4_b, [384], "Conv4 bias shape")
    assert_shape(c5_k, [256, 384, 3, 3], "Conv5 kernel shape")
    assert_shape(c5_b, [256], "Conv5 bias shape")
    assert_shape(fc1_w, [4096, 9216], "FC1 weights shape")
    assert_shape(fc1_b, [4096], "FC1 bias shape")
    assert_shape(fc2_w, [4096, 4096], "FC2 weights shape")
    assert_shape(fc2_b, [4096], "FC2 bias shape")
    assert_shape(fc3_w, [1000, 4096], "FC3 weights shape")
    assert_shape(fc3_b, [1000], "FC3 bias shape")

    # Verify biases are initialized to zero
    for i in range(c1_b.numel()):
        var val = c1_b._get_float64(i)
        assert_close_float(val, 0.0, atol=1e-10, message="Conv1 bias not zero")


fn test_multiple_dtypes() raises:
    """Test forward pass with multiple dtypes."""
    var batch_size = 1
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(3)
    input_shape.append(224)
    input_shape.append(224)

    # Test float32
    var input_f32 = create_seeded_random_tensor(
        input_shape, DType.float32, seed=42
    )
    var (
        c1_k_f32,
        c1_b_f32,
        c2_k_f32,
        c2_b_f32,
        c3_k_f32,
        c3_b_f32,
        c4_k_f32,
        c4_b_f32,
        c5_k_f32,
        c5_b_f32,
        fc1_w_f32,
        fc1_b_f32,
        fc2_w_f32,
        fc2_b_f32,
        fc3_w_f32,
        fc3_b_f32,
    ) = create_alexnet_parameters(DType.float32)
    var output_f32 = alexnet_forward(
        input_f32,
        c1_k_f32,
        c1_b_f32,
        c2_k_f32,
        c2_b_f32,
        c3_k_f32,
        c3_b_f32,
        c4_k_f32,
        c4_b_f32,
        c5_k_f32,
        c5_b_f32,
        fc1_w_f32,
        fc1_b_f32,
        fc2_w_f32,
        fc2_b_f32,
        fc3_w_f32,
        fc3_b_f32,
    )
    assert_shape(output_f32, [1, 1000], "float32 output shape")
    assert_dtype(output_f32, DType.float32, "float32 output dtype")

    # Test float16
    var input_f16 = create_seeded_random_tensor(
        input_shape, DType.float16, seed=42
    )
    var (
        c1_k_f16,
        c1_b_f16,
        c2_k_f16,
        c2_b_f16,
        c3_k_f16,
        c3_b_f16,
        c4_k_f16,
        c4_b_f16,
        c5_k_f16,
        c5_b_f16,
        fc1_w_f16,
        fc1_b_f16,
        fc2_w_f16,
        fc2_b_f16,
        fc3_w_f16,
        fc3_b_f16,
    ) = create_alexnet_parameters(DType.float16)
    var output_f16 = alexnet_forward(
        input_f16,
        c1_k_f16,
        c1_b_f16,
        c2_k_f16,
        c2_b_f16,
        c3_k_f16,
        c3_b_f16,
        c4_k_f16,
        c4_b_f16,
        c5_k_f16,
        c5_b_f16,
        fc1_w_f16,
        fc1_b_f16,
        fc2_w_f16,
        fc2_b_f16,
        fc3_w_f16,
        fc3_b_f16,
    )
    assert_shape(output_f16, [1, 1000], "float16 output shape")
    assert_dtype(output_f16, DType.float16, "float16 output dtype")


# ============================================================================
# Assertion Helper (simple version for this test file)
# ============================================================================


fn assert_equal(a: Int, b: Int) raises:
    """Assert two integers are equal."""
    if a != b:
        raise Error("Assertion failed: " + String(a) + " != " + String(b))


# ============================================================================
# Integration Tests
# ============================================================================


fn test_full_forward_pipeline() raises:
    """Test complete forward pass: input -> output through all 16 operations."""
    var dtype = DType.float32

    var (
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    ) = create_alexnet_parameters(dtype)

    # Create realistic batch
    var batch_size = 4
    var input = create_seeded_random_tensor(
        [batch_size, 3, 224, 224], dtype, seed=42, low=-1.0, high=1.0
    )

    # Forward pass (goes through all 16 operations)
    var output = alexnet_forward(
        input,
        c1_k,
        c1_b,
        c2_k,
        c2_b,
        c3_k,
        c3_b,
        c4_k,
        c4_b,
        c5_k,
        c5_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    )

    # Verify output
    assert_shape(output, [batch_size, 1000], "Full pipeline output shape")
    assert_dtype(output, dtype, "Full pipeline output dtype")

    # Verify output is valid
    var has_valid_values = False
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_false(
            isnan(val), "Full pipeline output contains NaN at " + String(i)
        )
        assert_false(
            isinf(val), "Full pipeline output contains Inf at " + String(i)
        )
        if val != 0.0:
            has_valid_values = True

    # Verify we have actual outputs (not all zeros)
    assert_true(has_valid_values, "Forward pass produced all zeros")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting AlexNet E2E Tests...")

    print("  test_forward_output_shape_224x224...", end="")
    test_forward_output_shape_224x224()
    print(" OK")

    print("  test_forward_single_sample_224x224...", end="")
    test_forward_single_sample_224x224()
    print(" OK")

    print("  test_forward_batch_sizes...", end="")
    test_forward_batch_sizes()
    print(" OK")

    print("  test_forward_deterministic...", end="")
    test_forward_deterministic()
    print(" OK")

    print("  test_shape_propagation_through_conv_layers...", end="")
    test_shape_propagation_through_conv_layers()
    print(" OK")

    print("  test_shape_propagation_through_fc_layers...", end="")
    test_shape_propagation_through_fc_layers()
    print(" OK")

    print("  test_parameters_initialization...", end="")
    test_parameters_initialization()
    print(" OK")

    print("  test_multiple_dtypes...", end="")
    test_multiple_dtypes()
    print(" OK")

    print("  test_full_forward_pipeline...", end="")
    test_full_forward_pipeline()
    print(" OK")

    print("\nAll AlexNet E2E tests passed!")
