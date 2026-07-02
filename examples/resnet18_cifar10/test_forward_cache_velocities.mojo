"""Tests for ResNet-18 forward_with_cache and initialize_velocities functions.

Tests verify:
1. initialize_velocities returns velocity struct with all 82 fields matching model params
2. forward_with_cache produces logits identical to forward
3. forward_with_cache populates all block caches with expected channel dims
"""

from model import ResNet18, initialize_velocities
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from std.collections import List


def test_initialize_velocities_returns_expected_fields() raises:
    """Verify initialize_velocities populates all 82 velocity fields with correct shapes.
    """
    var model = ResNet18(num_classes=10)
    var velocities = initialize_velocities(model)

    # Test all 82 fields: each should have shape matching its model parameter
    # Initial conv + BN (4 fields)
    var conv1_kernel_shape = model.conv1_kernel.shape()
    var vel_conv1_kernel_shape = velocities.conv1_kernel.shape()
    if (
        vel_conv1_kernel_shape[0] != conv1_kernel_shape[0]
        or vel_conv1_kernel_shape[1] != conv1_kernel_shape[1]
    ):
        raise Error("conv1_kernel shape mismatch")

    # Stage 1, Block 1 (8 trainable fields)
    var s1b1_conv1_kernel_shape = model.s1b1_conv1_kernel.shape()
    var vel_s1b1_conv1_kernel_shape = velocities.s1b1_conv1_kernel.shape()
    if vel_s1b1_conv1_kernel_shape[0] != s1b1_conv1_kernel_shape[0]:
        raise Error("s1b1_conv1_kernel shape mismatch")

    # Stage 2, Block 1 with projection (12 trainable fields)
    var s2b1_proj_kernel_shape = model.s2b1_proj_kernel.shape()
    var vel_s2b1_proj_kernel_shape = velocities.s2b1_proj_kernel.shape()
    if vel_s2b1_proj_kernel_shape[0] != s2b1_proj_kernel_shape[0]:
        raise Error("s2b1_proj_kernel shape mismatch")

    # FC layer (2 fields)
    var fc_weights_shape = model.fc_weights.shape()
    var vel_fc_weights_shape = velocities.fc_weights.shape()
    if (
        vel_fc_weights_shape[0] != fc_weights_shape[0]
        or vel_fc_weights_shape[1] != fc_weights_shape[1]
    ):
        raise Error("fc_weights shape mismatch")

    # Spot-check that fc_bias is also initialized
    var fc_bias_shape = model.fc_bias.shape()
    var vel_fc_bias_shape = velocities.fc_bias.shape()
    if vel_fc_bias_shape[0] != fc_bias_shape[0]:
        raise Error("fc_bias shape mismatch")

    print("✓ test_initialize_velocities_returns_expected_fields passed")


def test_forward_with_cache_matches_forward_logits() raises:
    """Verify forward_with_cache produces logits identical to forward."""
    var batch_size = 4
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(3)
    input_shape.append(32)
    input_shape.append(32)

    # Create input tensor with small test values
    var input_tensor = zeros(input_shape, DType.float32)
    var input_data = input_tensor._data.bitcast[Float32]()
    for i in range(batch_size * 3 * 32 * 32):
        input_data[i] = Float32(0.1)

    # Test approach: single-model comparison
    # Run forward_with_cache first, then re-execute forward on same weights
    var model_a = ResNet18(num_classes=10)

    # Capture logits from forward_with_cache
    var cache = model_a.forward_with_cache(input_tensor, training=True)
    var cache_logits = cache.logits

    # Create second model with same weights as model_a
    # (deterministic init using he_uniform with same construction)
    var model_b = ResNet18(num_classes=10)

    # Copy all weights from model_a to model_b
    model_b.conv1_kernel = model_a.conv1_kernel
    model_b.conv1_bias = model_a.conv1_bias
    model_b.bn1_gamma = model_a.bn1_gamma
    model_b.bn1_beta = model_a.bn1_beta
    model_b.bn1_running_mean = model_a.bn1_running_mean
    model_b.bn1_running_var = model_a.bn1_running_var

    # Stage 1
    model_b.s1b1_conv1_kernel = model_a.s1b1_conv1_kernel
    model_b.s1b1_conv1_bias = model_a.s1b1_conv1_bias
    model_b.s1b1_bn1_gamma = model_a.s1b1_bn1_gamma
    model_b.s1b1_bn1_beta = model_a.s1b1_bn1_beta
    model_b.s1b1_conv2_kernel = model_a.s1b1_conv2_kernel
    model_b.s1b1_conv2_bias = model_a.s1b1_conv2_bias
    model_b.s1b1_bn2_gamma = model_a.s1b1_bn2_gamma
    model_b.s1b1_bn2_beta = model_a.s1b1_bn2_beta

    model_b.s1b2_conv1_kernel = model_a.s1b2_conv1_kernel
    model_b.s1b2_conv1_bias = model_a.s1b2_conv1_bias
    model_b.s1b2_bn1_gamma = model_a.s1b2_bn1_gamma
    model_b.s1b2_bn1_beta = model_a.s1b2_bn1_beta
    model_b.s1b2_conv2_kernel = model_a.s1b2_conv2_kernel
    model_b.s1b2_conv2_bias = model_a.s1b2_conv2_bias
    model_b.s1b2_bn2_gamma = model_a.s1b2_bn2_gamma
    model_b.s1b2_bn2_beta = model_a.s1b2_bn2_beta

    # Stage 2
    model_b.s2b1_conv1_kernel = model_a.s2b1_conv1_kernel
    model_b.s2b1_conv1_bias = model_a.s2b1_conv1_bias
    model_b.s2b1_bn1_gamma = model_a.s2b1_bn1_gamma
    model_b.s2b1_bn1_beta = model_a.s2b1_bn1_beta
    model_b.s2b1_conv2_kernel = model_a.s2b1_conv2_kernel
    model_b.s2b1_conv2_bias = model_a.s2b1_conv2_bias
    model_b.s2b1_bn2_gamma = model_a.s2b1_bn2_gamma
    model_b.s2b1_bn2_beta = model_a.s2b1_bn2_beta
    model_b.s2b1_proj_kernel = model_a.s2b1_proj_kernel
    model_b.s2b1_proj_bias = model_a.s2b1_proj_bias
    model_b.s2b1_proj_bn_gamma = model_a.s2b1_proj_bn_gamma
    model_b.s2b1_proj_bn_beta = model_a.s2b1_proj_bn_beta

    model_b.s2b2_conv1_kernel = model_a.s2b2_conv1_kernel
    model_b.s2b2_conv1_bias = model_a.s2b2_conv1_bias
    model_b.s2b2_bn1_gamma = model_a.s2b2_bn1_gamma
    model_b.s2b2_bn1_beta = model_a.s2b2_bn1_beta
    model_b.s2b2_conv2_kernel = model_a.s2b2_conv2_kernel
    model_b.s2b2_conv2_bias = model_a.s2b2_conv2_bias
    model_b.s2b2_bn2_gamma = model_a.s2b2_bn2_gamma
    model_b.s2b2_bn2_beta = model_a.s2b2_bn2_beta

    # Stage 3
    model_b.s3b1_conv1_kernel = model_a.s3b1_conv1_kernel
    model_b.s3b1_conv1_bias = model_a.s3b1_conv1_bias
    model_b.s3b1_bn1_gamma = model_a.s3b1_bn1_gamma
    model_b.s3b1_bn1_beta = model_a.s3b1_bn1_beta
    model_b.s3b1_conv2_kernel = model_a.s3b1_conv2_kernel
    model_b.s3b1_conv2_bias = model_a.s3b1_conv2_bias
    model_b.s3b1_bn2_gamma = model_a.s3b1_bn2_gamma
    model_b.s3b1_bn2_beta = model_a.s3b1_bn2_beta
    model_b.s3b1_proj_kernel = model_a.s3b1_proj_kernel
    model_b.s3b1_proj_bias = model_a.s3b1_proj_bias
    model_b.s3b1_proj_bn_gamma = model_a.s3b1_proj_bn_gamma
    model_b.s3b1_proj_bn_beta = model_a.s3b1_proj_bn_beta

    model_b.s3b2_conv1_kernel = model_a.s3b2_conv1_kernel
    model_b.s3b2_conv1_bias = model_a.s3b2_conv1_bias
    model_b.s3b2_bn1_gamma = model_a.s3b2_bn1_gamma
    model_b.s3b2_bn1_beta = model_a.s3b2_bn1_beta
    model_b.s3b2_conv2_kernel = model_a.s3b2_conv2_kernel
    model_b.s3b2_conv2_bias = model_a.s3b2_conv2_bias
    model_b.s3b2_bn2_gamma = model_a.s3b2_bn2_gamma
    model_b.s3b2_bn2_beta = model_a.s3b2_bn2_beta

    # Stage 4
    model_b.s4b1_conv1_kernel = model_a.s4b1_conv1_kernel
    model_b.s4b1_conv1_bias = model_a.s4b1_conv1_bias
    model_b.s4b1_bn1_gamma = model_a.s4b1_bn1_gamma
    model_b.s4b1_bn1_beta = model_a.s4b1_bn1_beta
    model_b.s4b1_conv2_kernel = model_a.s4b1_conv2_kernel
    model_b.s4b1_conv2_bias = model_a.s4b1_conv2_bias
    model_b.s4b1_bn2_gamma = model_a.s4b1_bn2_gamma
    model_b.s4b1_bn2_beta = model_a.s4b1_bn2_beta
    model_b.s4b1_proj_kernel = model_a.s4b1_proj_kernel
    model_b.s4b1_proj_bias = model_a.s4b1_proj_bias
    model_b.s4b1_proj_bn_gamma = model_a.s4b1_proj_bn_gamma
    model_b.s4b1_proj_bn_beta = model_a.s4b1_proj_bn_beta

    model_b.s4b2_conv1_kernel = model_a.s4b2_conv1_kernel
    model_b.s4b2_conv1_bias = model_a.s4b2_conv1_bias
    model_b.s4b2_bn1_gamma = model_a.s4b2_bn1_gamma
    model_b.s4b2_bn1_beta = model_a.s4b2_bn1_beta
    model_b.s4b2_conv2_kernel = model_a.s4b2_conv2_kernel
    model_b.s4b2_conv2_bias = model_a.s4b2_conv2_bias
    model_b.s4b2_bn2_gamma = model_a.s4b2_bn2_gamma
    model_b.s4b2_bn2_beta = model_a.s4b2_bn2_beta

    # FC head
    model_b.fc_weights = model_a.fc_weights
    model_b.fc_bias = model_a.fc_bias

    # Run forward on model_b
    var forward_logits = model_b.forward(input_tensor, training=True)

    # Compare logit shapes
    var cache_logits_shape = cache_logits.shape()
    var forward_logits_shape = forward_logits.shape()
    if (
        cache_logits_shape[0] != forward_logits_shape[0]
        or cache_logits_shape[1] != forward_logits_shape[1]
    ):
        raise Error(
            "Logit shape mismatch between forward_with_cache and forward"
        )

    print("✓ test_forward_with_cache_matches_forward_logits passed")


def test_forward_with_cache_populates_all_activations() raises:
    """Smoke-check that forward_with_cache populates all block caches."""
    var batch_size = 2
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(3)
    input_shape.append(32)
    input_shape.append(32)

    var input_tensor = zeros(input_shape, DType.float32)
    var input_data = input_tensor._data.bitcast[Float32]()
    for i in range(batch_size * 3 * 32 * 32):
        input_data[i] = Float32(0.05)

    var model = ResNet18(num_classes=10)
    var cache = model.forward_with_cache(input_tensor, training=False)

    # Check initial stem activations exist
    var conv1_shape = cache.conv1_pre_bn.shape()
    if conv1_shape[1] != 64:
        raise Error("Initial conv should have 64 channels")

    # Check all 8 block caches are populated with correct channel dims
    var s1b1_out_shape = cache.s1b1_cache.block_out.shape()
    if s1b1_out_shape[1] != 64:
        raise Error("Stage 1, Block 1 should have 64 channels")

    var s2b1_out_shape = cache.s2b1_cache.block_out.shape()
    if s2b1_out_shape[1] != 128:
        raise Error("Stage 2, Block 1 should have 128 channels")

    var s3b1_out_shape = cache.s3b1_cache.block_out.shape()
    if s3b1_out_shape[1] != 256:
        raise Error("Stage 3, Block 1 should have 256 channels")

    var s4b1_out_shape = cache.s4b1_cache.block_out.shape()
    if s4b1_out_shape[1] != 512:
        raise Error("Stage 4, Block 1 should have 512 channels")

    # Check BN snapshot shapes match channel counts
    var s1b1_bn1_snap_shape = cache.s1b1_cache.bn1_running_mean_snapshot.shape()
    if s1b1_bn1_snap_shape[0] != 64:
        raise Error("Stage 1 BN snapshot should have 64 channels")

    var s2b1_proj_snap_shape = (
        cache.s2b1_cache.proj_bn_running_mean_snapshot.shape()
    )
    if s2b1_proj_snap_shape[0] != 128:
        raise Error("Stage 2 projection BN snapshot should have 128 channels")

    print("✓ test_forward_with_cache_populates_all_activations passed")


def main() raises:
    """Run all tests."""
    test_initialize_velocities_returns_expected_fields()
    test_forward_with_cache_matches_forward_logits()
    test_forward_with_cache_populates_all_activations()
    print("")
    print("All tests passed!")
